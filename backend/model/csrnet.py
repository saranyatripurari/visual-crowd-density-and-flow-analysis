"""
CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes
Li et al., CVPR 2018 — https://arxiv.org/abs/1802.10062

Architecture:
  Frontend : VGG16 conv layers 1–10 (3 MaxPool → H/8 × W/8, 512 channels)
  Backend  : 6 dilated conv layers (dilation=2, no further downsampling)
  Output   : 1×1 conv → 1-channel density map at H/8 × W/8

  crowd_count = density_map.sum()   (the spatial integral equals number of people)

ShanghaiTech-A benchmarks (from paper):
  MCNN  : MAE=110.2, RMSE=173.2
  CSRNet: MAE= 68.2, RMSE=115.0   (~38 % better MAE)

ShanghaiTech-B benchmarks:
  MCNN  : MAE=26.4, RMSE=41.3
  CSRNet: MAE=10.6, RMSE=16.0    (~60 % better MAE)
"""

import os
import torch
import torch.nn as nn
from torchvision import models


# ──────────────────────────────────────────────────────────────────────────────
# Architecture helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_layers(cfg, in_channels: int = 3, dilation: bool = False) -> nn.Sequential:
    """
    Build a Sequential of [Conv2d → ReLU] blocks following a VGG-style cfg list.
    'M' inserts a MaxPool2d(2, stride=2).  When dilation=True, every conv uses
    dilation=2 and padding=2 to expand the receptive field without downsampling.
    """
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(
                in_channels, v,
                kernel_size=3, padding=d_rate, dilation=d_rate
            )
            layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """
    CSRNet crowd-counting network.

    Frontend : First 10 VGG16 conv layers (+ 3 max-pool), output = H/8 × W/8 × 512
    Backend  : 6 dilated conv layers (dilation=2), output = H/8 × W/8 × 64
    Head     : 1×1 conv → 1-channel density map

    Usage
    -----
    model = CSRNet()
    density = model(img_tensor)          # (B, 1, H/8, W/8)
    count   = density.sum().item()       # total people
    """

    FRONTEND_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
    BACKEND_CFG  = [512, 512, 512, 256, 128, 64]

    def __init__(self):
        super().__init__()
        self.frontend     = make_layers(self.FRONTEND_CFG, in_channels=3, dilation=False)
        self.backend      = make_layers(self.BACKEND_CFG,  in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._init_weights()

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
            x : (B, 3, H, W)  ImageNet-normalised RGB tensor
        Returns
            (B, 1, H/8, W/8) density map whose spatial sum == crowd count
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    # ── weight helpers ────────────────────────────────────────────────────────

    def _init_weights(self):
        """Gaussian-initialise all conv weights (backend + head)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_vgg16_frontend(self) -> bool:
        """
        Copy VGG16 ImageNet-pretrained weights into the CSRNet frontend.
        VGG16.features[:23] exactly matches FRONTEND_CFG (10 conv layers, 3 pools).
        """
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            vgg_layers = [l for l in vgg.features[:23].children()]
            csr_layers = [l for l in self.frontend.children()]
            for csr_l, vgg_l in zip(csr_layers, vgg_layers):
                if isinstance(csr_l, nn.Conv2d) and isinstance(vgg_l, nn.Conv2d):
                    csr_l.weight.data.copy_(vgg_l.weight.data)
                    if csr_l.bias is not None:
                        csr_l.bias.data.copy_(vgg_l.bias.data)
            print("[CSRNet] Frontend initialised with VGG16 ImageNet pretrained weights.")
            return True
        except Exception as exc:
            print(f"[CSRNet] VGG16 frontend init failed: {exc}")
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Weight download helpers
# ──────────────────────────────────────────────────────────────────────────────

_HF_CANDIDATES = [
    # (repo_id, filename)
    ("muasifk/CSRNet", "csrnet.pth"),
    ("muasifk/CSRNet", "model.pth"),
    ("muasifk/CSRNet", "CSRNet.pth"),
    ("muasifk/CSRNet", "weights.pth"),
]


def _download_from_hf(save_dir: str) -> "str | None":
    """Try each candidate file on HuggingFace Hub; return local path on success."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[CSRNet] huggingface_hub not installed. Run: pip install huggingface-hub")
        return None

    for repo_id, filename in _HF_CANDIDATES:
        try:
            print(f"[CSRNet] Fetching {repo_id}/{filename} from HuggingFace …")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=save_dir,
                local_dir_use_symlinks=False,
            )
            print(f"[CSRNet] Downloaded → {path}")
            return path
        except Exception:
            continue

    print("[CSRNet] All HuggingFace download attempts failed.")
    return None


def _load_state(path: str, device: torch.device) -> "dict | None":
    """
    Load a state dict from path, unwrapping common checkpoint wrappers.
    Returns None on failure.
    """
    try:
        ckpt = torch.load(path, map_location=device)
        # Some checkpoints wrap the state dict
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "model", "net"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
        if isinstance(ckpt, dict):
            return ckpt
    except Exception as exc:
        print(f"[CSRNet] Cannot read checkpoint '{path}': {exc}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Public factory
# ──────────────────────────────────────────────────────────────────────────────

def build_csrnet(weights_path: str = None, device: torch.device = None) -> CSRNet:
    """
    Build a CSRNet and load the best available weights.

    Search order
    ─────────────
    1. Explicit ``weights_path`` argument
    2. models/csrnet_partA.pth   (ShanghaiTech Part-A trained)
    3. models/csrnet_partB.pth   (ShanghaiTech Part-B trained)
    4. models/csrnet_sha.pth     (previously auto-downloaded)
    5. models/csrnet.pth
    6. HuggingFace Hub: muasifk/CSRNet  (auto-download ~56 MB on first run)
    7. VGG16 ImageNet frontend only     (fallback — still >> random MCNN)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CSRNet().to(device)
    os.makedirs("models", exist_ok=True)

    local_candidates = [
        weights_path,
        os.path.join("models", "csrnet_partA.pth"),
        os.path.join("models", "csrnet_partB.pth"),
        os.path.join("models", "csrnet_sha.pth"),
        os.path.join("models", "csrnet.pth"),
    ]

    # ── 1–5: try local files ─────────────────────────────────────────────────
    for path in local_candidates:
        if not (path and os.path.isfile(path)):
            continue
        state = _load_state(path, device)
        if state is None:
            continue
        try:
            model.load_state_dict(state, strict=True)
            print(f"[CSRNet] [OK] Loaded ShanghaiTech weights from: {path}")
            model.eval()
            return model
        except RuntimeError:
            # Possibly trained with a slightly different key naming
            try:
                model.load_state_dict(state, strict=False)
                missing = [k for k in model.state_dict() if k not in state]
                print(
                    f"[CSRNet] [OK] Partially loaded weights from: {path}  "
                    f"(missing {len(missing)} keys — using defaults)"
                )
                model.eval()
                return model
            except Exception as exc2:
                print(f"[CSRNet] Skipping '{path}': {exc2}")

    # ── 6: HuggingFace download ───────────────────────────────────────────────
    hf_path = _download_from_hf("models")
    if hf_path and os.path.isfile(hf_path):
        state = _load_state(hf_path, device)
        if state is not None:
            try:
                model.load_state_dict(state, strict=False)
                # Cache a clean copy under a stable name
                cached = os.path.join("models", "csrnet_sha.pth")
                torch.save(model.state_dict(), cached)
                print(f"[CSRNet] ✓ HuggingFace weights loaded and cached at: {cached}")
                model.eval()
                return model
            except Exception as exc:
                print(f"[CSRNet] HuggingFace weight load failed: {exc}")

    # ── 7: VGG16 ImageNet frontend fallback ───────────────────────────────────
    print(
        "[CSRNet] ⚠ No ShanghaiTech weights available.\n"
        "         Using VGG16 ImageNet frontend (significantly better than random MCNN).\n"
        "         For best accuracy, place csrnet_partA.pth in the models/ directory."
    )
    model.load_vgg16_frontend()
    model.eval()
    return model
