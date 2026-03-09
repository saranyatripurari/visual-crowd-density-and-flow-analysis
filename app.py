from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from scipy.ndimage import gaussian_filter

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "mp4", "avi", "mov"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ================= HEAD DETECTION (IMPROVED & STABLE) =================
def detect_people_heads(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    h, w = gray.shape

    max_corners = int((h * w) / 4000)
    max_corners = max(30, min(max_corners, 500))

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.03,
        minDistance=10,
        blockSize=5,
        useHarrisDetector=False
    )

    if corners is None:
        return []

    points = [(int(x), int(y)) for x, y in corners.reshape(-1, 2)]

    # remove close duplicates
    filtered = []
    for (x, y) in points:
        keep = True
        for (fx, fy) in filtered:
            if np.hypot(x - fx, y - fy) < 12:
                keep = False
                break
        if keep:
            filtered.append((x, y))

    return filtered


# ================= GAUSSIAN KERNEL =================
def generate_adaptive_gaussian_kernel(size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    return kernel @ kernel.T


# ================= GT DENSITY MAP (STABLE HEATMAP) =================
def generate_gt_density_map(img, sigma=None):
    height, width = img.shape[:2]

    if sigma is None:
        sigma = max(height, width) / 150

    points = detect_people_heads(img)

    density_map = np.zeros((height, width), dtype=np.float32)

    for (x, y) in points:
        if 0 <= x < width and 0 <= y < height:
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1

            y_min = max(0, int(y - kernel_size // 2))
            y_max = min(height, int(y + kernel_size // 2) + 1)
            x_min = max(0, int(x - kernel_size // 2))
            x_max = min(width, int(x + kernel_size // 2) + 1)

            kernel = generate_adaptive_gaussian_kernel(kernel_size, sigma)

            k_y_min = y_min - (int(y - kernel_size // 2))
            k_x_min = x_min - (int(x - kernel_size // 2))
            k_y_max = k_y_min + (y_max - y_min)
            k_x_max = k_x_min + (x_max - x_min)

            density_map[y_min:y_max, x_min:x_max] += kernel[k_y_min:k_y_max, k_x_min:k_x_max]

    density_map = gaussian_filter(density_map, sigma=0.8)

    if density_map.max() > 0:
        density_map = density_map / density_map.max()

    density_vis = (density_map * 255).astype(np.uint8)
    density_colored = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)

    crowd_count = len(points)

    return density_colored, crowd_count, density_map, points


# ================= POINT MAP =================
def generate_point_map(img, points):
    h, w = img.shape[:2]
    point_map = np.zeros((h, w, 3), dtype=np.uint8)

    for (x, y) in points:
        cv2.circle(point_map, (x, y), 3, (255, 255, 255), -1)

    return point_map


# ================= OVERLAY =================
def overlay_detections(img, points):
    overlay = img.copy()

    for (x, y) in points:
        cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(overlay, (x, y), 8, (0, 255, 0), 2)

    return overlay


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html")

    if not allowed_file(file.filename):
        return render_template("index.html")

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    ext = filename.rsplit(".", 1)[1].lower()
    is_video = ext in ["mp4", "avi", "mov"]

    # ================= IMAGE =================
    if not is_video:
        img = cv2.imread(input_path)

        if img is None:
            return render_template("index.html")

        density_img, crowd_count, density_map, points = generate_gt_density_map(img)

        overlay_img = overlay_detections(img, points)
        point_map = generate_point_map(img, points)

        density_path = os.path.join(OUTPUT_FOLDER, f"density_{filename}")
        overlay_path = os.path.join(OUTPUT_FOLDER, f"overlay_{filename}")
        point_path = os.path.join(OUTPUT_FOLDER, f"points_{filename}")

        cv2.imwrite(density_path, density_img)
        cv2.imwrite(overlay_path, overlay_img)
        cv2.imwrite(point_path, point_map)

        flow_text = "Static Image"
        output_display_path = density_path

    # ================= VIDEO =================
    else:
        cap = cv2.VideoCapture(input_path)
        ret, frame1 = cap.read()

        if not ret:
            cap.release()
            return render_template("index.html")

        density_img, crowd_count, density_map, points = generate_gt_density_map(frame1)

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        ret, frame2 = cap.read()
        if not ret:
            frame2 = frame1.copy()

        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        fx, fy = flow[..., 0], flow[..., 1]
        mean_x = np.mean(fx)
        mean_y = np.mean(fy)

        if abs(mean_x) > abs(mean_y):
            flow_text = "→ Moving Right" if mean_x > 0 else "← Moving Left"
        else:
            flow_text = "↓ Moving Down" if mean_y > 0 else "↑ Moving Up"

        h, w = density_img.shape[:2]
        start = (w // 2, h // 2)
        end = (int(start[0] + mean_x * 40), int(start[1] + mean_y * 40))

        cv2.arrowedLine(density_img, start, end, (255, 255, 255), 3)
        cv2.putText(density_img, flow_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(density_img, f"Crowd: {crowd_count}", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        density_path = os.path.join(OUTPUT_FOLDER, "video_density_output.jpg")
        cv2.imwrite(density_path, density_img)
        output_display_path = density_path

        cap.release()

    return render_template(
        "index.html",
        input_path=input_path,
        output_path=output_display_path,
        crowd_count=crowd_count,
        is_video=is_video,
        flow_text=flow_text if is_video else "Static Image"
    )


if __name__ == "__main__":
    app.run(debug=True)
