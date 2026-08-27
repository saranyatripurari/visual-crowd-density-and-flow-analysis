# Visual Crowd Density and Flow Analysis

A FastAPI application for crowd counting from images and crowd movement analysis from videos.

## Structure

```text
crowdDensity/
├── frontend/
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       └── js/app.js
├── backend/
│   ├── app.py                  # API and frontend connection
│   ├── inference.py            # Image analysis and density-map output
│   ├── video_crowd_system.py   # Video detection, tracking, and flow analysis
│   ├── density_map.py          # Density-map utilities
│   └── model/csrnet.py         # CSRNet architecture and loader
├── models/                     # Model checkpoints
├── part_A_final/               # ShanghaiTech Part A data
├── part_B_final/               # ShanghaiTech Part B data
├── outputs/                    # Runtime and test output files
├── requirements.txt
└── README.md
```

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
python -m backend.app
```

Open `http://127.0.0.1:8000` in a browser.

## API

- `GET /health` checks the server and loaded models.
- `POST /predict/image` accepts JPG, JPEG, PNG, or WEBP files.
- `POST /predict/video` accepts MP4, AVI, MOV, MKV, or WEBM files.

The image pipeline generates a people count, density map, and annotated output. The video pipeline reports the estimated crowd count, movement status, and dominant flow direction.

## Notes

`model/` contains source code. `models/` contains trained or downloaded weights. `outputs/` is the single location for generated results. Training and evaluation scripts are not included in the compact runtime version.
