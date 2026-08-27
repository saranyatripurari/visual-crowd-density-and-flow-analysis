# Visual Crowd Density and Flow Analysis

An end-to-end **Computer Vision and Deep Learning web application** that performs crowd counting from images and crowd movement analysis from videos.

## Project Overview

This project is designed to analyze crowded images and videos using deep learning and computer vision techniques.

The image analysis component predicts the number of people in an image and generates a crowd density map.

The video analysis component detects and tracks people and analyzes their movement using optical flow and tracking techniques.

The application provides an interactive web interface where users can upload images or videos and view the analysis results.

## Features

- Crowd counting from images
- Crowd density map generation
- Image analysis using CSRNet
- Video crowd detection
- Person tracking
- Crowd movement analysis
- Optical flow-based movement analysis
- Stationary movement detection
- Left, Right, Forward and Backward movement analysis when significant movement is detected
- Image and video upload through web interface
- Generated analysis results

## Machine Learning & Computer Vision

The project uses **CSRNet** for crowd counting and density-map generation.

For video analysis, the system uses computer vision techniques including:

- Person Detection
- Object Tracking
- Optical Flow
- Movement Analysis

The video system is designed to avoid treating small movements as dominant directional movement. When people are mostly stationary, the output is considered **Stationary**.

## Dataset

The crowd counting model is based on the **ShanghaiTech Crowd Counting Dataset**.

The project includes:

- ShanghaiTech Part A
- ShanghaiTech Part B

## Tech Stack

### Programming Language

- Python

### Deep Learning & Computer Vision

- PyTorch
- OpenCV
- CSRNet
- NumPy
- SciPy

### Backend

- Flask

### Frontend

- HTML5
- CSS3
- JavaScript

## Project Structure

```text
Visual-Crowd-Density-and-Flow-Analysis/
│
├── frontend/
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
│
├── backend/
│   ├── app.py
│   ├── inference.py
│   ├── video_crowd_system.py
│   ├── density_map.py
│   └── model/
│       └── csrnet.py
│
├── models/
│
├── part_A_final/
│
├── part_B_final/
│
├── outputs/
│
├── requirements.txt
├── .gitignore
└── README.md
