# 🏍️ Two-Wheeler Traffic Violation Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)
![EasyOCR](https://img.shields.io/badge/EasyOCR-License%20Plate-00C853?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

**An AI-powered computer vision pipeline that detects traffic rule violations by two-wheeler riders using YOLOv8 and EasyOCR.**

</div>

---

## 📌 Overview

This project is a **Final Year Project (FYP)** that automates traffic violation detection for two-wheeler riders. It processes video footage frame by frame using three independently trained YOLOv8 models combined into a unified inference pipeline.

The system detects violations, identifies the offending motorcycle, and reads its license plate — all automatically.

---

## 🚦 Violations Detected

| Violation | Description |
|-----------|-------------|
| 🔴 **Wrong Lane** | Motorcycle is rear-facing (driving away from camera / against traffic flow) |
| 🟠 **No Helmet** | One or more riders detected without a helmet |
| 🟡 **Triple Riding** | More than two riders on a single motorcycle |

---

## 🧠 Pipeline Flow

```
INPUT VIDEO FRAME
        │
        ▼
┌─────────────────────┐
│   Helmet Model      │ ──► Detects motorcyclists, helmets, license plates
└────────┬────────────┘
         │  for each motorcycle crop
    ┌────┴──────────────────┐
    ▼                       ▼
┌──────────┐        ┌──────────────┐
│Lane Model│        │  Face Model  │ + Helmet Model (reused)
└────┬─────┘        └──────┬───────┘
     │                     │
rear-facing?         count faces & helmets
     │                     │
Wrong Lane ✗        Violation Logic
                  ┌─────────────────────┐
                  │ No Helmet?          │
                  │ Triple Riding?      │
                  └────────┬────────────┘
                           │ if any violation
                           ▼
                  ┌─────────────────────┐
                  │  EasyOCR (plate)    │
                  └────────┬────────────┘
                           ▼
                  Save image + plate + log
```

---

## 📁 Project Structure

```
finalyearproject/
│
├── models/
│   ├── helmet_best.pt          ← Trained: motorcyclist, helmet, license_plate
│   ├── face_best.pt            ← Trained: face
│   └── lane_best.pt            ← Trained: front-facing, rear-facing motorcycle
│
├── datasets/
│   ├── helmet_detection/       ← Dataset 1 (YOLOv8 format)
│   ├── face_detection/         ← Dataset 2
│   └── wrong_lane/             ← Dataset 3
│
├── videos/                     ← Input video files go here
│
├── output/
│   ├── violations/             ← Cropped violation images + .txt logs
│   └── plates/                 ← Cropped license plate images
│
├── config.py                   ← All paths, class indices, thresholds
├── detector.py                 ← Model loading + inference wrapper
├── violation_checker.py        ← Violation logic (helmet, lane, triple)
├── ocr_handler.py              ← EasyOCR license plate reading
├── utils.py                    ← IoU, file saving, helpers
├── main.py                     ← Entry point — processes full video
├── test_single.py              ← Quick test on a single image
└── checkclasses.py             ← Prints class names of each loaded model
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA (recommended — GTX 1650 or better)
- CUDA 11.8+ installed

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/two-wheeler-violation-detection.git
cd two-wheeler-violation-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# With GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install the rest
pip install ultralytics easyocr opencv-python numpy requests
```

> **No GPU?** Skip the first line and just `pip install torch torchvision torchaudio`. Then set `OCR_GPU = False` in `config.py`. Inference will work, just slower.

### 4. Verify GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### 5. Add Model Weights

Place your trained `.pt` files in the `models/` folder:

```
models/
├── helmet_best.pt
├── face_best.pt
└── lane_best.pt
```

> Don't have weights yet? See [Training Your Own Models](#-training-your-own-models).

---

## 🚀 Running the Project

### ▶ Process a Video

```bash
python main.py --video videos/your_video.mp4
```

Results saved to `output/violations/` and `output/plates/`.

### 🖼️ Test on a Single Image

```bash
python test_single.py
```

Edit `image_path` inside `test_single.py` to point to your image.

### 🔍 Check Model Class Indices

```bash
python checkclasses.py
```

Always run this after adding new `.pt` files to verify `config.py` indices are correct.

---

## 🔧 Configuration (`config.py`)

```python
# Model paths
HELMET_MODEL = "models/helmet_best.pt"
FACE_MODEL   = "models/face_best.pt"
LANE_MODEL   = "models/lane_best.pt"

# Class indices — verify with checkclasses.py after training
HELMET_CLASS = {'helmet': 0, 'license_plate': 1, 'motorcyclist': 2}
FACE_CLASS   = {'face': 0}
LANE_CLASS   = {'front': 0, 'rear': 1}

# Detection thresholds
CONF_THRESHOLD    = 0.4   # minimum confidence for a valid detection
OVERLAP_THRESHOLD = 0.6   # face-helmet IoU to avoid double counting
FRAME_SKIP        = 5     # process every Nth frame (speed vs accuracy)

# EasyOCR
OCR_GPU = True            # set False if VRAM runs out
```

> ⚠️ **Common mistake:** mismatched class indices between `config.py` and your trained model will cause all violations to fire incorrectly. Always run `checkclasses.py` first.

---

## 🏋️ Training Your Own Models

### Dataset Format (YOLOv8)

```
dataset_name/
├── train/images/ & train/labels/
├── valid/images/ & valid/labels/
└── data.yaml
```

### `data.yaml` Example

```yaml
path: /absolute/path/to/dataset
train: train/images
val: valid/images

nc: 3
names: ['helmet', 'license_plate', 'motorcyclist']
```

### Training Commands

Run each separately — one terminal session per model:

```bash
# Model 1 — Helmet Detection
yolo task=detect mode=train model=yolov8n.pt \
  data=datasets/helmet_detection/data.yaml \
  epochs=50 imgsz=640 batch=16 device=0 \
  project=runs name=helmet_model

# Model 2 — Face Detection
yolo task=detect mode=train model=yolov8n.pt \
  data=datasets/face_detection/data.yaml \
  epochs=50 imgsz=640 batch=16 device=0 \
  project=runs name=face_model

# Model 3 — Lane / Orientation Detection
yolo task=detect mode=train model=yolov8n.pt \
  data=datasets/wrong_lane/data.yaml \
  epochs=50 imgsz=640 batch=16 device=0 \
  project=runs name=lane_model
```

After training, copy weights to `models/`:

```bash
cp runs/helmet_model/weights/best.pt models/helmet_best.pt
cp runs/face_model/weights/best.pt   models/face_best.pt
cp runs/lane_model/weights/best.pt   models/lane_best.pt
```

> Running out of VRAM? Try `batch=8` or `batch=4`.

---

## 📊 Output Format

For each detected violation:

```
output/
├── violations/
│   ├── frame_00042_moto.jpg     ← cropped motorcycle region
│   └── frame_00042_info.txt     ← violation details
└── plates/
    └── frame_00042_plate.jpg    ← cropped license plate
```

**Sample `info.txt`:**
```
Violations: Wrong Lane, No Helmet
Plate Text: MH12AB1234
```

---

## 🧩 Model Roles

| Model | Classes | Used For |
|-------|---------|----------|
| `helmet_best.pt` | `motorcyclist`, `helmet`, `license_plate` | Detecting bikes, counting helmets, locating plates |
| `face_best.pt` | `face` | Counting riders by visible faces |
| `lane_best.pt` | `front`, `rear` | Determining riding direction |

> The helmet model is called **3 times** per motorcycle: once for motorcycle detection, once for helmet counting, once for plate localization. Same weights, different class filters each call.

---

## ⚡ Performance Notes

- `FRAME_SKIP = 5` processes every 5th frame — good balance of speed and coverage
- OCR requires plate crops of at least ~80px wide — low-res footage will return `None`
- On a **GTX 1650 (4GB VRAM)**: expect ~15–20 FPS inference with all 3 models
- If VRAM is tight: set `OCR_GPU = False` — EasyOCR runs on CPU, YOLO stays on GPU

---

## 🛠️ Tech Stack

| Tool | Role |
|------|------|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | License plate OCR |
| [OpenCV](https://opencv.org/) | Video processing & image I/O |
| [PyTorch](https://pytorch.org/) | Deep learning backend |
| [Roboflow](https://roboflow.com/) | Dataset management |

---

## 🙋 FAQ

**Violations firing incorrectly even though riders have helmets?**
Run `python checkclasses.py` and make sure the indices in `config.py` match your model's actual output. This is the #1 bug.

**Plate text always `None`?**
Expected for low-resolution footage. Plates must be at least ~80px wide for EasyOCR to read them. Use higher-resolution input.

**CUDA out of memory during training?**
Lower batch size: `batch=8` or `batch=4`.

**Can I run without a GPU?**
Yes. Set `OCR_GPU = False` in config and remove `device=0` from training commands. Slower but works.

---

## 👥 Team

| Name | Role |
|------|------|
| **Aryan** | Pipeline architecture, model integration, OCR, full codebase |
| *Teammate 2* | *Add contribution* |
| *Teammate 3* | *Add contribution* |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [Pratham Jaiswal](https://github.com/prathamjaiswal) — original Two Wheeler Lane Detection dataset concept
- **Arnav Rawat** and **Shubham Sharma** — dataset co-contributors
- [Roboflow](https://roboflow.com/) for dataset hosting and annotation

---

<div align="center">

Made with 💻 + ☕ as a Final Year Project

*Star ⭐ the repo if this helped you!*

</div>