<<<<<<< HEAD
# 🚀 IdVest AI

**IdVest AI** is an AI-powered surveillance system designed to monitor ID card presence and dress code compliance in real time using computer vision.

It provides intelligent detection, automated violation capture, and a modern dashboard for monitoring and reporting.

---

## ✨ Key Features

* 🎥 Real-time camera monitoring
* 🪪 ID card detection (Missing / Present)
* 👔 Dress code compliance detection
* 📸 Automatic violation image capture
* 🚨 Alert and logging system
* 📊 Live dashboard with statistics
* 📄 PDF report generation

---

## 🧠 Tech Stack

* **Backend:** FastAPI
* **AI Model:** YOLOv8
* **Computer Vision:** OpenCV
* **Frontend:** HTML, CSS, JavaScript
* **Reporting:** ReportLab
=======
# 🎓 IDVest AI — Smart Dress Code & ID Compliance System

A real-time AI-powered system that monitors dress code compliance and ID card usage in a college environment using computer vision.

---

## 🚀 Features

* 👤 **Person Detection (YOLOv8)**
* 🧠 **Unique Person Tracking** (no duplicate detections)
* 👔 **Attire Classification**

  * Formal (blazer / formal shirt)
  * Faculty (casual red t-shirt)
  * Not Formal
* 🪪 **ID Card Detection**

  * Student ID validation
  * Faculty ID validation
* ⚠️ **Violation Detection**

  * Missing ID
  * Improper attire
* 📊 **Real-time Dashboard UI**
* 📧 **Alert System (Optional Email Notifications)**

---

## 🧠 System Architecture

Camera → YOLO Detection → Tracking → Logic Engine → Dashboard + Alerts

---

## 📁 Project Structure

```
IDVestAI/
│
├── app/                  # Core modules
│   ├── detector/         # YOLO detection
│   ├── logic/            # Dress code + compliance logic
│   ├── utils/            # Alerts, helpers
│
├── dataset/              # Training dataset
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── models/               # Trained models (.pt)
│   └── idvest_best.pt
│
├── dashboard/            # UI (HTML / Streamlit)
├── logs/                 # Detection logs
├── tests/                # Unit tests
│
├── main.py               # Main app
├── detector.py           # YOLO wrapper
├── config.py             # Configuration
├── api_server.py         # FastAPI backend
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/idvest-ai.git
cd idvest-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
>>>>>>> 1a34ec5e4105418bddedca8714520744d63ed80c

---

## ▶️ Run the Project

<<<<<<< HEAD
```bash
pip install -r requirements.txt
uvicorn api_server:app --reload
```

Open:

```
http://127.0.0.1:8000
=======
### Run main detection system:

```bash
python main.py
```

### OR run API server:

```bash
uvicorn api_server:app --reload
```

### OR run Streamlit UI (recommended):

```bash
streamlit run app.py
>>>>>>> 1a34ec5e4105418bddedca8714520744d63ed80c
```

---

<<<<<<< HEAD
## 🌐 Main Endpoints

* `/` → Dashboard
* `/video` → Live stream
* `/stats` → Real-time stats
* `/logs_json` → Logs
* `/report` → Download PDF report

---

## 📸 Violation System

When a violation is detected:

* Image is captured automatically
* Stored in `/violations/`
* Logged and reported

---

## 📌 Overview

IdVest AI is designed as a **smart compliance monitoring system** for environments where identity verification and dress standards are essential.

---

## 👤 Author

**IDVestAI Pro** — Academic Computer Vision Project  
Built with: Python · YOLOv8 · OpenCV · FastAPI · HTML/CSS/JS
=======
## 🧪 Model Training (Optional)

```bash
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
```

---

## 📌 Detection Logic

### 👔 Attire Rules

* Blazer → Formal
* Formal Shirt → Formal
* Faculty Casual (Red T-shirt) → Faculty
* Otherwise → Not Formal

### 🪪 ID Rules

* Students must wear **ID Card**
* Faculty must wear **Faculty ID**

### 📅 Monday Rule

* Faculty must wear **red casual t-shirt**

---

## 🔥 Key Improvements

* ✅ Prevents duplicate detection using tracking
* ✅ Reduces false positives using confidence filtering
* ✅ Optimized for real-time CPU performance
* ✅ Modular and scalable architecture

---

## 📸 Demo

* Live webcam detection
* Bounding boxes with labels
* Real-time compliance status
* Dashboard metrics

---

## 🛠️ Tech Stack

* Python 🐍
* YOLOv8 (Ultralytics)
* OpenCV
* Streamlit / FastAPI
* NumPy

---

## ⚠️ Known Limitations

* Model accuracy depends on training data
* Lighting conditions affect detection
* Simple tracking (can be improved with DeepSORT)

---

## 🚀 Future Enhancements

* Face recognition for identity tracking
* Database integration (attendance system)
* Advanced tracking (DeepSORT / ByteTrack)
* Mobile app integration

---

## 👨‍💻 Author

**Priyanshi Dwivedi**
BTech IT Student

---

## 📜 License

This project is for academic purposes only.
>>>>>>> 1a34ec5e4105418bddedca8714520744d63ed80c
