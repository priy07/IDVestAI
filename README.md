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

---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
uvicorn api_server:app --reload
```

Open:

```
http://127.0.0.1:8000
```

---

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
