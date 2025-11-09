# 🌿 Plant Disease Detection using Deep Learning  

### 🔬 Developed at **NIT Jamshedpur (NIT-JSR)**  

This project focuses on **real-time plant disease detection** using advanced **Deep Learning** and **Hybrid Machine Learning** models.  
The system is capable of identifying various plant leaf diseases (such as **potato leaf diseases**) with **95%+ accuracy**, enabling early diagnosis and improving agricultural productivity.  

---

## 🚀 Project Overview  

This repository contains multiple models and approaches for detecting plant leaf diseases using **CNN-based architectures** like:  
- **MobileNetV2**  
- **EfficientNetV2-B3 / B4**  
- **Hybrid Model (EfficientNet + LightGBM / XGBoost)**  

These models are trained on leaf image datasets to classify plant health status — e.g., **Healthy**, **Early Blight**, **Late Blight**, etc.  

---

## 🧠 Model Highlights  

| Model | Type | Accuracy | Description |
|-------|------|-----------|--------------|
| MobileNetV2 | CNN | ~94% | Lightweight model ideal for real-time detection (Raspberry Pi compatible) |
| EfficientNetB2/B3/B4 | CNN | 95–98% | High-accuracy model with better feature extraction |
| Hybrid Model | CNN + ML (LightGBM/XGBoost) | 96–98% | Combines deep features with ML classifiers for robust results |

---

## ⚙️ Tech Stack  

**Machine Learning / Deep Learning:**  
- TensorFlow / Keras  
- Scikit-learn  
- LightGBM, XGBoost  

**Backend (for Deployment):**  
- FastAPI / Flask  
- TensorFlow Serving  

**Frontend (for Visualization):**  
- HTML, CSS, JavaScript  
- Real-time integration with Raspberry Pi camera module  

**Other Tools:**  
- Google Colab / Kaggle for training  
- Firebase / Google Cloud for deployment and data storage  

---

## 🧩 Features  

✅ Detects diseases from **real-time or uploaded leaf images**  
✅ Supports **multiple CNN architectures**  
✅ Works on **Raspberry Pi / edge devices**  
✅ Provides **JSON API for integration** with websites or mobile apps  
✅ **Hybrid model** combines CNN + ML classifier for enhanced performance  
✅ **Grad-CAM visualization** for feature interpretability  

---

## 📊 Results  

- Achieved **95%+ overall accuracy**  
- High precision and recall across all disease classes  
- Optimized for **low latency inference** on edge devices  

---

## 🧪 How to Run  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/<your-username>/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```
### 2️⃣ Install dependencies & Run FastApi Backend
```bash
pip install -r requirements.txt
uvicorn main:app --reload
http://127.0.0.1:8000
```
