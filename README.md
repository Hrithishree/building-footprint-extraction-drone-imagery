# 🏗️ Building Footprint Extraction from Drone Imagery

## 📌 Overview
This project presents an AI-based pipeline for extracting building footprints from high-resolution aerial orthophotos using DeepLabV3+.

The system converts raw drone imagery into clean, structured building boundaries suitable for geospatial applications such as rural planning and asset mapping.

---

## 🚀 Pipeline

Orthophoto → DeepLabV3+ → Threshold (0.4) → Post-processing → Edge Refinement → Polygonization → Final Output


---

## 🧠 Key Features
- Clean polygon-like building outputs  
- Boundary refinement using morphological operations  
- Noise removal and structure preservation  
- Building count estimation  
- Scalable tile-based processing  
- GIS-ready outputs  

---

## 📊 Sample Results

Buildings are highlighted in **red** in overlay outputs.

*(See sample images in `sample_output/` folder)*

---

## ⚙️ Model Details
- Model: DeepLabV3+  
- Backbone: ResNet50  
- Loss: BCE + Dice  
- Input Size: 512 × 512  
- Threshold: 0.40  

---

## 📈 Performance
- IoU: ~0.81  ---

## 🧠 Key Features
- Clean polygon-like building outputs  
- Boundary refinement using morphological operations  
- Noise removal and structure preservation  
- Building count estimation  
- Scalable tile-based processing  
- GIS-ready outputs  

---

## 📊 Sample Results

Buildings are highlighted in **red** in overlay outputs.

*(See sample images in `sample_output/` folder)*

---

## ⚙️ Model Details
- Model: DeepLabV3+  
- Backbone: ResNet50  
- Loss: BCE + Dice  
- Input Size: 512 × 512  
- Threshold: 0.40  

---

## 📈 Performance
- IoU: ~0.81  
- Dice Score: ~0.88  

---

## ⚠️ Limitations
- Nearby buildings may merge (semantic segmentation limitation)  
- Small structures may be partially detected  
- Instance-level separation is future work  

---

## 🧪 Run Inference

Before running, update:
- `MODEL_PATH`
- `INPUT_IMAGE`

Then run:
python inference.py

---

## 📌 Applications
- Panchayat-level asset mapping  
- Rural planning and development  
- Solar panel potential estimation  
- Infrastructure analysis  

---

## 🔮 Future Work
- Road extraction  
- Waterbody extraction  
- Roof-type classification  
- Instance segmentation for building separation  

---
