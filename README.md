# Fabric Defect Detection using PatchCore 

This project implements **PatchCore** (CVPR 2021), an unsupervised anomaly detection method, to distinguish between **defective** and **non-defective** fabric images.  

Unlike supervised approaches, PatchCore does **not require defective samples** during training. Instead, it learns from normal (good) samples and detects anomalies based on feature distances.

---

## Methodology

### 1. Preprocessing
- Resize images to **224 × 224**
- Normalize using **ImageNet statistics**
- Extract patch embeddings from **ResNet-50 backbone**

### 2. Training Phase
- Use only **normal (non-defective) images**
- Subsample **10% of patch embeddings** to form a **compact memory bank** (max 15k patches)
- Fit a **k-nearest neighbor model** on the memory bank

### 3. Testing Phase
- Extract patch embeddings from test images
- Compute **nearest-neighbor distance** for each patch
- Image-level anomaly score = **maximum patch distance**
- If score > threshold → classify as **defective**

---

## Evaluation
- **Metrics:** AUROC, ROC Curve, F1-Score, Confusion Matrix  
- **Threshold Selection:** Youden’s J statistic & F1-score maximization  
- **Result:** Achieved **perfect detection (AUROC = 1.0)** with clear separation between good and defective samples  

---

## Tech Stack
- Python 3.x  
- PyTorch & PyTorch Geometric  
- NumPy, SciPy, scikit-learn  
- Matplotlib, Seaborn  

---

## 📂 Repository Structure
