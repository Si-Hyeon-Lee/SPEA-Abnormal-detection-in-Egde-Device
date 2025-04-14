# 📦 Lightweight Student Model for Real-Time Industrial Inference via Knowledge Distillation

- This repository presents a complete pipeline for deploying a lightweight deep learning classifier on SPEA's resource-constrained semiconductor testing devices.
- Used **Knowledge Distillation (KD)** and **Model Compression** techniques. 
- The task focuses on binary classification of tabular data derived from semiconductor inspection processes (e.g., pass/fail signal waveform vectors).
- Data is confidential, but eight tests are conducted on a single semiconductor item, and 28 electrical related data are generated for each test, 
i.e. 28 input features.

---

## 🛠️ Pipeline Summary

### **1. Teacher Model Training**
Three high-capacity models were independently trained on the same structured tabular dataset:
- Tree based , `XGBoostClassifier`
- Fully-connected , `MLP`
- Attention-based , `TabNetClassifier`

Each model was optimized using validation AUC as the key metric, and their outputs were ensembled to serve as **soft labels** for knowledge distillation.

### **2. Student Model Training via Knowledge Distillation**
A shallow `MLP` (2-layer architecture) was trained using the ensemble of teacher predictions as soft targets. The student model was optimized using a composite loss:
- Binary Cross-Entropy with ground-truth labels
- BCE with soft probabilities (temperature-scaled)
  
This enabled the student to capture generalizable knowledge from the teacher ensemble.

### **3. Model Compression: Pruning and Quantization**
To further reduce latency and memory footprint:
- **Unstructured L1-pruning** was applied to the student model, sparsifying 50% of the weights in each layer.
- **Post-training dynamic quantization** was performed, converting all `Linear` layers to `int8` operations.

The final student model maintains high predictive performance while being suitable for edge deployment.

### **4. Real-Time Deployment on SPEA Edge Device**
A minimal Python daemon using `watchdog` monitors an industrial device output file.

---


## 📊 Evaluation

| Model           | Type            | Size(Byte)   | ROC-AUC |
|----------------|-----------------|--------|---------|
| XGBoost         | Teacher         | 4959812 | 0.8302  |
| TabNet          | Teacher         | 1756819(zip file) | 0.8688  |
| MLP             | Teacher         | 50528 | 0.8423  |
| MLP (KD)    | Student  | 9680 | 0.8227  |
| MLP (pruned)| Student| 9864 | 0.7668  |
| MLP (int8 Quantized)  |Student| 5776 | 0.7620  |


