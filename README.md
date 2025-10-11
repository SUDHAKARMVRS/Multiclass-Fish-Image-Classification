# 🐟 Multiclass Fish Image Classification using CNN & Transfer Learning

This project focuses on classifying fish species using Convolutional Neural Networks (CNNs) and various pre-trained models such as **VGG16**, **ResNet50**, **MobileNet**, **InceptionV3**, and **Xception**.  
It also includes a **Streamlit web application** for easy image-based prediction.

---

## 📁 Project Structure

```
📦 Fish_Classification_Project
├── dataset/
│   ├── train/
│   ├── test/
│   └── val/                # optional (if available)
├── models/
│   ├── cnn_model.h5
│   ├── best_model.h5
│   └── pretrained/
│       ├── vgg16_model.h5
│       ├── resnet50_model.h5
│       ├── mobilenet_model.h5
│       ├── inceptionv3_model.h5
│       └── Xception_model.h5
├── app/
│   └── app.py              # Streamlit app
├── require.txt
└── README.md
```

---

## ⚙️ Installation

### Step 1️⃣: Clone the Repository
```bash
git clone https://github.com/yourusername/fish-classification.git
cd fish-classification
```

### Step 2️⃣: Install Dependencies
```bash
pip install -r require.txt
```
---

## 🧠 Model Training

### Data Preprocessing & Augmentation
- Images are **rescaled** to [0,1].
- Applied **rotation**, **zoom**, and **horizontal flip** for augmentation.

### Models Trained
1. CNN (custom model built from scratch)
2. VGG16
3. ResNet50
4. MobileNet
5. InceptionV3
6. Xception

### Fine-tuning
Each pre-trained model was fine-tuned on the fish dataset.

### Model Saving
The model with **highest validation accuracy** is saved as:
```bash
models/best_model.h5
```

---

## 📊 Model Evaluation

Evaluation metrics used:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
---

## 🚀 Deployment (Streamlit App)

### Run Locally
```bash
streamlit run app/app.py
```
---

## 🖼️ Streamlit App Features
✅ Upload fish image  
✅ Predict fish species  
✅ Display model confidence score  

---

---

## 🧾 Author
**Sudhakar M**  
---
