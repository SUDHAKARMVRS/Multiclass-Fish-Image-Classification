# 🐟 Multiclass Fish Image Classifier

This is a **Deep Learning-based web application** that can classify multiple fish species from an image. The app provides **Top-3 predictions**, **confidence scores**, **fish information**, **nutrition facts**, and **Wikipedia summaries**. Built with **Python, TensorFlow, Keras, and Streamlit**, this project demonstrates **image classification, transfer learning, and interactive dashboard design**.

---

## 🔍 Features

- Upload any fish image (jpg, png, jpeg)  
- Predict the fish species using pre-trained models  
- View **Top-3 predictions** with confidence percentages  
- Explore **fish info** and **nutrition facts**  
- Read a **Wikipedia summary** of the fish  
- Interactive and visually appealing **Streamlit dashboard**  

---

## 🛠️ Tech Stack

- **Python 3.x**  
- **TensorFlow / Keras**  
- **Streamlit**  
- **PIL / NumPy / Pandas**  
- **Wikipedia API**  

**Pre-trained Models Included:**  
- MobileNet (Best Performing)  
- InceptionV3  
- VGG16  
- ResNet50  
- Xception  

---

## 📁 Project Structure

```
fish-classifier/
│
├─ models/               # Pre-trained .h5 models
│  ├─ MobileNet.h5
│  ├─ InceptionV3.h5
│  ├─ VGG16.h5
│  ├─ ResNet50.h5
│  └─ Xception.h5
│
├─ app.py                # Main Streamlit application
├─ requirements.txt      # Python dependencies
├─ README.md
└─ assets/               # Images, logos, icons (optional)
```

---

## 🚀 How to Run

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/fish-classifier.git
cd fish-classifier
```

2. **Create a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

5. **Upload an image** and see the predictions and fish details in the dashboard.  

---

## 📊 Model Explainability

- The **confidence score** represents the probability of the prediction.  
- **Lower confidence** indicates uncertainty due to factors such as image quality, lighting, or visual similarity between fish species.  
- Users are encouraged to check **Top-3 predictions** when confidence is low.  

---

## 🌟 Future Improvements

- Add **more fish species** to increase coverage  
- Integrate **real-time camera input** for live classification  
- Enhance **UI/UX** with more interactive visualizations  
- Add **multi-language support** for fish info  

---
## 📷 Snapshot
<img width="1910" height="999" alt="Dashboard (4)" src="https://github.com/user-attachments/assets/f191e042-4fc7-4681-8850-8c75d0e06fa9" />
<img width="1913" height="1004" alt="Dashboard (1)" src="https://github.com/user-attachments/assets/c73ed896-c6b9-4fe1-96c0-68000ee216df" />
<img width="1919" height="1001" alt="Dashboard (2)" src="https://github.com/user-attachments/assets/d6c76b86-64ba-496e-bc07-6bd9972ecbcb" />
<img width="1916" height="1000" alt="Dashboard (3)" src="https://github.com/user-attachments/assets/56f62733-b497-429d-bc2d-1c6da0386a5a" />

---

## 👨‍💻 Author

**Sudhakar M**
[LinkedIn](https://www.linkedin.com/in/sudhakar-m-657ba787/)

---

**Demo Video:** (https://www.linkedin.com/posts/sudhakar-m-657ba787_datascience-machinelearning-deeplearning-activity-7420103229302247425-BSUA?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAABKCEOMB652hawyOtWY3dSUOQiLjCTcOG-4)

