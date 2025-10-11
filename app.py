import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Model options
model_files = {
    "InceptionV3": r"C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\models\InceptionV3.h5",
    "VGG16": r"C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\models\VGG16.h5",
    "ResNet50": r"C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\models\ResNet50.h5",
    "MobileNet": r"C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\models\MobileNet.h5",
    "Xception": r"C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\models\Xception.h5"
}

class_names = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
    'fish sea_food sea_bass', 'fish sea_food shrimp',
    'fish sea_food striped_red_mullet', 'fish sea_food trout'
]

st.title("🐟 Multiclass Fish Image Classifier")

# Model selection
st.sidebar.title("blue.:Model Selection")
selected_model = st.sidebar.selectbox("Select Model", list(model_files.keys()))
model_path = model_files[selected_model]
model = tf.keras.models.load_model(model_path, compile=False)

uploaded_file = st.sidebar.file_uploader("Upload a fish image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    st.success(f"Prediction: {class_names[idx]} ({prediction[0][idx]*100:.2f}%)")