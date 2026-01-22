import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import wikipedia


# ---------------- MODEL PATHS ----------------
model_files = {
    "InceptionV3": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\InceptionV3.h5",
    "VGG16": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\VGG16.h5",
    "ResNet50": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\ResNet50.h5",
    "MobileNet": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\MobileNet.h5",
    "Xception": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\Xception.h5"
}

# ---------------- CLASS NAMES ----------------
class_names = [
    "Animal Fish",
    "Bass",
    "Black Sea Sprat",
    "Gilt Head Bream",
    "Horse Mackerel",
    "Red Mullet",
    "Red Sea Bream",
    "Sea Bass",
    "Shrimp",
    "Striped Red Mullet",
    "Trout"
]

# ---------------- FISH INFO ----------------
fish_info = {
    "animal fish": "General category for aquatic animals with gills and fins.",
    "bass": "Bass are freshwater fish known for firm texture and mild flavor.",
    "black sea sprat": "A small oily fish commonly smoked or fried.",
    "gilt head bream": "Popular Mediterranean fish with delicate white flesh.",
    "horse mackerel": "Fast-swimming fish used in Asian and European dishes.",
    "red mullet": "Rich-flavored fish often grilled or pan-fried.",
    "red sea bream": "Premium fish in Japanese cuisine.",
    "sea bass": "Versatile fish with tender flesh.",
    "shrimp": "Popular seafood used worldwide.",
    "striped red mullet": "Sweeter variant of red mullet.",
    "trout": "Freshwater fish, often grilled or smoked."
}

# ---------------- NUTRITION FACTS ----------------
nutrition_facts = {
    "animal fish": {
        "Calories": 120,
        "Protein": "20g",
        "Fat": "4g"
    },
    "bass": {
        "Calories": 124,
        "Protein": "23g",
        "Fat": "2.6g"
    },
    "black sea sprat": {
        "Calories": 137,
        "Protein": "21g",
        "Fat": "6g"
    },
    "gilt head bream": {
        "Calories": 96,
        "Protein": "19g",
        "Fat": "2g"
    },
    "horse mackerel": {
        "Calories": 131,
        "Protein": "20g",
        "Fat": "5g"
    },
    "red mullet": {
        "Calories": 117,
        "Protein": "19g",
        "Fat": "4g"
    },
    "red sea bream": {
        "Calories": 98,
        "Protein": "20g",
        "Fat": "2g"
    },
    "sea bass": {
        "Calories": 97,
        "Protein": "20g",
        "Fat": "2g"
    },
    "shrimp": {
        "Calories": 99,
        "Protein": "24g",
        "Fat": "0.3g"
    },
    "striped red mullet": {
        "Calories": 120,
        "Protein": "20g",
        "Fat": "4.2g"
    },
    "trout": {
        "Calories": 148,
        "Protein": "20g",
        "Fat": "6g"
    }
}



# ---------------- FUNCTIONS ----------------
def get_wikipedia_summary(name):
    try:
        wikipedia.set_lang("en")
        return wikipedia.summary(name + " fish", sentences=2)
    except:
        return "Wikipedia summary not available."

def confidence_color(conf):
    if conf >= 90:
        return "#00FF7F"
    elif conf >= 75:
        return "#1E90FF"
    elif conf >= 50:
        return "#FFA500"
    else:
        return "#FF4500"

# ---------------- UI STYLING ----------------
st.markdown("""
<style>
.neon-glow {
    font-size: 52px;          /* 🔥 BIGGER */
    font-weight: 900;
    text-align: center;
    padding: 18px 10px;
    margin-bottom: 30px;
    color: #ffffff;
    background: linear-gradient(90deg, #7c3aed, #06b6d4, #22c55e);
    border-radius: 18px;
    letter-spacing: 1px;
    box-shadow:
        0 0 12px #7c3aed,
        0 0 24px #06b6d4,
        0 0 36px #22c55e;
}
)</style>""", unsafe_allow_html=True)

st.markdown("""
<style>
.rank-container {
    margin-bottom: 18px;
}

.rank-label {
    font-weight: bold;
    margin-bottom: 6px;
}

.rank-bar-bg {
    width: 100%;
    background: #020617;
    border-radius: 10px;
    padding: 4px;
}

.rank-bar {
    height: 18px;
    border-radius: 8px;
    animation: growBar 1.4s ease-out forwards;
}

@keyframes growBar {
    from { width: 0%; }
    to { width: var(--width); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""<style>.aura-glow {
        font-size: 26px;
        font-weight: 700;
        text-align: center;
        padding: 16px;
        margin-bottom: 28px;
        color: #e0f2fe;
        background: #020617;
        border-radius: 16px;
        box-shadow:
            0 0 0 2px rgba(56, 189, 248, 0.4),
            0 0 20px rgba(56, 189, 248, 0.7),
            0 0 40px rgba(56, 189, 248, 0.5);
    }
    </style>""", unsafe_allow_html=True)

st.markdown("""
<style>
.confidence-container {
    width: 100%;
    background: #020617;
    border-radius: 12px;
    padding: 4px;
    margin-top: 10px;
    box-shadow: 0 0 10px rgba(34, 211, 238, 0.6);
}

.confidence-bar {
    height: 18px;
    border-radius: 10px;
    background: linear-gradient(90deg, #22c55e, #06b6d4, #6366f1);
    width: 0%;
    animation: fillBar 1.5s ease-out forwards;
}

@keyframes fillBar {
    from { width: 0%; }
    to { width: var(--confidence); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.glow-title {
    font-size: 40px;
    text-align: center;
    color: #00FFFF;
    text-shadow: 0 0 10px #00FFFF, 0 0 20px #1E90FF;
    font-weight: bold;
}
.glow-box {
    font-size: 22px;
    color: #00FFFF;
    text-shadow: 0 0 5px #00FFFF;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-bottom:30px;">
  <div class="neon-glow" style="font-size:56px;">
    🐟 Multiclass Fish Image Classifier
  </div>
  <div style="
    font-size:20px;
    color:#94a3b8;
    margin-top:8px;">
    Deep Learning • CNN • Transfer Learning
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()
# ---------------- SIDEBAR ----------------

BEST_MODEL_NAME = "MobileNet"


selected_model = st.sidebar.selectbox("Choose Model", model_files.keys())

if selected_model == BEST_MODEL_NAME:
    st.sidebar.markdown("✅ **This is the Best Model**")
model = tf.keras.models.load_model(model_files[selected_model], compile=False)
uploaded_file = st.sidebar.file_uploader("📤 Upload Fish Image", type=["jpg", "png", "jpeg"])

col1, col2,  = st.columns(2)
with col1:
    st.success(f"🏆 Best Performing Model: {BEST_MODEL_NAME}")
with col2:
    st.success(f"Using Model: {selected_model}")
# ---------------- PREDICTION ----------------
prediction = None
img = None

if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    confidence = prediction[0][idx] * 100
    species_name = class_names[idx]
    species_key = species_name.lower()

    show_top3 = st.sidebar.checkbox("Show Top-3 Predictions", value=True)
    prediction = model.predict(img_array) if uploaded_file else None

st.divider()
# ---------------- TOP-3 PREDICTIONS ----------------
st.markdown("""
<style>
.neon-glow {
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    padding: 10px;
    color: #ffffff;
    background: linear-gradient(90deg, #7c3aed, #06b6d4, #22c55e);
    border-radius: 12px;
    box-shadow:
        0 0 8px #7c3aed,
        0 0 16px #06b6d4,
        0 0 24px #22c55e;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="neon-glow">🏆 Top-3 Predictions</div>', unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
if prediction is not None and show_top3:

    medals = ["🥇", "🥈", "🥉"]
    colors = ["#facc15", "#94a3b8", "#fb923c"]  # gold, silver, bronze

    top3_idx = np.argsort(prediction[0])[::-1][:3]

    for rank, idx in enumerate(top3_idx):
        conf = prediction[0][idx] * 100

        st.markdown(f"""
        <div class="rank-container">
            <div class="rank-label">
                {medals[rank]} {class_names[idx]} — {conf:.2f}%
            </div>
            <div class="rank-bar-bg">
                <div class="rank-bar"
                     style="--width:{conf:.2f}%;
                            background:{colors[rank]};">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
st.markdown("""
<div style="
margin-top:20px;
padding:12px;
background:rgba(2,6,23,0.6);
border-left:4px solid #38bdf8;
border-radius:8px;
color:#e5e7eb;
font-size:15px;
">
🧠 <b>Model Explainability Note</b><br><br>
The confidence scores represent the model’s probability estimates.
Lower confidence indicates uncertainty due to factors such as
image quality, lighting conditions, or visual similarity between fish species.
Users are encouraged to review the Top-3 predictions when confidence is low.
</div>
""", unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
st.divider()
# ---------------- COLUMNS ----------------
col1, col2, col3 = st.columns(3)

# -------- IMAGE --------
with col1:

    st.markdown('<div class="aura-glow">📷 Uploaded Image</div>', unsafe_allow_html=True)

    if img:
        st.image(img, width=500)
    else:
        st.info("Upload an image")



# -------- PREDICTION --------
with col2:
    st.markdown('<div class="aura-glow">🔮 Prediction</div>', unsafe_allow_html=True)
    if prediction is not None:
        bg = confidence_color(confidence)
        st.markdown(f"""
        <div style="
        background-color:{bg};
        padding:15px;
        border-radius:10px;
        color:white;
        text-align:center;
        font-size:40px;
        font-weight:bold;
        box-shadow:0 0 15px {bg};
        ">
        {species_name}<br>
        Confidence: {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="confidence-container">
            <div class="confidence-bar" style="--confidence:{confidence:.2f}%"></div>
        </div>
        <div style="text-align:right; font-size:24px; color:#94a3b8;">
            {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)


    else:
        st.info("Upload an image to see prediction")

# -------- ABOUT --------
with col3:

    st.markdown('<div class="aura-glow">📘 About</div>', unsafe_allow_html=True)
    if prediction is not None:
        info = fish_info.get(species_key, "Information not available.")
        wiki = get_wikipedia_summary(species_name)
        nutrition = nutrition_facts.get(species_key)

        st.markdown(f"""
<div style="
background-color:#0f172a;
padding:15px;
border-radius:10px;
font-size:16px;
color:#e5e7eb;
box-shadow:0 0 15px #38bdf8;
">
<b>{species_name}</b><br><br>
{info}<br><br>
<b>📚 Wikipedia</b><br>{wiki}<br><br>
<b>🍽️ Nutrition (per 100g)</b><br>
{"Calories: "+str(nutrition["Calories"])+" kcal<br>Protein: "+nutrition["Protein"]+"<br>Fat: "+nutrition["Fat"] if nutrition else "Not available"}
</div>
""", unsafe_allow_html=True)

