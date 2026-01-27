import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import wikipedia
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# ---------------- MODEL PATHS ----------------
model_files = {
    "CNN Scratch" : r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\cnn_scratch.h5",
    "InceptionV3": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\InceptionV3.h5",
    "VGG16": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\VGG16.h5",
    "ResNet50": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\ResNet50.h5",
    "MobileNet": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\MobileNet.h5",
    "Xception": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\Xception.h5"
}

# ---------------- CLASS NAMES ----------------
class_names = [
    "Animal Fish",
    "Animal Fish Bass",
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
    "animal fish bass": "Bass are freshwater fish known for firm texture and mild flavor.",
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
    "animal fish bass": {
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
/* ===== OCEAN BACKGROUND ===== */
.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(56,189,248,0.18), transparent 40%),
        radial-gradient(circle at 80% 30%, rgba(124,58,237,0.18), transparent 40%),
        radial-gradient(circle at 50% 80%, rgba(34,197,94,0.18), transparent 45%),
        linear-gradient(135deg, #020617, #020617);
    background-size: 200% 200%;
    animation: oceanMove 20s ease infinite;
    overflow: hidden;
}

@keyframes oceanMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ===== FISH LAYER ===== */
.fish-layer {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    pointer-events: none;
    z-index: 0;
}

/* ===== FISH ===== */
.fish {
    position: absolute;
    font-size: 42px;
    animation-name: swim;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    filter: drop-shadow(0 0 8px rgba(56,189,248,0.8));
}

@keyframes swim {
    0%   { transform: translateX(-15vw) translateY(0); }
    50%  { transform: translateX(50vw) translateY(-25px); }
    100% { transform: translateX(115vw) translateY(0); }
}


/* Individual fish */
.fish1 { top: 20%; animation-duration: 22s; font-size: 46px; }
.fish2 { top: 40%; animation-duration: 30s; font-size: 36px; }
.fish3 { top: 65%; animation-duration: 26s; font-size: 40px; }
.fish4 { top: 80%; animation-duration: 34s; font-size: 30px; }
</style>

<div class="fish-layer">
    <div class="fish fish1">🐳</div>
    <div class="fish fish2">🐠</div>
    <div class="fish fish4">🦈</div>
    <div class="fish fish2">🐬</div>
    <div class="fish fish3">🐙</div>
    <div class="fish fish4">🦑</div>  
</div>
""", unsafe_allow_html=True)



st.markdown("""
<style>
.glass-glow {
    font-size: 38px;
    font-weight: 900;
    text-align: center;
    padding: 16px 24px;
    margin-bottom: 26px;
    color: #e5e7eb;
    background: rgba(15, 23, 42, 0.55);
    border-radius: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow:
        0 8px 30px rgba(0, 0, 0, 0.45),
        inset 0 0 12px rgba(56, 189, 248, 0.25);
}
</style>
""", unsafe_allow_html=True)

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
  <div class="glass-glow" style="font-size:56px;">
    🐠 Multiclass Fish Image Classifier 🦈
  </div>
  <div style="
    font-size:20px;
    color:#94a3b8;
    margin-top:8px;">
    Deep Learning • CNN • Transfer Learning
  </div>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.gradient-glow {
    font-size: 34px;
    font-weight: 800;
    text-align: center;
    padding: 14px 20px;
    margin-bottom: 20px;
    color: #ffffff;
    background: #020617;
    border-radius: 18px;
    border: 2px solid transparent;
    background-image:
        linear-gradient(#020617, #020617),
        linear-gradient(90deg, #7c3aed, #06b6d4, #22c55e);
    background-origin: border-box;
    background-clip: padding-box, border-box;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.gradient-text-glow {
    font-size: 40px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 28px;
    background: linear-gradient(90deg, #7c3aed, #06b6d4, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow:
        0 0 12px rgba(124, 58, 237, 0.6),
        0 0 24px rgba(6, 182, 212, 0.5);
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
@keyframes pulseGlow {
    0% { box-shadow: 0 0 8px rgba(34,211,238,0.5); }
    50% { box-shadow: 0 0 28px rgba(34,211,238,0.9); }
    100% { box-shadow: 0 0 8px rgba(34,211,238,0.5); }
}

.pulse-glow {
    font-size: 34px;
    font-weight: 800;
    text-align: center;
    padding: 14px 20px;
    margin-bottom: 20px;
    color: #e0f2fe;
    background: #020617;
    border-radius: 16px;
    animation: pulseGlow 2s infinite;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    @keyframes colorchange {
        0% {color: #ff0000;}
        25% {color: #ffa500;}
        50% {color: #00ff00;}
        75% {color: #00ffff;}
        100% {color: #ff0000;}
    }
    .animated-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        animation: colorchange 5s infinite;
    }
    </style>
    """,
    unsafe_allow_html=True
)
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



# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="🐟 Image Classifier", layout="wide")
# ---------------- SIDEBAR ----------------

BEST_MODEL_NAME = "MobileNet"


selected_model = st.sidebar.selectbox("Choose Model", model_files.keys())

if selected_model == BEST_MODEL_NAME:
    st.sidebar.markdown("✅ **This is the Best Model**")
model = tf.keras.models.load_model(model_files[selected_model], compile=False)
uploaded_file = st.sidebar.file_uploader("📤 Upload Fish Image", type=["jpg", "png", "jpeg"])

# ---------------- HEADER ----------------


st.markdown(f'<div class="animated-title">🏆 Best Performing Model:✨{BEST_MODEL_NAME}🔥</div>', unsafe_allow_html=True)

st.divider()
# ---------------- SELECTED MODEL ----------------
st.markdown(f'<div class="gradient-glow">Selected Model: {selected_model}</div>', unsafe_allow_html=True)

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


st.markdown('<div class="glass-glow">⭐ Top-3 Predictions </div>', unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
if prediction is not None and show_top3:

    medals = ["🥇 🐬 ", "🥈 🐟", "🥉🐳"]
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
else:
        st.info("Upload an image and enable Top-3 Predictions to see results.")


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
        padding:30px;
        border-radius:10px;
        color:white;
        text-align:center;
        font-size:40px;
        font-weight:bold;
        box-shadow:0 0 15px {bg};
        ">
        🐳{species_name}<br>
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
<b>🐬{species_name}🐡</b><br><br>
{info}<br><br>
<b>📚 Wikipedia</b><br>{wiki}<br><br>
<b>🍽️ Nutrition (per 100g)</b><br>
{"Calories: "+str(nutrition["Calories"])+" kcal<br>Protein: "+nutrition["Protein"]+"<br>Fat: "+nutrition["Fat"] if nutrition else "Not available"}
</div>
""", unsafe_allow_html=True)

    else:
        st.info("Upload an image to info about the fish species")

@st.cache_resource
def load_all_models(model_files):
    models = {}
    for name, path in model_files.items():
        models[name] = tf.keras.models.load_model(path, compile=False)
    return models


def predict_all_models(models, img_array, class_names):
    results = []
    for model_name, model in models.items():
        preds = model.predict(img_array, verbose=0)
        idx = np.argmax(preds)
        results.append({
            "Model": model_name,
            "Predicted Class": class_names[idx],
            "Confidence (%)": round(preds[0][idx] * 100, 2)
        })
    return pd.DataFrame(results)
# ---------------- MULTI-MODEL PREDICTION ----------------
st.divider()
all_models = load_all_models(model_files)

st.markdown('<div class="gradient-glow">🧠 Compare Models</div>', unsafe_allow_html=True)

if uploaded_file:
    compare_df = predict_all_models(all_models, img_array, class_names)

    colA, colB = st.columns([2, 1])

    with colA:
        st.markdown('<div class="pulse-glow">📊 Prediction Comparison</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.bar_chart(
            compare_df.set_index("Model")["Confidence (%)"],
            color="Confidence (%)",
            height=420
        )


    with colB:
        st.markdown('<div class="pulse-glow">Confidence Values</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
            compare_df,
            use_container_width=True
        )

else:
    st.info("Upload an image to compare model predictions")
# ---------------- MODEL UNCERTAINTY ----------------
st.divider()

def prediction_entropy(probabilities):
    probs = np.clip(probabilities, 1e-9, 1)
    return -np.sum(probs * np.log(probs))

entropy_results = []

if uploaded_file is not None:
    for model_name, model in all_models.items():
        preds = model.predict(img_array, verbose=0)[0]
        entropy = prediction_entropy(preds)

        entropy_results.append({
            "Model": model_name,
            "Entropy (Uncertainty)": round(entropy, 4)
         })
    entropy_df = pd.DataFrame(entropy_results)


    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown('<div class="pulse-glow">🧠 Model Entropy Comparison</div>', unsafe_allow_html=True)
        fig_entropy = px.bar(
        entropy_df,
        x="Model",
        y="Entropy (Uncertainty)",
        color="Entropy (Uncertainty)",
        color_continuous_scale="Inferno",
        labels={"Entropy (Uncertainty)": "Entropy (Uncertainty)"},
        height=420)
        st.plotly_chart(fig_entropy, use_container_width=True)
        st.info("Lower entropy indicates higher confidence in predictions.")

    with colB:
        st.markdown('<div class="pulse-glow">Entropy Values</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(
        entropy_df,
        use_container_width=True)

else:
    st.info("Upload an image to compare model predictions")


# Keep val_data as the DirectoryIterator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_data = datagen.flow_from_directory(
    r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\images.cv_jzk6llhf18tm3k0kyttxz\data\val",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(val_data.class_indices.keys())


st.markdown(f'<div class="aura-glow">{selected_model}</div>',unsafe_allow_html=True)


colc , cold = st.columns([2, 2])

if uploaded_file is not None:
    with colc :

        st.markdown('<div class="pulse-glow">📊 Confusion Matrix</div>', unsafe_allow_html=True)

        # Predictions
        y_true = val_data.classes
        y_pred = np.argmax(model.predict(val_data), axis=1)

# Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {selected_model}")
        st.pyplot(fig)


    with cold:
        st.markdown('<div class="pulse-glow">📊 Classification Report</div>', unsafe_allow_html=True)

# Load only the chosen model
        model_path = model_files[selected_model]
        model = tf.keras.models.load_model(model_path)


# Classification Report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        st.dataframe(report)

else:
        st.info("Upload an image to compare model predictions")


st.divider()
# ----------------------------------------------
# FOOTER
# ----------------------------------------------
st.markdown("""
<div class="footer glow">
    Built with 💙 & 🧠 using <b>Streamlit</b> | Designed by <b>Sudhakar M</b><br>
    <small>© 2026 Fish Image Classifier</small>
</div>
""", unsafe_allow_html=True)


