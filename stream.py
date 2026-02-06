import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import wikipedia
import time
import os
# ---------------- TENSORFLOW OPTIMIZATIONS ----------------

# ---------------- STREAMLIT APP ----------------

st.set_page_config(page_title="🐟 Image Classifier", layout="wide")


# ---------------- MODEL PATHS ----------------

page = st.sidebar.selectbox(
    "🧭 Choose Your Destination:", 
    ["🏠Home", "🔍 Prediction"],
    key="navigation"
)

model_files = {
    "MobileNet": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\MobileNet.h5",
}

compare_models = {
    "MobileNet": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\MobileNet.h5",
    "CNN from Scratch": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\cnn_scratch.h5",
    "ResNet50": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\ResNet50.h5",
    "EfficientNetB0": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\EfficientNetB0.h5",
    "VGG16": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\VGG16.h5",
    "InceptionV3": r"C:\Users\jaish\OneDrive\Documents\Desktop\Sudhakar\Fish Image Classification\models\InceptionV3.h5"
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

fish_images = {
    "animal fish": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/animal fish/0AUE3U3PPXVL.jpg",
    "animal fish bass": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/animal fish bass/8HCN0EX2B24L.jpg",
    "black sea sprat": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/fish sea_food black_sea_sprat/2QMAGGO98PYZ.jpg",
    "gilt head bream": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/fish sea_food gilt_head_bream/4RG9LIF4V2IW.jpg",
    "horse mackerel": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/fish sea_food hourse_mackerel/1TM1YQ60T75L.jpg",
    "red mullet": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/fish sea_food red_mullet/1X3J2KU6FCXA.jpg",
    "red sea bream": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/fish sea_food red_sea_bream/0GMID1XYY1OD.jpg",
    "sea bass": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/fish sea_food sea_bass/2MBQJVRBKI3N.jpg",
    "shrimp": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/fish sea_food shrimp/2VWK0OVYLR6X.jpg",
    "striped red mullet": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/fish sea_food striped_red_mullet/1YEBAPBBZYHJ.jpg",
    "trout": "images.cv_jzk6llhf18tm3k0kyttxz/data/test/fish sea_food trout/4C1OKKUQ6LEA.jpg"}


wiki_facts = {
    "animal fish": "A fish is an aquatic, anamniotic, gill-bearing vertebrate animal with swimming fins and a tough cranium to protect the brain, but lacking limbs with digits. Fish can be grouped into the more basal jawless fish and the more common jawed fish, the latter including all living cartilaginous and bony fish, as well as the extinct placoderms and acanthodians.",
    "animal fish bass": "Bass (;pl.: bass) is a common name shared by many species of ray-finned fish from the large clade Percomorpha, mainly belonging to the orders Perciformes and Moroniformes, encompassing both freshwater and marine species. The word bass comes from Middle English bars, meaning perch, despite that none of the commonly referred bass species belong to the perch family Percidae. The term bass is also used for some species outside of these orders, such as the giant sea bass (Stereolepis gigas) and the black sea bass (Centropristis striata), which are in the order Centrarchiformes, and the Australian bass (Macquaria novemaculeata) and the Tasmanian bass (Percalates novemaculeata), which are in the order Perciformes but not in the same clade as the other basses. The term bass is also used for some species outside of these orders, such as the giant sea bass (Stereolepis gigas) and the black sea bass (Centropristis striata), which are in the order Centrarchiformes, and the Australian bass (Macquaria novemaculeata) and the Tasmanian bass (Percalates novemaculeata), which are in the order Perciformes but not in the same clade as the other basses.",
    "black sea sprat": "Sprat is the common name applied to a group of forage fish belonging to the genus Sprattus in the family Clupeidae. The term also is applied to a number of other small sprat-like forage fish (Clupeoides, Clupeonella, Corica, Ehirava, Hyperlophus, Microthrissa, Nannothrissa, Platanichthys, Ramnogaster, Rhinosardinia, and Stolothrissa). They are small oily fish that are commonly smoked or fried and eaten whole.",
    "gilt head bream": "The gilt-head bream (Sparus aurata), also known as the gilthead, dourade, gilt-head seabream, European seabream or silver seabream, is a species of marine ray-finned fish belonging to the family Sparidae, the seabreams or porgies. This fish is found in the Eastern Atlantic and the Mediterranean.",
    "horse mackerel": "The Atlantic horse mackerel (Trachurus trachurus), also known as the European horse mackerel or common scad, is a species of jack mackerel in the family Carangidae, which includes the jacks, pompanos and trevallies. It is found in the eastern Atlantic Ocean off Europe and Africa and into the south-eastern Indian Ocean.",
    "red mullet": "Red mullet is a rich-flavored fish that is often grilled or pan-fried and is popular in Mediterranean cuisine.",
    "red sea bream": "Bream (, US also ) are species of freshwater fish belonging to a variety of genera including Abramis (e.g., A. brama, the common bream), Ballerus, Blicca, Chilotilapia, Etelis, Lepomis, Gymnocranius, Lethrinus, Nemipterus, Pharyngochromis, Rhabdosargus, Scolopsis, or Serranochromis. Although species from all of these genera are called bream, the term does not imply a degree of relatedness between them.",
    "sea bass": "Sea bass is a common name for a variety of species of marine fish. Many fish species of various families have been called sea bass.",
    "shrimp": "A shrimp (pl.: shrimp (US) or shrimps (UK)) is a common name typically used for crustaceans with an elongated body and a primarily swimming mode of locomotion – usually decapods belonging to the Caridea or Dendrobranchiata, although some crustaceans outside of this order are also referred to as shrimp. More narrow definitions may be restricted to Caridea, to smaller species of either of the aforementioned groups, or only the marine species.",
    "striped red mullet": "The striped red mullet or surmullet (Mullus surmuletus) is a species of goatfish found in the Mediterranean Sea, eastern North Atlantic Ocean, and the Black Sea. They can be found in water as shallow as 5 metres (16 ft) or as deep as 409 metres (1,342 ft) depending upon the portion of their range that they are in.",
    "trout": "Trout (pl.: trout) is a generic common name for numerous species of carnivorous freshwater fishes belonging to the genera Oncorhynchus, Salmo, and Salvelinus, all of which are members of the subfamily Salmoninae in the family Salmonidae. The word trout is also used for some similar-shaped but non-salmonid fish, such as the spotted seatrout/speckled trout (Cynoscion nebulosus, which is actually a croaker)"
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
/* ===== ICE BLUE / CYAN OCEAN BACKGROUND ===== */
.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(0,255,255,0.28), transparent 42%),
        radial-gradient(circle at 80% 30%, rgba(56,189,248,0.22), transparent 42%),
        radial-gradient(circle at 50% 80%, rgba(34,211,238,0.18), transparent 48%),
        linear-gradient(135deg, #020617, #020617);
    background-size: 200% 200%;
    animation: oceanMove 18s ease infinite;
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
    inset: 0;
    pointer-events: none;
    z-index: 0;
}

/* ===== FISH (ICE BLUE NEON GLOW) ===== */
.fish {
    position: absolute;
    font-size: 42px;
    animation-name: swim;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    filter:
        drop-shadow(0 0 6px rgba(0,255,255,0.9))
        drop-shadow(0 0 14px rgba(56,189,248,0.8))
        drop-shadow(0 0 28px rgba(34,211,238,0.6));
}

@keyframes swim {
    0%   { transform: translateX(-15vw) translateY(0); }
    50%  { transform: translateX(50vw) translateY(-25px); }
    100% { transform: translateX(115vw) translateY(0); }
}

/* Individual fish */
.fish1 { top: 18%; animation-duration: 22s; font-size: 46px; }
.fish2 { top: 38%; animation-duration: 30s; font-size: 36px; }
.fish3 { top: 62%; animation-duration: 26s; font-size: 40px; }
.fish4 { top: 82%; animation-duration: 34s; font-size: 30px; }
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
<style>
.top3-heading {
    font-family: 'Poppins', sans-serif;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
    margin: 12px 0 20px 0;
    color: #67e8f9; /* ice-blue */
    text-shadow:
        0 0 8px rgba(103,232,249,0.8),
        0 0 18px rgba(34,211,238,0.6);
    animation: top3SlideGlow 1.8s ease-out;
}

@keyframes top3SlideGlow {
    0% {
        opacity: 0;
        transform: translateY(-18px);
        text-shadow: none;
    }
    60% {
        opacity: 1;
        transform: translateY(4px);
    }
    100% {
        transform: translateY(0);
        text-shadow:
            0 0 8px rgba(103,232,249,0.8),
            0 0 18px rgba(34,211,238,0.6);
    }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div style="text-align:center; margin-bottom:-150px;">
  <div class="glass-glow" style="font-size:56px;">
    🐠 Seeing the Sea Through AI Eyes 🦈
  </div>
  <div style="
    font-size:20px;
    color:#99e9ff;
    margin-top:10px;">
    Deep Learning • CNN • Transfer Learning
  </div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.gradient-glow {
    font-size: 34px;
    font-weight: 900;
    text-align: center;
    padding: 0px 0px;
    margin-bottom: -50px;
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
st.markdown("""
<style>
.floating-speed {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 9999;
    padding: 18px 22px;
    border-radius: 18px;
    font-weight: bold;
    color: white;
    background: linear-gradient(
        90deg,
        #000004,
        #420a68,
        #932667,
        #dd513a,
        #fca50a,
        #fcffa4
    );
    box-shadow:
        0 0 20px rgba(252,165,10,0.9),
        0 0 40px rgba(221,81,58,0.7);
    animation: top3SlideGlow 1.8s ease-out,
           pulseGlow 2.5s infinite alternate;
;
}

@keyframes floatPulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* 🔥 Animated (first time only) */
.top3-heading-animate {
    font-family: 'Poppins', sans-serif;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
    margin: 12px 0 20px 0;
    color: #67e8f9;
    text-shadow:
        0 0 8px rgba(103,232,249,0.8),
        0 0 18px rgba(34,211,238,0.6);
    animation:
        top3SlideGlow 1.8s ease-out,
        pulseGlow 2.5s infinite alternate;
}

/* 🧊 Static (after first upload) */
.top3-heading-static {
    font-family: 'Poppins', sans-serif;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
    margin: 12px 0 20px 0;
    color: #67e8f9;
    text-shadow:
        0 0 8px rgba(103,232,249,0.8),
        0 0 18px rgba(34,211,238,0.6);
}

/* Animations */
@keyframes top3SlideGlow {
    0% {
        opacity: 0;
        transform: translateY(-18px);
        text-shadow: none;
    }
    60% {
        opacity: 1;
        transform: translateY(4px);
    }
    100% {
        transform: translateY(0);
    }
}

@keyframes pulseGlow {
    from { text-shadow: 0 0 6px rgba(103,232,249,0.6); }
    to   { text-shadow: 0 0 18px rgba(103,232,249,1); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""<style>
.holo-glow {
    font-size: 40px;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg, #22d3ee, #818cf8, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow:
        0 0 12px rgba(34,211,238,0.6),
        0 0 24px rgba(99,102,241,0.6);
    animation: holoShift 4s linear infinite;
}

@keyframes holoShift {
    0% { filter: hue-rotate(0deg); }
    100% { filter: hue-rotate(360deg); }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""<style>)/* Fish class tab grid */
.fish-tab-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 18px;
    margin-top: 20px;
}

/* Fish tab card – SAME ocean feel */
.fish-tab {
    background: linear-gradient(135deg,
        rgba(0, 255, 255, 0.15),
        rgba(0, 120, 255, 0.08));
    backdrop-filter: blur(18px);
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    border: 2px solid rgba(0, 255, 255, 0.35);
    box-shadow:
        0 0 18px rgba(0, 255, 255, 0.35),
        inset 0 0 12px rgba(255, 255, 255, 0.08);
    transition: all 0.35s ease;
}

/* Hover = glowing tab */
.fish-tab:hover {
    transform: translateY(-6px) scale(1.05);
    box-shadow:
        0 0 30px rgba(0, 255, 255, 0.75),
        inset 0 0 18px rgba(255, 255, 255, 0.15);
}

}
/* COLUMN 1 – ICE CYAN */
.fish-tab.col-1 {
    box-shadow:
        inset 0 0 14px rgba(34,211,238,0.6),
        0 0 26px rgba(34,211,238,0.7);
    border: 1px solid rgba(34,211,238,0.6);
}

/* COLUMN 2 – SKY BLUE */
.fish-tab.col-2 {
    box-shadow:
        inset 0 0 14px rgba(56,189,248,0.6),
        0 0 26px rgba(56,189,248,0.7);
    border: 1px solid rgba(56,189,248,0.6);
}

/* COLUMN 3 – AQUA GREEN */
.fish-tab.col-3 {
    box-shadow:
        inset 0 0 14px rgba(34,197,94,0.55),
        0 0 26px rgba(34,197,94,0.65);
    border: 1px solid rgba(34,197,94,0.6);
}

}
</style>""", unsafe_allow_html=True)

st.markdown("""
<style>
.glow-title {
    text-align: center;
    font-size: 38px;
    font-weight: 800;
    color: #22d3ee;
    text-shadow:
        0 0 5px #22d3ee,
        0 0 10px #22d3ee,
        0 0 20px #0ea5e9,
        0 0 40px #0284c7;
    margin-top: 20px;
    margin-bottom: 20px;
}

.glow-sub {
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    color: #a5f3fc;
    text-shadow:
        0 0 4px #67e8f9,
        0 0 10px #22d3ee;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 25px;
    margin-top: 30px;
}

.fish-card {
    background: linear-gradient(145deg, #062b45, #0a4f75);
    border-radius: 18px;
    padding: 25px 20px;
    text-align: center;
    box-shadow:
        0 0 15px rgba(34,211,238,0.25),
        inset 0 0 10px rgba(255,255,255,0.05);
    transition: all 0.3s ease-in-out;
}

.fish-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow:
        0 0 25px rgba(34,211,238,0.6),
        0 0 60px rgba(14,165,233,0.4);
}

.fish-icon {
    font-size: 38px;
    margin-bottom: 12px;
}

.fish-title {
    font-size: 20px;
    font-weight: 700;
    color: #e0f2fe;
    margin-bottom: 12px;
}

.fish-line {
    width: 70%;
    height: 3px;
    margin: auto;
    border-radius: 5px;
    background: linear-gradient(90deg, #22d3ee, #38bdf8);
}
</style>
""", unsafe_allow_html=True)



# ===== EMOJIS PER FISH =====
fish_emojis = [
    "🐟",  # Black Sea Sprat
    "🐳",  # Gilt-Head Bream
    "🐬",  # Hourse Mackerel
    "🐟",  # Red Mullet
    "🦈",  # Red Sea Bream
    "🐠",  # Sea Bass
    "🦐",  # Shrimp
    "🦈",  # Striped Red Mullet
    "🐬"   # Trout
    ]

# ===== ICE-BLUE GLOW COLORS =====
fish_colors = [
    "#67E8F9",  # cyan
    "#38BDF8",  # sky blue
    "#22D3EE",  # aqua
    "#60A5FA",  # blue
    "#7DD3FC",  # light ice
    "#0EA5E9",  # deep blue
    "#A5F3FC",  # pale ice
    "#38BDF8",  # sky blue
    "#67E8F9"   # cyan
    ]

fish_cards = [
    ("🐠", "Animal Fish"),
    ("🐟", "Animal Fish Bass"),
    ("🦐", "Black Sea Sprat"),
    ("🐡", "Gilt Head Bream"),
    ("🦈", "Horse Mackerel"),
    ("🐙", "Red Mullet"),
    ("🦞", "Red Sea Bream"),
    ("🐚", "Sea Bass"),
    ("🦑", "Shrimp"),
    ("🐬", "Striped Red Mullet"),
    ("🐳", "Trout")
]


model_icons = [
    ("🧱", "VGG16"),
    ("⚡", "MobileNet"),
    ("🔬", "Xception"),
    ("🚀", "EfficientNet"),
    ("🧠", "ResNet"),
    ("🛠️", "CNN (From Scratch)")
]

model_details = {
    "VGG16": "VGG16 is a deep CNN with 16 layers that uses very small (3x3) convolution filters. It is known for its simplicity and strong feature extraction ability, making it popular for transfer learning and baseline image classification tasks.",

    "MobileNet": "MobileNet is a lightweight and fast neural network designed for mobile and edge devices. It uses depthwise separable convolutions to drastically reduce parameters and computation while maintaining good accuracy.",

    "Xception": "Xception is an extreme version of Inception that fully relies on depthwise separable convolutions. It provides better accuracy than Inception with fewer parameters and is widely used in advanced image classification problems.",

    "EfficientNet": "EfficientNet scales depth, width, and resolution in a balanced way using compound scaling. It achieves high accuracy with fewer parameters, making it one of the most efficient CNN architectures available.",

    "ResNet": "ResNet introduces residual (skip) connections that allow very deep networks to be trained without vanishing gradient issues. It is highly effective for complex image recognition tasks.",

    "CNN (From Scratch)": "CNN trained from scratch is built without pretrained weights. It learns features directly from the dataset and is useful for understanding core deep learning concepts and when domain-specific features are required."
}




if page == "🏠Home":

    st.divider()
    st.markdown('<div class="holo-glow">🐟 Fish Classes</div>', unsafe_allow_html=True)

    cards_html = '<div class="card-grid">'

    for icon, name in fish_cards:
        cards_html += f"""
    <div class="fish-card">
        <div class="fish-icon">{icon}</div>
        <div class="fish-title">{name}</div>
        <div class="fish-line"></div>
        <div style="font-size:14px; color:#94a3b8; margin-top:8px;line-height:1.5">
            {fish_info.get(name.lower(), "Information not available.")}
        </div>
    """

        cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)
    

    st.divider()

    st.markdown('<div class="holo-glow">🧠 Models Used </div>', unsafe_allow_html=True)

    cards_html = '<div class="card-grid">'

    for icon, name in model_icons:
        info = model_details.get(name, "Model information not available.")

        cards_html += f"""
    <div class="fish-card">
        <div class="fish-icon">{icon}</div>
        <div class="fish-title">{name}</div>
        <div class="fish-line"></div>
        <div style="
            font-size:14px;
            color:#94a3b8;
            margin-top:8px;
            line-height:1.5;
        ">
            {info}
        </div>
        </div>"""

    cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
<div class="holo-glow">🔮 About the Fish Species Predictor</div>
<div style="
background:#020617;
margin-top:30px;
padding:18px;
border-radius:12px;
color:#e5e7eb;
font-size:16px;
line-height:1.8;
box-shadow:0 0 18px #22d3ee;
">
This AI-powered Fish Species Predictor uses a deep learning CNN models to automatically identify fish species from images.

The system is optimized for:
⚡ **High accuracy**
⚡ **Fast inference**
⚡ **Low hardware dependency**

It is designed to work efficiently on **real-world images** captured from
cameras, mobile devices, and underwater systems.

</div>
""", unsafe_allow_html=True)

    st.divider()
    
    st.markdown("""
<div class="holo-glow">🚀 What This Can Become</div>

<div style="
background:#020617;
margin-top:30px;
padding:18px;
border-radius:12px;
color:#e5e7eb;
font-size:16px;
line-height:1.9;
box-shadow:0 0 18px #22c55e;
">

🐟 <b>Smart Fisheries Management</b>  
• Automatic species identification  
• Catch monitoring & regulation  

🌊 <b>Marine Biodiversity Research</b>  
• Species population tracking  
• Ecosystem health analysis  

📱 <b>Mobile AI Applications</b>  
• Fishermen & traders can identify species instantly  
• Runs on low-end devices using MobileNet  

🛒 <b>Seafood Quality Control</b>  
• Detect mislabelled fish species  
• Prevent food fraud  

🤖 <b>Autonomous Underwater Drones</b>  
• Real-time fish detection & classification  
• Ocean exploration & surveillance  

</div>
""", unsafe_allow_html=True)


    st.divider()

mobilenet_metrics = {
    "Accuracy": "99.53%",
    "F1 Score": "96.91%",
    "Precision": "97%",
    "Recall": "99%"
}


if page == "🔍 Prediction":

    BEST_MODEL_NAME = "MobileNet"

    selected_model = st.sidebar.selectbox("Choose Model", model_files.keys())

    if selected_model == BEST_MODEL_NAME:
        st.sidebar.markdown("✅ **This is the Best Model**")
    model = tf.keras.models.load_model(model_files[selected_model], compile=False)
    uploaded_file = st.sidebar.file_uploader("📤 Upload Fish Image", type=["jpg", "png", "jpeg"])

    st.divider()
# ---------------- HEADER ----------------

    st.markdown(f'<div class="animated-title">🏆 Best Performing Model:✨{BEST_MODEL_NAME}🔥</div>', unsafe_allow_html=True)


    mobilenet_metrics = {
    "Accuracy": 99.53,
    "F1 Score": 96.91,
    "Precision": 99.72,
    "Recall": 95.40
}

# Define custom icons for each metric
    metric_icons = {
    "Accuracy": "✅",
    "F1 Score": "🎯",
    "Precision": "📏",
    "Recall": "🔄"
}

    cards_html = '<div class="card-grid">'

    for metric, value in mobilenet_metrics.items():
        icon = metric_icons.get(metric, "📈")  # fallback icon if not defined
        cards_html += f"""
    <div class="fish-card">
        <div class="fish-icon">{icon}</div>
        <div class="fish-title">{metric}</div>
        <div class="fish-line"></div>
        <div style="font-size:24px; color:#94a3b8; margin-top:8px; line-height:2.5;">
            {value:.2f}%
    </div>
    """

        cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
    prediction = None
    img = None
    inference_ms = None
    fps = None


    if uploaded_file:

        img = Image.open(uploaded_file).resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        _ = model.predict(img_array, verbose=0)  # warm-up

        start_time = time.perf_counter()
        prediction = model.predict(img_array, verbose=0)
        end_time = time.perf_counter()

        inference_ms = (end_time - start_time) * 1000
        fps = 1000 / inference_ms if inference_ms > 0 else 0

        idx = np.argmax(prediction)
        confidence = prediction[0][idx] * 100
        species_name = class_names[idx]
        species_key = species_name.lower()


    st.divider()

    if "top3_animated" not in st.session_state:
        st.session_state.top3_animated = False

# ---------------- TOP-3 PREDICTIONS ----------------


    st.markdown('<div class="holo-glow">⭐ Top-3 Predictions</div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    if prediction is not None :

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
        font-size:30px;
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
            wiki = wiki_facts.get(species_key, "Wikipedia summary not available.")
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


    if inference_ms is not None:
        st.markdown(f"""
    <div class="floating-speed">
        ⚡ {inference_ms:.2f} ms<br>
        🚀 {fps:.1f} FPS
    </div>
    """, unsafe_allow_html=True)

    st.divider()
# ----------------------------------------------
# FOOTER
# ----------------------------------------------
st.markdown("""
            <div style="text-align:center; color:#94a3b8; font-size:16px; margin-top:20px;">
<div class="footer glow">
    Built with 💙 & 🧠 using <b>Streamlit</b> | Designed by <b>Sudhakar M</b><br>
    <small>© 2026 Fish Image Classifier</small>,
</div>
""", unsafe_allow_html=True)


