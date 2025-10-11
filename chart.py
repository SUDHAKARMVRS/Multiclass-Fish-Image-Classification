import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_data = datagen.flow_from_directory(
    r"C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\val",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(val_data.class_indices.keys())
model_dir = "models/"

for model_file in os.listdir(model_dir):
    if model_file.endswith(".h5"):
        print(f"\n📌 Evaluating {model_file}...")
        model = tf.keras.models.load_model(os.path.join(model_dir, model_file))

        y_true = val_data.classes
        y_pred = np.argmax(model.predict(val_data), axis=1)

        print(classification_report(y_true, y_pred, target_names=class_names))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        plt.title(f"Confusion Matrix - {model_file}")
        plt.savefig(f"{model_file}_cm.png")
        plt.close()
