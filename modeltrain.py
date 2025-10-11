# TRANSFER LEARNING
from tensorflow.keras.applications import VGG16,MobileNetV2,ResNet50,InceptionV3,Xception,DenseNet121

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, Xception
from tensorflow.keras import layers, models

# Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    r"C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\train",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    r"C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\val",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Pretrained Models
pretrained_models = {
    "VGG16": VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3)),
    "ResNet50": ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3)),
    "MobileNet": MobileNet(weights="imagenet", include_top=False, input_shape=(224,224,3)),
    "InceptionV3": InceptionV3(weights="imagenet", include_top=False, input_shape=(224,224,3)),
    "Xception": Xception(weights="imagenet", include_top=False, input_shape=(224,224,3))
}

results = {}

# Train each model
for name, base_model in pretrained_models.items():
    print(f"\n🔹 Training {name}...")
    base_model.trainable = False # Freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(train_data.num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(train_data, validation_data=val_data, epochs=5, verbose=1)

    # Save model
    model.save(f"models/{name}.h5")

    # Track last validation accuracy
    val_acc = history.history['val_accuracy'][-1]
    results[name] = val_acc

# Print results
print("\n📊 Validation Accuracies:")
for k,v in results.items():
    print(f"{k}: {v:.4f}")

best_model = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model} with {results[best_model]:.4f}")
print(f"✅ Model saved at models/{best_model}.h5")