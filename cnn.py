import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Params
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    r"C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    r"C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(train_data.num_classes, activation="softmax")
])

# Compile & Train
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save
os.makedirs("models", exist_ok=True)
model.save("models/cnn_scratch.h5")
print("✅ CNN model saved at models/cnn_scratch.h5")