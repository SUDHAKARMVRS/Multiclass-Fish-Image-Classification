import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tr_path = r'C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\train'
va_path = r'C:\Users\SRI SHANKARI\Desktop\Sudhakar\Data Science\Project\Multiclass Fish Image Classification\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\val'
train_dataset=keras.utils.image_dataset_from_directory(
    tr_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(224,224),
    batch_size=32
)
val_dataset=keras.utils.image_dataset_from_directory(
    va_path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224,224),
    batch_size=32
)

model=Sequential()

# Block 1
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 4
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.0001),metrics=['accuracy'])

model.summary()