import os
import numpy as np
import matplotlib.pyplot as plt
import ssl
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# Disable SSL verification to avoid SSL certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

# Enable mixed precision training
set_global_policy('mixed_float16')

# Disable XLA due to compatibility issues with Metal plugin
# tf.config.optimizer.set_jit(False)

# Ensure TensorFlow uses the GPU (M1/M2) with tensorflow-metal
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0].name}")
    else:
        print("No GPU detected. Running on CPU.")
except RuntimeError as e:
    print(e)

# Paths to your dataset
train_dir = "Datasets/Training"  # Update with your path if needed
test_dir = "Datasets/Testing"    # Update with your path if needed

# 1. Data Loading and Preprocessing
# Data preparation for binary classification (notumor vs. combined tumor types)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),  # Additional augmentation
    shear_range=0.2  # Additional augmentation
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(512, 512),
    batch_size=16,
    class_mode="binary",  # Binary classification for tumor vs. no tumor
    classes=['notumor', 'glioma', 'meningioma', 'pituitary'],  # Combine glioma, meningioma, pituitary into a generic 'tumor' class
    subset=None
)

# Test data generator (no augmentation)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(512, 512),
    batch_size=16,
    class_mode="binary",  # Binary classification for tumor vs. no tumor
    classes=['notumor', 'glioma', 'meningioma', 'pituitary'],  # Combine glioma, meningioma, pituitary into a generic 'tumor' class
    subset=None,
    shuffle=False
)

# 2. Model Building for Tumor Detection
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(512, 512, 3))
base_model.trainable = False  # Freeze the base model initially
model_tumor_detection = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary classification: tumor vs. no tumor
])

model_tumor_detection.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

# Model Training for Tumor Detection
history_tumor_detection = model_tumor_detection.fit(
    train_data,
    validation_data=test_data,
    epochs=20,
    steps_per_epoch=train_data.samples // train_data.batch_size + (train_data.samples % train_data.batch_size > 0),
    validation_steps=test_data.samples // test_data.batch_size + (test_data.samples % test_data.batch_size > 0),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), ModelCheckpoint('best_model.keras', save_best_only=True)]
)

# Save the Tumor Detection Model
model_tumor_detection.save("tumor_detection_model.keras")

# 3. Predict Tumor Presence
test_data.reset()  # Reset the generator for consistent predictions
predictions = model_tumor_detection.predict(test_data)
predicted_labels = (predictions > 0.5).astype(int)

# Extract tumor-positive images for further classification
tumor_indices = np.where(predicted_labels == 1)[0]
tumor_images = [test_data.filepaths[i] for i in tumor_indices]

# Load only tumor images for classification
tumor_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
tumor_data = tumor_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': tumor_images}),
    x_col='filename',
    target_size=(512, 512),
    batch_size=16,
    class_mode=None,
    shuffle=False
)

# 4. Model Building for Tumor Type Classification
base_model_tumor_type = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(512, 512, 3))
base_model_tumor_type.trainable = False  # Freeze the base model initially

model_tumor_type = Sequential([
    base_model_tumor_type,
    GlobalAveragePooling2D(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.6),
    Dense(3, activation="softmax")  # Adjust for the number of tumor classes (glioma, meningioma, pituitary)
])

# Unfreeze a portion of the base model for fine-tuning
for layer in base_model_tumor_type.layers[-len(base_model_tumor_type.layers) // 4:]:
    layer.trainable = True

model_tumor_type.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=[categorical_accuracy])

# Model Training for Tumor Type Classification
tumor_train_dir = "Datasets/Tumor_Training"  # Update with directory containing only tumor images
tumor_test_dir = "Datasets/Tumor_Testing"  # Update with directory containing only tumor images

tumor_train_data = train_datagen.flow_from_directory(
    tumor_train_dir,
    target_size=(512, 512),
    batch_size=16,
    class_mode="categorical"
)

tumor_validation_data = train_datagen.flow_from_directory(
    tumor_test_dir,
    target_size=(512, 512),
    batch_size=16,
    class_mode="categorical"
)

history_tumor_type = model_tumor_type.fit(
    tumor_train_data,
    validation_data=tumor_validation_data,
    epochs=30,
    steps_per_epoch=tumor_train_data.samples // tumor_train_data.batch_size + (tumor_train_data.samples % tumor_train_data.batch_size > 0),
    validation_steps=tumor_validation_data.samples // tumor_validation_data.batch_size + (tumor_validation_data.samples % tumor_validation_data.batch_size > 0),
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), ModelCheckpoint('best_tumor_model.keras', save_best_only=True)]
)

# Save the Tumor Type Classification Model
model_tumor_type.save("tumor_type_classifier.keras")

# Continue with evaluations and visualizations as appropriate...
