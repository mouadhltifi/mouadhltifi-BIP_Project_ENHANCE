import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import math

# Define the model using Functional API with MaxPooling2D
inputs = Input(shape=(512, 512, 3), name="input_layer")

# First convolutional block with pooling
x = Conv2D(64, (5, 5), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)  # Pooling reduces feature map dimensions

# Second convolutional block with pooling
x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Third convolutional block with pooling
x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
x = BatchNormalization()(x)

# Fourth convolutional block with pooling
x = Conv2D(512, (3, 3), activation='relu', padding='same', strides=2)(x)
x = BatchNormalization()(x)

# Final convolutional block for heatmap generation
x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3))(x)

# Global average pooling to reduce tensor size
x = GlobalAveragePooling2D()(x)

# Fully connected layers
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.4)(x)
outputs = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs, name="brain_tumor_binary_classifier")

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set up data generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Define directories for training and testing data
train_data_dir = 'Datasets/Binary/Training'  # Replace with actual path
test_data_dir = 'Datasets/Binary/Testing'  # Replace with actual path

# Prepare training, validation, and testing datasets
train_dataset = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(512, 512),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_dataset = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(512, 512),
    batch_size=16,
    class_mode='binary',
    subset='validation'
)


test_dataset = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(512, 512),
    batch_size=16,
    class_mode='binary'
)

# Calculate steps per epoch
train_steps = math.ceil(train_dataset.samples / train_dataset.batch_size)
val_steps = math.ceil(val_dataset.samples / val_dataset.batch_size)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint(
    'binary_temp_best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train the model
model.fit(
    train_dataset,
    steps_per_epoch=train_steps,
    epochs=200,
    validation_data=val_dataset,
    validation_steps=val_steps,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Restore best weights (for safety)
model.load_weights('binary_temp_best_model.keras')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the final model explicitly
model.save('binary_classifier_v1.keras')