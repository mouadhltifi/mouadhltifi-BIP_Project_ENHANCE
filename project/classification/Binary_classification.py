import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up TF_CONFIG environment variable if doing multi-worker training (this is usually done automatically in distributed setups)
# Example (replace with your cluster's information):
# os.environ['TF_CONFIG'] = '{
#     "cluster": {
#         "worker": ["worker1.example.com:12345", "worker2.example.com:23456"],
#         "chief": ["chief.example.com:12345"]
#     },
#     "task": {"type": "worker", "index": 0}
# }'

# Define the model without a distribution strategy
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(512, 512, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # Convolutional layer for image input
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout to reduce overfitting
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set up data generators for training, validation, and testing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split training data into training and validation
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Define directories for training and testing data
train_data_dir = 'Datasets/Binary/Training'  # Replace with actual path

test_data_dir = 'Datasets/Binary/Testing'  # Replace with actual path

# Prepare training, validation, and testing datasets
train_dataset = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(512, 512),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_dataset = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(512, 512),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

test_dataset = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(512, 512),
    batch_size=32,
    class_mode='binary'
)

# Train the model
model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)

# Save the trained model
model.save('brain_tumor_classifier.keras')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')