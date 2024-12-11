import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('binary_classifier_v1.keras')

# Set up data generators for testing
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Define directory for testing data
test_data_dir = 'Datasets/Binary/Testing'  # Replace with actual path

# Prepare the testing dataset
test_dataset = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(512, 512),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Ensure the order of images matches the true labels
)

# Predict the class probabilities for the test dataset
y_pred_probs = model.predict(test_dataset)

# Get the predicted classes (use a threshold of 0.5 for binary classification)
y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()

# Get the true labels from the test dataset
y_true = test_dataset.classes

# Create a confusion matrix using TensorFlow
conf_matrix = tf.math.confusion_matrix(y_true, y_pred_classes)

# Convert confusion matrix to percentages
conf_matrix_percentage = conf_matrix / tf.reduce_sum(conf_matrix) * 100

# Plot the confusion matrix with percentages
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percentage, annot=True, cmap='Blues', fmt='.2f', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentages)')
plt.show()

# To include probabilities for matches and mismatches
for i in range(len(y_true)):
    print(f"True Label: {y_true[i]}, Predicted Label: {y_pred_classes[i]}, Probability: {y_pred_probs[i][0]}")

# Optionally: Additional evaluation metrics
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['Negative', 'Positive']))
