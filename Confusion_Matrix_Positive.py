import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('brain_tumor_multiclass_classifier_v4.keras')

# Set up data generators for testing
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Define directory for testing data
test_data_dir = 'Datasets/Positive_Classification/Testing'  # Replace with actual path

# Prepare the testing dataset
test_dataset = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(256, 256),
    batch_size=16,
    class_mode='categorical',
    shuffle=False  # Ensure the order of images matches the true labels
)

# Predict the class probabilities for the test dataset
y_pred_probs = model.predict(test_dataset)

# Get the predicted classes
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Get the true labels from the test dataset
y_true = test_dataset.classes

# Create a confusion matrix using TensorFlow
conf_matrix = tf.math.confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=test_dataset.class_indices.keys(), yticklabels=test_dataset.class_indices.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# To include probabilities for matches and mismatches
for i in range(len(y_true)):
    print(f"True Label: {y_true[i]}, Predicted Label: {y_pred_classes[i]}, Probabilities: {y_pred_probs[i]}")

# Optionally: Additional evaluation metrics
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=test_dataset.class_indices.keys()))
