import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
import cv2
import os

# Load the trained model
model = load_model('multiclass_classifier_v1.keras')

# Function to preprocess the input image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(512, 512))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to generate a heatmap
def generate_heatmap(model, image_path, last_conv_layer_name):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Create a model that maps the input image to the activations of the last conv layer and the predictions
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for the input image with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_pred_index = tf.argmax(predictions[0])  # Get index of top prediction
        loss = predictions[:, top_pred_index]  # Use the top predicted class

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Compute channel-wise mean of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by the "importance" of the channel with regard to the predicted class
    conv_outputs = conv_outputs[0]
    heatmap = np.mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize the heatmap using percentile for better focus
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

# Function to overlay the heatmap on the original image
def overlay_heatmap(heatmap, image_path, alpha=0.2):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))

    # Resize heatmap to match the image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)  # Use a less intense colormap

    # Overlay the heatmap on the image
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return overlay

# Main function to classify an image and generate a heatmap
def classify_image_with_heatmap(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Predict the class
    predictions = model.predict(img_array)
    class_indices = ['Glioma', 'Meningioma', 'Pituitary']  # Update with your class names
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_indices[predicted_class_index]

    # Generate heatmap
    heatmap = generate_heatmap(model, image_path, last_conv_layer_name='conv2d_4')

    # Overlay heatmap on the original image
    overlay = overlay_heatmap(heatmap, image_path)

    # Save the heatmap with a similar name to the input image
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    heatmap_path = f"{name}_heatmap{ext}"
    cv2.imwrite(heatmap_path, overlay)
    print(f"Heatmap saved to {heatmap_path}")

    # Save the original image with a resized version
    resized_image_path = f"{name}_resized{ext}"
    resized_img = cv2.imread(image_path)
    resized_img = cv2.resize(resized_img, (512, 512))
    cv2.imwrite(resized_image_path, resized_img)
    print(f"Resized image saved to {resized_image_path}")

    # Display the results
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(load_img(image_path, target_size=(512, 512)))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Heatmap - {predicted_class}")
    plt.imshow(overlay)
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'Datasets/Multiclass/Training/pituitary/Tr-pi_0123.jpg'  # Replace with your image path
classify_image_with_heatmap(image_path)
