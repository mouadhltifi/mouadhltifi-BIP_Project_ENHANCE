import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to dynamically detect all Conv2D layers
def get_all_conv_layers(model):
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append(layer.name)
    if not conv_layers:
        raise ValueError("No Conv2D layers found in the model.")
    print(f"Conv2D layers found: {conv_layers}")
    return conv_layers

# Generate heatmap using multiple convolutional layers
def generate_composite_heatmap(model, img_path, output_path="heatmap_output.png"):
    # Get all Conv2D layers
    conv_layer_names = get_all_conv_layers(model)

    # Load and preprocess the image
    img = load_img(img_path, target_size=(512, 512))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize and add batch dimension

    # Convert image array to tf.Tensor
    img_tensor = tf.convert_to_tensor(img_array)

    # Create a gradient model for all selected Conv2D layers
    outputs = [model.get_layer(layer_name).output for layer_name in conv_layer_names] + [model.output]
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=outputs)

    # Compute gradients for the top predicted class
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(img_tensor)
        conv_outputs = grad_model(img_tensor, training=False)  # Get all feature maps and predictions
        feature_maps = conv_outputs[:-1]  # Exclude model output
        predictions = conv_outputs[-1]  # Model predictions
        top_class_index = tf.argmax(predictions[0])
        top_class_channel = predictions[:, top_class_index]

    # Aggregate heatmaps from all Conv2D layers
    composite_heatmap = None
    for feature_map in feature_maps:
        grads = tape.gradient(top_class_channel, feature_map)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Importance weights
        feature_map = feature_map[0]  # Remove batch dimension
        layer_heatmap = feature_map @ pooled_grads[..., tf.newaxis]  # Weighted sum
        layer_heatmap = tf.squeeze(layer_heatmap)
        layer_heatmap = tf.maximum(layer_heatmap, 0) / tf.reduce_max(layer_heatmap)  # Normalize

        if composite_heatmap is None:
            composite_heatmap = tf.image.resize(layer_heatmap[..., tf.newaxis], (512, 512))  # Resize to match input
        else:
            composite_heatmap += tf.image.resize(layer_heatmap[..., tf.newaxis], (512, 512))  # Add resized heatmap

    # Cleanup the persistent GradientTape
    del tape

    # Normalize the composite heatmap
    composite_heatmap = composite_heatmap / tf.reduce_max(composite_heatmap)

    # Display and save the heatmap
    plt.matshow(composite_heatmap.numpy())
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)  # Save as PNG
    print(f"Heatmap saved to {output_path}")
    plt.show()

# Classify and generate composite heatmap
def classify_and_generate_composite_heatmap(model, img_path, output_path="heatmap_output.png"):
    # Load and preprocess the image
    img = load_img(img_path, target_size=(512, 512))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Add batch dimension and normalize

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_labels = ["Class 0", "Class 1", "Class 2"]  # Replace with your class names
    print(f'Predicted Class: {class_labels[predicted_class]}')

    # Generate and save composite heatmap
    generate_composite_heatmap(model, img_path, output_path)


# Example usage
img_path = 'Datasets/Multiclass/Testing/meningioma/Te-me_0298.jpg'
output_path = 'heatmap_output.png'

# Load the saved model
try:
    model = tf.keras.models.load_model('best_model.keras')
    classify_and_generate_composite_heatmap(model, img_path, output_path)
except Exception as e:
    print(f"An error occurred: {e}")
