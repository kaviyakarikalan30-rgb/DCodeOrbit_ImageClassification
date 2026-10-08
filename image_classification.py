# CodeOrbit Tech - Image Classification
# Task 3: Artificial Intelligence Internship

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.utils import load_img, img_to_array

# Load pretrained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Images to classify
image_paths = ["sample.jpg", "apple.jpg"]

for image_path in image_paths:

    print("\n------------------------------")
    print("Image:", image_path)
    print("------------------------------")

    # Load and prepare image
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)

    # Display top 3 predictions
    results = decode_predictions(predictions, top=3)[0]

    print("Image Classification Results:")

    for _, label, confidence in results:
        print(f"{label}: {confidence * 100:.2f}%")