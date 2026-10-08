# Image Classification Using a Pretrained Model

## CodeOrbit Tech - AI Internship
### Task 3: Image Classification

This project uses a pretrained MobileNetV2 model with TensorFlow to classify sample images.

## Features

- Uses a pretrained MobileNetV2 model
- Classifies images using ImageNet labels
- Displays the top 3 predictions
- Shows prediction confidence percentages
- Classifies both cat and apple images

## Technologies Used

- Python
- TensorFlow
- Keras
- Pillow
- MobileNetV2

## How It Works

1. Load the pretrained MobileNetV2 model.
2. Load the input image.
3. Resize the image to 224 x 224 pixels.
4. Preprocess the image.
5. Send the image to the model.
6. Display the top 3 predicted labels and confidence scores.

## Sample Images

### Cat Image

File name: `sample.jpg`

Prediction:
- tabby: 50.16%
- tiger_cat: 38.33%
- lynx: 2.02%

### Apple Image

File name: `apple.jpg`

Prediction:
- pomegranate: 56.99%
- strawberry: 8.03%
- fig: 6.68%

## Project Structure

```text
DCodeOrbit_ImageClassification/
│
├── image_classification.py
├── sample.jpg
├── apple.jpg
└── README.md