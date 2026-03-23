# Streamlit Web App

> A multi-page Streamlit web application for image classification and object detection using MobileNetV2, VGG19, PyTorch CNN, and YOLOv3.

## Pages

| Page | Description | Model |
|------|-------------|-------|
| Image Classifier | Upload an image and classify it into 1000+ ImageNet categories | MobileNetV2 (Keras) |
| Number Recognizer | Draw a digit on canvas and get real-time MNIST recognition | Custom CNN (PyTorch) |
| OpenCV Classifier | Upload an image for deep classification with similarity score | VGG19 (Keras) |
| Object Detection | Detect objects in images or videos with bounding boxes | YOLOv3 |

## Prerequisites

- Python 3.8+
- pip

## Local Installation

```shell
# 1. Clone the repo and navigate into it

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download required model files (see below)

# 4. Run the app
streamlit run Image_Classifier.py
```

The app will be available at http://localhost:8501

## Downloading Model Files

The large model/weight files are not included in the repository. Download them and place them in the correct locations before running the app.

### MNIST CNN Model (`mnist_cnn.pt`)

Train the model locally using the included training script:

```shell
python train.py
```

This generates `mnist_cnn.pt` in the project root.

### YOLOv3 Weights (`Yolo_Weights_Config/yolov3.weights`)

Download the official YOLOv3 pre-trained weights (237 MB) from [pjreddie.com](https://pjreddie.com/darknet/yolo/) and place the file at:

```
Yolo_Weights_Config/yolov3.weights
```

### VGG19 Model (`image_classification.hdf5`)

This file is auto-generated on first run of the OpenCV Classifier page — VGG19 weights are downloaded from Keras and saved locally.

## Project Structure

```
Streamlit-Web-App/
├── Image_Classifier.py          # Main app (MobileNetV2 classifier)
├── layout.py                    # Shared UI utilities / footer
├── train.py                     # MNIST CNN training script
├── requirements.txt
├── pages/
│   ├── 2_Number_Recognizer.py   # MNIST digit drawing + recognition
│   ├── 3_OpenCV_Classifier.py   # VGG19 image classifier
│   └── 4_Object_Detection.py    # YOLOv3 image & video detection
└── Yolo_Weights_Config/
    └── yolov3.cfg               # YOLOv3 architecture config
```

## Customization

### Use your own Keras model

Replace the model loading line with your own `.h5` file saved via `model.save()`. See `pages/3_OpenCV_Classifier.py` for the pattern.

### Use a different pre-trained model

See [Keras Applications](https://keras.io/api/applications/) for available models such as DenseNet, ResNet, EfficientNet, etc.

## Dependencies

Key libraries used:

- [Streamlit](https://streamlit.io/) — web framework
- [TensorFlow / Keras](https://www.tensorflow.org/) — VGG19 & MobileNetV2
- [PyTorch](https://pytorch.org/) — MNIST CNN
- [OpenCV](https://opencv.org/) — image/video processing
- [streamlit-drawable-canvas](https://pypi.org/project/streamlit-drawable-canvas/) — digit drawing widget
