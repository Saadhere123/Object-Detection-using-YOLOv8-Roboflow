The goal of this project is to train a YOLOv8 model on a custom dataset and achieve high-accuracy object detection.
The workflow includes:

Dataset hosting and preprocessing with Roboflow

Training YOLOv8 (yolov8s.pt) on the custom dataset

Validating the trained model

Running predictions on test images

Visualizing results such as loss curves, accuracy metrics, and detection outputs

🔧 Tech Stack

Python

Google Colab

YOLOv8 (Ultralytics)

Roboflow

PyTorch

📂 Dataset

The dataset is imported directly from Roboflow:

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("workspace-name").project("project-name")
version = project.version(1)
dataset = version.download("yolov8")


Dataset format: YOLOv8

Contains train, valid, and test splits

Auto-generated data.yaml file used for training and validation

🧠 Model Training

YOLOv8 training command used:

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=10 imgsz=640

✔ Training Highlights:

Loss values continually decreased (model improving)

Precision, Recall, and mAP values increased (better detection)

Confusion matrix & training curves confirm good model learning

🧪 Model Validation
!yolo task=detect mode=val model=/content/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml


Validation ensures the model performs well on unseen data.

🔍 Prediction on Test Images
!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.5 source={dataset.location}/test/images


The results are saved in:

/content/runs/detect/predict/


All detected images are displayed using:

for image_path in glob.glob('/content/runs/detect/predict/*.jpg'):
    display(Image(filename=image_path))

📊 Output Visuals Included

Confusion Matrix

Training Curves (loss, precision, recall, mAP)

Prediction Examples

These visuals help understand how well the model learned and how accurately it detects objects.

🏆 Results

The model successfully detects objects with good accuracy.

Metrics improved steadily during training.

Predictions on test images are clear and accurate.

This confirms that the YOLOv8 model training was successful.

📦 Future Improvements

Increase dataset size

Train for more epochs

Use larger YOLO versions (YOLOv8m / YOLOv8l)

Deploy model using FastAPI or Streamlit

Convert model to ONNX / TensorRT for faster inference
