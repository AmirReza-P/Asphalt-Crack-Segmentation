# CNN for Asphalt Crack Segmentation

![Project Timeline](https://img.shields.io/badge/Timeline-Spring%202025-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-green)

## 📌 Project Overview

This project implements an end-to-end deep learning pipeline for infrastructure maintenance, specifically designed to detect and segment cracks in asphalt surfaces. It features a custom **Encoder-Decoder Convolutional Neural Network (CNN)** built from scratch in **PyTorch**, capable of performing precise pixel-wise segmentation.

The system is designed to automate the assessment of road conditions, helping to streamline maintenance efforts by accurately identifying surface defects.

## 🚀 Key Features

* **Custom Deep Learning Architecture**: Designed and trained an Encoder-Decoder CNN from scratch to capture both local features and global context for accurate segmentation.
* **End-to-End Pipeline**: Implemented a full pipeline handling everything from raw image loading to model inference.
* **Custom Data Generator**: Built a robust PyTorch `Dataset` class (`AsphaltDataset`) capable of parsing COCO-format annotations and generating binary masks dynamically.
* **Image Preprocessing & Augmentation**: Integrated OpenCV and PIL for preprocessing and data augmentation (rotation, scaling, noise) to improve model generalization.
* **Performance Metrics**: Evaluated model accuracy using the **Intersection over Union (IoU)** metric to ensure high-quality segmentation results.

## 🛠️ Technologies Used

* **Deep Learning**: PyTorch, Torchvision
* **Image Processing**: OpenCV (cv2), PIL (Python Imaging Library)
* **Data Handling**: NumPy, Pandas, PyCOCOTools
* **Visualization**: Matplotlib
* **Environment**: Jupyter Notebook / Python

## 📂 Dataset

The project uses the **Asphalt Segmentation Dataset** (COCO format).
* **Input**: RGB images of asphalt surfaces.
* **Annotations**: JSON file containing segmentation masks in COCO format.
* **Resolution**: Images are resized to `512x512` for training.

## ⚙️ Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/asphalt-crack-segmentation.git](https://github.com/yourusername/asphalt-crack-segmentation.git)
    cd asphalt-crack-segmentation
    ```

2.  Install the required dependencies:
    ```bash
    pip install torch torchvision opencv-python pandas matplotlib pycocotools tqdm
    ```

## 🧠 Model Architecture

The model follows a classic **Encoder-Decoder** structure:
1.  **Encoder**: Convolutional layers extract hierarchical features from the input image, downsampling spatial dimensions while increasing depth.
2.  **Decoder**: Transposed convolutions (or upsampling layers) reconstruct the spatial resolution to generate a pixel-wise mask.
3.  **Output**: A final activation layer produces a binary mask representing crack vs. background.

## 💻 Usage

To run the training pipeline, execute the Jupyter Notebook:

1.  Open the notebook:
    ```bash
    jupyter notebook asphalt-crack-segmentation.ipynb
    ```
2.  Update the `IMAGE_DIR` and `ANNOTATION_FILE` paths in the second cell to point to your local dataset.
3.  Run all cells to train the model.

## 📊 Evaluation

The model performance is tracked using:
* **Training/Validation Loss**: Cross Entropy / Dice Loss.
* **IoU (Intersection over Union)**: Measures the overlap between the predicted crack area and the ground truth.
