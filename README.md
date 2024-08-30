# Image Classification with Convolutional Neural Networks

This repository contains two Jupyter notebooks that implement Convolutional Neural Networks (CNNs) for image classification. The models are designed to classify images into six different categories:

- **Buildings**
- **Forest**
- **Glacier**
- **Mountain**
- **Sea**
- **Street**

[**Here is the dataset**](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)


## Notebooks

### 1. `cnn_model.ipynb`
This notebook contains the implementation of a CNN model built from scratch using popular deep learning libraries such as TensorFlow or PyTorch. The model is trained on a dataset consisting of images from the six classes listed above.

#### Key Sections:
- **Data Loading and Preprocessing**: Images are loaded, resized, and normalized to prepare them for training.
- **Model Architecture**: A Convolutional Neural Network is constructed with multiple layers including convolutional, pooling, and fully connected layers.
- **Training**: The model is trained on the dataset, and key metrics such as accuracy and loss are recorded.
- **Evaluation**: The trained model is evaluated on a validation set to assess its performance.

### 2. `pretrained_cnn_model.ipynb`
This notebook uses a pre-trained CNN model, specifically **Inception V3**, for the same image classification task. Transfer learning is utilized to adapt the pre-trained model to our specific dataset.

#### Key Sections:
- **Pre-trained Model Loading**: The Inception V3 model is loaded and its top layers are modified to fit the current classification task.
- **Fine-tuning**: The model is fine-tuned on the dataset to improve accuracy.
- **Evaluation**: The fine-tuned model is evaluated on the validation set and compared with the scratch model.

## Dataset
The dataset used in these notebooks consists of images categorized into six classes: Buildings, Forest, Glacier, Mountain, Sea, and Street. Each image is pre-processed (resized, normalized) before being fed into the models.

## Usage
1. Clone this repository.
2. Ensure that you have all the required dependencies installed. You can install the necessary libraries using:
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter notebooks in your preferred environment (e.g., Jupyter Lab, Jupyter Notebook).
4. Run the cells sequentially to train and evaluate the models.

## Results
The performance of both the scratch-built CNN model and the pre-trained model is compared based on their accuracy and loss on the validation dataset.

## Conclusion
The notebooks demonstrate how to build and fine-tune CNN models for image classification tasks. The model built from scratch showed signs of **overfitting**, which can be attributed to the model's insufficient complexity and the lack of data augmentation techniques. Users can experiment with different architectures and pre-trained models to achieve better performance on the given dataset.
