# 🧠 CNN for Image Classification using CIFAR-10

🧑‍💻 Author
Vanshika Sood
B.Tech CSE | Deep Learning Project (October 2025)


## 📘 Problem Statement
The objective of this project is to design and implement a **Convolutional Neural Network (CNN)** capable of classifying images from the **CIFAR-10 dataset** into 10 distinct object categories.  
The aim is to understand the working of CNN layers, feature extraction, and performance evaluation on real-world image data.

---

## 📊 Dataset Used
- **Dataset Name:** CIFAR-10  
- **Source:** Available in Keras (`tf.keras.datasets.cifar10`)  
- **Number of Images:** 60,000  
  - 50,000 for training  
  - 10,000 for testing  
- **Image Size:** 32x32 pixels (RGB)  
- **Number of Classes:** 10  
  - airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

## 🧩 Model Architecture
The CNN model is built using **TensorFlow/Keras** and includes the following layers:

| Layer Type | Details | Activation |
|-------------|----------|-------------|
| Conv2D | 32 filters, 3x3 kernel | ReLU |
| MaxPooling2D | 2x2 pool size | - |
| Conv2D | 64 filters, 3x3 kernel | ReLU |
| MaxPooling2D | 2x2 pool size | - |
| Conv2D | 128 filters, 3x3 kernel | ReLU |
| Flatten | Converts 2D → 1D | - |
| Dense | 128 units | ReLU |
| Dropout | 0.5 | - |
| Dense | 10 units | Softmax |

**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  
**Metrics:** Accuracy  

---

## ⚙️ Instructions to Run the Code

### 🪜 Step-by-Step Guide

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/CNN-Image-Classification.git
   cd CNN-Image-Classification

2. Install Dependencies
pip install -r requirements.txt

3. Run the notebook or script
python cnn_cifar10.ipynb

4. Train and evaluate the model
cnn_cifar10_model.h5



Evaluation Metrics and Results
| Metric              | Description                     | Value (approx.) |
| ------------------- | ------------------------------- | --------------- |
| Training Accuracy   | Accuracy on training data       | 85%             |
| Validation Accuracy | Accuracy on validation data     | 78%             |
| Test Accuracy       | Accuracy on unseen data         | 77–80%          |
| Loss                | Sparse Categorical Crossentropy | ~0.75           |


🔍 Sample Results:

Model successfully classifies most images with high accuracy.

Overfitting was reduced using Dropout and normalization.

Confusion matrix visualized category-wise performance.

📊 Visualizations

1. Accuracy vs. Epochs


2. Confusion Matrix


💡 Key Insights

Increasing convolutional layers improves feature extraction.

Dropout (0.5) effectively prevents overfitting.

With more epochs or transfer learning (VGG16/ResNet), accuracy can exceed 90%.

🧠 Future Enhancements

Add data augmentation (rotation, flipping, zoom).

Try Transfer Learning with pre-trained models like VGG16 or MobileNet.

Deploy model as a web app using Flask or Streamlit.
