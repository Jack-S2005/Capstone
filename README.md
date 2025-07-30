# Celebrity Face Recognition Project

A comprehensive machine learning project that implements and compares multiple approaches for celebrity face recognition using a dataset of 17 celebrities with approximately 100 images each.

## 📋 Project Overview

This project explores various machine learning techniques for face recognition, including traditional methods like K-Nearest Neighbors (KNN) and modern deep learning approaches like Convolutional Neural Networks (CNN) and Transfer Learning. The goal is to identify which celebrities are present in facial images with high accuracy.

## 🎯 Key Features

- **Multiple ML Approaches**: KNN, Custom CNN, Transfer Learning (ResNet50V2), and Face Recognition Library
- **Comprehensive Evaluation**: Per-celebrity accuracy analysis, confusion matrices, and detailed performance metrics
- **Advanced Techniques**: Triplet loss implementation for improved embeddings
- **Data Preprocessing**: Face detection, cropping, and image normalization
- **Grid Search Optimization**: Automated hyperparameter tuning for KNN

## 📊 Dataset

- **Source**: Celebrity Face Image Dataset from Kaggle
- **Size**: ~1,700 images across 17 celebrities
- **Format**: RGB images, resized to 128x128 pixels
- **Classes**: 17 different celebrities with balanced representation

### Dataset Statistics
- Total images: ~1,700
- Number of celebrities: 17
- Average images per celebrity: ~100
- Image resolution (Downsampled): 128x128x3 (RGB) 

## 🛠️ Technical Implementation

### 1. Data Preprocessing
- **Face Detection**: Using `face_recognition` library to detect and crop faces
- **Image Resizing**: Standardized to 128x128 pixels
- **Normalization**: Pixel values scaled to [0,1] range
- **Train/Test Split**: 80/20 split with stratification

### 2. Machine Learning Models

#### K-Nearest Neighbors (KNN)
- **Best Parameters**: k=11, metric='manhattan', weights='distance'
- **Grid Search**: Comprehensive hyperparameter optimization
- **Performance**: Baseline comparison method

#### Custom Convolutional Neural Network
- **Architecture**: 4 convolutional layers with dropout and batch normalization
- **Input**: 128x128x3 RGB images
- **Output**: 17-class classification (softmax)

#### Transfer Learning (ResNet50V2)
- **Base Model**: Pre-trained ResNet50V2 on ImageNet
- **Fine-tuning**: Custom classification layers
- **Advantages**: Leverages pre-trained features

#### Face Recognition Library
- **Method**: Face encodings with distance-based matching
- **Features**: 128-dimensional face embeddings
- **Tolerance**: Optimized threshold for face matching

#### Triplet Loss Implementation
- **Purpose**: Learn better face embeddings
- **Architecture**: Custom triplet loss with embedding model
- **Training**: Anchor-positive-negative triplets

## 📈 Results

### Model Performance Comparison
| Method | Accuracy | Rank | Notes |
|--------|----------|------|-------|
| Face Recognition Library | 0.998 | 1 | Best performing method |
| ResNet50V2 (Transfer Learning) | 0.569 | 2 | Best deep learning approach |
| Custom CNN | 0.527 | 3 | Baseline deep learning |
| KNN (k=11, manhattan) | 0.293 | 4 | Traditional machine learning |

### Performance Analysis
- **Mean accuracy** across all methods: 0.597
- **Standard deviation**: 0.254
- **Best performing method**: Face Recognition Library (0.998)

### Key Findings
1. **Face Recognition Library** significantly outperforms all other methods with 99.8% accuracy
2. **Transfer Learning (ResNet50V2)** is the best deep learning approach at 56.9%
3. **Custom CNN** provides baseline deep learning performance at 52.7%
4. **KNN** shows lower performance at 29.3%
5. **Face cropping and preprocessing** were crucial for the face recognition library's success

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Installation
1. Clone the repository
2. Install dependencies
3. Download the dataset using Kaggle API
4. Run the Jupyter notebook

### Usage
```python

# Run face recognition analysis
# See capstone.ipynb for complete implementation
```

## 📁 Project Structure
```
├── capstone.ipynb              # Main analysis notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file

```

## 🔧 Dependencies

### Core Libraries
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **OpenCV**: Image processing
- **face_recognition**: Face detection and encoding
- **matplotlib/seaborn**: Visualization
- **numpy/pandas**: Data manipulation

### Key Packages
```
tensorflow==2.19.0
scikit-learn==1.7.0
opencv-python==4.12.0.88
face-recognition==1.3.0
matplotlib==3.10.3
numpy==2.1.3
pandas==2.3.0
```

## 📊 Analysis Features

### 1. Per-Celebrity Performance
- Individual accuracy analysis for each celebrity
- Identification of challenging cases
- Performance visualization with color-coded bars

### 2. Confusion Matrices
- Detailed error analysis
- Identification of commonly confused celebrities
- Model comparison through confusion patterns

### 3. Training History
- Loss and accuracy curves
- Overfitting detection
- Model convergence analysis

### 4. Hyperparameter Optimization
- Grid search for KNN parameters
- Cross-validation results
- Optimal parameter identification

## 🎯 Future Improvements

1. **Data Augmentation**: Expand dataset with synthetic variations
2. **Data Extraction**: Explore the use of different facial extractors on the training
3. **Use Embeddings**: Optimize own CNN to use embeddings and see if an improvment materializes

## 📝 Methodology

### Experimental Design
1. **Data Preparation**: Face detection, cropping, and normalization
2. **Model Training**: Multiple approaches with consistent train/test split
3. **Evaluation**: Comprehensive metrics and visualizations
4. **Comparison**: Comparison across all methods
5. **Optimization**: Hyperparameter tuning for best performance

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Per-class Accuracy**: Individual celebrity performance
- **Confusion Matrix**: Detailed error analysis


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kaggle for providing the celebrity face dataset
- TensorFlow and scikit-learn communities for excellent documentation
- Face recognition library developers for the face detection tools

---

**Note**: This project is for educational and research purposes. Please ensure compliance with data privacy regulations when using face recognition technology.
