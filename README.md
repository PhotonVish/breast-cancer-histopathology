-> Developed a Deep Learning-based Histopathology Classifier using PyTorch and the ConvNeXt architecture to detect Invasive Ductal Carcinoma (IDC) with ~90% Accuracy and 0.95 AUC

-> Engineered a robust preprocessing pipeline using Elastic Transformations and Stochastic Depth regularization to handle high intra-class variance in medical imaging.

-> Optimized model performance using Transfer Learning and Test-Time Augmentation (TTA), reducing false positives while maintaining high sensitivity on an imbalanced dataset of 277k+ patches.

Breast Histopathology Image Classification

This project implements a deep learning pipeline to classify breast cancer histopathology images (IDC vs. Non-IDC) using the convnext_tiny architecture. It is specifically designed to maximize Accuracy while implementing robust strategies to prevent Overfitting in medical imaging datasets.
🛠️ Key Features

    Slide-Level Splitting: Ensures that all patches from a single patient (slide) stay within the same set (Train/Val/Test), preventing data leakage.

    Heavy Augmentation: Utilizes aggressive geometric and color transforms to improve model generalization.

    Regularization: Implements Dropout and Stochastic Depth (Drop Path) to mitigate memorization.

    Mixed Precision & Multi-GPU: Optimized for NVIDIA GPUs using torch.amp and DataParallel.

    Advanced Scheduling: Uses Cosine Annealing learning rate scheduling for smooth convergence.

📊 Project Workflow

The following flowchart illustrates the end-to-end pipeline from data ingestion to model evaluation:
Code snippet

graph TD
    A[Raw Data: Breast Histopathology Images] --> B{Slide-Level Grouping}
    B --> C[Stratified Split: Train/Val/Test]
    
    subgraph Training_Phase
    C --> D[Heavy Data Augmentation]
    D --> E[ConvNeXt-Tiny Model]
    E --> F[Loss Calculation: Weighted CrossEntropy]
    F --> G[Backpropagation & AdamW]
    G --> H[Cosine Annealing Scheduler]
    H --> E
    end
    
    subgraph Evaluation_Phase
    E --> I[Validation Monitoring: Accuracy/AUC]
    I -->|Best Metric| J[Checkpoint: best_model.pth]
    J --> K[Final Test Evaluation]
    end
    
    K --> L[Metrics: ACC, AUC, F1, Precision, Recall]
    L --> M[Visualization: Training Curves]

🚀 Getting Started
Prerequisites

    Python 3.8+

    PyTorch 2.0+

    timm (PyTorch Image Models)

    NVIDIA GPU with CUDA support

Installation
Bash

pip install torch torchvision timm pandas scikit-learn matplotlib pillow

Dataset Structure

The project supports two common Kaggle layouts:

    Layout A: slide_id/class_id/patch_name.png

    Layout B: Flat directory with filenames like slideID_class0.png

Update the KAGGLE_INPUT_DIR in the notebook to point to your data source.
🧠 Model Configuration

The model is initialized via the timm library with specific settings to combat overfitting:
Python

model = timm.create_model(
    "convnext_tiny", 
    pretrained=True, 
    num_classes=2, 
    drop_rate=0.3,       # Final layer dropout
    drop_path_rate=0.2   # Stochastic depth
)

📈 Performance Monitoring

The notebook tracks several key metrics to ensure clinical relevance:

    Accuracy: The primary target for this version.

    ROC-AUC: Measures the model's ability to distinguish between classes.

    F1-Score: Balances Precision and Recall, crucial for medical diagnosis.

📂 File Description

    v3-1-overfitting-acc.ipynb: The main training and evaluation notebook.

    best_model.pth: Full checkpoint including optimizer state.

    best_model_weights_only.pth: Lightweight version for deployment.

📝 License

This project is for educational and research purposes. Please ensure compliance with the original dataset license when using histopathology images.
