# Breast Histopathology Image Classification (IDC Detection)

This project implements a deep learning pipeline to classify breast cancer histopathology images as Invasive Ductal Carcinoma (IDC) or non-IDC. The model is built using the `convnext_tiny` architecture and is specifically optimized to maximize accuracy while preventing overfitting through advanced regularization and data augmentation techniques.

## 🚀 Key Technical Features

- **Model Architecture:** Utilizes `convnext_tiny` from the `timm` library with pretrained weights.
- **Overfitting Prevention:** - **Slide-Level Splitting:** Ensures that all image patches from a single patient (slide) are grouped together in either train, validation, or test sets to prevent data leakage.
    - **Heavy Augmentation:** Includes RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, and GaussianBlur.
    - **Regularization:** Implements Dropout (0.3) and Stochastic Depth (0.2).
- **Optimization:** Uses `AdamW` optimizer with a `CosineAnnealingLR` scheduler.
- **Hardware Acceleration:** Supports Mixed Precision training (`torch.amp`) and Multi-GPU setups (`DataParallel`).

## 📊 Pipeline Workflow

The following diagram illustrates the data processing and training pipeline:

```mermaid
graph TD
    A[Input: Breast Histopathology Patches] --> B{Slide-Level Grouping}
    B --> C[Stratified Dataset Split]
    
    subgraph "Training Pipeline"
    C --> D[Heavy Data Augmentation]
    D --> E[ConvNeXt-Tiny Architecture]
    E --> F[Weighted CrossEntropy Loss]
    F --> G[Backpropagation & AdamW]
    G --> H[Cosine Annealing Scheduler]
    H --> E
    end
    
    subgraph "Evaluation"
    E --> I[Validation Set Tracking]
    I -->|Save Best| J[best_model.pth]
    J --> K[Final Test Evaluation]
    end
    
    K --> L[Metrics: Accuracy, AUC, F1-Score]
