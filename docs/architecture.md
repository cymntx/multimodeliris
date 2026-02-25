# Architecture Documentation

## Project Overview
The project is a **Multimodal Biometric Recognition System** designed to process and analyze biometric data (iris and fingerprint) using machine learning. It emphasizes modular Python code, scalable data pipelines, and production-quality MLOps practices.

---

## System Architecture

### High-Level Overview
The system follows a modular architecture with clear separation of concerns:
- **Data Layer**: Handles data loading, transformation, and augmentation
- **Model Layer**: Defines neural network architectures for multimodal processing
- **Training Layer**: Manages the training pipeline with optimization and scheduling
- **Inference Layer**: Provides prediction and evaluation capabilities
- **Testing Layer**: Ensures code quality and correctness through unit tests

---

## Folder Structure

### Root Directory
```
/mlproject/
├── README.md              # Project overview and setup instructions
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project metadata and configuration
├── model.pth             # Pre-trained model weights
└── .github/workflows/    # CI/CD configuration
    └── ci.yml           # Automated testing and linting pipeline
```

### `src/` Directory (Core Implementation)
```
src/
├── config.yaml           # Configuration file for dataset paths and hyperparameters
├── data.py              # Dataset abstraction and data loaders
├── model.py             # Multimodal neural network architecture
├── train.py             # Training pipeline and optimization loop
└── infer.py             # Inference pipeline and evaluation
```

**Purpose**: Contains all core application logic including data handling, model definitions, and training/inference workflows.

### `dataset/` Directory (Data Storage)
```
dataset/
├── 1/
│   ├── Fingerprint/     # Fingerprint images for person 1
│   ├── left/           # Left iris images
│   └── right/          # Right iris images
├── 2/
│   ├── Fingerprint/
│   ├── left/
│   └── right/
└── ... (45 total people)
```

**Purpose**: Stores organized biometric data, with each person ID containing three modalities.

### `tests/` Directory (Unit Tests)
```
tests/
├── test_basic.py       # Model initialization and forward pass tests
└── test_data.py        # Dataset loading and transformation tests
```

**Purpose**: Validates core functionality through automated testing.

### `benchmark/` Directory (Performance Analysis)
```
benchmark/
└── data_loading.py     # Data loading performance benchmarking
```

**Purpose**: Measures and optimizes data pipeline performance.

---

## Core Components

### Data Pipeline (`src/data.py`)

#### BiometricDataset Class
- **Responsibility**: Handles multimodal data loading and preprocessing
- **Key Features**:
  - Loads fingerprint (3-channel RGB) and iris (1-channel grayscale) images
  - Supports data augmentation for training
  - Applies image transformations (resize, normalize)
  - Validates dataset integrity
  - Tracks loading performance with timing

#### Data Loaders
- **Training DataLoader**: Includes augmentation, 70% of data
- **Validation DataLoader**: Without augmentation, 15% of data
- **Test DataLoader**: Without augmentation, 15% of data
- Configurable batch size and worker count

---

### Model Architecture (`src/model.py`)

#### IrisBranch
- Processes **1-channel grayscale iris images**
- Architecture:
  - Conv2d(1 → 16) + BatchNorm + ReLU + MaxPool2d
  - Conv2d(16 → 32) + BatchNorm + ReLU + MaxPool2d
  - Adaptive Average Pooling (GAP)
- Output: 32-dimensional feature vector

#### FingerBranch
- Processes **3-channel RGB fingerprint images**
- Architecture:
  - Conv2d(3 → 16) + BatchNorm + ReLU + MaxPool2d
  - Conv2d(16 → 32) + BatchNorm + ReLU + MaxPool2d
  - Adaptive Average Pooling (GAP)
- Output: 32-dimensional feature vector

#### MultiModalNet (Fusion Network)
- **Input**: Concatenates features from both branches (64 features)
- **Fusion Strategy**:
  - Fully connected layer: 64 → 128 + ReLU + BatchNorm + Dropout
  - Output layer: 128 → num_classes
- **Dropout**: Configurable (default 0.3) for regularization

---

### Training Pipeline (`src/train.py`)

#### Key Functions
1. **set_seed()**: Ensures reproducibility across runs
2. **train_epoch()**: Single epoch of training with loss and accuracy tracking
3. **val_epoch()**: Validation loop without gradient updates
4. **main()**: Orchestrates the full training workflow

#### Training Configuration (from `config.yaml`)
- **Batch Size**: 16
- **Learning Rate**: 0.0001
- **Epochs**: 50
- **Patience**: 5 (for early stopping)
- **Weight Decay**: 1e-5
- **Dropout**: 0.3
- **Optimizer**: Adam (implicit)
- **LR Scheduler**: ReduceLROnPlateau

---

### Inference Pipeline (`src/infer.py`)

#### Key Functions
1. **infer()**: Evaluates model on test data
   - Disables gradients for efficiency
   - Returns predictions and ground truth labels
   - Supports batch processing

2. **main()**: End-to-end inference workflow
   - Loads trained model from checkpoint
   - Runs evaluation on test set
   - Generates classification metrics

---

## Configuration Management (`src/config.yaml`)

```yaml
# Data Configuration
data_path: '/Users/cymntx/mlproject/dataset'
num_people: 36

# Data Loading
batch_size: 16
num_workers: 2
train_split: 0.7
val_split: 0.15
augment_train: true

# Training Hyperparameters
epochs: 50
lr: 0.0001
weight_decay: 1e-5
patience: 5
dropout: 0.3

# Reproducibility
seed: 42
```

---

## Data Flow

```
Raw Dataset (45 people)
    ↓
BiometricDataset (loads fingerprint + iris)
    ↓
Data Transformations (resize, normalize, augment)
    ↓
DataLoaders (train/val/test split)
    ↓
MultiModalNet (dual-branch processing)
    ↓
Model Output (person classification)
    ↓
Metrics (accuracy, confusion matrix, classification report)
```

---

## Testing Strategy

### Unit Tests (`tests/`)
1. **test_basic.py**:
   - Model initialization with various configurations
   - Forward pass with expected tensor shapes
   - Output shape validation

2. **test_data.py**:
   - Dataset creation and loading
   - Transform application
   - DataLoader functionality
   - Edge cases (missing data, invalid paths)

---

## CI/CD Pipeline (`.github/workflows/ci.yml`)
- Automated testing on push/pull request
- Code linting for style consistency
- Type checking for Python code quality

---

## Performance Benchmarking

### Data Loading Benchmark (`benchmark/data_loading.py`)
- Measures throughput with different batch sizes
- Tests impact of number of workers
- Identifies I/O bottlenecks
- Reports timing statistics

---

## Key Design Principles

1. **Modularity**: Each component (data, model, training, inference) is independent
2. **Configuration-Driven**: Hyperparameters externalized to YAML
3. **Reproducibility**: Fixed random seeds and deterministic operations
4. **Scalability**: Supports variable number of people and modalities
5. **Production-Ready**: Logging, error handling, type hints throughout
6. **Testability**: Comprehensive unit tests for core functionality

---

## Dependencies
- **PyTorch**: Deep learning framework
- **Torchvision**: Image transformations and utilities
- **Scikit-learn**: Metrics and evaluation
- **PIL**: Image loading and manipulation
- **NumPy**: Numerical operations
- **YAML**: Configuration parsing

---

## Execution Flow

### Training
1. Load configuration from `config.yaml`
2. Set random seed for reproducibility
3. Initialize BiometricDataset and DataLoaders
4. Create MultiModalNet model
5. Define loss function (CrossEntropyLoss) and optimizer (Adam)
6. For each epoch:
   - Train on training set
   - Validate on validation set
   - Update learning rate if plateau detected
   - Save best model checkpoint
7. Generate final metrics on test set

### Inference
1. Load configuration and pretrained model
2. Load test dataset via DataLoaders
3. Run inference on test batch
4. Generate predictions and evaluation metrics
5. Output classification report and confusion matrix

