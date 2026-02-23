# Multimodal Biometric Recognition System

## Overview
This project implements a scalable, production-quality machine learning infrastructure for multimodal biometric recognition using iris and fingerprint data. The focus is on Python engineering, machine learning workflows, data pipelines, and MLOps practices.

## Features
- Modular and clean Python code.
- Dataset abstraction for multimodal data.
- Training and inference pipelines.
- Configuration-driven design for scalability.
- CI/CD pipeline for automated testing and linting.

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Clone this repository:
  ```bash
  git clone <repository-url>
  cd mlproject
  ```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configuration
Update the `config.yaml` file with the appropriate dataset path and parameters.

### Run Training
```bash
python src/train.py
```

### Run Inference
```bash
python src/infer.py
```

## Testing and Linting

### Run Tests
```bash
pytest --cov=src tests/
```

### Run Linters
```bash
flake8 src tests
black --check src tests
isort --check-only src tests
```

## CI/CD Pipeline
This project uses GitHub Actions for automated testing and linting. The pipeline runs on every push and pull request to the `main` branch.
