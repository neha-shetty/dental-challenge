# 🦷 Dental Wisdom Tooth Presence Classifier

This repository contains a deep learning model designed to detect whether a dental X-ray contains at least one wisdom tooth.

It uses transfer learning with ResNet50 and the Dentex Challenge 2023 dataset for binary classification.

## Project Motivation

This project actually started from a personal story — my cousin had a wisdom tooth problem, and the dentist mentioned they couldn’t predict when the tooth might erupt.

That conversation sparked my curiosity: could machine learning help forecast tooth eruption stages automatically?

Originally, the plan was to build a model to predict when a wisdom tooth might erupt based on dental X-rays. However, due to data limitations — mainly the lack of detailed eruption-stage labels and longitudinal scans — this initial version focuses on a simpler but important step: detecting whether a wisdom tooth is present or not in an X-ray.

With more complete data, this could evolve into a much more interesting eruption prediction challenge in the future.

## 🧩 Project Overview

- **Dataset**: Dentex Challenge 2023 (Quadrant & Enumeration annotations)
- **Task**: Binary classification — Wisdom tooth present vs. not present
- **Model**: ResNet50 (ImageNet weights, fine-tuned)
- **Framework**: TensorFlow / Keras
- **Metrics**: Accuracy, AUC, Precision, Recall

## 🏗️ Repository Structure
```
dental-wisdom-detector/
├─ data/
│  ├─ raw/                         # Original dataset (not committed)
│  └─ processed/                   # Generated CSVs, subsets for testing
├─ notebooks/
│  └─ eda.ipynb                    # Exploratory analysis
├─ src/
│  ├─ data/
│  │  ├─ prepare.py                # Build labels from JSON annotations
│  │  └─ dataset.py                # tf.data or generator pipeline
│  ├─ models/
│  │  └─ model.py                  # Model architecture builder
│  ├─ train.py                     # Training pipeline
│  └─ evaluate.py                  # Evaluation & metrics
├─ tests/
│  └─ test_dataset.py              # Unit tests for loaders
├─ .gitignore
├─ requirements.txt
├─ README.md
└─ LICENSE
```

Note: The current repo contains a single notebook implementing the pipeline. The above structure is a suggested future layout as the project grows.

## ⚙️ Setup Instructions

1. Environment setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Prepare data
```bash
python src/data/prepare.py --data-root /path/to/training_data --output data/processed/labels.csv
```

3. Train the model
```bash
python src/train.py --config configs/train.yaml
```

4. Evaluate results
```bash
python src/evaluate.py --model checkpoints/best_model.h5 --data data/processed/labels_val.csv
```

## 📊 Future Work

- Extend the model to predict eruption stages instead of simple presence detection.
- Integrate bounding-box information to localize teeth precisely.
- Build a larger annotated dataset with temporal (before-after) X-rays for eruption forecasting.
- Explore vision transformers (ViT) or multi-task learning for detection + staging.
