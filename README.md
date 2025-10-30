# Dental X-ray Wisdom Tooth Detection (Classification)

This project builds a binary image classifier to detect the presence of wisdom teeth in dental panoramic X-rays. It leverages annotations from the Dentex Challenge 2023 dataset to label each image as:
- 1: Image contains at least one wisdom tooth
- 0: No wisdom tooth present

A ResNet50 backbone (pretrained on ImageNet) is fine-tuned to perform the classification using class balancing, light augmentation, and early stopping.

## Contents
- `dental-model (1).ipynb`: End-to-end notebook with data loading, labeling, visualization, training, and plotting
- `README.md`: This file

## Data
The notebook expects the Dentex Challenge training data to be available locally with the following structure (as used on Kaggle):
```
/kaggle/input/dentex-challenge-2023/training_data/training_data/
  ├── quadrant_enumeration/
  │   ├── train_quadrant_enumeration.json
  │   └── xrays/
  └── quadrant-enumeration-disease/
      ├── train_quadrant_enumeration_disease.json
      └── xrays/
```

Adjust `base_path` in the notebook if your dataset lives elsewhere.

## Approach
1. Load `images` and `annotations` from both JSON sources.
2. Filter annotations where `category_id_2 == 7` (wisdom tooth).
3. Label images: 1 if any wisdom-tooth annotation exists for the image, else 0.
4. Balance classes by upsampling the minority class.
5. Split into train/validation (stratified).
6. Train a ResNet50-based classifier with light augmentation and early stopping.
7. Plot training/validation accuracy and loss.

## Environment
Recommended Python 3.10+ with the following key packages:
- tensorflow
- opencv-python
- scikit-learn
- pandas
- matplotlib
- numpy
- tqdm

Example install:
```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow opencv-python scikit-learn pandas matplotlib numpy tqdm
```

If you have a compatible GPU, consider installing a GPU-enabled TensorFlow build for faster training.

## Running
1. Open the notebook `dental-model (1).ipynb`.
2. Update `base_path` to point to your dataset root if needed.
3. Run all cells to:
   - Load and label data
   - Visualize wisdom-tooth bounding boxes (sample)
   - Train the classifier
   - Plot training curves

## Results
- The notebook prints dataset counts and shows training history plots.
- Early stopping is used to select the best validation loss.
- You can extend the notebook with evaluation cells (confusion matrix, ROC/AUC) as needed.

## Notes and Limitations
- This project reframes detection as image-level classification based on presence of any wisdom tooth.
- Input images are resized to 224×224, which may lose fine details compared to detection approaches.
- Class balance is achieved with upsampling; consider alternative strategies (e.g., focal loss) for robustness.

## Extending
- Add evaluation metrics and confusion matrix on a held-out test set
- Export/snapshot the trained model (`model.save(...)`)
- Try different backbones (EfficientNet, DenseNet) or fine-tuning strategies
- Move to an object detector (e.g., RetinaNet, YOLO) for localization tasks

## Acknowledgements
- Dentex Challenge 2023 dataset.
- Keras/TensorFlow for model training utilities.

## License
Specify your license here (e.g., MIT). If unsure, leave this section and decide before publishing.
