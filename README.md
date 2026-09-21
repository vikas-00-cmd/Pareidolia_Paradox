# Pareidolia Paradox - Lunar Surface Classification

This repository contains the training and inference code for the Pareidolia Paradox challenge, an image classification task to distinguish between lunar craters (Depth) and mounds (Rise).

## Methodology Summary
**Handling the Topographic Inversion Illusion:**
The primary challenge of this dataset is topographic inversion, where a crater illuminated from one direction appears identical to a mound illuminated from the opposite direction. 

To solve this, our pipeline includes a **Physics Normalization Engine**. Before any neural network training occurs, we read the `sun_azimuth_angle` from the metadata. We then use OpenCV to dynamically rotate every single image counter-clockwise by `-sun_azimuth_angle` using reflection padding. 
This geometric transformation ensures that the illumination vector (the direction of the sunlight) is completely uniform (pointing left-to-right) across the entire dataset. By physically neutralizing the lighting variance, our `ConvNeXt-Tiny` model accurately learns the true morphological features of craters and mounds without being fooled by inverted shadows.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure your data is structured as follows:
   ```
   ├── Train/
   │   ├── train_images/
   │   └── train_metadata.csv
   ├── Test/
   │   ├── eval_images/
   │   └── test_metadata.csv
   ```

## Usage
**Training:**
Run the training script to normalize the images and train the 5-fold ConvNeXt-Tiny ensemble.
```bash
python train.py
```

**Inference:**
Run the inference script to predict on the test set and generate `submission.csv`.
```bash
python inference.py
```

## Model Weights
The pre-trained weights for the 5-fold ensemble can be downloaded here:
**https://drive.google.com/file/d/12__1olwQ3HOfmKlDP5cpm_c8lA2xa58a/view?usp=drive_link** 

*(Make sure to place the downloaded `.pt` files in the `working/checkpoints/` directory before running inference).*
