# Plant Pathology Classification

Classifies plant leaf images into 4 disease categories using a fine-tuned EfficientNet-B2 model.

**Dataset:** [Kaggle Plant Pathology 2020 FGVC7](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data)
**Model:** [HuggingFace — nahuelnb/plant-pathology-efficientnetb2](https://huggingface.co/nahuelnb/plant-pathology-efficientnetb2)

## Stack
- Python, PyTorch, torchvision
- EfficientNet-B2 (pretrained on ImageNet)
- HuggingFace Hub

## Classes
- `healthy`
- `multiple_diseases`
- `rust`
- `scab`

## Model

EfficientNet-B2 pretrained on ImageNet, with the classifier head replaced by a linear layer for 4-class output (~7.7M parameters).

**Training config:**
- Image size: 260x260
- Batch size: 32
- Epochs: 15
- Optimizer: Adam (lr=0.001, with ReduceLROnPlateau)
- Augmentation: horizontal/vertical flip, rotation, color jitter

**Results (test set, 181 images):**

| Class             | Precision | Recall | F1   |
|-------------------|-----------|--------|------|
| healthy           | 0.96      | 0.96   | 0.96 |
| multiple_diseases | 1.00      | 0.67   | 0.80 |
| rust              | 0.98      | 1.00   | 0.99 |
| scab              | 0.95      | 0.98   | 0.97 |
| **accuracy**      |           |        | **0.97** |

Best validation accuracy: **96.04%** (epoch 9)

## How to run

```bash
kaggle competitions download -c plant-pathology-2020-fgvc7
unzip plant-pathology-2020-fgvc7.zip -d plant-pathology-2020-fgvc7/
rm -rf plant-pathology-2020-fgvc7.zip
python organize_images.py
pip install -r requirements.in  # or requirements-cuda.in for GPU (CUDA 11.1/2)
```

Then run the notebooks in order:

1. [data_cleaning.ipynb](data_cleaning.ipynb) — cleans and organizes the dataset
2. [plant_pathology.ipynb](plant_pathology.ipynb) — **main notebook**: trains EfficientNet-B2 and evaluates on the test set
3. [experiments.ipynb](experiments.ipynb) — compares EfficientNet-B1/B2, VGG16, ResNet50, and DINO using TensorBoard
4. [deploy.py](deploy.py) — uploads the trained model to HuggingFace
5. [download_and_test.ipynb](download_and_test.ipynb) — downloads the model from HuggingFace and runs inference
