"""
Deployment script for Plant Pathology EfficientNet-B2 model to Hugging Face Hub.
This script converts the PyTorch model to safetensors format and uploads it with all necessary configs.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo
import json
from pathlib import Path


class PlantPathologyModel(nn.Module):
    def __init__(self, num_classes=4):
        super(PlantPathologyModel, self).__init__()
        self.backbone = efficientnet_b2(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def create_config():
    """Create model configuration file"""
    config = {
        "architectures": ["PlantPathologyModel"],
        "model_type": "efficientnet",
        "num_classes": 4,
        "image_size": 260,
        "id2label": {
            "0": "healthy",
            "1": "multiple_diseases",
            "2": "rust",
            "3": "scab"
        },
        "label2id": {
            "healthy": 0,
            "multiple_diseases": 1,
            "rust": 2,
            "scab": 3
        },
        "backbone": "efficientnet_b2",
        "pretrained_backbone": True,
        "dropout": 0.3,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
    return config


def create_readme():
    """Create model card README"""
    readme = """---
license: apache-2.0
tags:
- image-classification
- plant-pathology
- efficientnet
- pytorch
datasets:
- plant-pathology-2020-fgvc7
metrics:
- accuracy
library_name: pytorch
---

# Plant Pathology EfficientNet-B2

This model classifies plant diseases using EfficientNet-B2 architecture. It was trained on the Plant Pathology 2020 FGVC7 dataset.

## Model Description

- **Architecture**: EfficientNet-B2 (pretrained on ImageNet)
- **Task**: Multi-class image classification (4 classes)
- **Input Size**: 260x260 RGB images
- **Classes**:
  - healthy
  - multiple_diseases
  - rust
  - scab

## Performance

- **Validation Accuracy**: 96.04%
- **Test Accuracy**: 97.00%

### Requirements

```bash
pip install torch torchvision Pillow safetensors
```

### Inference Code

```python
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b2
from PIL import Image
from safetensors.torch import load_file

# Define the model architecture
class PlantPathologyModel(nn.Module):
    def __init__(self, num_classes=4):
        super(PlantPathologyModel, self).__init__()
        self.backbone = efficientnet_b2(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Load model
model = PlantPathologyModel(num_classes=4)
state_dict = load_file("plant-pathology-efficientnetb2.safetensors")
model.load_state_dict(state_dict)
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Inference
image = Image.open("your_plant_image.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

# Class names
class_names = ["healthy", "multiple_diseases", "rust", "scab"]
print(f"Predicted: {class_names[predicted_class]}")
print(f"Confidence: {probabilities[predicted_class]:.2%}")
```

## Training Details

### Training Data
- Dataset: Plant Pathology 2020 FGVC7
- Training samples: 1,310
- Validation samples: 328
- Test samples: 181

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 15
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Data Augmentation**:
  - Random horizontal flip
  - Random vertical flip
  - Random rotation (±20°)
  - Color jitter (brightness=0.2, contrast=0.2)

### Hardware
- GPU training on CUDA-enabled device

## Limitations

- Model is trained specifically on apple leaf diseases from the Plant Pathology 2020 dataset
- Performance may vary on other plant species or different imaging conditions
- Requires consistent image preprocessing (resize to 260x260, normalize with ImageNet stats)

## Citation

If you use this model, please cite:

```bibtex
@misc{plant-pathology-efficientnetb2,
  author = {Nahuel},
  title = {Plant Pathology EfficientNet-B2},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/nahuelnb/plant-pathology-efficientnetb2}}
}
```

## License

Apache 2.0
"""
    return readme


def convert_and_upload(model_path="best_model.pth", repo_id="nahuelnb/plant-pathology-efficientnetb2"):
    """
    Main function to convert model to safetensors and upload to HF Hub
    """
    print("Plant Pathology Model Deployment")

    output_dir = Path("./hf_model")
    output_dir.mkdir(exist_ok=True)
    print(f"\n1. Created output directory: {output_dir}")

    print(f"\n2. Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')

    print("\n3. Initializing model architecture...")
    model = PlantPathologyModel(num_classes=4)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("\n4. Converting to safetensors format...")
    safetensors_path = output_dir / "plant-pathology-efficientnetb2.safetensors"
    save_file(model.state_dict(), safetensors_path)
    print(f"   - Saved to: {safetensors_path}")

    print("\n5. Creating config.json...")
    config = create_config()
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   - Saved to: {config_path}")

    print("\n6. Creating README.md (model card)...")
    readme = create_readme()
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme)
    print(f"   - Saved to: {readme_path}")

    print(f"\n7. Uploading to Hugging Face Hub: {repo_id}")
    print("   NOTE: You need to be logged in to HF Hub.")
    print("   Run: huggingface-cli login")

    try:
        api = HfApi()

        print(f"   - Creating/accessing repository...")
        create_repo(repo_id, exist_ok=True, private=False)

        print(f"   - Uploading files...")
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"SUCCESS! Model deployed to: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"\n   ERROR: Failed to upload to Hugging Face")
        print(f"   {str(e)}")
        print(f"\n   Files are saved locally in {output_dir}/")
        print(f"   You can manually upload them to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    convert_and_upload(
        model_path="best_model.pth",
        repo_id="nahuelnb/plant-pathology-efficientnetb2"
    )
