Data source: https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data
HuggingFace repo: https://huggingface.co/nahuelnb/plant-pathology-efficientnetb2
## How to run:
- kaggle competitions download -c plant-pathology-2020-fgvc7
- unzip plant-pathology-2020-fgvc7.zip -d plant-pathology-2020-fgvc7/
- rm -rf plant-pathology-2020-fgvc7.zip
- python organize_images.py
- pip install -r requirements.in or pip install -r requirements-cuda.in (if using a GPU) compatible w/ CUDA 11.1/2
- run data_cleaning.ipynb notebook
- Now you can check:
    - experiments.ipynb to see some quick experiments that compare Efficientnetb1, Efficientnetb2, Vgg16, Resnet50 and Dino using tensorboard.
    - plant_pathology.ipynb the main notebook that shows how the model was trained using Efficientnetb2.
    - deploy.py model deploy script to HuggingFace
    - download_and_test.ipynb download the model from HuggingFace and test it.