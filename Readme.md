https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data
- kaggle competitions download -c plant-pathology-2020-fgvc7
- unzip plant-pathology-2020-fgvc7.zip -d plant-pathology-2020-fgvc7/
- rm -rf plant-pathology-2020-fgvc7.zip
- python organize_images.py
- pip install -r requirements.in or pip install -r requirements-cuda.in (if using a GPU) compatible w/ CUDA 11.1/2
- run data_cleaning.ipynb notebook
