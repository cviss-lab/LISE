<img align="left" src="misc/logo.jpg">
    
    
<br /><br /><br /><br /><br /><br />
    
    
### [CVISS Research](http://www.cviss.net/)
    
## Learning based Image Scale Estimation (LISE) for Quantitative Visual Inspection

### Introduction

This repository contains the source code used by the authors for training and validating the LISE network developed for the [paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12613). LISE is a CNN regression model trained to predict image scale using texture patches. Once estimated, the image scales can be used to quantify features on images (example image shown below).

<img align="left" src="misc/Fig_inference.png">

This repository strictly deals with the generation of the patch-scale image dataset, and the training of the CNN model. Note that for the pretrained models, all models use greyscale patch size of 850 X 850 pixels as input. The training framework is shown in the following image. The data generation algorithm uses ArUco markers as a method to calculate image scale.

<img align="left" src="misc/Fig_training.png">

### Dependencies

LISE was built using the following dependencies (**requirements.txt**).

Python version: **3.7.7**

```
Keras==2.4.3
opencv_contrib_python==4.5.1.48
pandas==1.0.3
tqdm==4.46.0
numpy==1.18.4
matplotlib==3.1.3
Shapely==1.7.1
tensorflow-gpu==2.3.0
scikit_learn==0.24.1
```

**NOTE** for training with a cpu, use tensorflow instead of tensorflow-gpu

## Sample usage example

### Creating the training dataset from collected images

#### Step 1: unzip the sample dataset into the "datasets" folder

Download **sample_PED.zip** from the [data repository](https://dataverse.scholarsportal.info/dataset.xhtml?persistentId=doi%3A10.5683%2FSP2%2FYKUZHG&version=DRAFT) and unzip to the **dataset** folder .

#### Step 2: Detect markers and generate the patch-scale dataset

In **create_montage_markers_by_scene.py**,  add this command to the main section of the code (at the bottom) like so:

```python
if __name__ == '__main__':
	create_n_by_n_markers(n_crops=1, m_images=50, raw_folder='../datasets/PED/', out_folder='../datasets/PED/2_detected_imgs', marker_len=9.4)
```

n_crops: number of patches to include in one image (should be N^2)

m_images: number of images to extract

raw_folder: path to the collected dataset

out_folder: output folder of detected results

marker_len: in units of cm, the physical dimensions of the marker.

**NOTE:** in the case the algorithm cannot find the marker, the problem will visualize the image and ask the user to manually select the four points. In this case, simply click the 4 corners of the marker and click "**c**" to continue. To reset the corner selections, click "**r**".

#### Step 3: Split the dataset into training and testing 

There should be **crop_dataset.csv** and **img_dataset.csv** in **datasets/PED/2_detected_imgs**.

**crop_dataset.csv**: each row contains the a patch image file path and its scale (used to train the model)

**img_dataset.csv**: each row contains a image file path and its scale (used to rapidly generate the crop dataset without rerunning the detection algorithm. See function **create_n_by_n_markers_from_df** in **create_montage_markers_by_scene.py**)

Change the **pth** variable in **split_data.py** to "../datasets/PED/2_detected_imgs" and run it.

**Note:** the validation_split ratio can also be changed to control the ratio of images between training and testing datasets.

```
python split_data.py
```

This will split the img_dataset.csv and crop_dataset.csv into training and testing portions.

```
test_1_data_crop_dataset.csv
train_1_data_crop_dataset.csv
```

The two csv files are used to train the scale estimation model.

### Training a image scale estimation model

Using the crop dataset generated in the previous section, we can train a patch-based scale estimator:

In "**model_training.py**",

three variables can be adjusted to train/validate models:

- train (list of tuples): each tuple contains, in order the training configurations specific to the model being trained.
- train_config (dict): contains the training configurations applicable to all models
- test (list of tuples): each tuple contains configurations specific to a model test

For this example, ensure that the **train** and **train_config** variable looks as follows:

```python
train = [
        # Sample
        ('../output/PED_sample',  # Output folder
        "../datasets/PED/2_detected_imgs/train_1_data_crop_dataset.csv",  # Path to the training crop csv
        'mape',  # Loss function to use
        0.001,   # Learning rate
        'reg',   # does nothing
        "../datasets/PED/2_detected_imgs/test_1_data_crop_dataset.csv"),  # Path to the test crop csv
]
train_config = {
        "epochs": 250,  # number of epoches
        "output_pth": '',
        "pth_to_labels": "",
        'img_norm': '-1_to_+1',  # image normalization
        'norm_labels': False,   # Normalize labels?
        'greyscale': True,   # Color or greyscale images?
        "lf_setting": 'mape', 
        'learning_rate': 0.001,
        "image_augmentations": {  # image augmentations
            "channel_shift_range": 50.0,
            "brightness_range": [0.8, 1.2],
            "horizontal_flip": True,
            "vertical_flip": True,
        }
    }
```

All model training results will be output in the **output** folder, which contains:

- best_model.h5: best performing model
- model.h5: most recent model
- hist.csv: Loss history
- training_config.json: training configuration used
- results: folder containing image results of patches
- train.csv and test.csv: the csvs used for training and validation - test.csv also contains model.h5 predictions 
- loss.jpg: loss curves.

### BibTeX Citation

```
@article{park2020LISE,
    author = {Park, Ju An and Yeum, Chul Min and Hrynyk, Trevor D.},
    title = {Learning-based image scale estimation using surface textures for quantitative visual inspection of regions-of-interest},
    journal = {Computer-Aided Civil and Infrastructure Engineering},
    volume = {36},
    number = {2},
    pages = {227-241},
    doi = {https://doi.org/10.1111/mice.12613},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12613},
    year = {2021}
}
```
