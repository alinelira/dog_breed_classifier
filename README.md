# Dog Breed Classifier

## Table of Contents

1. [Project Overview](#overview)
2. [Model Performance](#model)
3. [Files Description](#files)
4. [Instructions](#instructions)

## Project Overview <a name="overview"></a>

This project consists of building a Convolutional Neural Network (CNN) pipeline to analyze user-provided images. The goal is to identify the breed of a dog when given an image of one, and to identify the resembling dog breed when given an image of a person.

Additionally, a  web app was created to receive an image as input and the output will be the dog breed (if the image provided is a dog) or resembling dog breed (if the image provided is a human).

Below are some screenshots of the web app.

![image](https://user-images.githubusercontent.com/48845915/226145683-41bcd708-a7fd-4234-940d-a4ae4e1d22f1.png)

![image](https://user-images.githubusercontent.com/48845915/226145728-7b40768f-8ab6-4512-93c0-d23f04062262.png)

## Model Performance <a name="model"></a>

## Files <a name="files"></a>

```bash
├── app
│   ├── config_app.py
│   ├── dog_names.json
│   ├── requirements.txt
│   └── run_app.py
│   └── templates
│       └── master.html
├── bottleneck_features
│   └── DogResnet50Data.npz
└── haarcascade
    └── haarcascade_frontalface_alt.xml
├── trained_model
    └── model_Resnet50.hdf5
├── README.md
├── dog_app.ipynb


```
## Instructions <a name="instructions"></a>

Follow these steps to run the application on a local machine

1. Clone the repository onto your own computer

2. Install the app requirements

        pip install -r "app/requirements.txt"

3. Run the following command in the main directory to create your web app

        python app/run.py
        
4.  Go to the web app that was generated in the previous step
        
        http://127.0.0.1:5000/

5. Upload an image and get the classification of the image
