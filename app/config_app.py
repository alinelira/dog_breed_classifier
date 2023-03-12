import numpy as np
import cv2
import json 
import tensorflow as tf

from PIL import Image
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image                  
from tqdm import tqdm
from glob import glob
from cv2 import CascadeClassifier, cvtColor


# Extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

# Load trained model
Resnet50_model = load_model('trained_model/model_Resnet50.hdf5')

# Initialize default graph
graph = tf.get_default_graph()


def face_detector(img_path):
    """
    Detects if there is a face in the image located at 'img_path'
    
    Args:
        img_path (str): file path to an image
        
    Returns:
        bool: True if a face is detected, otherwise False
    
    """
    img = Image.open(img_path)
    gray = cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):
    """
    Loads an image from a file path and returns it as a 4D tensor
    
    Args:
        img_path (str): file path to an image
    
    Returns: 
        ndarray: a 4D tensor of shape (1, 224, 224, 3) representing the loaded image
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) 
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    """
    Takes a list of image paths as input and returns a 4D tensor 
    
    Args:
        img_paths (List[str]): A list of file paths to the images that will be converted to 4D tensor
        
    Returns:
        ndarray: A 4D tensor of shape (num_images, 224, 224, 3) 
    
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    """
    Returns prediction vector for image located at 'img_path' using the ResNet50 model
    
    Args:
        img_path (str) : file path to the image to be predicted
    
    Returns:
        int: The index of the predicted clas
    
    """
    # Define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')
    
    img = preprocess_input(path_to_tensor(img_path))
    with graph.as_default():
        prediction = np.argmax(ResNet50_model.predict(img))
        
    return prediction


def dog_detector(img_path):
    """
    Detects if there is a dog in the image located at img_path
    
    Args:
        img_path (str): file path to an image
        
    Returns:
        bool: True if a dog is detected, otherwise False
    
    """
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def extract_Resnet50(tensor):
    """ 
    Extracts features from an input tensor using the ResNet50 convolutional neural network model

    Args:
        tensor (ndarray): an input tensor

    Returns:
        ndarray: preprocessed tensor
    """

    # Define ResNet50 model
    ResNet50_base = ResNet50(weights='imagenet', include_top=False)
    with graph.as_default():
        output = ResNet50_base.predict(preprocess_input(tensor))
        
    return output


def dog_breed_prediction(img_path):
    """
    Predicts dog breed for an image located at 'img_path'
    
    Args:
        img_path (str) : path to the image to be classified
        
    Returns:
        str: dog breed prediction
    """
    # Extract the bottleneck features
    bottleneck_features = extract_Resnet50(path_to_tensor(img_path))
    
    # Predict dog breed (index)
    with graph.as_default():
        pred_vector = Resnet50_model.predict(bottleneck_features)
    
    # Return dog breed name
    with open("dog_names.json", "r") as f:
        dog_names = json.load(f)

    return ' '.join(dog_names[np.argmax(pred_vector)].split(".")[-1].split("_"))


def classify_image(img_path):
    """
    Returns classification and breed predicted of the image provided at 'img_pth'
    
    Args:
        image_path (str): local path of image to be classified 
        
    Returns:
        str: string containing classification of the image (e.g. dog, human, neither)
            and the predicted dog breed/resenbling dog breed
    """
    
    # Predict dog breed
    dog_breed_name = " ".join(dog_breed_prediction(img_path).split(".")[-1].split("_"))
            
    # Classify image
    
    if dog_detector(img_path):
        return f"Looks like this is a {dog_breed_name} dog"
    
    elif face_detector(img_path):
        return f"Looks like this is a human that looks like a {dog_breed_name}"
    
    else:
        return "Ooops, looks like you haven't provided an image of a dog or a human. Please provide a different image"