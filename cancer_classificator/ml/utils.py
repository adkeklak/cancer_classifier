import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image
from glob import glob
from keras.preprocessing import image

def load_yaml(file_path='cancer_classificator/ml/config.yaml'):    
    with open(file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    model_type = config['model_type']
    model_name = config['model_name']
    path = config['path']
    version = config['version']
    epochs = config['epochs']
    verbose = config['verbose']
    
    return (model_type, model_name, path, version, epochs, verbose)    

def decode(list):
    index = np.argmax(list[0]) 
    match index:
        case 0:
            return "normal"
        case 1:
            return "bengin"
        case 2:
            return "malignant"
    

def load_data(file_path = "ml_data/lungs"):
    data = glob(f'{file_path}/*/*.jpg')
    X, y = preprocess_data(data)
    return X, y

def preprocess_data(data):
    images = [path.replace('\\', '/') for path in data]
    df = pd.DataFrame({'filepath':images})
    df['label'] = df['filepath'].str.split('/',expand=True)[2]
    df['label_num'] = np.where(df['label'] == 'Normal cases', 0, \
                        np.where(df['label'] == 'Bengin cases', 1, 2))
    
    X, y = df['filepath'], df['label_num']
    return X, y

def decode_image(file_path, label=None, augment=False):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [512, 512])
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = augment_image(img)
    
    return img, label

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    return image

def preprocess_image(file_path):
    img = Image.open(file_path)
    img = img.resize((512, 512))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array