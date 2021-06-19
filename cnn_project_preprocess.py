"""
Cnn helper functions used for data preparations and preprocessing. 
The functions to help with our cnn project 
with image classification of south African road images with or without potholes
"""
import pandas as pd
import numpy as np
import os
import shutil
import pathlib
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

##########################
def get_data(path):
    """
    provide the csv file path
    """
    data = pd.read_csv(path)

    return data

#############################
def get_class_groups(data, number_class, target_column):
    """
    Returns a list of dataframe of length `number_class' 
    """
    class_ = []
    for i in range(number_class):
        class_.append(data[target_column ==i])

    return class_


################################
def check_image_files(path):
    """
    Confirming the directories and appropriate number of image files in the directories
    path == the parent directory with sub directories where files are kept.
    """
    for dirpath, dirnames, filename in os.walk(path):
        print(f"There are {len(dirnames)} directories and \
            {len(filename)} images in '{dirpath}''.")

#################
def separate_train_test_images(train_labels, test_labels):
    """
    We need the complete names of the image files to segregate them between directories
    and therefore we need to rewrite each image file with the right name by appending
    the '.JPG' extension to each image to make a match and collect them individually
    """
    train_class_0 = train_labels[0].Image_ID.values
    train_class_1 = train_labels[1].Image_ID.values
    test_label = test_labels.Image_ID.values

    train_class_0ID = []
    train_class_1ID = []
    test_IDS = []
    for id in train_class_0:
        train_class_0ID.append(id + '.JPG')
    for id in train_class_1:
        train_class_1ID.append(id + '.JPG')
    for id in test_label:
        test_IDS.append(id + '.JPG')
    return train_class_0ID, train_class_1ID, test_IDS

###################################################
def create_image_directories(image_file_list, parent_path):
    """
    Give the list of the images, created by the `separate_train_test` function
    and the parent folder where the images are mixed together.
    """
    directories = []
    for _, a, filename in os.walk(parent_path):
        for img_name in image_file_list:
            for image_file in filename:
                if image_file == img_name:
                    img_path = parent_path + '/' + image_file
                    directories.append(img_path)
    return directories

###########################################################################
def move_image_file(source_path, new_directory):
    """
    Expects a list of the file paths, that is, `source_paths` to move to a given new directory
    """
    for file_path in source_path:
        shutil.move(file_path, new_directory)

#########################################################
def get_validation_data(image_source, num_images, val_dir):
    """
    Dividing the train dataset into train set and validation set.

    num_images == the number of images taking from training set for validation
    val_dir == the directory of the image class, pothole or no_pothole
    image_source == the directory containing the particular class of images
    """
    source_path = pathlib.Path(image_source)
    path_list = list(source_path.glob('*'))

    val_data_list = []
    for i in range(num_images):
        img = str(np.random.choice(path_list))
        if img not in val_data_list:
            val_data_list.append(img)
    move_image_file(val_data_list, val_dir)


# val_pothole_path = '/content/drive/MyDrive/KOBBY/ML and AI data/CNN/train_data/pothole'
# val_no_pothole_path = '/content/drive/MyDrive/KOBBY/ML and AI data/CNN/train_data/no_pothole'

# val_pothole_target_dir = '/content/drive/MyDrive/KOBBY/ML and AI data/CNN/validation_data/potholes'
# val_no_pothole_target_dir = '/content/drive/MyDrive/KOBBY/ML and AI data/CNN/validation_data/no_potholes'


# get_validation_data(val_pothole_path, 242, val_pothole_target_dir)

# get_validation_data(val_no_pothole_path, 650, val_no_pothole_target_dir)

######################################################################
def random_image_view(target_directory, target_class):
    """
    Randomly select an image from the target directory for visual.
    """
    ### targer directory 
    target_folder = target_directory+ '/' + target_class

    ### get a random image
    random_img = random.sample(os.listdir(target_folder), 1)

    ### read image 
    img = mpimg.imread(target_folder + '/' + random_img[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis('off')

    print(f"Image: {img.shape}")

    return img
#################################################################
def plot_training_history(history):
    """
    Returns a plot of the training performance and validation
    """
    ## Get train_loss, val_loss
    loss = history.history['loss']
    val_loss = history.history['val_accuracy']

    ## Get train_accuracy, val_accuracy
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    ### Get the epochs
    epochs = range(len(history.history['loss']))

    ### Plot loss
    plt.plot(epochs, loss, label = 'training_loss')
    plt.plot(epochs, val_loss, label = 'val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    ##### Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label = 'training_accuracy')
    plt.plot(epochs, val_accuracy, label = 'val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

