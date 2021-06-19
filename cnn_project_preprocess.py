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
        val_data_list.append(img)
    move_image_file(val_data_list, val_dir)
