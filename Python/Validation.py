'''
CS 6375.501 Course Project
This is Deep Convolution Neural Networks approach to the Dog Breed Classification Project.
Authors : Gurudutt Durgadas Shetti,
          Vishrut Sharma,
          Amandeep Singh,
          Faustina Dominic
'''

import os
import random
import shutil
import pandas as pd

PATH = 'Kaggle/'

n = 0.2

labels = pd.read_csv('Kaggle/labels.csv')


def utility_folders():
    if not os.path.exists(f'{PATH}valid'):
        os.mkdir(f'{PATH}valid')
    if not os.path.exists(f'{PATH}saved_models'):
        os.mkdir(f'{PATH}saved_models')


def create_valid_subfolders():
    for f in labels.breed.unique():
        if not os.path.exists(f'{PATH}valid/{f}'):
            os.mkdir(f'{PATH}valid/{f}')


def create_validation_data(path):
    for f in os.listdir(f'{PATH}train_sep'):
        pics = os.listdir(f'{PATH}train_sep/{f}')
        numpics = len(pics)
        numvalpics = round(n * numpics)

        val_pics = random.sample(pics, numvalpics)

        for p in val_pics:
            file_path = os.path.abspath(f'{PATH}train_sep/{f}/{p}')
            val_path = os.path.abspath(f'{PATH}valid/{f}/{p}')
            shutil.move(file_path, val_path)


def file_rename(path):
    i = 0
    for f in os.listdir(path):
        src = os.path.abspath(f'{path}/{f}')
        i += 1
        prefix = str(i).zfill(3)
        dst = os.path.abspath(f'{path}/{prefix}.{f}')
        print(src, dst)
        os.rename(src, dst)


file_rename(path=f'{PATH}valid')

print("it's done")
