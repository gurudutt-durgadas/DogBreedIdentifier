'''
CS 6375.501 Course Project
This is Deep Convolution Neural Networks approach to the Dog Breed Classification Project.
Authors : Gurudutt Durgadas Shetti,
          Vishrut Sharma,
          Amandeep Singh,
          Faustina Dominic
'''
from __future__ import print_function
import pandas as pd
import shutil
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

labels = pd.read_csv('Kaggle/labels.csv')

train_dir = 'Kaggle/train/'
train_sep_dir = 'train_sep/'
if not os.path.exists(train_sep_dir):
    os.mkdir(train_sep_dir)


def create_train_data():
    for filename, class_name in labels.values:
        if not os.path.exists(train_sep_dir + class_name):
            os.mkdir(train_sep_dir + class_name)
        src_path = os.path.abspath(os.path.join(train_dir, filename + '.jpg'))
        dst_path = os.path.abspath(os.path.join(train_sep_dir + class_name + '/' + filename + '.jpg'))
        try:
            shutil.move(src_path, dst_path)
        except IOError as e:
            print('Unable to copy file {} to {}'
                  .format(src_path, dst_path))
        except:
            print('When try copy file {} to {}, unexpected error: {}'
                  .format(src_path, dst_path, sys.exc_info()))


def plt_chart():
    ax = pd.value_counts(labels['breed'], ascending=True).plot(kind='barh',
                                                               fontsize="40",
                                                               title="Class Distribution",
                                                               figsize=(50, 100))
    ax.set(xlabel="Images per class", ylabel="Classes")
    ax.xaxis.label.set_size(40)
    ax.yaxis.label.set_size(40)
    ax.title.set_size(60)
    plt.show()
    plt.waitforbuttonpress(10)


def plt_chart2():
    yy = pd.value_counts(labels['breed'])
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    sns.set_style("whitegrid")

    ax = sns.barplot(x=yy.index, y=yy, data=labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set(xlabel='Dog Breed', ylabel='Count')
    ax.set_title('Distribution of the Dog Breeds')
    plt.show()
    plt.waitforbuttonpress(10)


# plt_chart()

plt_chart2()
