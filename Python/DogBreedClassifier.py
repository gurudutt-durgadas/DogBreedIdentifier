'''
CS 6375.501 Course Project
This is Deep Convolution Neural Networks approach to the Dog Breed Classification Project.
Authors : Gurudutt Durgadas Shetti,
          Vishrut Sharma,
          Amandeep Singh,
          Faustina Dominic
'''
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.preprocessing import image
from tqdm import tqdm
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential

from scipy import interp
from itertools import cycle
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, classification_report

class DogBreedIdentifier:
	
	def __init__(self):
		self.drive = '/floyd/input/dogbreeddata'
		self.path = self.drive + '/dogImages/'


	def loading_dataset(self, path):
	    dog_data = load_files(path)
	    dog_files = np.array(dog_data['filenames'])
	    dog_targets = np_utils.to_categorical(np.array(dog_data['target']), 133)
	    return dog_files, dog_targets


	def dataset_characteristics(self):
	    print('There are %d total dog categories.' % len(dog_names))
	    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
	    print('There are %d training dog images.' % len(train_files))
	    print('There are %d validation dog images.' % len(valid_files))
	    print('There are %d test dog images.' % len(test_files))


	def filepath_tensor(self, img_path):
	    img = image.load_img(img_path, target_size=(224, 224))
	    x = image.img_to_array(img)
	    return np.expand_dims(x, axis=0)


	def filepaths_tensors(self, img_paths):
	    list_of_tensors = [self.filepath_tensor(img_path) for img_path in tqdm(img_paths)]
	    return np.vstack(list_of_tensors)


	def create_tensors(self):
	    train_tensors = self.filepaths_tensors(train_files).astype('float32') / 255
	    valid_tensors = self.filepaths_tensors(valid_files).astype('float32') / 255
	    test_tensors = self.filepaths_tensors(test_files).astype('float32') / 255
	    return train_tensors, valid_tensors, test_tensors


	def cnn_model_generation(self, model, target_count):
	    model.add(Conv2D(16, (3, 3), padding='same', use_bias=False, input_shape=(224, 224, 3)))
	    model.add(BatchNormalization(axis=3, scale=False))
	    model.add(Activation("relu"))
	    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
	    model.add(Dropout(0.2))

	    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
	    model.add(BatchNormalization(axis=3, scale=False))
	    model.add(Activation("relu"))
	    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
	    model.add(Dropout(0.2))

	    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
	    model.add(BatchNormalization(axis=3, scale=False))
	    model.add(Activation("relu"))
	    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
	    model.add(Dropout(0.2))

	    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
	    model.add(BatchNormalization(axis=3, scale=False))
	    model.add(Activation("relu"))
	    model.add(Flatten())
	    model.add(Dropout(0.2))

	    model.add(Dense(512, activation='relu'))
	    model.add(Dense(target_count, activation='softmax'))
	    model.summary()
	    return model


	def CNN_from_scratch(self):
	    model = Sequential()

	    class_count = len(dog_names)
	    model = self.cnn_model_generation(model, class_count)

	    from keras.callbacks import ModelCheckpoint

	    EPOCHS = 20
	    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	    checkpointer = ModelCheckpoint(filepath=path + 'saved_models/weights.best.from_scratch.hdf5',
	                                   verbose=1, save_best_only=True)
	    model.fit(train_tensors, train_targets,
	              validation_data=(valid_tensors, valid_targets),
	              epochs=EPOCHS, batch_size=32, callbacks=[checkpointer], verbose=1)

	    model.load_weights(path + 'saved_models/weights.best.from_scratch.hdf5')
	    dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
	    test_accuracy = 100 * np.sum(np.array(dog_breed_predictions) == np.argmax(test_targets, axis=1)) / len(
	        dog_breed_predictions)
	    print('Test accuracy: %.4f%%' % test_accuracy)


	def plot_roc_auc(self, y_true, y_pred):
	    """
	    This function plots the ROC curves and provides the scores.
	    """

	    # initialize dictionaries and array
	    fpr = dict()
	    tpr = dict()
	    roc_auc = dict()
	    lw = 2

	    # prepare for figure

	    # for both classification tasks (categories 1 and 2)
	    for i in range(1, len(dog_names)):
	        # obtain ROC curve
	        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
	        # obtain ROC AUC
	        roc_auc[i] = auc(fpr[i], tpr[i])

	    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
	    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(1, len(dog_names))]))
	    mean_tpr = np.zeros_like(all_fpr)
	    for i in range(1, len(dog_names)):
	        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
	    mean_tpr /= len(dog_names)
	    fpr["macro"] = all_fpr
	    tpr["macro"] = mean_tpr
	    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	    # Plot all ROC curves
	    plt.figure(1)
	    #     plt.plot(fpr["micro"], tpr["micro"],
	    #              label='micro-average ROC curve (area = {0:0.2f})'
	    #                    ''.format(roc_auc["micro"]),
	    #              color='deeppink', linestyle=':', linewidth=4)

	    plt.plot(fpr["macro"], tpr["macro"],
	             label='macro-average ROC curve (area = {0:0.2f})'
	                   ''.format(roc_auc["macro"]),
	             color='navy', linestyle=':', linewidth=4)

	    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	    #     for i, color in zip(range(1,len(dog_names)), colors):
	    #         plt.plot(fpr[i], tpr[i], color=color, lw=lw,
	    #                  label='ROC curve of class {0} (area = {1:0.2f})'
	    #                  ''.format(i, roc_auc[i]))

	    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	    plt.xlim([0.0, 1.0])
	    plt.ylim([0.0, 1.05])
	    plt.xlabel('False Positive Rate')
	    plt.ylabel('True Positive Rate')
	    plt.title('ROC curve')
	    plt.legend(loc="lower right")
	    plt.show()


	def extract_VGG19(self, file_paths):
	    tensors = filepaths_tensors(file_paths).astype('float32')
	    preprocessed_input = preprocess_input_vgg19(tensors)
	    return VGG19(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)


	def extract_Resnet50(self, file_paths):
	    tensors = filepaths_tensors(file_paths).astype('float32')
	    preprocessed_input = preprocess_input_resnet50(tensors)
	    return ResNet50(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)


	def transfer_branches(self, input_shape=None):
	    size = int(input_shape[2] / 4)

	    transfer_branch_input = Input(shape=input_shape)
	    transfer_branch = GlobalAveragePooling2D()(transfer_branch_input)
	    transfer_branch = Dense(size, use_bias=False, kernel_initializer='uniform')(transfer_branch)
	    transfer_branch = BatchNormalization()(transfer_branch)
	    transfer_branch = Activation("relu")(transfer_branch)
	    return transfer_branch, transfer_branch_input


	def model_history_accuracy(self, history):
	    plt.plot(history.history['acc'])
	    plt.plot(history.history['val_acc'])
	    plt.title('model accuracy')
	    plt.ylabel('accuracy')
	    plt.xlabel('epoch')
	    plt.legend(['train', 'valid'], loc='upper left')
	    plt.show()


	def model_history_loss(self, history):
	    plt.plot(history.history['loss'])
	    plt.plot(history.history['val_loss'])
	    plt.title('model loss')
	    plt.ylabel('loss')
	    plt.xlabel('epoch')
	    plt.legend(['train', 'valid'], loc='upper left')
	    plt.show()


	def utility_functions(self):
	    test_names = []
	    for item in dog_names:
	        #     print(item.split('/')[3][4:])
	        test_names.append(item.split('/')[3][4:])

	    test_names2 = []
	    count = 1
	    for num in range(len(test_names)):
	        test_names2.append(count)
	        count += 1
	    labels = []

	    for item in test_files:
	        #     print(int(item.split('/')[6][:3]))
	        labels.append(int(item.split('/')[6][:3]))

	    return test_names2


def main():

	dogIdentify = DogBreedIdentifier()
	train_tensors, valid_tensors, test_tensors = dogIdentify.create_tensors()

	train_files, train_targets = dogIdentify.loading_dataset(dogIdentify.path + 'train')
	valid_files, valid_targets = dogIdentify.loading_dataset(dogIdentify.path + 'valid')
	test_files, test_targets = dogIdentify.loading_dataset(dogIdentify.path + 'test')

	dog_names = [item[20:-1] for item in sorted(glob(dogIdentify.path + "train/*/"))]
	dogIdentify.dataset_characteristics()

	train_vgg19 = dogIdentify.dogIdentify.dogIdentify.extract_VGG19(train_files)
	valid_vgg19 = dogIdentify.dogIdentify.extract_VGG19(valid_files)
	test_vgg19 = dogIdentify.dogIdentify.extract_VGG19(test_files)
	vgg_shape = train_vgg19.shape[1:]
	print("\nVGG19 shape", vgg_shape)

	train_resnet50 = dogIdentify.extract_Resnet50(train_files)
	valid_resnet50 = dogIdentify.extract_Resnet50(valid_files)
	test_resnet50 = dogIdentify.extract_Resnet50(test_files)
	resnet50_shape = train_resnet50.shape[1:]
	print("\nResnet50 shape", resnet50_shape)

	vgg19_branch, vgg19_input = dogIdentify.transfer_branches(input_shape=vgg_shape)
	resnet50_branch, resnet50_input = dogIdentify.transfer_branches(input_shape=resnet50_shape)
	concatenate_branches = Concatenate()([vgg19_branch, resnet50_branch])

	network = Dropout(0.3)(concatenate_branches)
	network = Dense(640, use_bias=False, kernel_initializer='uniform')(network)
	network = BatchNormalization()(network)
	network = Activation("relu")(network)
	network = Dropout(0.3)(network)
	network = Dense(133, kernel_initializer='uniform', activation="softmax")(network)

	model = Model(inputs=[vgg19_input, resnet50_input], outputs=[network])
	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
	checkpointer = ModelCheckpoint(filepath='saved_models/bestmodel.hdf5',
	                           verbose=1, save_best_only=True)
	history = model.fit([train_vgg19, train_resnet50], train_targets,
	                validation_data=([valid_vgg19, valid_resnet50], valid_targets),
	                epochs=10, batch_size=4, callbacks=[checkpointer], verbose=1)

	dogIdentify.model_history_accuracy(history)
	dogIdentify.model_history_loss(history)

	model.load_weights('saved_models/bestmodel.hdf5')

	predictions = model.predict([test_vgg19, test_resnet50])
	breed_predictions = [np.argmax(prediction) for prediction in predictions]
	breed_true_labels = [np.argmax(true_label) for true_label in test_targets]
	print('Test accuracy: %.4f%%' % (accuracy_score(breed_true_labels, breed_predictions) * 100))
	report = classification_report(breed_true_labels, breed_predictions)
	print(report)

	dogIdentify.plot_roc_auc(test_targets, predictions)

if __name__== "__main__":
  main()