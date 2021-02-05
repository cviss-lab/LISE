# import the necessary packages
import os, shutil, pickle, cv2, errno, json
from tensorflow.keras import applications
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, Iterator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tqdm import tqdm
from tensorflow.keras.layers import LeakyReLU
from custom_loss_functions import mape_0_to_1, weighted_mape
from custom_data_generator import MultiPatchDataGenerator

class CNN_model:
    def __init__(self,
                 prefix='',
                 validation_split=0.2,
                 learning_rate=0.0001,
                 lf_setting='mean_squared_error',
                 epochs=20,
                 batch_size=32,
                 pth_to_labels="../datasets/V2/multilabels.csv", image_augmentations={},
                 model_img_width=299, model_img_height=299, model_img_channels=3,
                 activation='relu',
                 img_norm='-1_to_+1',
                 model_name='model.h5',
                 kfold=None,
                 greyscale=False,
                 output_pth='../output/1_model',
                 random_state=42,
                 norm_labels=True,
                 retrain=None,
                 dataset_size=None,
                 testing=False,
                 tmp=None,
                 x_name='file',
                 y_name='pix_per_len',
                 shuffle=True,
                 model_type='single',
                 N_patches_per_set=3,
                 shuffle_patches=True):
        """
        A basic keras-based CNN regression model trainer.

        Some assumptions:
            - The csv file containing path to images and the corresponding target value should be:
                path to images --> "file"
                target value --> "pix_per_len"
            - The output files with model predictions, the column name containing predicted values will be:
                predicted values --> "predicted_pix_per_len"

        Definitions:

        :string prefix: Can add a prefix to all output files
            eg. prefix='', will output files like "data.csv".
                prefix='1' will output files like "1_data.csv".
        :float validation_split: Specify what is the training/test split (range from 0 to 1). 0.3 specifies 30% of data
            will be used for testing and 70% for training
        :float learning_rate: The learning rate of the classifer
        :string lf_setting: Specify the loss function in https://keras.io/losses/.
            For custom loss functions, add a function to custom_loss_functions.py and setup a if-statement in
            __loss_function to set it to the custom loss function given a keyword of your choice.
        :int epochs: Number of cycles to go through the dataset
        :int batch_size: The amount of images to use at once to update weights
        :string pth_to_labels: path to csv containing path to images and corresponding labels.
        :dict image_augmentations: dictionary containing keyword-value pairs corresponding to:
            https://keras.io/preprocessing/image/ that informs the augmentation module how to augment the image.
        :int model_img_width: Input width into model
        :int model_img_height: Input height into model
        :int model_img_channels: Input channel into model
        :strint activation: Last activation function as specified in: https://keras.io/activations/
        :string img_norm: keyword instruction on how to normalize the image. Current implementations:
            '-1_to_+1': Convert image range from 0 to 255 to -1 to +1
            '0_to_+1': Convert image range from 0 to 255 to 0 to +1
            'mean_and_std': Normalize image range from 0 to 255 by mean and std of the image such that
                pixel = (pixel - mean)/std
        :string model_name: Model file name.
        :boolean greyscale: Keyword
        :string output_pth:
        :int random_state:
        :boolean norm_labels:
        :None||string retrain:
        :int dataset_size:
        :bool shuffle: Set true to shuffle training dataset at the end of each epoch
        :int or None kfold: if a number is specified, training proceeds as a k-fold
        :str model_type: "single" or "multi", used to indicate whether to train a single patch or multipatch input model
        :int N_patches_per_set: if model_type is "multi", the value of N_patches_per_set is the number of patches per set
        :bool patches_shuffle: For multipatch model. Set true to shuffle the patches within each set of the training dataset at the end of each epoch
        """
        # Append "_" to prefix if user specifies one
        if prefix != '':
            self.prefix = f'{prefix}_'
        else:
            self.prefix = prefix
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.lf_setting = lf_setting
        self.epochs = epochs
        self.batch_size = batch_size
        self.pth_to_labels = pth_to_labels
        self.image_augmentations = image_augmentations
        self.model_img_width = model_img_width
        self.model_img_height = model_img_height
        self.model_img_channels = model_img_channels
        self.tmp = tmp
        if greyscale:
            self.model_img_channels = 1
        self.activation = activation
        self.img_norm = img_norm
        self.model_name = self.__append_prefix(model_name)
        self.greyscale = greyscale
        self.output_pth = output_pth
        self.random_state = random_state
        self.retrain = retrain
        self.dataset_size = dataset_size
        self.norm_labels = norm_labels
        self.mean_pt = None
        self.std_pt = None
        self.img_mean_pt = None
        self.img_std_pt = None
        self.kfold=kfold
        self.testing = testing
        self.x_name = x_name
        self.y_name = y_name
        self.model_type = model_type
        self.shuffle = shuffle
        self.shuffle_patches = shuffle_patches
        self.N_patches_per_set = N_patches_per_set
        # Update all paths related to output_path
        self.update_output_pth(output_pth)

    def update_output_pth(self, output_pth):
        self.output_pth = output_pth
        self.model_pth = os.path.join(output_pth, self.model_name)
        self.model_best_pth = os.path.join(output_pth, "best_"+self.model_name)
        self.results_pth = os.path.join(output_pth, self.__append_prefix('results'))

    def plot_history(self, x1, x2, t, xlabel, ylabel, legend, name='plot.jpg'):
        plt.figure(dpi=400)
        plt.plot(x1, 'b')
        plt.plot(x2, 'c')
        plt.title(t)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.yscale('log')
        plt.legend(legend, loc='upper left')
        plt.savefig(os.path.join(self.output_pth, self.__append_prefix(name)))

    def plot_actual_to_predicted(self, data, name='plot_a_vs_p.jpg'):
        plt.figure(dpi=400)
        plt.scatter(data[self.y_name].values, data[f'predicted_{self.y_name}'].values, color='blue', s=2)
        order = np.argsort(data[self.y_name].values)
        xs = np.array(data[self.y_name].values)[order]
        plt.plot(xs, xs, color='red')
        plt.xlim([data[[f'predicted_{self.y_name}', self.y_name]].values.min(),
                  data[[f'predicted_{self.y_name}', self.y_name]].values.max()])
        plt.ylim([data[[f'predicted_{self.y_name}', self.y_name]].values.min(),
                  data[[f'predicted_{self.y_name}', self.y_name]].values.max()])
        plt.xlabel(self.y_name)
        plt.ylabel(f"predicted_{self.y_name}")
        y_true = data[self.y_name].values
        y_pred = data[f'predicted_{self.y_name}'].values
        plt.title(f"MAE: {np.round(mean_absolute_error(y_true, y_pred))}")
        plt.savefig(os.path.join(self.output_pth, self.__append_prefix(name)))

    def load_dataset(self):
        data = pd.read_csv(self.pth_to_labels)
        if self.dataset_size is not None:
            data = data.sample(self.dataset_size)
            data.reset_index(inplace=True, drop=True)
        # If normalized labels are needed, extract it
        if self.norm_labels and not self.testing:
            mean_pt = data[self.y_name].mean()
            std_pt = data[self.y_name].std()
            self.mean_pt = mean_pt
            self.std_pt = std_pt
            # # Temporary global parameters
            # if self.tmp == 'inv':
            #     self.mean_pt = 0.9
            #     self.std_pt = 10
            # elif self.tmp == 'reg':
            #     self.mean_pt = 10
            #     self.std_pt = 300
        return data

    def __read_image_dataset(self, data):
        # loop over the input images
        imgs = []
        for idx, row in tqdm(data.iterrows()):
            # load the image, pre-process it, and store it in the data list
            image = cv2.imread(row[self.x_name])
            image = cv2.resize(image, (self.model_img_width, self.model_img_height))
            if self.greyscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = img_to_array(image)
            imgs.append(image)
            # extract the class label from the image path
        return np.array(imgs, dtype=np.float)

    def read_image_dataset(self, data):
        # Read in images
        if type(data) == list:
            return [self.__read_image_dataset(d) for d in data]
        elif type(data) == tuple:
            return tuple(self.__read_image_dataset(d) for d in data)
        else:
            return self.__read_image_dataset(data)

    def get_image_norm_parameters(self, data):
        # Get image normalization parameters if training phase
        if not self.testing and self.img_norm == 'mean_and_std':
            # Get normalized parameters
            self.img_mean_pt, self.img_std_pt = self.get_image_mean_std(data)

    def get_image_mean_std(self, data):

        # Initialize variables
        if self.greyscale:
            mean = 0
            std = 0
        else:
            mean = np.zeros(3)
            std = np.zeros(3)
        # Compute mean and std
        print(f'Calculating image mean and std for {len(data)} images.')
        for idx, row in tqdm(data.iterrows()):
            x = cv2.imread(row[self.x_name])
            if self.greyscale:
                x_mean = np.mean(x)
                x_std = np.std(x)
            else:
                x_mean = np.mean(x, axis=(0, 1))
                x_std = np.std(x, axis=(0, 1))
            mean += x_mean
            std += x_std
        if self.greyscale:
            return mean / len(data), std / len(data)
        else:
            return list(mean/len(data)), list(std/len(data))

    def __norm_image_dataset(self, imgs):
        if self.testing==True:
            if self.img_norm == '-1_to_+1':
                imgs /= 127.5
                imgs -= 1.
            elif self.img_norm == '0_to_+1':
                imgs /= 255.0
            elif self.img_norm == 'mean_and_std':
                if self.greyscale:
                    imgs -= self.img_mean_pt
                    imgs /= self.std_pt
                else:
                    imgs[:, :, :, 0] -= self.img_mean_pt[0]
                    imgs[:, :, :, 1] -= self.img_mean_pt[1]
                    imgs[:, :, :, 2] -= self.img_mean_pt[2]
                    imgs[:, :, :, 0] /= self.img_std_pt[0]
                    imgs[:, :, :, 1] /= self.img_std_pt[1]
                    imgs[:, :, :, 2] /= self.img_std_pt[2]
        else:
            print("Images are not manually normalized as it is the training process right now. Instead, it is done in the image data generator.")

        return imgs

    def normalize_image_dataset(self, imgs):
        if type(imgs) == list:
            return [self.__norm_image_dataset(i) for i in imgs]
        elif type(imgs) == tuple:
            return tuple(self.__norm_image_dataset(i) for i in imgs)
        else:
            return self.__norm_image_dataset(imgs)

    def __get_scenes(self, data):
        split_labels = [[pth.split('/')[-3], pth.split('/')[-2]] for pth in data['original_fp'].values]
        df = pd.DataFrame(split_labels, columns=['scene', 'num'])
        scene_df = df.drop_duplicates()
        # Total number of scenes
        num_scenes = len(scene_df)
        return df, scene_df, num_scenes

    def __get_test_scenes(self, scene_df, num_scenes):
        num_test_scenes = int(self.validation_split * num_scenes)
        test_scene_ind = np.random.choice(scene_df['num'].values, num_test_scenes, replace=False)
        test_scenes = scene_df['num'][scene_df['num'].isin(test_scene_ind)]
        return test_scenes

    def __get_df_indexes(self, df, data, test_scenes):
        # Create indices
        test_idx = df['num'].isin(list(test_scenes))
        train_idx = ~df['num'].isin(list(test_scenes))
        # Return train and test dataframes
        return data[train_idx], data[test_idx]

    def split_training_and_testing(self, data):
        """
        Split training and testing sets by:
            (1) by crop (I think this is wrong)
            (2) by scene
        :param imgs:
        :param labels:
        :param data:
        :return:
        """

        # (2) Split into scenes
        df, scene_df, num_scenes = self.__get_scenes(data)
        if self.kfold is not None:
            kf = KFold(n_splits=self.kfold)
            train_data = []
            test_data = []
            for train_index, test_index in kf.split(scene_df):
                test_scenes = scene_df['num'].iloc[test_index]
                tmp_train_data, tmp_test_data = self.__get_df_indexes(df, data, test_scenes)
                train_data.append(tmp_train_data)
                test_data.append(tmp_test_data)
            return train_data, test_data
        else:
            # Total number of test scenes
            test_scenes = self.__get_test_scenes(scene_df, num_scenes)
            # Create train and test datasets
            train_data, test_data = self.__get_df_indexes(df, data, test_scenes)
            return train_data, test_data

        # # (1) Split by crop
        # (train_data, test_data) = train_test_split(data, test_size=self.validation_split, random_state=self.random_state)
        # return train_data, test_data

    def load_model(self, train_data):
        print(f'norm_labels: {self.norm_labels}')
        if self.norm_labels:
            train_data[self.y_name] = (train_data[self.y_name] - self.mean_pt) / self.std_pt
        if self.lf_setting == 'weighted_mape':
            self.weight = train_data[self.y_name].max()

        if self.retrain is None:  # Train new model
            # Import Base Model from Bag of Models
            base_model = applications.MobileNetV2(
                input_shape=(self.model_img_width, self.model_img_height, self.model_img_channels),
                weights=None, include_top=False)
            x = GlobalAveragePooling2D()(base_model.output)
            x = Dense(1280)(x)  # add a fully-connected layer
            x = Dense(1, name='predictions')(x)
            predictions = LeakyReLU(alpha=0.3)(x)

            # Create model
            model = Model(base_model.input, predictions)
            model.summary()

            sgd = SGD(lr=self.learning_rate, decay=0.01, momentum=0.9, nesterov=True)
            model.compile(loss=self.__loss_function(), optimizer=sgd)
        else:  # Re-train model
            print(self.retrain)
            model = load_model(self.retrain, custom_objects={'weighted_mape': weighted_mape(250)})

        return model

    def reset_output_pth(self):
        # remove all previous files from destination directory
        if os.path.exists(self.output_pth):
            shutil.rmtree(self.output_pth)
        os.makedirs(self.output_pth)

    def train_model(self, model, train_data, test_data):
        # If normalized labels are needed, extract it
        if self.norm_labels:
            train_data[self.y_name] = (train_data[self.y_name] - self.mean_pt) / self.std_pt
            test_data[self.y_name] = (test_data[self.y_name] - self.mean_pt) / self.std_pt
        if self.lf_setting == 'weighted_mape':
            self.weight = train_data[self.y_name].max()

        # Create Data Generators
        self.test_image_augmentations = {}
        if 'zca_whitening' in self.image_augmentations.keys():
            if self.image_augmentations['zca_whitening']:
                self.test_image_augmentations['zca_whitening'] = True
        if self.img_norm == 'mean_and_std':
            self.image_augmentations['featurewise_center'] = True
            self.image_augmentations['featurewise_std_normalization'] = True
            self.test_image_augmentations['featurewise_center'] = True
            self.test_image_augmentations['featurewise_std_normalization'] = True

        datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input, **self.image_augmentations)
        test_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input, **self.test_image_augmentations)
        if self.img_norm == 'mean_and_std':
            if self.greyscale:
                datagen.mean = np.array(self.img_mean_pt, dtype=np.float32)
                datagen.std = np.array(self.img_std_pt, dtype=np.float32)
                test_datagen.mean = np.array(self.img_mean_pt, dtype=np.float32)
                test_datagen.std = np.array(self.img_std_pt, dtype=np.float32)
            else:
                datagen.mean = np.array(self.img_mean_pt, dtype=np.float32).reshape((1, 1, 3))  # ordering: [R, G, B]
                datagen.std = np.array(self.img_std_pt, dtype=np.float32).reshape((1, 1, 3))  # ordering: [R, G, B]
                test_datagen.mean = np.array(self.img_mean_pt, dtype=np.float32).reshape(
                    (1, 1, 3))  # ordering: [R, G, B]
                test_datagen.std = np.array(self.img_std_pt, dtype=np.float32).reshape((1, 1, 3))  # ordering: [R, G, B]

        # Set-up callback
        callbacks_list = [ModelCheckpoint(self.model_pth, verbose=1, period=5),
                          ModelCheckpoint(self.model_best_pth, verbose=1, save_best_only=True),
                          CSVLogger(os.path.join(self.output_pth, 'hist.csv'))]
        if self.model_type == 'single':
            # Create Data Generators
            if self.greyscale:
                train_generator = datagen.flow_from_dataframe(dataframe=train_data, directory=None,
                                                              x_col=self.x_name, y_col=self.y_name,
                                                              batch_size=self.batch_size, seed=self.random_state,
                                                              shuffle=self.shuffle,
                                                              class_mode="raw",
                                                              target_size=(self.model_img_width, self.model_img_height), color_mode='grayscale')
                valid_generator = test_datagen.flow_from_dataframe(dataframe=test_data, directory=None,
                                                                   x_col=self.x_name, y_col=self.y_name,
                                                                   batch_size=self.batch_size, seed=self.random_state,
                                                                   shuffle=False,
                                                                   class_mode="raw",
                                                                   target_size=(
                                                                   self.model_img_width, self.model_img_height), color_mode='grayscale')
            else:
                train_generator = datagen.flow_from_dataframe(dataframe=train_data, directory=None,
                                                              x_col=self.x_name, y_col=self.y_name,
                                                              batch_size=self.batch_size, seed=self.random_state, shuffle=self.shuffle,
                                                              class_mode="raw",
                                                              target_size=(self.model_img_width, self.model_img_height))
                valid_generator = test_datagen.flow_from_dataframe(dataframe=test_data, directory=None,
                                                                   x_col=self.x_name, y_col=self.y_name,
                                                                   batch_size=self.batch_size, seed=self.random_state, shuffle=False,
                                                                   class_mode="raw",
                                                                   target_size=(self.model_img_width, self.model_img_height))
        elif self.model_type == 'multi':
            train_generator = MultiPatchDataGenerator(train_data, self.N_patches_per_set, datagen, self.greyscale, self.batch_size, (self.model_img_width, self.model_img_height), self.shuffle, self.shuffle_patches)
            valid_generator = MultiPatchDataGenerator(test_data, self.N_patches_per_set, datagen, self.greyscale, self.batch_size, (self.model_img_width, self.model_img_height), False, False)
        else:
            raise ValueError(f"Unknown model type {self.model_type}! Please choose either 'single' or 'multi'.")


        H = model.fit_generator(generator=train_generator,
                                steps_per_epoch=len(train_data) // self.batch_size,
                                validation_data=valid_generator,
                                validation_steps=len(test_data) // self.batch_size,
                                epochs=self.epochs, verbose=1,
                                callbacks=callbacks_list, workers=12)
        return H, model

    def __append_prefix(self, name):
        return f'{self.prefix}{name}'

    def __loss_function(self):
        """Check if I need to set a """
        if self.lf_setting == 'mape_0_to_1':
            return mape_0_to_1
        elif self.lf_setting == 'weighted_mape':
            return weighted_mape(self.weight)
        else:
            return self.lf_setting

    def generate_predictions(self, model, data):
        self.test_image_augmentations = {}
        if 'zca_whitening' in self.image_augmentations.keys():
            if self.image_augmentations['zca_whitening']:
                self.test_image_augmentations['zca_whitening'] = True
        if self.img_norm == 'mean_and_std':
            self.test_image_augmentations['featurewise_center'] = True
            self.test_image_augmentations['featurewise_std_normalization'] = True
        test_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_input, **self.test_image_augmentations)
        if self.img_norm == 'mean_and_std':
            if self.greyscale:
                test_datagen.mean = np.array(self.img_mean_pt, dtype=np.float32)
                test_datagen.std = np.array(self.img_std_pt, dtype=np.float32)
            else:
                test_datagen.mean = np.array(self.img_mean_pt, dtype=np.float32).reshape((1, 1, 3))  # ordering: [R, G, B]
                test_datagen.std = np.array(self.img_std_pt, dtype=np.float32).reshape((1, 1, 3))  # ordering: [R, G, B]
        # Create Data Generators
        if self.model_type == 'single':
            if self.greyscale:
                valid_generator = test_datagen.flow_from_dataframe(dataframe=data, directory=None,
                                                               x_col=self.x_name, y_col=self.y_name,
                                                               batch_size=self.batch_size, seed=self.random_state, shuffle=False,
                                                               class_mode="other",
                                                               target_size=(self.model_img_width, self.model_img_height), color_mode='grayscale')
            else:
                valid_generator = test_datagen.flow_from_dataframe(dataframe=data, directory=None,
                                                               x_col=self.x_name, y_col=self.y_name,
                                                               batch_size=self.batch_size, seed=self.random_state,
                                                               shuffle=False,
                                                               class_mode="other",
                                                               target_size=(self.model_img_width, self.model_img_height))
        elif self.model_type == 'multi':
            valid_generator = MultiPatchDataGenerator(data, self.N_patches_per_set, test_datagen, self.greyscale, self.batch_size, (self.model_img_width, self.model_img_height), False, False)
        else:
            raise ValueError(f"Model type: {self.model_type} not implemented.")

        # Predict
        if self.norm_labels:
            print(f"mean: {self.mean_pt} std: {self.std_pt}")
            predictions = model.predict_generator(valid_generator, steps=np.ceil(len(data)/self.batch_size), verbose=1) * self.std_pt + self.mean_pt
            if self.model_type == 'multi':
                return valid_generator._get_patch_set_dataframe(), predictions
            else:
                return predictions
        else:
            predictions = model.predict_generator(valid_generator, steps=np.ceil(len(data)/self.batch_size), verbose=1)
            if self.model_type == 'multi':
                return valid_generator._get_patch_set_dataframe(), predictions
            else:
                return predictions

    def assess_predictions(self, imgs, actual, predicted):
        # Remove old predicted folder
        if os.path.exists(os.path.dirname(self.results_pth + '/')):
            shutil.rmtree(self.results_pth)

    def save_dataframe_results(self, data, name='dat.csv'):
        data.to_csv(os.path.join(self.output_pth, self.__append_prefix(name)), index=False)

    def save_cnn_training_parameters(self, name='training_config.json'):
        with open(os.path.join(self.output_pth, self.__append_prefix(name)), 'w') as f:
            json.dump(self.__dict__, f)

    def output_image_results(self, test_data, actual, predicted):
        # Write labels to image
        for idx_row, a, p in zip(test_data[[self.x_name, self.y_name]].iterrows(), actual, predicted):
            idx = idx_row[0]
            row = idx_row[1]
            img = cv2.imread(row[self.x_name])
            fc = (0, 0, 0)  # Font color
            fs = 0.001  # Font size
            # Get image shape
            if self.greyscale:
                h, w, c = img.shape
            else:
                h, w, c = img.shape
            # Put text on image
            img = cv2.putText(img, "Actual: {} pix/cm".format(np.round(row[self.y_name], 1)), (0, 25), cv2.QT_FONT_NORMAL, fs * h, fc, 1, cv2.LINE_AA)
            img = cv2.putText(img, 'Prediction: {} pix/cm'.format(np.round(float(p), 1)), (0, 50), cv2.QT_FONT_NORMAL, fs * h, fc, 1, cv2.LINE_AA)
            # Save image
            new_filename = os.path.join(self.results_pth, str(idx)+'.jpg')
            if not os.path.exists(os.path.dirname(new_filename)):
                try:
                    os.makedirs(os.path.dirname(new_filename))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            cv2.imwrite(new_filename, img)

    def pkl_training_history(self, H):
        with open(os.path.join(self.output_pth, self.__append_prefix('train_hist.json')), 'w') as f:
            json.dump(H.history, f)

    def preprocess_input(self, image, n_crops=9):
        # """
        #
        # :param image:
        # :param n_crops: number of crops in the montage
        # :return:
        # """
        # # If file is a crop, randomly shuffle
        # if self.x_name == 'file':
        #     # Get Crop
        #     crop_idx = np.arange(n_crops)
        #     rows = int(n_crops ** 0.5)
        #     # Shuffle idx
        #     np.random.shuffle(crop_idx)
        #     new_crop_image = np.zeros(image.shape, dtype='uint8')
        #     for i in range(rows):
        #         for j in range(rows):
        #             curr_idx = crop_idx[i * rows + j]
        #             i_im = int((curr_idx) / rows)
        #             j_im = (curr_idx) % rows
        #             new_crop_image[i * 299:(i + 1) * 299, j * 299:(j + 1) * 299, :] = image[
        #                                                                               i_im * 299:(i_im + 1) * 299,
        #                                                                               j_im * 299:(j_im + 1) * 299,
        #                                                                               :]
        #     return new_crop_image
        # # If file is a image, randomly montage
        # elif self.x_name == 'original_fp':
        #     img = row['original_fp']
        #     # Read image
        #     image = cv2.imread(img)
        #     # Create flexible subdirectory path
        #     f_list = img.split('/')
        #     idx = f_list.index(data_folder)
        #     f_list = f_list[idx - len(f_list) + 1:-1]
        #     # Detect marker
        #     corners = re.sub(' +', ' ',
        #                      row['corners'].replace('\n', ' ').replace('[', ' ').replace(']', ' ')).strip().split(
        #         ' ')
        #     corners = np.array([(float(corners[0]), float(corners[1])),
        #                         (float(corners[2]), float(corners[3])),
        #                         (float(corners[4]), float(corners[5])),
        #                         (float(corners[6]), float(corners[7]))])
        #     if isinstance(corners, bool):
        #         continue
        #     else:
        #         cont = False
        #         for c in corners:
        #             if c[0] > 5000 or c[1] > 5000 or c[0] < 0 or c[1] < 0:
        #                 cont = True
        #                 break
        #         if cont:
        #             continue
        #
        #     # Get len_per_pix
        #     dist = []
        #     for c in corners:
        #         tmp_dist = []
        #         for c_2 in corners:
        #             tmp_dist.append(((c_2[0] - c[0]) ** 2 + (c_2[1] - c[1]) ** 2) ** 0.5)
        #         tmp_dist = sorted(tmp_dist)[1:-1]
        #         dist.extend(tmp_dist)
        #     pix_per_len = np.average(dist) / marker_len
        #     if pix_per_len < 10 or pix_per_len > 500:
        #         # print(f"corners: {corners}")
        #         print(f"Skipped, pix_per_len = {pix_per_len}")
        #         continue
        #     # Get Crop
        #     new_crops = get_random_crops(image, crop_height, crop_width, corners, n_crops, m_images)
        #     # Save crop and crop information
        #     for c in range(len(new_crops)):
        #         # Crop
        #         new_cropped_fp = os.path.join(out_folder, 'cropped', '/'.join(f_list),
        #                                       f"{img.split('.')[-2].split('/')[-1]}_crop_{c + len(new_crops)}.JPG")
        #         cv2.imwrite(new_cropped_fp, new_crops[c])
        #         # Crop information
        #         df_crop['original_fp'].append(img)
        #         df_crop['file'].append(new_cropped_fp)
        #         df_crop['pix_per_len'].append(pix_per_len)
        #         df_crop['units'].append(units)
        #     return image
        # else:
        #     raise ValueError(f'{self.x_name} is not a valid column in the dataset.')
        return image


def __load_and_run_model_and_save_results(cnn, train_data, test_data):
    # train_data = train_data[:100]
    # test_data = test_data[:100]
    # test_data = train_data.copy()
    # Get image norm parameters
    cnn.get_image_norm_parameters(train_data)
    # Save training parameters
    cnn.save_cnn_training_parameters()
    # Load and Train model
    model = cnn.load_model(train_data)
    # Train model
    H, model = cnn.train_model(model, train_data, test_data)
    # Load best model from training as the fit_generator returns the model from the last trained epoch :(
    clear_session()  # Clear GPU memory
    cnn.retrain = cnn.model_pth
    model = cnn.load_model(train_data)
    # Generate test predictions
    if cnn.model_type == 'multi':
        test_data, predictions = cnn.generate_predictions(model, test_data)
    else:
        predictions = cnn.generate_predictions(model, test_data)
    # Output image results
    # TODO: Work from here downwards
    cnn.output_image_results(test_data, test_data['pix_per_len'].values, predictions)
    # Output DataFrame results
    test_data['predicted_pix_per_len'] = predictions
    cnn.save_dataframe_results(train_data, 'train.csv')
    cnn.save_dataframe_results(test_data, 'test.csv')
    # Generate and save plots
    cnn.plot_history(H.history['loss'], H.history['val_loss'], 'model loss', 'epoch', 'loss', ['train', 'test'],
                           'loss.jpg')
    # Save Training History
    cnn.pkl_training_history(H)
    # Save actual vs predicted plot
    cnn.plot_actual_to_predicted(test_data, 'plot.jpg')
    # Aggregate and plot results
    agg_test_data = test_data.groupby(["original_fp"]).mean()
    cnn.save_dataframe_results(agg_test_data, "test_agg_mean.csv")
    cnn.plot_actual_to_predicted(agg_test_data, 'plot_agg_mean.jpg')
    # Aggregate and plot results
    agg_test_data = test_data.groupby(["original_fp"]).median()
    cnn.save_dataframe_results(agg_test_data, "test_agg_median.csv")
    cnn.plot_actual_to_predicted(agg_test_data, 'plot_agg_median.jpg')
    # Clear GPU memory
    clear_session()


def train_model(config, test_data_pth=None):
    """
    :return:
    """
    config['testing'] = False
    # Initialize model trainer
    train_cnn = CNN_model(**config)
    # Read and normalize labels
    data = train_cnn.load_dataset()
    # Generate training data
    if test_data_pth is not None:
        test_data = pd.read_csv(test_data_pth)
        train_data = data.copy()
    else:
        train_data, test_data = train_cnn.split_training_and_testing(data)

    if train_cnn.kfold is not None:
        original_output_pth = train_cnn.output_pth
        for i, train, test in zip(range(len(train_data)), train_data, test_data):
            train_cnn.update_output_pth(f'{original_output_pth}_fold_{i}')
            # Delete and recreate output folder
            train_cnn.reset_output_pth()
            # Run training
            __load_and_run_model_and_save_results(train_cnn, train, test)
    else:
        # Delete and recreate output folder
        train_cnn.reset_output_pth()
        # Run training
        __load_and_run_model_and_save_results(train_cnn, train_data, test_data)


def test_model(config):
    """
    Assumes the folder location of the model contains the model tranining information.
    :param config:
    :return:
    """
    # Initialize model trainer
    with open(os.path.join('/'.join(config['retrain'].split('/')[:-1]), 'training_config.json'), 'r') as f:
        old_config = json.load(f)
    # Load useful parameters
    config['testing'] = True
    if not 'retrain' in config.keys():
        config['retrain'] = old_config['model_pth']
    # Delete and recreate output folder
    config['img_norm'], config['model_name'], \
        config['greyscale'], \
        config['norm_labels'] = old_config['img_norm'], old_config['model_name'], \
                                   old_config['greyscale'], old_config['norm_labels']
    test_cnn = CNN_model(**config)
    # Remove old results
    test_cnn.reset_output_pth()
    test_cnn.mean_pt, test_cnn.std_pt = old_config['mean_pt'], old_config['std_pt']
    test_cnn.img_mean_pt, test_cnn.img_std_pt = old_config['img_mean_pt'], old_config['img_std_pt']

    # Read and normalize labels
    data = test_cnn.load_dataset()
    # Load and Train model
    model = test_cnn.load_model(data)
    # Generate test predictions
    predictions = test_cnn.generate_predictions(model, data)
    # Output image results
    test_cnn.output_image_results(data, data['pix_per_len'], predictions)
    # Output DataFrame results
    data['predicted_pix_per_len'] = predictions
    test_cnn.save_dataframe_results(data, 'data.csv')
    # Save actual vs predicted plot
    test_cnn.plot_actual_to_predicted(data, 'plot.jpg')
    # Aggregate and plot results
    agg_test_data = data.groupby("original_fp").mean()
    test_cnn.save_dataframe_results(agg_test_data, "test_agg_mean.csv")
    test_cnn.plot_actual_to_predicted(agg_test_data, 'plot_agg_mean.jpg')

    test_cnn.save_cnn_training_parameters('test_config.json')
    # Clear GPU memory
    clear_session()

if __name__ == '__main__':
    """
    Training: parameters you want to specify are: epoches, output_pth, pth_to_labels, img_norm, lf_setting, learning_rate, image_augmentations
    Testing: parameters you want to specify are: output_pth, retrain, pth_to_labels
    """
    # Montage Training - uwb
    train = [
        # Sample
        ('../output/PED_sample', "../datasets/PED/2_detected_imgs/train_1_data_crop_dataset.csv", 'mape', 0.001, 'reg', "../datasets/PED/2_detected_imgs/test_1_data_crop_dataset.csv"),
        # # ASH_V2
        # ('../output/ASH_V2_speed_test', "../datasets/ASH_V2/3_train_final/crop_dataset.csv", 'mape', 0.001, 'reg', "../datasets/ASH_V2/3_test_final/crop_dataset.csv"),
        # # 850
        # ('../output/PED_V2_850', "../datasets/PED_V2/3_train_850_final/crop_dataset.csv", 'mape', 0.001, 'reg', "../datasets/PED_V2/3_test_850_final/crop_dataset.csv"),
        # # BW
        # ('../output/BW', "../datasets/BW/3_train_final/crop_dataset.csv", 'mape', 0.001, 'reg', "../datasets/BW/3_test_final/crop_dataset.csv"),
        # # 100
        # ('../output/PED_V2_100', "../datasets/PED_V2/3_train_100_final/crop_dataset.csv", 'mape', 0.001, 'reg', "../datasets/PED_V2/3_test_100_final/crop_dataset.csv"),
        # # 350
        # ('../output/PED_V2_350', "../datasets/PED_V2/3_train_350_final/crop_dataset.csv", 'mape', 0.001, 'reg', "../datasets/PED_V2/3_test_350_final/crop_dataset.csv"),
    ]
    train_config = {
        "epochs": 250,
        "output_pth": '',
        "pth_to_labels": "",
        'img_norm': '-1_to_+1',
        'norm_labels': False,
        'greyscale': True,
        "lf_setting": 'mape',
        'learning_rate': 0.001,
        "image_augmentations": {
            "channel_shift_range": 50.0,
            "brightness_range": [0.8, 1.2],
            "horizontal_flip": True,
            "vertical_flip": True,
        }
    }
    for t in train:
        train_config['output_pth'] = t[0]
        train_config['pth_to_labels'] = t[1]
        train_config['lf_setting'] = t[2]
        train_config['learning_rate'] = t[3]
        train_config['tmp'] = t[4]
        train_model(train_config, t[5])  # Custom test data set

    # Test DIFF and ZOOM on PED model
    # Montage Training - uwb
    test = [
        # ('../output/DIFF',    "../datasets/DIFF/3_test_final/crop_dataset.csv", '../output/PED_V2_850/model.h5'),
        # ('../output/ZOOM',    "../datasets/ZOOM/3_test_final/crop_dataset.csv", '../output/PED_V2_850/model.h5'),
    ]
    test_config = {
        "output_pth": '',
        "pth_to_labels": "",
        'retrain': ''
    }
    for t in test:
        test_config['output_pth'] = t[0]
        test_config['pth_to_labels'] = t[1]
        test_config['retrain'] = t[2]
        test_model(test_config)

