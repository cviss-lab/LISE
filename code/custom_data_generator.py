import numpy as np
import pandas as pd
import cv2, re, math, os, errno, time, functools, random
from tensorflow.keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

class MultiPatchDataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, data, N, img_transformer, grayscale=True, batch_size=16, dim=(299, 299), shuffle=True, shuffle_patches=True):
        """Initialization
        eg. https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        :param data: dataframe containing file paths, and labels
        :param N: number of patches in a set
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param shuffle: True to shuffle label indexes after every epoch

        """
        self.data = data
        self.N = N
        self.grayscale = grayscale
        self.batch_size = batch_size
        self.dim = dim
        if grayscale:
            self.n_channels = 1
        else:
            self.n_channels = 3
        self.shuffle = shuffle
        self.shuffle_patches = shuffle_patches
        if not shuffle_patches:
            self._set_patch_sets()
        self.on_epoch_end()
        # Here is our beloved image augmentator <3
        self.transformer = img_transformer

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.ceil(len(self.patch_sets) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        idx_min = index * self.batch_size
        idx_max = min(idx_min + self.batch_size, len(self.patch_sets))
        indexes = self.indexes[idx_min: idx_max]

        # Find list of DFs, each DF contains a set of patches
        list_DFs_temp = [self.patch_sets[k] for k in indexes]

        # Generate image data
        X, y = self._generate_Xy(list_DFs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle_patches:
            self._set_patch_sets()

        self.indexes = np.arange(len(self.patch_sets))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_Xy(self, list_DFs_temp):
        """Generates data containing batch_size images
        :param list_DFs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((len(list_DFs_temp), *self.dim, self.N*self.n_channels))
        y = np.empty((len(list_DFs_temp)))
        # Generate data
        for i, DF in enumerate(list_DFs_temp):
            # If it is the full image and not a montage, extract montages
            for j, patch_row in DF.iterrows():
                # Store sample
                tmp = self._load_image(patch_row['file'])
                if self.grayscale:
                    tmp = np.expand_dims(cv2.resize(cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY), self.dim, interpolation=cv2.INTER_NEAREST), axis=-1)
                else:
                    tmp = cv2.resize(tmp, self.dim, interpolation=cv2.INTER_NEAREST)
                start_idx = i * self.N + j * self.n_channels
                end_idx = i * self.N + (j + 1) * self.n_channels
                X[i, :, :, start_idx:end_idx] = tmp

            # Store y
            y[i] = DF['pix_per_len'].iloc[0]

        # Transform the inputs and correct the outputs accordingly
        for i, x in enumerate(X):
            transform_params = self.transformer.get_random_transform(x.shape)
            X[i] = self.transformer.standardize(self.transformer.apply_transform(x, transform_params))

        return X, y


    def _set_patch_sets(self):
        # Create patch sets
        patch_sets = []

        if self.shuffle_patches:
            self.data = self.data.sample(frac=1)

        for idx, grp_df in self.data.groupby(['original_fp', 'pix_per_len'], sort=False):
            patch_sets.extend(split_given_size(grp_df, self.N))
            # Fill leftover patches with repeats of other patches
            if len(patch_sets[-1]) != self.N:
                num_sample = self.N - len(patch_sets[-1])
                patch_sets[-1] = pd.concat([patch_sets[-1], grp_df.iloc[:-num_sample].sample(num_sample)])
        self.patch_sets = patch_sets

    def _load_image(self, image_path):
        """Load grayscale image
        :param image_path: path to image to load
        :return: loaded image
        """
        img = cv2.imread(image_path)
        return img

    def _get_patch_set_dataframe(self):
        columns = self.patch_sets[0].columns.tolist()
        out_df = {col: [] for col in columns}
        for i in self.indexes:
            for col in columns:
                if col == 'file':
                    out_df[col].append(self.patch_sets[i][col].tolist())
                else:
                    out_df[col].append(self.patch_sets[i][col].values[0])

        return pd.DataFrame(out_df)
