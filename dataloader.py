import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras 

class DataGenerator(keras.utils.Sequence):

  # Initialize the data generator
  def __init__(self, image_paths, facs_features, emotion_labels, batch_size, input_size, shuffle=True, Augment=True):
    # Initialize variables
    self.image_paths = image_paths
    self.facs_features = facs_features
    self.emotion_labels = emotion_labels
    self.dim = (224, 224, 3)
    self.batch_size = batch_size
    self.input_size = input_size
    self.shuffle = shuffle
    self.Augment = Augment
    self.indexes = np.arange(len(self.image_paths))
    self.data_mean = 0
    self.data_std = 255.0

  # Get the number of batches
  def __len__(self):
    return int(np.ceil(len(self.image_paths) / self.batch_size))

  # Get a batch of data
  def __getitem__(self, index):
    # Select batch indexes
    batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

    # Load and preprocess images
    batch_image_paths = [self.image_paths[i] for i in batch_indexes]
    batch_images = np.array([self.load_and_preprocess_image(path) for path in batch_image_paths])

    # Get batch facs features and emotion labels
    batch_facs_features = self.facs_features[batch_indexes]
    batch_emotion_labels = self.emotion_labels[batch_indexes]
    batch_emotion_labels_one_hot = to_categorical(batch_emotion_labels, num_classes=3)

    return batch_images, {'facs_output': batch_facs_features, 'emotion_output': batch_emotion_labels_one_hot}

  # Shuffle data at the end of each epoch
  def on_epoch_end(self):
    if self.shuffle:
      np.random.shuffle(self.indexes)

  # Load and preprocess a single image
  def load_and_preprocess_image(self, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))

    # Apply data augmentation if enabled
    if self.Augment:
      image = tf.keras.preprocessing.image.random_rotation(image, 10)
      image = tf.keras.preprocessing.image.random_shift(image, 0.15, 0.15)
      image = tf.keras.preprocessing.image.random_zoom(image, (0.8, 1.2))
      image = tf.image.random_flip_left_right(image)

      image = tf.image.random_brightness(image, max_delta=0.1)

      image = tf.image.resize_with_pad(image, 224, 224)
      image = tf.image.random_crop(image, size=self.dim)

    image = (image - self.data_mean) / self.data_std

    return image