import numpy as np
import tensorflow as tf
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# const from library

image = tf.keras.preprocessing.image
ResNet50 = tf.keras.applications.resnet50.ResNet50
preprocess_input = tf.keras.applications.resnet50.preprocess_input
model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))


class FeatureExtractor:

    def extract_features(img_path):
        input_shape = (224, 224, 3)
        img = image.load_img(img_path, target_size=(
            input_shape[0], input_shape[1]))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        features = model.predict(preprocessed_img)
        flattened_features = features.flatten()
        normalized_features = flattened_features / np.linalg.norm(flattened_features)
        return normalized_features
