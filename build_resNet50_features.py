import os
from tqdm import notebook
import pickle
import models.resNet50 as resNet_class

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def get_file_list(root_dir):
    file_list = []
    counter = 1
    for root, directories, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(ext in filename for ext in extensions):
                file_list.append(os.path.join(root, filename))
                counter += 1
    return file_list


root_dir = 'static/data'
file_names = sorted(get_file_list())

feature_list = []
for index in notebook.tqdm(range(len(file_names))):
    feature_list.append(resNet_class.FeatureExtractor.extract_features(file_names[index]))

pickle.dump(feature_list, open('static/feature/features-caltech101-resnet.pickle', 'wb'))
