import os
from tqdm import notebook
import pickle
import models.vgg16 as vgg16_model

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
file_names = sorted(get_file_list(root_dir))

feature_list = []
for index in notebook.tqdm(range(len(file_names))):
    feature_list.append(vgg16_model.FeartureExtractor_VGG16.extract_features(file_names[index]))

pickle.dump(feature_list, open('static/feature/features-vgg16-resnet.pickle', 'wb'))
pickle.dump(file_names, open('static/feature/filenames-caltech101.pickle', 'wb'))
