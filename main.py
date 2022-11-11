import pickle

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from datetime import datetime
from flask import Flask, request, render_template
import models.vgg16 as VGG16
from pathlib import Path

app = Flask(__name__)

# Read image features
fe = VGG16.FeartureExtractor_VGG16
features = []
img_paths = []
filenames = pickle.load(open('static/feature/filenames-caltech101.pickle', 'rb'))
feature_list = pickle.load(open('static/feature/features-vgg16-resnet.pickle', 'rb'))

neighbors = NearestNeighbors(n_neighbors=100,
                             algorithm='brute',
                             metric='euclidean').fit(feature_list)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract_features(img_path=uploaded_img_path)
        distances, indices = neighbors.kneighbors([query])  # L2 distances to features
        scores = [(distances[0][index],  filenames[indices[0][index]].replace("\\", "/" )) for index in list(range(100))]
        print(scores)
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores
                               )
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run("0.0.0.0")
