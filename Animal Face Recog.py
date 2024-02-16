#ISSUE ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 3 dimensions. The detected shape was (2400, 80, 80) + inhomogeneous part.
#need to use a picklefile rather than two array for the images and the labels then resize them where the try loop is

import os
from matplotlib import pyplot as plt
import skimage
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.transform import rescale
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.color import rgb2gray
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import numpy as np

image_folder_dir = "Animal Recog Proj/Image"
file_name = []
label = []
images = []
height = None 
width=150
height = height if height is not None else width


pklname = f"{pklname}_{width}x{height}px.pkl"

#goes through data folder and assigns what animal each image is in a dicionary
for folder in os.listdir(image_folder_dir):
    try:    #using try avoids the .DS_Store file on mac which causes problems
        for file in os.listdir(os.path.join(image_folder_dir,folder)):
            file_name.append(file)
            label.append(folder)
            image = imread(os.path.join(image_folder_dir,folder, file))
            image = resize(image, (width, height)) #all images get resized to 80x80
            images.append(image)

    except Exception as e:
            print("Issue with image {}".format(file))
joblib.dump(data, pklname)

X = np.array(images)
y = np.array(label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    shuffle=True,random_state=42)


#using HOG-SVM - must transform the image to greyscale first then HOG
class MakeGrey(BaseEstimator, TransformerMixin): 
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for img in X:
            img = rescale(img, 1/3, mode='reflect')
            #variable2 = np.asarray(variable1, dtype="object")
            X_grey = np.append(skimage.color.rgb2gray(img))
        print(X_grey)
        #return (np.array([skimage.color.rgb2gray(img) for img in X])


class HogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, y = None, orientations=9, pixel_per_cell=(8, 8),
                 cell_per_block=(3, 3), block_norm="L2-Hys"):
        self.y = y
        self.orientations = orientations
        self.pixel_per_cell = pixel_per_cell
        self.cell_per_block = cell_per_block
        self.block_norm = block_norm

    def fit(self):
        return self

    def transform(self, X, y=None):

        def hog_fn(X):
            return hog(X, orientations = self.orientations,
                    pixel_per_cell = self.pixel_per_cell, 
                    cell_per_block = self.cell_per_block, 
                    block_norm = self.block_norm)
        try: 
            return np.array([hog_fn(img) for img in X])
        except:
            return np.array([hog_fn(img) for img in X])

#instance of grey and hog class
make_grey = MakeGrey()
image_hog = HogTransform(
    pixel_per_cell=(14, 14), 
    cell_per_block=(2,2), 
    orientations=9, 
    block_norm="L2-Hys"
)
scale = StandardScaler()

#fit transforming X_train
X_train_grey = make_grey.fit_transform(X_train)
X_train_hog = image_hog.fit_transform(X_train_grey)
X_train_transformed = scale.fit_transform(X_train_hog)

sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train_transformed, y_train)

X_test_grey = make_grey.transform(X_test)
X_test_hog = image_hog.transform(X_test_grey)
X_test_tranformed = scale.transform(X_test_hog)

y_pred = sgd_clf.predict(X_test_tranformed)
print(np.array(y_pred == y_test)[:25])
print("Accuracy (%): ", 100*(np.sum(y_pred == y_test)/len(y_test)))

