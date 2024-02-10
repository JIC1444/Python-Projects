import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#importing test and train data
imagefile = "/Users/jakecordery/VSCode Projects/NatSci Coding WS/t10k-images.idx3-ubyte"
imagearray = idx2numpy.convert_from_file(imagefile)

labelfile = "/Users/jakecordery/VSCode Projects/NatSci Coding WS/t10k-labels.idx1-ubyte"
labelarray = idx2numpy.convert_from_file(labelfile)

X_train, X_test, y_train, y_test = train_test_split(imagearray, labelarray, test_size=0.2)

X_test = X_test.reshape((2000,784))
X_train = X_train.reshape((8000,784))

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

pred = knn.predict(X_test)
print(accuracy_score(y_test,pred))

#prints a few examples to show it works
X_train = X_train.reshape(8000,28,28)
for i in range(0,3):
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    print("Predicted: ",y_train[i])
    plt.show()
