from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

class Subject():
    MATHEMATICS = "MATHEMATICS"
    BIOLOGY = "BIOLOGY"


#training data (all modified to be the same length so may not nessessarily be true)
X_train =[
    "Algebra is mathematics",
    "HWE is biology",
    "Calculus not biology",
    "Evolution like biology"
]

y_train = [
    Subject.MATHEMATICS,
    Subject.BIOLOGY,
    Subject.MATHEMATICS,
    Subject.BIOLOGY
]

#finds all unique words turns them into vectors
vectoriser = CountVectorizer()
X_train_vectors = vectoriser.fit_transform(X_train)

#lists unique words as array
words = vectoriser.get_feature_names_out(X_train)
print(words)

#prints array of the vector values of all the words and the ones correspond to the utterance of the word in the array
print(X_train_vectors.toarray())

#classifies whether a sentence is biology or mathematics and then can be used on further data
#using support vector machine SVM as it is used in classification problems
clf_svm = svm.SVC(kernel="linear")
clf_svm.fit(X_train_vectors,y_train)

#test data
X_test = [
    "Integrals is calculus mathematics, not biology like HWE/Evolution",
    "Not biology"
]

#using the SVM, predict the subject that the test data is
X_test_vectors = vectoriser.fit_transform(X_test)
pred = clf_svm.predict(X_test_vectors)
print(pred)

#two predictions are correct, model could be developed further with more data
#however limitation that the train and test data have to contain each other
#using a text file and NLTK is next step to make the algorithm more advanced


