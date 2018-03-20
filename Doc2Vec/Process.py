# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy as np

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# random, itertools, matplotlib
import random
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Load model
model = Doc2Vec.load('./imdb.d2v')
#Test
#print(model.most_similar('good'))

# Classifying Sentiments

# Extract vector from Doc2Vec
# We have 25000 reviews (12500 pos and 12500 neg)
X_train = np.zeros((25000, 100))	#vectors
y_train = np.zeros(25000)			#labels

for i in range(12500):
    prefix_train_pos = 'TRAIN_POS_' + str(i)
    prefix_train_neg = 'TRAIN_NEG_' + str(i)
    X_train[i] = model.docvecs[prefix_train_pos]
    X_train[12500 + i] = model.docvecs[prefix_train_neg]
    y_train[i] = 1
    y_train[12500 + i] = 0

# Testing Vectors
X_test = np.zeros((25000, 100))
y_test = np.zeros(25000)

for i in range(12500):
    prefix_test_pos = 'TEST_POS_' + str(i)
    prefix_test_neg = 'TEST_NEG_' + str(i)
    X_test[i] = model.docvecs[prefix_test_pos]
    X_test[12500 + i] = model.docvecs[prefix_test_neg]
    y_test[i] = 1
    y_test[12500 + i] = 0

# Classification
# Using Logistic Regression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

# Using SVM
# classifier = SVC()
# classifier.fit(X_train, y_train)

print('Accuracy', classifier.score(X_test, y_test))
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)
plot_confusion_matrix(cm, classes=['neg', 'pos'])





