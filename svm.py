import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from numpy.linalg import inv

def svc(training_points, training_labels):
    clf = SVC()
    clf.fit(training_points, training_labels)
    return clf


def nn(training_points, training_labels):
    # MLP classification
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,), random_state=1)
    clf.fit(training_points, training_labels)
    return clf


def logistic_regression():
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf


letters = pd.read_csv('letter-recognition.txt')

#  (SVM : oke
#  , NN,
#  CART,
#  LR) on

training_points = np.array(letters[:15000].drop(['letter'], 1))
training_labels = np.array(letters[:15000]['letter'])

print(training_labels[:3])

test_points = np.array(letters[15000:].drop(['letter'], 1))
test_labels = np.array(letters[15000:]['letter'])

clf = nn(training_points, training_labels)

expected = test_labels
predicted = clf.predict(test_points)

print("testing ...")
print(expected[:5])


accuracy = clf.score(test_points, test_labels)

print(float(accuracy))

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

#----------------------------------------------------------------------

# predicted = clf.predict(test_points)
# accuracy = clf.score(test_points, test_labels)


# tập hợp các điểm trong class i
#
def distance_metric_A(classes_list, test_point):
    """
    :param classes_list: np.array, a list of lists of points which are in the same class.
    :param test_point: np.array,  a vector of size n (n is the # of features)
    :return:
    """
    total = 0
    for each_class in len(classes_list):
        s = sum([np.matmul(test_point - sample, (test_point - sample).transpose()) for sample in each_class])/len(each_class)
        total += s
    return inv(total)


def get_distance(test_point, classes_list, samples, k=5):
    """
    return the k nearest neighbors distance values
    """
    dis = [np.array(sample-test_point).transpose()@distance_metric_A(classes_list, test_point)@np.array(sample-test_point) for sample in samples]
    return sorted(dis)[:k]

