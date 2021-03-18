import pandas as pd

# blending ensemble for classification using hard voting
from numpy import hstack
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.neural_network import MLPClassifier
from numpy.linalg import inv
from sklearn import preprocessing
from tqdm import tqdm
import sys

def train_test_split(data):
    """
    split data into three set train, test and valid
    return:
        dict: python dictionary
    """
    training_points = np.array(letters[:12000].drop(['letter'], 1))
    training_labels = np.array(letters[:12000]['letter'])

    le = preprocessing.LabelEncoder()
    le.fit(training_labels)
    training_labels = le.transform(training_labels)

    test_points = np.array(letters[16000:].drop(['letter'], 1))
    test_labels = np.array(letters[16000:]['letter'])
    test_labels = le.transform(test_labels)

    valid_points = np.array(letters[12000: 16000].drop(['letter'], 1))
    valid_labels = np.array(letters[12000: 16000]['letter'])
    valid_labels = le.transform(valid_labels)

    return le, {
        'train_x': training_points,
        'train_y': training_labels,
        'valid_x': valid_points,
        'valid_y': valid_labels,
        'test_x': test_points,
        'test_y': test_labels
    }


# get a list of base models
def get_models():
    models = list()
    models.append(('lr',  LogisticRegression()))
    models.append(('mlp',  MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,), random_state=43)))
    models.append(('cart', DecisionTreeClassifier()))
    models.append(('svm', SVC(probability=True)))
    return models


def get_classes_list_point(train_x, train_y):
    """
    get claases list in trainset
    """
    classes_list_points_dict = {}
    for (x, y) in zip(train_x, train_y):
        if y not in classes_list_points_dict.keys():
            classes_list_points_dict[y] = [x]
        else:
            classes_list_points_dict[y].append(x)

    classes_list = []
    for k in classes_list_points_dict.keys():
        classes_list.append(classes_list_points_dict[k])

    return classes_list


# tập hợp các điểm trong class i
def get_distance_metric_A(classes_list):
    """
    :param classes_list: np.array, a list of lists of points which are in the same class.
    :return:

    """
    total = 0
    for each_class in classes_list:  # each_class: shape: n x n_feature
        class_center = np.array(each_class).mean(axis=0)
        s = np.sum(np.array(
            [np.matmul((sample - class_center).reshape(16, 1), (sample - class_center).reshape(1, 16)).tolist() for
             sample in each_class]) / len(each_class), axis=0)
        total += s

    return inv(total)


def get_knn_point_index(test_point, matrix_A, samples, k=5):
    """
    return the k nearest neighbors distance values
    test point laf mot diem
    """
    # matrix_A = distance_metric_A(classes_list)
    dis = np.array([((np.array(sample - test_point).reshape(1, 16)) @ matrix_A @ (
        np.array(sample - test_point).reshape(16, 1))).reshape(-1)[0] for sample in samples])
    idx = np.argpartition(dis, k)
    return idx[:k]


def get_bias_for_point(clf, knn_index, train_x, train_y):
    """
    estimate bias for test point
    """
    knn_points = train_x[knn_index, :]
    knn_y = train_y[knn_index]
    knn_y_onehot = np.zeros((knn_y.shape[0], 26))
    knn_y_onehot[:, knn_y] = 1

    y_hat_knn_points = clf.predict_proba(knn_points)
    bias = (y_hat_knn_points - knn_y_onehot)
    # print(np.argmax(y_hat_knn_points, axis=1), knn_y)
    bias = np.mean(bias, axis=0)
    return bias


# def predict_and_bias_correction(clf, X, classes_list, train_x, train_y):
#     Y = []
#     distance_matrix_A = get_distance_metric_A(classes_list)
#     for x in tqdm(X):
#         knn_idx = get_knn_point_index(x, distance_matrix_A, train_x)
#         bias = get_bias_for_point(clf, knn_idx, train_x, train_y)
#         pred = clf.predict_proba([x]).reshape(-1)
#         pred = pred - bias
#         Y.append(pred.tolist())
#
#     Y = np.array(Y)
#     return np.argmax(Y, axis=1).reshape(len(Y), 1)


def predict_proba_and_bias_correction(clf, X, classes_list, train_x, train_y):
    """
    predict probabilites for all data in test
    """
    Y = []
    distance_matrix_A = get_distance_metric_A(classes_list)
    for x in tqdm(X):
        knn_idx = get_knn_point_index(x, distance_matrix_A, train_x)
        bias = get_bias_for_point(clf, knn_idx, train_x, train_y)
        pred = clf.predict_proba([x]).reshape(-1)
        pred = pred - bias
        Y.append(pred.tolist())

    Y = np.array(Y)
    return Y


def fit_ensemble(models, X_train, X_val, y_train, y_val):
    """
    fit the blending ensemble
    """
    # fit all models on the training set and predict on hold out set
    meta_X = list()
    classes_list = get_classes_list_point(X_train, y_train)
    print("Fitting model ...")
    for name, model in models:
        # fit in training set
        model.fit(X_train, y_train)
        # predict on hold out set
        # yhat = predict_and_bias_correction(model, X_val, classes_list, X_train, y_train)
        yhat = predict_proba_and_bias_correction(model, X_val, classes_list, X_train, y_train)
        print(yhat.shape)

        # store predictions as input for blending
        meta_X.append(yhat)
    # create 2d array from predictions, each set is an input feature
    meta_X = hstack(meta_X)
    # define blending model
    blender = LogisticRegression()
    # fit on predictions from base models
    blender.fit(meta_X, y_val)
    return blender


def predict_ensemble(models, blender, X_train, y_train, X_test):
    """
    make a prediction with the blending ensemble
    """
    # make predictions with base models
    meta_X = list()
    bias_corrections_out = {} # dict of bias correction predict for models
    classes_list = get_classes_list_point(train_x, train_y)
    for name, model in models:
        # predict with base model
        yhat = predict_proba_and_bias_correction(model, X_test, classes_list, X_train, y_train)
        # # reshape predictions into a matrix with one column
        # yhat = yhat.reshape(len(yhat), 1)
        # store prediction
        meta_X.append(yhat)
        bias_corrections_out[name] = yhat
    # create 2d array from predictions, each set is an input feature
    meta_X = hstack(meta_X)
    # predict
    return blender.predict(meta_X), bias_corrections_out


if __name__ == '__main__':
    # define log file
    f = open('log.txt', 'w')
    sys.stdout = f

    # define dataset
    letters = pd.read_csv('letter-recognition.txt')
    le, split = train_test_split(letters)
    train_x, train_y, valid_x, valid_y, test_x, test_y = split['train_x'], split['train_y'], split['valid_x'], split['valid_y'], split['test_x'], split['test_y']

    X_train, X_val, X_test, y_train, y_val, y_test = train_x, valid_x, test_x, train_y, valid_y, test_y

    # summarize data split
    print('Train: %s, Val: %s, Test: %s' % (X_train.shape, X_val.shape, X_test.shape))

    # create the base models
    models = get_models()
    # train the blending ensemble
    blender = fit_ensemble(models, X_train, X_val, y_train, y_val)
    # make predictions on test set
    yhat, bias_corrections_out = predict_ensemble(models, blender, X_train, y_train, X_test)
    # evaluate predictions
    score = accuracy_score(y_test, yhat)
    print('Blending Accuracy: %.3f' % (score*100))

    for name, model in models:
        yhat = model.predict(X_test)
        score = accuracy_score(y_test, yhat)
        print(f'{name} accuracy: %.3f' % (score*100))

    for key in bias_corrections_out.keys():
        yhat = bias_corrections_out[key]
        yhat = np.argmax(yhat, axis=1)
        score = accuracy_score(y_test, yhat)
        print(f'{key} with bias correction accuracy: %.3f' % (score*100))

    yhat = None
    for key in bias_corrections_out.keys():
        if yhat is None:
            yhat = bias_corrections_out[key]
        else:
            yhat += bias_corrections_out[key]
    yhat = yhat / len(bias_corrections_out.keys())
    yhat = np.argmax(yhat, axis=1)
    score = accuracy_score(y_test, yhat)
    print('essemble model with bias correction: %.3f' % (score*100))



