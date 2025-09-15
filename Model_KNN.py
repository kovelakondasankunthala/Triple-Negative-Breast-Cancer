from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from Evaluation import evaluation
import random as rn


def Model_KNN(Train_data, train_target, Test_data, test_target):
    scaler = StandardScaler()

    train_data = np.reshape(Train_data,
                            (Train_data.shape[0], (Train_data.shape[1] * Train_data.shape[2] * Train_data.shape[3])))
    test_data = np.reshape(Test_data,
                           (Test_data.shape[0], (Test_data.shape[1] * Test_data.shape[2] * Test_data.shape[3])))

    scaler.fit(train_data)

    # X_train = scaler.transform(train_data)
    # X_test = scaler.transform(test_data)
    classifier = KNeighborsClassifier(n_neighbors=1)

    classifier.fit(train_data, train_target)

    # Predict Output
    out = classifier.predict(test_data)
    predicted = np.round(out)

    Eval = evaluation(predicted.reshape(-1, 1), test_target.reshape(-1, 1))

    return np.asarray(Eval).ravel(), predicted
