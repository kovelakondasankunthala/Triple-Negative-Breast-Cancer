import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from Evaluation import evaluation


def Model_Adaboost(Train_data, train_target, Test_data, test_target):

    train_data = np.reshape(Train_data, (Train_data.shape[0], (Train_data.shape[1]*Train_data.shape[2]*Train_data.shape[3])))
    test_data = np.reshape(Test_data, (Test_data.shape[0], (Test_data.shape[1]*Test_data.shape[2]*Test_data.shape[3])))

    model = AdaBoostClassifier(n_estimators=30, random_state=0)
    pred = np.zeros(test_target.shape)


    for i in range(test_target.shape[1]):
        model.fit(train_data, train_target[:, i])
        pred[:, i] = model.predict(test_data)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)
    return Eval, pred