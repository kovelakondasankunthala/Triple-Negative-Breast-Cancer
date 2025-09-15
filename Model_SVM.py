from sklearn.svm import SVC  # "Support Vector Classifier"
import numpy as np
from Evaluation import evaluation

def Model_SVM(train_data, train_target, test_data, test_target):
    # if Epoch is None:
    #     Epoch = 100
    # Data = [82, 70]
    # Train_X = np.resize(train_data, (train_data[0], (train_data[1] * train_data[2] * train_data[3]))).any()
    train_samples, train_height, train_width, train_channels = train_data.shape
    test_samples, test_height, test_width, test_channels = test_data.shape

    # Reshape the data
    Train_X = train_data.reshape((train_samples, train_height * train_width * train_channels))
    Test_X = test_data.reshape((test_samples, test_height * test_width * test_channels))

    # # Ensure target arrays are 1-dimensional
    # train_target = train_target.ravel()
    # test_target = test_target.ravel()

    svm_model = SVC(kernel='linear', C=2.0, random_state=42)
    svm_model.fit(Train_X, train_target)

    Y_pred = svm_model.predict(Test_X)
    pred = np.asarray(Y_pred).reshape(-1, 1)

    Eval = evaluation(pred, test_target)
    return Eval, pred


# if __name__ == '__main__':
#     Image = np.load('Images_1.npy', allow_pickle=True)
#     Target = np.load('Target_1.npy', allow_pickle=True)
#     Hid_Neuron = round(Image.shape[0] * 0.75)
#     Train_Data = Image[:Hid_Neuron, :]
#     Train_Target = Target[:Hid_Neuron, :]
#     Test_Data = Image[Hid_Neuron:, :]
#     Test_Target = Target[Hid_Neuron:, :]
#     Eval, pred = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target)
#     iosdhg = 56

