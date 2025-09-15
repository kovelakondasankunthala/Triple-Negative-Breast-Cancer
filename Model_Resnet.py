import numpy as np
from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dense
from Evaluation import evaluation


def Model_RESNET(train_data, train_target, test_data, test_target):
    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    base_model.add(Dense(units=train_target.shape[1], activation='relu'))
    base_model.summary()
    base_model.compile(loss='binary_crossentropy', metrics=['acc'])
    base_model.fit(train_data, train_target, epochs=1)
    pred = base_model.predict(test_data)
    return pred


def Model_Resnet(train_data, train_target, test_data, test_target):
    pred = Model_RESNET(train_data, train_target, test_data, test_target)

    pred = np.asarray(pred)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)

    return Eval, pred
