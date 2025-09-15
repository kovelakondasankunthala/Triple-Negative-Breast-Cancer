from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten
import numpy as np
from keras import backend as K
from keras import layers, models
from Evaluation import evaluation


def get_new_model(input_shape):
    '''
    This function returns a compiled CNN with specifications given above.
    '''

    # Defining the architecture of the CNN
    input_layer = Input(shape=[32, 32, 1], name='input')
    h = Conv2D(filters=16, kernel_size=(3, 3),
               activation='relu', padding='same', name='conv2d_1')(input_layer)
    h = Conv2D(filters=16, kernel_size=(3, 3),
               activation='relu', padding='same', name='conv2d_2')(h)

    h = MaxPool2D(pool_size=(2, 2), name='pool_1')(h)

    h = Conv2D(filters=16, kernel_size=(3, 3),
               activation='relu', padding='same', name='conv2d_3')(h)
    h = Conv2D(filters=16, kernel_size=(3, 3),
               activation='relu', padding='same', name='conv2d_4')(h)

    h = MaxPool2D(pool_size=(2, 2), name='pool_2')(h)

    h = Conv2D(filters=16, kernel_size=(3, 3),
               activation='relu', padding='same', name='conv2d_5')(h)
    h = Conv2D(filters=16, kernel_size=(3, 3),
               activation='relu', padding='same', name='conv2d_6')(h)

    h = Dense(64, activation='relu', name='dense_1')(h)
    h = Dropout(0.5, name='dropout_1')(h)
    h = Flatten(name='flatten_1')(h)
    output_layer = Dense(10, activation='softmax', name='dense_2')(h)

    # To generate the model, we pass the input layer and the output layer
    model = Model(inputs=input_layer, outputs=output_layer, name='model_CNN')

    # Next we apply the compile method
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def Model_CNN(Data):
    IMG_SIZE = 32
    train_data1 = np.zeros((len(Data), IMG_SIZE, IMG_SIZE))
    Data = np.resize(Data, (train_data1.shape[0], IMG_SIZE * IMG_SIZE))
    for n in range(len(Data)):
        train_data1[n, :, :] = np.resize(Data[n], (IMG_SIZE, IMG_SIZE))
    X = np.reshape(train_data1, (len(Data), IMG_SIZE, IMG_SIZE, 1))
    # Train_Data = Data.reshape(32, 32, 1)
    # model = get_new_model(X.shape)
    benchmark_layers = get_new_model(X.shape)
    benchmark_input = X.shape
    inp = benchmark_layers.input  # input placeholder
    outputs = [layer.output for layer in benchmark_layers.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layerNo = 4
    Feature = []
    for i in range(X.shape[0]):
        test = X[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()
        Feature.append(layer_out)
    CNN_Feat = np.asarray(Feature)
    Feat = np.reshape(CNN_Feat, (CNN_Feat.shape[0], CNN_Feat.shape[1]*CNN_Feat.shape[2]*CNN_Feat.shape[3]))
    return Feat

#---------------------------------------------------------------
def Model_Cls(X, Y, test_x, test_y, Batch_size):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # 'relu'
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 'relu'
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 'relu'
    model.summary()
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(test_y.shape[-1]))
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])  # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.fit(X, Y, epochs=50, batch_size=Batch_size, validation_data=(test_x, test_y))
    pred = model.predict(test_x)
    return pred


def Model_CNN_Cls(train_data, train_target, test_data, test_target, Batch_size=None):
    if Batch_size is None:
        Batch_size = 16
    IMG_SIZE = 32
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))
    pred = Model_Cls(Train_X, train_target, Test_X, test_target, Batch_size)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)
    return Eval, pred

