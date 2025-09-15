import numpy as np
import random as random
from scipy.stats import truncnorm
from Evaluation import evaluation


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


def Model_ANN_Feat(Data):
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
    ANN_Feat = np.asarray(Feature)
    Feat = np.reshape(ANN_Feat, (ANN_Feat.shape[0], ANN_Feat.shape[1]*ANN_Feat.shape[2]*ANN_Feat.shape[3]))
    return Feat


@np.vectorize
def sigmoid(x):
    return random.random() #1/(1 + np.e ** -x)


activation_function = sigmoid


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class NeuralNetwork:

    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate, bias=None):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.bias = bias

        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes

        self.no_of_hidden_nodes = no_of_hidden_nodes

        self.learning_rate = learning_rate
        self.bias = bias
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural
        network with optional bias nodes"""

        bias_node = 1 if self.bias else 0

        rad = 1 / np.sqrt(self.no_of_in_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_nodes,
                                        self.no_of_in_nodes + bias_node))

        rad = 1 / np.sqrt(self.no_of_hidden_nodes + bias_node)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_nodes,
                                         self.no_of_hidden_nodes + bias_node))

    def train(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray

        bias_node = 1 if self.bias else 0
        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [self.bias]))

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        output_vector1 = np.dot(self.weights_in_hidden, input_vector)
        output_vector_hidden = activation_function(output_vector1)

        if self.bias:
            output_vector_hidden = np.concatenate((output_vector_hidden, [[self.bias]]))

        output_vector2 = np.dot(self.weights_hidden_out, output_vector_hidden)
        output_vector_network = activation_function(output_vector2)

        output_errors = target_vector - output_vector_network
        # update the weights:
        tmp = output_errors * output_vector_network * (1.0 - output_vector_network)
        tmp = self.learning_rate * np.dot(tmp, output_vector_hidden.T)
        self.weights_hidden_out += tmp

        # calculate hidden errors:
        hidden_errors = np.dot(self.weights_hidden_out.T, output_errors)
        # update the weights:
        tmp = hidden_errors * output_vector_hidden * (1.0 - output_vector_hidden)
        if self.bias:
            x = np.dot(tmp, input_vector.T)[:-1, :]  # ???? last element cut off, ???
        else:
            x = np.dot(tmp, input_vector.T)
        self.weights_in_hidden += self.learning_rate * x

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray

        if self.bias:
            # adding bias node to the end of the inpuy_vector
            input_vector = np.concatenate((input_vector, [1]))
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = activation_function(output_vector)

        if self.bias:
            output_vector = np.concatenate((output_vector, [[1]]))

        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector


def train_dnn(data, labels, test_data, sol):
    simple_network = NeuralNetwork(no_of_in_nodes=data.shape[1],
                                   no_of_out_nodes=1,
                                   no_of_hidden_nodes=int(sol),
                                   learning_rate=0.1,
                                   bias=None)

    for _ in range(20):
        for i in range(len(data)):
            simple_network.train(data[i, :], labels[i])

    pred = simple_network.run(test_data)
    predict = np.zeros((pred.shape[1])).astype('int')
    for i in range(pred.shape[1]):
        if pred[0, i] > 0.5:
            predict[i] = 1
        else:
            predict[i] = 0

    return predict, simple_network



def Model_Cas_ANN(Data,Target):

    Feature = Model_ANN_Feat(Data)

    Hid_Neu = [100, 200, 300, 400, 500]
    for learn in range(len(Hid_Neu)):
        Hid_Neuron = round(Feature.shape[0] * 0.75)
        Train_Data = Feature[:Hid_Neuron, :]
        Train_Target = Target[:Hid_Neuron, :]
        Test_Data = Feature[Hid_Neuron:, :]
        Test_Target = Target[Hid_Neuron:, :]

    simple_network = NeuralNetwork(no_of_in_nodes=Train_Data.shape[1],
                                   no_of_out_nodes=Train_Target.shape[1],
                                   no_of_hidden_nodes=1,
                                   learning_rate=0.1,
                                   bias=None)

    for _ in range(20):
        for i in range(len(Train_Data)):
            simple_network.train(Train_Data[i, :], Train_Target[i, :])

    pred = simple_network.run(Test_Data)
    predict = np.zeros((pred.shape[1], pred.shape[0])).astype('int')
    for i in range(pred.shape[1]):
        for j in range(pred.shape[0]):
            if pred[j, i] > 0.5:
                predict[i, j] = 1
            else:
                predict[i, j] = 0
    Eval = evaluation(predict, Test_Target)
    # return np.asarray(Eval).ravel()
    return Eval, predict