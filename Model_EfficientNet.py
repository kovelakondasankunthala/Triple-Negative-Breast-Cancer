from keras import Sequential
from keras.applications import EfficientNetB3
from keras.layers import Dense, GlobalAveragePooling2D, Reshape, Multiply, Conv2D
import numpy as np
from Evaluation import evaluation
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf


# Squeeze-and-Excitation block
def se_block(input_tensor, reduction=16):
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]

    se_shape = (1, 1, filters)
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(filters // reduction, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([input_tensor, se])
    return x


def Model_AAENet(Train_Data, Train_Target, Test_Data, Test_Target, sol=None):
    if sol is None:
        sol = [50, 15]
    IMG_SIZE = [32, 32, 3]
    Feat1 = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Feat1[i, :] = np.resize(Train_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Train_Data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((Test_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Test_Data.shape[0]):
        Feat2[i, :] = np.resize(Test_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Test_Data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    base_model = EfficientNetB3(
        weights='imagenet',
        input_shape=(32, 32, 3),
        include_top=False
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)
    x = se_block(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=sol[0], activation='relu')(x)
    x = Dense(units=sol[1], activation='relu')(x)
    outputs = Dense(units=Train_Target.shape[1], activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(Train_Data, Train_Target, epochs=1, batch_size=32, validation_data=(Test_Data, Test_Target))
    pred = model.predict(Test_Data)
    Eval = evaluation(pred, Test_Target)
    return Eval, pred


# --------------------------------------------------------------------------------

def Model_EfficientNet_Feat(Data, Target, sol=None):
    if sol is None:
        sol = [5, 5, 5, 5]
    IMG_SIZE = [32, 32, 3]
    # Split dataset for training and testing
    Train_Data, Test_Data, Train_Target, Test_Target = train_test_split(Data, Target, test_size=0.2, random_state=42)
    Feat1 = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Feat1[i, :] = np.resize(Train_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Train_Data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((Test_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Test_Data.shape[0]):
        Feat2[i, :] = np.resize(Test_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Test_Data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    efficient_net = EfficientNetB3(
        weights='imagenet',
        input_shape=(32, 32, 3),
        include_top=False,
        pooling='max'
    )

    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units=int(sol[0]), activation='relu'))
    model.add(Dense(units=Train_Target.shape[1], activation='relu'))
    # model.add(Dense(units=Test_Target.shape[2], activation='sigmoid'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(Train_Data, Train_Target, epochs=int(sol[1]), batch_size=32, validation_data=(Test_Data, Test_Target))
    layerNo = 0
    get_layer_output = K.function([model.layers[0].input], [model.layers[layerNo].output])

    data = np.concatenate((Train_Data, Test_Data), axis=0)
    Feats = []
    for i in range(data.shape[0]):
        print(i, data.shape[0])
        test = data[i, :, :][np.newaxis, ...]
        layer_out = get_layer_output([test])[0]
        Feats.append(layer_out)
    Feats = np.asarray(Feats)
    return Feats

