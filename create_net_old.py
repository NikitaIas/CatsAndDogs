#TEMP FILE
import keras
from keras import optimizers
from keras import regularizers
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import load_model, Model

def create_network4():
    input_img = Input(shape=input_dimensions)

    x = dimension_reduction_inception(input_img)
    x = dimension_reduction_inception(x)
    x = MaxPooling2D((2, 2))(x)
    x = dimension_reduction_inception(x)
    x = dimension_reduction_inception(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_img, outputs=predictions)

    opt = optimizers.SGD(lr=0.01, momentum=True)
    model.compile(loss='logcosh', 
              optimizer=opt,
              metrics=['binary_accuracy'])
    #opt = optimizers.Adam(lr=0.001)
    #model.compile(loss='mean_squared_error', 
    #          optimizer=opt,
    #          metrics=['mae', 'acc'])
    return model

def create_network3():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_dimensions, activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5), activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5),activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5),activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(32, bias_initializer='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, bias_initializer='glorot_uniform'))
    model.add(Activation('sigmoid'))

    opt = optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', 
              optimizer=opt,
              metrics=['binary_accuracy'])
    return model

def create_network2():
    trained_classifier = load_model('networks\\classificator.nn')
    old_layers = trained_classifier.layers[:-5]
    for layer in old_layers:
        layer.trainable = False
    x = old_layers[-1].output
    x = Dense(100, bias_initializer='glorot_uniform', activation='relu')(x)
    #x = Dense(64, bias_initializer='glorot_uniform', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, bias_initializer='glorot_uniform', activation='relu')(x)
    #x = Dropout(0.5)(x)
    predictions = Dense(4, bias_initializer='glorot_uniform', activation='sigmoid')(x)
    model = Model(inputs=old_layers[0].input, outputs=predictions)
    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='mse', 
              optimizer=opt,
              metrics=['accuracy'])
    return model

def create_network1():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_dimensions, activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3),activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3),activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu',
        bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Conv2D(32, (3, 3), activation='relu',
    #    bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Conv2D(32, (3, 3), activation='relu',
    #    bias_initializer='glorot_uniform', padding='same'))#, kernel_regularizer=regularizers.l2()))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, bias_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, bias_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(4, bias_initializer='glorot_uniform', activation='sigmoid'))

    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='mse', 
              optimizer=opt,
              metrics=['accuracy'])
    return model

def dimension_reduction_inception(inputs):
    tower_one = MaxPooling2D((3,3), strides=(1,1), padding='same')(inputs)
    tower_one = Conv2D(6, (1,1), activation='relu', padding='same')(tower_one)
    #tower_one = MaxPooling2D((2, 2))(tower_one)

    tower_two = Conv2D(6, (1,1), activation='relu', padding='same')(inputs)
    tower_two = Conv2D(6, (3,3), activation='relu', padding='same')(tower_two)
    #tower_two = MaxPooling2D((2, 2))(tower_two)

    tower_three = Conv2D(6, (1,1), activation='relu', padding='same')(inputs)
    tower_three = Conv2D(6, (5,5), activation='relu', padding='same')(tower_three)
    #tower_three = MaxPooling2D((2, 2))(tower_three)

    x = concatenate([tower_one, tower_two, tower_three], axis=3)
    return x