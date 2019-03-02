### IMPORTS
print('Loading libraries')
import keras
from keras import backend as tf
from keras import optimizers
from keras import regularizers
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.models import load_model, Model
from keras_preprocessing.image import ImageDataGenerator
import os
import random
import numpy as np
import cv2
import gc
import plots

### GLOBAL VARIABLES
train_set_dir = "data\\train\\"
valid_set_dir = "data\\valid\\"
networks_dir = "networks\\"
rows = 200
columns = 200
depth = 3
input_dimensions = (rows, columns, depth)
input_count = 0
val_count = 0
batch = 64
batch_count = 0
autosave_threshold = 5


### FUNCTIONS
def create_network2():
    trained_classifier = load_model('networks\\trained_regressor.nn')
    old_layers = trained_classifier.layers[:-5]
    for layer in old_layers:
        layer.trainable = False
    x = old_layers[-1].output
    x = Flatten()(x)
    x = Dense(64, bias_initializer='glorot_uniform', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, bias_initializer='glorot_uniform', activation='relu')(x)
    #x = Dropout(0.5)(x)
    predictions = Dense(1, bias_initializer='glorot_uniform', activation='sigmoid')(x)
    model = Model(inputs=old_layers[0].input, outputs=predictions)
    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', 
              optimizer=opt,
              metrics=['accuracy'])
    return model
def create_network():
    input_img = Input(shape=input_dimensions)

    x = Conv2D(16,(3,3), activation='relu',padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(32,(3,3), activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32,(3,3), activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(3,3), activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64,(3,3), activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64,(3,3), activation='relu',padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    #x = Conv2D(32,(3,3), activation='relu',padding='same')(x)
    #x = BatchNormalization()(x)
    #x = Conv2D(32,(3,3), activation='relu',padding='same')(x)
    #x = BatchNormalization()(x)
    #x = Conv2D(32,(3,3), activation='relu',padding='same')(x)
    #x = BatchNormalization()(x)
    #x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)

    y = Dense(100, activation='relu')(x)
    y = Dropout(0.5)(y)
    y = Dense(25, activation='relu')(y)
    y = Dense(1, activation='sigmoid')(y)

    z = Dense(100, activation='relu')(x)
    z = Dropout(0.5)(z)
    z = Dense(100, activation='relu')(z)
    z = Dense(4, activation='sigmoid')(z)

    output = keras.layers.concatenate([y, z], axis=1)
    
    model = Model(inputs=input_img, outputs=output)

    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='mse',
              optimizer=opt)
    return model

def set_lr(value, flag):
    if flag:
        print('Original Learning rate value:' + str(keras.backend.get_value(model.optimizer.lr)))
    keras.backend.set_value(model.optimizer.lr, value)
    print('New Learning rate value:' + str(keras.backend.get_value(model.optimizer.lr)))
def set_decay(value, flag):
    if flag:
        print('Original Decay rate value:' + str(keras.backend.get_value(model.optimizer.decay)))
    keras.backend.set_value(model.optimizer.decay, value)
    print('New Decay rate value:' + str(keras.backend.get_value(model.optimizer.decay)))
def save():
    print('Enter filename:')
    model.save(networks_dir + input()+'.nn')
    print("Saved model to disk")

def load_data(path):
    # Returns image paths and loaded labels as 2 numpy arrays
    images = []
    labels = []
    for i in os.listdir(path):
        if i.endswith('.jpg'):
            images.append(path + i)
        else:
            labels.append(np.loadtxt(path + i))
    _labels = np.vstack(labels)
    #assign classes to 0 and 1
    _labels[:,0] = _labels[:,0] - 1
    _images = np.array(images)
    del images
    del labels
    gc.collect()
    return (_images, _labels)
def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        # what happens below
        # load image -> resize -> change BGR to RGB -> add to an array 
        images.append(cv2.cvtColor(cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (rows,columns), 
                                            interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2RGB))
    result = np.array(images) / 255
    del images
    gc.collect()
    return result

def train_regressor(epochs):
    val_errors = []
    train_errors = []
    for i in range(epochs):
        print('Epoch ' + str(i))
        shuffle_set(train_indexes)
        epoch_error = 0
        for X,y in batch_generator(train_image_paths, train_labels, train_indexes):
            #show_images(X,y)
            epoch_error += model.train_on_batch(X, y)[0]
        val_errors.append(model.evaluate(valid_images, valid_labels, verbose=0)[0])
        train_errors.append(epoch_error/batch_count)
        print('Train error: ' + str(train_errors[-1]) + '\tValid error: ' + str(val_errors[-1]))
    return (train_errors, val_errors)
def train_classifier(epochs):
    train_datagen = ImageDataGenerator(
        shear_range=0.1,zoom_range=0.1, rescale=1./255,
        horizontal_flip=True, rotation_range=10,
        width_shift_range=0.1,height_shift_range=0.1)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_set_dir, shuffle=True,
            target_size=(rows, columns, ),
            batch_size=batch,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            valid_set_dir,
            target_size=(rows, columns),
            batch_size=32,
            class_mode='binary')

    model.fit_generator(
            train_generator,
            use_multiprocessing=False,
            epochs=epochs,
            validation_data=validation_generator)
def train_global(epochs):
    val_errors = []
    train_errors = []
    for i in range(epochs):
        print('Epoch ' + str(i))
        shuffle_set(train_indexes)
        epoch_error = 0
        for X,y in batch_generator(train_image_paths, train_labels, train_indexes):
            #show_images(X,y)
            epoch_error += model.train_on_batch(X, y)#[y[:,0],y[:,1:]])[0]
        val_errors.append(model.evaluate(valid_images, valid_labels))#[valid_labels[:,0],valid_labels[:,1:]], verbose=0))
        train_errors.append(epoch_error/batch_count)
        print('Train error: ' + str(train_errors[-1]) + '\tValid error: ' + str(val_errors[-1]))
    return (train_errors, val_errors)

def shuffle_set(indexes):
    np.random.shuffle(indexes)
def batch_generator(images, labels, indexes):
    X = np.array(1)
    y = np.empty((batch, 5))
    global batch_count
    batch_count = len(images) // batch
    for i in range(batch_count):
        X = preprocess_images(np.take(images, indexes[i*batch:(i+1)*batch]))
        y = np.take(labels, indexes[i*batch:(i+1)*batch], axis=0)
        yield (X, y)

def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou
def mIoU():
    iou_list = []
    label_list = []
    for X,y in batch_generator(train_image_paths, train_labels, train_indexes):
        iou_list.append(model.predict(X))
        label_list.append(y)
    iou_list = np.vstack(iou_list)[:,1:]
    label_list = np.vstack(label_list)[:,1:]
    summ = 0
    for i in range(len(iou_list)):
        summ += bb_intersection_over_union(iou_list[i], label_list[i])
    print('MIOU on training set: ' + str((summ/len(iou_list)) * 100) + '%')

    iou_list2 = []
    summ2 = 0
    for i in range(len(valid_labels) // 32):
        iou_list2.append(model.predict(valid_images[32*i:32*(i+1)]))
    iou_list2 = np.vstack(iou_list2)[:,1:]
    for i in range(len(iou_list2)):
        summ2 += bb_intersection_over_union(iou_list2[i], valid_labels[i])
    print('MIOU on testing set: ' + str((summ2/len(iou_list2)) * 100) + '%')
    print('Total MIOU: ' + str((summ + summ2) / (len(iou_list2) + len(iou_list) )))
 def accuracy():
    pred_list = []
    label_list = []
    for X,y in batch_generator(train_image_paths, train_labels, train_indexes):
        pred_list.append(model.predict(X))
        label_list.append(y)
    pred_list = np.vstack(pred_list)[:,0]
    label_list = np.vstack(label_list)[:,0]

    def f(x):
        return (1 if x > 0.5 else 0)
    f = np.vectorize(f)
    pred_list = f(pred_list)

    error_train = np.sum(np.absolute(label_list - pred_list))
    print('Train set classification accuracy: ' + str((1 - error_train / len(label_list))* 100) + '%')

    pred_list2 = []
    for i in range(len(valid_labels) // 32):
        pred_list2.append(model.predict(valid_images[32*i:32*(i+1)]))
    pred_list2 = f(np.vstack(pred_list2)[:,0])

    error_test = np.sum(np.absolute(valid_labels[:576,0] - pred_list2))
    print('Test set classification accuracy: ' + str((1 - error_test / len(valid_labels))* 100) + '%')

    print('Total accuracy: ' + str(((1 - (error_test + error_test) / 
            ( len(label_list) + len(valid_labels))))* 100) + '%')


def mainloop():
    global batch
    counter = 0
    while(True):
        print('''
1. Keep training
2. Save ANN to file
3. Test
4. Set learning rate
5. Set batch size
6. Model summary
7. Unrelated images
8. Set decay
9. MIOU + Accuracy
0. Exit co console''')
        tmp = input()
        if tmp == '1':
            # keep training
            print('Train for how many epochs?')
            epochs = int(input())
            print('Training')
            #global tr_errors, vl_errors
            while(True):
                gc.collect()
                if epochs > autosave_threshold:
                    # if training for more than autosave_threshold - autosave
                    train_global(autosave_threshold)
                    model.save('iteration'+str(counter)+'.nn')
                    counter += 1
                else:
                    # otherwise train normally
                    train_global(epochs)
                    break
                epochs = epochs - autosave_threshold
            #popup.WindowsBalloonTip('Training complete', 'CNN')
        elif tmp == '2':
            save()
        elif tmp == '3':
            mid_point = len(valid_image_paths)//2
            val = random.randint(5, mid_point - 5)
            predictions = model.predict(np.vstack([valid_images[val:val+5], 
                                            valid_images[val+mid_point:val+mid_point+5]]))
            print(str(predictions))
            #show_images(valid_images[val:val+10], model.predict(valid_images[val:val+10]), categories=False)

            plots.show_all(valid_images[val:val+10], (rows, columns),
                                 model.predict(valid_images[val:val+10]), valid_labels[val:val+10])

            #count = 5
            #for X,y in batch_generator(train_image_paths, train_labels, train_indexes):
            #    predictions = model.predict(X)
            #    show_bounds(X, predictions, y)
            #    count -= 1
            #    if count < 1:
            #        break
            #gc.collect()

            #cats = os.listdir(train_set_dir + '0')
            #dogs = os.listdir(train_set_dir + '1')
            #val = random.randint(5, len(cats) - 5)
            #catpics = np.core.defchararray.add(train_set_dir + '0\\' , cats[val:val+5])
            #dogpics = np.core.defchararray.add(train_set_dir + '1\\' , dogs[val:val+5])
            #paths = np.concatenate([catpics, dogpics])
            #images = preprocess_images(paths)
            #predictions = model.predict(images)
            #print(str(predictions))
            #show_images_class(images,predictions)
        elif tmp == '4':
            print('Current Learning rate value:' + str(keras.backend.get_value(model.optimizer.lr)))
            print('Enter new LR value:')
            set_lr(float(input()), False)
        elif tmp == '5':
            print('Current batch size value:' + str(batch))
            print('Enter batch size value:')
            batch = int(input())
        elif tmp == '6':
            model.summary()
        elif tmp == '7':
            imgs = preprocess_images(['random_images\\' + i for i in os.listdir('random_images\\')])
            lbls_pred = model.predict(imgs)
            plots.show_unlabeled(imgs,(rows, columns),lbls_pred)
        elif tmp == '8':
            print('Current Decay rate value:' + str(keras.backend.get_value(model.optimizer.decay)))
            print('Enter new decay value:')
            set_decay(float(input()), False)
        elif tmp == '9':
            print('This will take a while')
            mIoU()
            accuracy()
        else:
            print('Exit?yn')
            if input() == 'y':
                break

### MAIN
#if __name__ == "__main__":
print('Loading data')
train_image_paths, train_labels = load_data(train_set_dir)
valid_image_paths, valid_labels = load_data(valid_set_dir)
train_indexes = np.arange(len(train_image_paths))
# might as well do it here for now
valid_images = preprocess_images(valid_image_paths)

print('Training set: ' + str(train_image_paths.shape))
print('Training labels: ' + str(train_labels.shape))
print('Validation set: ' + str(valid_image_paths.shape))
print('Validation labels: ' + str(valid_labels.shape))

print('\n1. Create new CNN\n2. Load from file')
tmp = input()
if tmp == '1':
    print('Creating ANN')
    model = create_network()
else:
    netfiles = []
    for file in os.listdir(networks_dir):
        if file.endswith('.nn'):
            netfiles.append(file)
    if len(netfiles) == 0:
        print('Input filename:')
        model = load_model(input())
    else:
        print('Choose network file:')
        [print(str(ct)+'. '+netfiles[ct]) for ct in range(0,len(netfiles))]
        print(str(len(netfiles)) + '. ' + 'Type filename manually')
        tmp = int(input())
        if tmp < len(netfiles):
            model = load_model(networks_dir + netfiles[tmp])
        else:
            print('Input filename:')
            model = load_model(input())
    print("Loaded model from disk")
mainloop()
