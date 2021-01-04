#import training_set_generator
import os
import data_get
import tensorflow as tf
import platform
import numpy as np
#from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, MaxPool2D, Dense, \
    TimeDistributed, GRU, Reshape, Input, Bidirectional, LSTM, \
        RepeatVector, Wrapper
import keras.optimizers
import models

def main():
    os.system('clear')  # clear the terminal on linux
    print("Using Tensorflow {}".format(tf.__version__))
    # train1 = training_set_generator.TrainingSetGenerator()
    if platform.system() == "Windows":
        db = r'C:\\msys64\\home\\dmmie\\.dashcam.software\\dashcam.index'
    else:
        db = os.environ['HOME']+'/.dashcam.software/dashcam.index'
    db = os.path.normpath(db)

    '''
    Addressing CUDA/cuDNN driver issues on Windows
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    print('================================================== TRAIN')
    '''
    Generators
    '''
    train_gen = data_get.tfGarminFrameGen(db, track_id=1)
    validation_gen = data_get.tfGarminFrameGen(db,file_id=5)
    '''
    # to match a particular file
    id, path = validation_gen.get_file_id_by_pattern('%Mt-Adams-11-nov-2020%GRMN0005.MP4')
    path = os.path.normpath(path)
    print('File "{}" id is "{}"'.format(path,id))
    return 
    '''

    '''
    # memory leak test loop
    for iter in range(20):
        for batch_idx in range(validation_gen.__len__()):
            x, y = validation_gen.__getitem__(batch_idx)
    '''

    '''
    Model
    '''
    opt = keras.optimizers.Adam(learning_rate=0.01)
    flownet = models.FlowNet()
    # data_get.tfGarminFrameGen.batch_size,
    inputs = Input(shape=(480, 640, 6))
    x = flownet(inputs) # flownet is a temporal model
    x = Reshape(((-1, 5 * 1 * 1024)))(x)
    x = Bidirectional(LSTM(3, return_sequences=True))(x)
    x = LSTM(3)(x)
    x = Dense(4096)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.1)(x)
    outputs = Dense(1)(x) # temporal dimension X V

    '''
    pgu = models.PoseConvGRUNet()
    outputs = pgu(x)
    '''

    model = keras.Model(inputs=inputs, outputs=outputs, name="egomotion")

    model.compile(loss='mean_squared_error', optimizer=opt)

    model.summary()
    #keras.utils.plot_model(model, show_shapes=True)
    #model.build(input_shape=(train_gen.seq_size,480, 640, 6))
    #print(model.summary())

    '''
    Checkpoints
    '''

    filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    '''
    Train
    '''
    model.fit(train_gen, validation_data=validation_gen,
              callbacks=callbacks_list)

    '''
    #debug
    use_multiprocessing=True,
    workers=4
    ) 
    '''

    print("Done!")

if __name__ == "__main__":
    main()
