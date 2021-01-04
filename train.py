#import training_set_generator
import os
import data_get
import tensorflow as tf
import platform
import numpy as np
#from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, MaxPool2D, Dense, \
    TimeDistributed, GRU, Reshape, Input, Bidirectional, LSTM, \
    RepeatVector, Wrapper, BatchNormalization
import keras.optimizers
import models
from keras.callbacks import ModelCheckpoint

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
        1535|5
        1692|8
        2482|4
        3208|12
        5478|7
        18256|6
    '''
    train_gen = data_get.tfGarminFrameGen(db, name='train', track_id=6) # 7, 12
    validation_gen = data_get.tfGarminFrameGen(db, name='validation',file_id=5)

    '''
    Model
    '''
    opt = keras.optimizers.Adam(learning_rate=0.01)

    # data_get.tfGarminFrameGen.batch_size,
    inputs = Input(shape=(480, 640, 6))

    x = Conv2D(64, kernel_size=7, strides=2, padding='same',input_shape=(480, 640, 6), name='conv1')(inputs)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, kernel_size=5, strides=2, padding='same', name='conv2')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, kernel_size=5, strides=2, padding='same', name='conv3')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding='same', name='conv3_1')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, kernel_size=3, strides=2,padding='same', name='conv4')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, kernel_size=3, strides=1, padding='same', name='conv4_1')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, kernel_size=3, strides=2,padding='same', name='conv5')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, kernel_size=3, strides=1, padding='same', name='conv5_1')(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(1024, kernel_size=3, strides=2,padding='same', name='conv6')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D(2, strides=2)(x)

    x = Reshape(((-1, 5 * 1 * 1024)))(x)
    x = Bidirectional(LSTM(1000, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = LSTM(1000)(x)
    x = BatchNormalization()(x)
    x = Dense(4096)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    outputs = Dense(1)(x) # temporal dimension X V

    '''
    pgu = models.PoseConvGRUNet()
    outputs = pgu(x)
    '''

    model = keras.Model(inputs=inputs, outputs=outputs, name="egomotion")
    model.compile(loss='mean_squared_error',optimizer=opt, metrics=['accuracy'])
    model.summary()

    if True:
        # Loads the weights
        checkpoint_path = 'save/egomotion-01-183971936.00.hdf5'
        model.load_weights(checkpoint_path)
    #keras.utils.plot_model(model, show_shapes=True)
    #model.build(input_shape=(train_gen.seq_size,480, 640, 6))
    #print(model.summary())

    '''
    Checkpoints
    '''

    filepath = "save/egomotion-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_loss',
        verbose=1, 
        save_best_only=True, 
        mode='auto')

    stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=1)

    '''
    Train
    '''
    model.fit(train_gen, validation_data=validation_gen,
              callbacks=[checkpoint, stop_callback],
              shuffle=False)

    '''
    #debug
    use_multiprocessing=True,
    workers=4
    ) 
    '''

    print("Done!")

if __name__ == "__main__":
    main()
