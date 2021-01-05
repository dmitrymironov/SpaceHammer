'''
temporal dataset compressor
'''

import tensorflow as tf
import tensorflow.keras.utils
import sqlite3
import os, platform
import cv2
import datetime
import numpy as np
from scipy import interpolate as I
#from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, MaxPool2D, Dense, \
    TimeDistributed, GRU, Reshape, Input, Bidirectional, LSTM, \
    RepeatVector, Wrapper, BatchNormalization
import keras.optimizers
import models
from keras.callbacks import ModelCheckpoint, EarlyStopping

'''
Create inputs and targets from videos and geo-tagged video sequence
Inputs is a set of sliding windows prepared for convolution as in PoseConvGRU
Targets are speed regression values
'''
class tfTemporalCompressor:
    name = "frame_generator"  # can be train, valid, etc

    connection = None  # database connection handler for GT
    cursor = None

    num_samples = -1  # number of frames in a whole track (or file)

    # feeding into the model
    num_batches: int = -1  # number of batches
    batch_size = 15  # N of temporal frame pairs sequences in the batch
    batch_stride = 4  # temporal stride between batches (in sequences)

    batch_x = None  # batch_x - batches RAM placeholders
    batch_y = None  # batch_y

    '''
                        INITIALIZE (generator method)
    '''

    def __init__(self, fn_idx, name):
        self.name = name
        # db operations
        #---------------------------------------------------------------
        self.index_file = fn_idx
        self.db_open(fn_idx)

    def get_tracks(self):
        self.cursor.execute(
            '''
            SELECT track_id,COUNT(*) as N FROM Locations
            GROUP BY track_id
            ORDER BY N DESC
            '''
        )
        return self.cursor.fetchall()

    def get_track(self,id):
        self.cursor.execute(
            '''
            SELECT timestamp/1000,X(coordinate),Y(coordinate),CAST(speed AS decimal)
            FROM Locations
            WHERE track_id={}
            ORDER BY timestamp
            '''.format(id)
        )
        data=np.asarray(self.cursor.fetchall())
        return data[:,0],data[:,1:]
    #
    def db_open(self, index_file):
        print("Loading database from '" + index_file + "'")
        assert os.path.isfile(index_file), "Database file is not readable"
        try:
            self.connection = sqlite3.connect(index_file)
        except sqlite3.Error:
            assert False, sqlite3.Error
        # load spatial extensions (GEOS based wkt etc)
        self.connection.enable_load_extension(True)
        self.cursor = self.connection.cursor()
        self.cursor.execute("SELECT load_extension('mod_spatialite')")

    # dtor
    def __del__(self):
        self.connection.close()

'''
Split training sequence
'''        


class dsSplit(tensorflow.keras.utils.Sequence):
    batch_size=1000 # batch length
    num_batches=0
    sz=0 # num samples
    name=None
    
    def __init__(self,name,x,y):
        self.name=name
        self.sz = x.shape[0]
        self.num_batches=int(np.ceil(self.sz/self.batch_size))
        self.x=x
        self.y=y
        pass

    def __len__(self):
        return self.num_batches

    def __getitem__(self,i):
        if i+1 == self.num_batches:
            bx = self.x[-self.batch_size:]
            by = self.y[-self.batch_size:]
        else:
            bx = self.x[i *self.batch_size:(i+1)*self.batch_size]
            by = self.y[i*self.batch_size:(i+1)*self.batch_size]
        return bx.reshape(-1, self.batch_size), by.reshape(-1, self.batch_size)

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
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    print('================================================== TRAIN')
    comp = tfTemporalCompressor(db,"comp")
    track_id = 6

    '''prepare generators'''
    t, y = comp.get_track(track_id)
    v = y[:, 2]
    Nsamples = t.shape[0]
    Nvalid = 1000
    gen = dsSplit("train", t[:Nsamples-Nvalid], v[:Nsamples-Nvalid])
    valid_gen = dsSplit("validation", t[-Nvalid:], v[-Nvalid:])

    ''' define model'''
    opt = keras.optimizers.Adam(learning_rate=0.01)
    inputs = Input(shape=(gen.batch_size))

    '''
    x = Reshape(((-1, gen.batch_size)))(inputs)
    x = Bidirectional(LSTM(gen.batch_size))(x)
    x = BatchNormalization()(x)
    '''

    x = inputs
    
    # compressing autoencoder
    for i in [4096,4096, 4096, 4096, 1024, 512, 256, 128, 64, 64, 128, 256, 512]:
        x = Dense(i)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)

    outputs = Dense(gen.batch_size)(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="track_{}".format(track_id))
    model.compile(loss='mean_squared_error',
                  optimizer=opt, metrics=['mse'])
    model.summary()

    es = EarlyStopping(monitor='val_loss', patience=30, mode='min',min_delta=0.01)

    model.fit(
        gen,
        validation_data=valid_gen,
        epochs=100,
        callbacks=[es])

if __name__ == "__main__":
    main()
