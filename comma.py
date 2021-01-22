import tensorflow as tf
import tensorflow.keras.utils
from tensorflow import keras
from keras.layers import Conv2D, LeakyReLU, MaxPool2D, Dense, \
    TimeDistributed, GRU, Reshape, Input, Bidirectional, LSTM, \
    RepeatVector, Wrapper, BatchNormalization, ReLU, Conv2D, Flatten, \
    AveragePooling1D
import keras.optimizers
import models
from keras.callbacks import ModelCheckpoint, EarlyStopping
import platform
import os
import cv2
import datetime
import numpy as np

class FileRecord:
    name = None  # file path
    cap = None  # video capture handler
    frameCount = 0  # num of frames in the file
    img = None  # Last frame accessed
    target_dim = (240,320)
    W = 0
    H = 0
    FPS = 0

    def get_frame(self, frame_idx):
        assert frame_idx >= 0 & frame_idx <= self.frameCount, "Illegal frame idx"
        # set to a proper frame
        actual_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if actual_pos != frame_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, self.img = self.cap.read()
        assert ret, "Broken video '{}'".format(self.name)
        self.img = cv2.resize(self.img, self.target_dim)
        return self.img

    def __init__(self, name):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.name = name
        self.cap = cv2.VideoCapture(self.name)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("Loaded {} fps {}x{} '{}' ".format(self.FPS,self.W,self.H,self.name))
        pass

    def reset(self):
        self.framePos = -1
        self.name = None

class FramePairsGenerator(tensorflow.keras.utils.Sequence):
    file = None
    name = "frame_generator"  # can be train, valid, etc
    speed_labels = None

    # feeding into the model
    num_batches: int = -1  # number of batches
    Xsize = 8  # N of temporal frame pairs in X
    stride = 4  # temporal stride between batches
    startFrame=0
    endFrame=0

    '''placeholders'''
    batch_x = None 
    batch_y = None 

    def __init__(self, name, video_fn, speed_labels, split=0.9):
        assert os.path.exists(
            video_fn), 'Video file {} does not exist'.format(video_fn)
        assert os.path.exists(
            speed_labels), 'Speed file {} is unreadable'.format(speed_labels)
        self.name = name
        self.file = FileRecord(video_fn)
        if (name == 'valid'):
            self.startFrame = int(self.file.frameCount*split)
            self.endFrame = self.file.frameCount
            self.Xsize = 10
            self.stride = 5
        elif (name == 'train'):
            self.startFrame = 0
            self.endFrame = int(self.file.frameCount*split)
            self.Xsize = 8
            self.stride = 2
        N = self.endFrame-self.startFrame
        self.num_batches = int(
            (N-self.Xsize)/self.stride)
        self.batch_x = np.zeros(
            (self.Xsize, self.file.target_dim[1], self.file.target_dim[0], 6),
            dtype='float16')
        self.batch_y = np.zeros((self.Xsize), dtype='float16')
        self.speed_labels = np.loadtxt(speed_labels)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, batch_idx: int):
        #print("Getting {} batch {}".format(self.name,batch_idx))
        assert batch_idx < self.num_batches, "incorrect batch number"
        frame1 = None
        frame2 = None
        # position in a current video file
        start = int(batch_idx*self.stride)+self.startFrame
        end = start + self.Xsize
        self.file.framePos = start
        for batch_pos in range(self.Xsize):
            if frame2 is None:
                frame1 = self.file.get_frame(self.file.framePos)
            else:
                frame1 = frame2
            self.file.framePos=self.file.framePos+1            
            frame2 = self.file.get_frame(self.file.framePos)
            self.batch_x[batch_pos] = tf.concat([frame1, frame2], axis=2)
        self.batch_y = self.speed_labels[start:end]
        return self.batch_x, self.batch_y

def main():

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

    #os.system('clear')  # clear the terminal on linux
    print('TensorFlow ' + tf.__version__)
    print('Keras ' + keras.__version__)

    if any(platform.win32_ver()):
        data_folder = 'D:/comma_ai.data/'
    else:
        data_folder = '/4tb/comma_ai.data/'
    train_file='{}train.mp4'.format(data_folder)
    train_labels = '{}train.txt'.format(data_folder)
    model_best = '{}/save/saved-model-07-72.51.hdf5'.format(data_folder)

    train = FramePairsGenerator('train', train_file, train_labels)
    valid = FramePairsGenerator('valid', train_file, train_labels)

    #opt = keras.optimizers.Nadam(learning_rate=0.01)
    opt = keras.optimizers.Adam(amsgrad=True, learning_rate=0.001)
    k=8
    input_shape = (train.file.target_dim[1], train.file.target_dim[0],6)
    model = keras.Sequential([
        Input(shape=input_shape),
        Conv2D(filters=64, kernel_size=7, strides=2,padding='same', name='Conv1'),
        BatchNormalization(),LeakyReLU(0.1),
        Conv2D(filters=128, kernel_size=5, strides=2,padding='same', name='Conv2'),
        BatchNormalization(),LeakyReLU(0.1),
        Conv2D(filters=256, kernel_size=5, strides=2, padding='same', name='Conv3'),
        BatchNormalization(),LeakyReLU(0.1),
        Conv2D(filters=256, kernel_size=3, strides=1,padding='same', name='Conv3_1'),
        BatchNormalization(), LeakyReLU(0.1),
        Conv2D(filters=512, kernel_size=3, strides=2,padding='same', name='Conv4'),
        BatchNormalization(), LeakyReLU(0.1),
        Conv2D(filters=512, kernel_size=3, strides=1,padding='same', name='Conv4_1'),
        BatchNormalization(), LeakyReLU(0.1),
        Conv2D(filters=512, kernel_size=3, strides=2,padding='same', name='Conv5'),
        BatchNormalization(), LeakyReLU(0.1),
        Conv2D(filters=512, kernel_size=3, strides=1,padding='same', name='Conv5_1'),
        BatchNormalization(), LeakyReLU(0.1),
        Conv2D(filters=1024, kernel_size=3, strides=2,padding='same', name='Conv6'),
        Reshape((-1, k*64)),
        LSTM(k*64, return_sequences=True),
        BatchNormalization(), LeakyReLU(0.1),
        LSTM(k*64),
        BatchNormalization(), LeakyReLU(0.1),
        Dense(1)
    ])

    model.compile(loss='mean_squared_error', optimizer=opt)
    model.summary()
    if os.path.exists(model_best):
        print("Loading weights from '{}'".format(model_best))
        model.load_weights(model_best)

    filepath = data_folder + "/save/saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_loss',
        verbose=1,
        save_best_only=False,
        mode='auto')

    es = EarlyStopping(monitor='val_loss', patience=200,
                       mode='auto', min_delta=0.01)

    model.fit(train, validation_data=valid, epochs=1000, callbacks=[es, checkpoint])
    pass

if __name__ == "__main__":
    main()
