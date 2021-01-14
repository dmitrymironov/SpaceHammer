import tensorflow as tf
import tensorflow.keras.utils
import os
import cv2
import datetime
import numpy as np

class FileRecord:
    name = None  # file path
    cap = None  # video capture handler
    frameCount = 0  # num of frames in the file
    img = None  # Last frame accessed
    W = 0
    H = 0
    FPS = 0

    def get_frame(self, frame_idx):
        assert frame_idx >= 0 & frame_idx <= self.frameCount, "Illegal frame idx"
        # set to a proper frame
        actual_pos = self.file.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if actual_pos != frame_idx:
            self.file.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, self.img = self.file.cap.read()
        assert ret, "Broken video '{}'".format(self.name)

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

    def reset(self):
        self.framePos = -1
        self.name = None

class FramePairsGenerator(tensorflow.keras.utils.Sequence):
    file = None
    name = "frame_generator"  # can be train, valid, etc
    speed_labels = None

    # feeding into the model
    num_batches: int = -1  # number of batches
    batch_size = 15  # N of temporal frame pairs in a batch
    batch_stride = 4  # temporal stride between batches

    '''placeholders'''
    batch_x = None 
    batch_y = None 

    def __init__(self, name, video_fn, speed_labels):
        assert os.path.exists(video_fn), 'Video file does not exist'
        assert os.path.exists(speed_labels), 'Speed file unreadable'
        self.name = name
        self.file = FileRecord(video_fn)
        self.num_batches = int(
            (self.file.frameCount-self.batch_size)/self.batch_stride)
        self.batch_x = np.zeros(
            (self.batch_size,self.file.H, self.file.W,6), 
            dtype='float16')
        self.batch_y = np.zeros((self.batch_size), dtype='float16')
        self.speed_labels = np.loadtxt(speed_labels)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, batch_idx: int):
        #print("Getting {} batch {}".format(self.name,batch_idx))
        assert batch_idx < self.num_batches, "incorrect batch number"
        frame1 = None
        frame2 = None
        # position in a current video file
        start = int(batch_idx*self.batch_stride)
        end = start + self.batch_size
        self.file.framePos = start
        for batch_pos in range(self.batch_size):
            if frame2 is None:
                frame1 = self.file.get_frame(self.file.framePos)
            else:
                frame1 = frame2
            self.file.framePos=self.file.framePos+1            
            frame2 = self.file.get_frame(self.file.framePos)
            self.batch_x[batch_pos] = tf.concat([frame1, frame2], axis=2)
        self.batch_y = self.speed_labels[start:end]

def main():
    os.system('clear')  # clear the terminal on linux

    train = FramePairsGenerator('train','data/train.mp4','data/train.txt')

if __name__ == "__main__":
    main()
