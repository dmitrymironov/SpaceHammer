import numpy as np
import sqlite3
import os
import cv2
import kinetic_model
import dataset_loader
import frame_generator

class TrainingSetGenerator:
    # Points in time, ms. Dimension: T
    time_points = None
    # 3-channel frames. Dimension: T, Nchannel, W, H (Channels: R, G, B)
    rgb_frames = None 
    # Optical flow vector field. Dimension: T, W, H
    optical_flow = None

    #
    # CTOR
    #

    def __init__(self, fn, target_dim, type):
        L = dataset_loader.DashamDatasetLoader()
        file_name = L.next()
        print("Loading '{}' file '{}'".format(L.file_type, file_name))
        framer = frame_generator.FrameGenerator(file_name)
        self.time_points = np.zeros((L.num_frames,1))
        Nchannels=3 # B, G, R
        self.rgb_frames = np.zeros(
            (self.time_points.shape[0], Nchannels, framer.W, framer.H))
        Tidx=0
        prevgray = None
        while True:
            ret, img = framer.next()
            if ret is not True:
                break
            # update time
            self.time_points[Tidx]=framer.pos_msec
            # Crop to particular format (remove text)
            img = self.crop(img)
            # Downsize image
            out_image=cv2.resize(img,target_dim)
            # update channels for the position
            self.rgb_frames[Tidx][2], \
                self.rgb_frames[Tidx][1], \
                self.rgb_frames[Tidx][0] = \
                        cv2.split(out_image) # b, g, r
            # set optical flow
            gray = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
            if prevgray == None:
                prevgray=gray
            self.optical_flow[Tidx] = cv2.calcOpticalFlowFarneback(
                prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # increment time dimension
            Tidx += 1

    #
    # Methods
    #

    def crop(self, file_type, img):
        assert file_type == 'garmin', 'unsupported file type ' + file_type
        if file_type == 'garmin':
            margin = 100
            return img[margin:-margin,margin:-margin]
        return img
