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
    # Optical flow vector field. Dimension: T, W, H, 2
    optical_flow = None

    #
    # CTOR
    #

    def __init__(self, target_dim=(640, 480), file_type='garmin'):
        L = dataset_loader.DashamDatasetLoader()
        file_name = L.next()
        framer = frame_generator.FrameGenerator(file_name, L.file_type)
        self.time_points = np.zeros((L.num_frames,1))
        Nchannels=3 # B, G, R
        self.rgb_frames = np.zeros(
            (self.time_points.shape[0], Nchannels, target_dim[0], target_dim[1]))
        self.optical_flow = np.zeros(
            (L.num_frames, target_dim[0], target_dim[1],2))
        Tidx = 0
        prevgray = None
        while True:
            ret, img = framer.next()
            if ret is not True:
                break
            # update time
            self.time_points[Tidx]=framer.pos_msec
            # Downsize image
            assert img.shape == (1080,1920,3), "Unexpected garmin dimensions"
            # Crop to particular format (remove text)
            img = self.crop(L.file_type, img, target_dim)
            # thats really weird but resize wants reversed HxW dimensions?
            out_image = cv2.resize(img, (target_dim[1],target_dim[0]))
            # update channels for the position
            self.rgb_frames[Tidx][2], \
                self.rgb_frames[Tidx][1], \
                self.rgb_frames[Tidx][0] = \
                        cv2.split(out_image) # b, g, r
            # set optical flow
            gray = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
            if prevgray is None:
                prevgray=gray
            self.optical_flow[Tidx] = cv2.calcOpticalFlowFarneback(
                prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # increment time dimension
            Tidx += 1

    #
    # Methods
    #

    def crop(self, file_type, img, target_dim):
        assert file_type == 'garmin', 'unsupported file type ' + file_type
        if file_type == 'garmin':
            # target aspect ratio
            ar = target_dim[0]/target_dim[1]
            sz = img.shape
            new_sz = [sz[0],sz[1]-50]
            if new_sz[0]/new_sz[1] > ar:
                # truncate width
                new_sz[0]=int(new_sz[1]*ar)
            else:
                # turncate height
                new_sz[1]=int(new_sz[0]/ar)
            dw = [int((sz[0]-new_sz[0])/2), int((sz[1]-new_sz[1])/2)]
            return img[dw[0]:sz[0]-dw[0],dw[1]:sz[1]-dw[1]]
        return img
