import numpy as np
import sqlite3
import os
import cv2
import kinetic_model
import dataset_loader
import frame_generator
from tqdm import tqdm

class TrainingSetGenerator:
    # Points in time, ms. Dimension: T
    time_points = None
    # 3-channel frames. Dimension: T, Nchannel, W, H (Channels: R, G, B)
    rgb_frames = None 
    # Optical flow vector field. Dimension: T, W, H, 2
    optical_flow = None
    # Ground truth Y - speed and angle. Dimensions: V
    y = None

    #
    # CTOR
    #

    def __init__(self, target_dim=(640, 480), file_type='garmin'):
        L = dataset_loader.DashamDatasetLoader()
        file_name = L.next()
        framer = frame_generator.FrameGenerator(file_name, L.file_type)
        self.time_points = np.zeros((L.num_samples*framer.FPS,1))
        Nchannels=3 # B, G, R
        self.rgb_frames = np.zeros(
            (self.time_points.shape[0], Nchannels, target_dim[1], target_dim[0]))
        self.optical_flow = np.zeros(
            (self.time_points.shape[0], target_dim[1], target_dim[0], 2))
        self.y = np.zeros(
            (self.time_points.shape[0],2))
        Tidx = 0
        prevgray = None
        # TODO: remove debug image showing when model works
        dbg_disp = False
        for t in tqdm(range(self.time_points.shape[0])):
            ret, img = framer.next()
            if ret is not True:
                break
            # update time
            self.time_points[Tidx]=framer.pos_msec
            # Downsize image
            assert img.shape == (1080,1920,3), "Unexpected garmin dimensions"
            '''
            # We should use tensorflow and CUDA acceleration for LA
            #-----------------------------------------------------------
            # Crop to particular format (remove text)
            img = self.crop(L.file_type, img, target_dim)
            # thats really weird but resize wants reversed HxW dimensions?
            out_image = cv2.resize(img, target_dim)
            # update channels for the position
            self.rgb_frames[Tidx][2], \
                self.rgb_frames[Tidx][1], \
                self.rgb_frames[Tidx][0] = \
                        cv2.split(out_image) # b, g, r
            # set optical flow
            gray = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('1', gray)
            if prevgray is None:
                prevgray=gray
            self.optical_flow[Tidx] = cv2.calcOpticalFlowFarneback(
                prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prevgray=gray
            if dbg_disp:
                cv2.imshow('flow', self.draw_flow(gray, self.optical_flow[Tidx]))
            #-----------------------------------------------------------
            '''
            # Set ground truth       
            self.y[Tidx][0] = L.kinetic_model.speed(framer.pos_msec)
            if Tidx == 0:
                continue
            
            inputs = tf.keras.Input(shape=(10, 128, 128, 3))

            # self.y[Tidx][1] = L.kinetic_model.angle(framer.pos_msec)
            # increment time dimension
            Tidx += 1
            if dbg_disp:
                ch = 0xFF & cv2.waitKey(1)

    #
    # Methods
    #

    def draw_flow(self,img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 0, 255))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def crop(self, file_type, img, target_dim):
        assert file_type == 'garmin', 'unsupported file type ' + file_type
        if file_type == 'garmin':
            # target aspect ratio, W/H
            ar = target_dim[0]/target_dim[1]
            # image size for opencv is H x W
            sz = img.shape
            Htxt=50
            new_sz = [sz[0]-Htxt, sz[1]]
            if new_sz[1]/new_sz[0] < ar:
                # truncate height
                new_sz[0] = int(new_sz[1]/ar)
                dw = [int((sz[0]-new_sz[0])/2), 0]
            else:
                # turncate width
                new_sz[1] = int(new_sz[0]*ar)
                dw = [Htxt, int((sz[1]-new_sz[1])/2)]
            return img[0:sz[0]-dw[0], 0:sz[1]-dw[1]]
        return img
