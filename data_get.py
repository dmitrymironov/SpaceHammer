import tensorflow as tf
import tensorflow.keras.utils
import sqlite3, os, cv2
import numpy as np
from scipy import interpolate as I

'''
Create inputs and targets from videos and geo-tagged video sequence
Inputs is a set of sliding windows prepared for convolution as in PoseConvGRU
Targets are speed regression values
'''

class tfGarminFrameGen(tensorflow.keras.utils.Sequence):
    connection = None # database connection handler for GT
    Vthreshold = 0.5  # km/h, GPS trheshold noise
    num_samples = -1 # number of frames in the sequence
    num_batches = -1 # number of batches
    batch_size = -1 # number of frames in the batch
    Nff = 1800 # Frames per file. On our dataset it's constant
    file_ids = [] # indexes of files in the dataset
    current_file_id = -1 # current file id
    current_file_pos = -1 # current file frame id
    cap = None # opencv video handler
    batch_x = None # batch_x - batches RAM placeholders
    batch_y = None # batch_y 

    Htxt = 50 # Garmin text cropping line height
    Wframe = 1080 # Garmin frame WxH
    Hframe = 1920
    CHframe = 6 # Num channels per X

    '''
                        INITIALIZE (generator method)
    Use file or track id to load data sequence
    '''
    def __init__(self, fn_idx, track_id=None, file_id=None):
        self.batch_size = 32
        assert self.Nff % self.batch_size == 0, "Batch size is not divisible"
        self.batch_x = np.zeros((self.batch_size, \
            self.Wframe, self.Hframe-self.Htxt, \
                self.CHframe))
        self.batch_y = np.zeros((self.batch_size))
        self.index_file = fn_idx
        self.db_open(fn_idx)
        # load ground truth data from the db
        if track_id is not None:
            self.preload_track(track_id)
        else:
            self.preload_file(file_id)
        self.num_samples = len(self.file_ids)*self.Nff
        self.num_batches = self.num_samples / self.batch_size

    '''
    number of batches (generator method)
    '''
    def __len__(self):
        return self.num_batches

    '''
    Garmin video:
        crop bottom with GPS text, rescale
    '''
    def garmin_crop(self, img, target_dim=(640, 480)):
        # target aspect ratio, W/H
        ar = target_dim[0]/target_dim[1]
        # image size for opencv is H x W
        sz = img.shape
        assert sz == (self.Wframe, self.Hframe, self.CHframe), "Unexpected image dimensions"
        new_sz = [sz[0]-self.Htxt, sz[1]]
        if new_sz[1]/new_sz[0] < ar:
            # truncate height
            new_sz[0] = int(new_sz[1]/ar)
            dw = [int((sz[0]-new_sz[0])/2), 0]
        else:
            # turncate width
            new_sz[1] = int(new_sz[0]*ar)
            dw = [self.Htxt, int((sz[1]-new_sz[1])/2)]
        return img[0:sz[0]-dw[0], 0:sz[1]-dw[1]]

    '''
    position to a particular file
    '''
    def select_file(self, file_id: int):
        # if we're already positioned in the file, return
        if file_id==self.current_file_id:
            return
        # close previously opened file
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.current_file_id=file_id
        self.current_file_pos = 0
        self.fn = self.file_name(self.current_file_id)
        self.cap = cv2.VideoCapture(self.fn)
        frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frameCount==self.Nff, "Unexpected frame count"
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("Loading '{}' {}x{} {}fps, {} frames".format(
            self.fn, self. W, self.H, self.FPS, self.Nff))

    '''
    Get next frame in the sequence
    '''
    def get_frame(self, frame_idx):

        '''
        Read next frame
        '''
        assert frame_idx >= 0 & frame_idx <= self.Nff, "Illegal frame idx"
        # set to a proper frame
        if self.cap.get(cv2.CV_CAP_PROP_POS_FRAMES) != frame_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = self.cap.read()
        assert ret, "Broken video '{}'".format(self.fn)
        return self.garmin_crop(img)

    '''
    Get current file_idx and frame idx
    '''
    def get_position(self):        
        assert self.cap is not None, "get_position expects opened file"
        self.current_file_pos = self.cap.get(cv2.CV_CAP_PROP_POS_FRAMES)
        return self.current_file_id, self.current_file_pos

    def move_on(self):
        if self.current_file_pos+1 >= self.Nff:
            # trigger switching to a next file
            self.file_ids_pos = self.file_ids_pos+1
            if self.file_ids_pos>=len(self.file_ids):
                # it's a last file, we need to stop
                self.current_file_id = -2
                self.current_file_pos = -2
                return False
            else:
                self.select_file(self.file_ids[self.file_ids_pos])
        else:
            self.current_file_pos = self.current_file_pos+1
        return True
    '''
    Get batch_x and batch_y for training (generator method)
    '''
    def __getitem__(self, batch_idx: int):
        frame1 = None
        frame2 = None
        self.file_ids_pos = int(batch_idx*self.batch_size/self.Nff)
        self.select_file(self.file_ids[self.file_ids_pos])
        self.current_file_pos=int(batch_idx*self.batch_size)%self.Nff
        fEndReached=False
        for batch_pos in range(self.batch_size):
            if frame2 is None:
                frame1 = self.get_frame(self.current_file_pos)
            else:
                frame1 = frame2
            if self.move_on(): 
                frame2 = self.get_frame(self.current_file_pos)
                # channels concatenate
                self.batch_x[batch_pos]=np.concatenate(frame1,frame2,axis=2)
                Tframe = (batch_idx*self.batch_size+batch_pos)*1000
                self.batch_y[batch_pos] = self.speed(Tframe)
            else:
                # We've reached the end, just repeating the last frame
                assert fEndReached, "We should not stack more than one end frame"
                self.batch_x[batch_pos] = np.concatenate(frame1, frame1, axis=2)
                self.batch_y[batch_pos] = self.speed(Tframe)
                fEndReached=True
        return self.batch_x, self.batch_y

    '''
    Database operations to provide ground truth
    '''
    def preload_track(self, track_id: int):
        cursor = self.connection.cursor()
        cursor.execute(
            '''
            SELECT timestamp/1000,CAST(speed AS real)
            FROM Locations 
            WHERE track_id=(?)
            ORDER BY timestamp
            ''',
            (track_id,)
        )
        self.load_speed_labels(cursor.fetchall())
        cursor.execute(
            '''
            SELECT DISTINCT(file_id) FROM Locations WHERE track_id=(?) 
            ORDER BY timestamp
            ''', (track_id,)
        )
        self.file_ids = cursor.fetchall()

    #
    def preload_file(self, file_id):
        cursor = self.connection.cursor()
        cursor.execute(
            '''
            SELECT timestamp/1000,CAST(speed AS real)
            FROM Locations 
            WHERE file_id=(?) AND type='garmin'
            ORDER BY timestamp
            ''',
            (file_id,)
        )        
        self.load_speed_labels(cursor.fetchall())
        cursor.execute(
            '''
            SELECT COUNT(*) FROM Locations WHERE file_id=(?)
            ''', (file_id,)
        )
        self.file_ids=[file_id]

    #
    def load_speed_labels(self,d):
        t_v=np.array(d)
        T = t_v[:,0]
        t0 = T[0]
        T -= t0
        gps_speed = t_v[:,1]
        self.Vinterpoated = I.splrep(T, gps_speed, s=0)

    # 
    def db_open(self,index_file):
        print("Loading database from '" + index_file + "'")
        assert os.path.isfile(index_file), "Database file is not readable"
        try:
            self.connection = sqlite3.connect(index_file)
        except sqlite3.Error:
            assert False, sqlite3.Error
        # load spatial extensions (GEOS based wkt etc)
        self.connection.enable_load_extension(True)
        self.connection.cursor().execute("SELECT load_extension('mod_spatialite')")

    #
    def speed(self, Tmsec: int):
        # use interpolated gps speed
        v = abs(I.splev(Tmsec, self.Vinterpoated, der=0))
        b = (v > self.Vthreshold).astype(int)
        return v*b

    # 
    def file_name(self,file_id: int) -> str:
        cursor = self.connection.cursor()
        cursor.execute(
            '''
            SELECT d.path || "/" || f.name
            FROM Files as f, Folders as d
            WHERE f.hex_digest IS NOT NULL AND f.path_id=d.id AND f.id=(?)
            ''', (file_id,)
        )
        return cursor.fetchall()[0][0]

    # dtor
    def __del__(self):
        self.connection.close()
