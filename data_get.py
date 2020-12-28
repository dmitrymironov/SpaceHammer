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
    connection = None
    Vthreshold = 0.5  # km/h, ignore GPS noise
    num_samples = 0
    num_batches = 0
    Nff = 1800 # Frames per file. On our dataset it's constant
    file_ids = []
    current_file_id = -1
    cap = None
    batch_x = None
    Wframe = 1080
    Hframe = 1920
    CHframe = 6
    '''
    Use file or track id to load data sequence
    '''
    def __init__(self, fn_idx, track_id=None, file_id=None):
        self.batch_size = 32
        self.batch_x = np.zeros((self.batch_size,self.Wframe,self.Hframe,self.CHframe))
        self.index_file = fn_idx
        self.db_open(fn_idx)
        # load ground truth data from the db
        if track_id is not None:
            self.preload_track(track_id)
        else:
            self.preload_file(file_id)
        self.num_samples = len(self.file_ids)*self.Nff
        self.num_batches = self.num_samples / self.batch_size

    def __len__(self):
        return self.num_batches

    '''
    Garmin video:
        crop bottom with GPS text
    '''
    def garmin_crop(self, img, target_dim=(640, 480)):
        # target aspect ratio, W/H
        ar = target_dim[0]/target_dim[1]
        # image size for opencv is H x W
        sz = img.shape
        Htxt = 50
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

    '''
    Get next frame in the sequence
    '''
    def get_frame(self,file_idx,idx):
        '''
        Open file if it was not yet opened
        '''
        if current_file_id != file_idx:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            current_file_id = file_idx
            fn = self.file_name(current_file_id)
            self.cap = cv2.VideoCapture(fn)
            self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
            print("Loading '{}' {}x{} {}fps, {} frames".format(
                fn, self. W, self.H, self.FPS, self.frameCount))
        '''
        Read next frame
        '''
        assert idx >= 0 & idx <= self.frameCount, "Illegal frame idx"
        # set to a proper frame
        if self.cap.get(CV_CAP_PROP_POS_FRAMES)!=idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, img = cap.read()
        assert ret, "Broken video '{}'".format(file_idx)
        return garmin_crop(img)

    '''
    return file_idx and frame idx
    '''
    def get_position(self):        
        if self.cap is None: 
            return -1,-1
        return current_file_id, self.cap.get(CV_CAP_PROP_POS_FRAMES)

    def __getitem__(self, batch_idx: int):


    #
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
