import tensorflow as tf
import tensorflow.keras.utils
import sqlite3, os
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
    Nff = 1800 # Frames per file. On our dataset it's constant
    file_ids = []
    current_file_id = None
    '''
    Use file or track id to load data sequence
    '''
    def __init__(self, fn_idx, track_id=None, file_id=None):
        self.batch_size = 32
        self.index_file = fn_idx
        self.db_open(fn_idx)
        # load ground truth data from the db
        if track_id is not None:
            self.preload_track(track_id)
        else:
            self.preload_file(file_id)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                         self.batch_size]

        return 0

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
        self.num_samples = len(self.file_ids)*self.Nff
        self.buffer_file(self.file_ids[0][0])

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
        self.num_samples = self.Nff*1
        file_ids=[file_id]
        self.buffer_file(file_id)

    def load_speed_labels(self,d):
        t_v=np.array(d)
        T = t_v[:,0]
        t0 = T[0]
        T -= t0
        gps_speed = t_v[:,1]
        self.Vinterpoated = I.splrep(T, gps_speed, s=0)

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

    def speed(self, Tmsec: int):
        # use interpolated gps speed
        v = abs(I.splev(Tmsec, self.Vinterpoated, der=0))
        b = (v > self.Vthreshold).astype(int)
        return v*b

    def file_name(self,file_id: int):
        cursor = self.connection.cursor()
        cursor.execute(
            '''
            SELECT d.path || "/" || f.name
            FROM Files as f, Folders as d
            WHERE f.hex_digest IS NOT NULL AND f.path_id=d.id AND f.id=(?)
            ''', (file_id,)
        )
        return cursor.fetchall()[0]

    def buffer_file(self,file_id: int):
        fn = self.file_name(file_id)
        cap = cv2.VideoCapture(self.file_name)
        frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("Loaded '{}' {}x{} {}fps, {} frames".format(
            fn, W, H, FPS, frameCount))
        # we'll save images in numpy array and let tensorflow do 
        # all the heavy lifting with image transofrmation later
        images = np.zeros((Nff,1080,1920,3))
        for idx in range(Nff):
            ret, images[idx] = cap.read()
            assert ret, "Broken video '{}'".format(file_id)
            #assert img.shape == (1080, 1920, 3), "Unexpected garmin dimensions"
            #self.pos_msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        self.images = tf.convert_to_tensor(images)
        self.current_file_id = file_id
    # dtor
    def __del__(self):
        self.connection.close()
