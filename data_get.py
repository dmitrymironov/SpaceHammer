import tf.keras.utils.Sequence as Sequence
import sqlite3, os
import numpy as np

'''
Create inputs and targets from videos and geo-tagged video sequence
Inputs is a set of sliding windows prepared for convolution as in PoseConvGRU
Targets are speed regression values
'''

class tfFrameGen(Sequence):
    connection = None
    Vthreshold = 0.5  # km/h, ignore GPS noise

    '''
    Use file or track id to load data sequence
    '''
    def __init__(self, fn_idx, track=None, fn=None):
        self.batch_size = 32
        self.index_file = fn_idx
        self.db_open(fn_idx)
        # load ground truth data from the db
        if track is not None:
            self.load_track(track)
        else:
            self.load_file(fn)

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                         self.batch_size]

        return 0

    def load_track(self, track_id):
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

    def load_file(self, file_id):
        cursor = self.connection.cursor()
        cursor.execute(
            '''
            SELECT timestamp/1000,CAST(speed AS real)
            FROM Locations 
            WHERE file_id=(?)
            ORDER BY timestamp
            ''',
            (file_id,)
        )        
        self.load_speed_labels(cursor.fetchall())

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

    def speed(self, Tmsec):
        # use interpolated gps speed
        v = abs(I.splev(Tmsec, self.Vinterpoated, der=0))
        b = (v > self.Vthreshold).astype(int)
        return v*b

    # dtor
    def __del__(self):
        self.connection.close()
