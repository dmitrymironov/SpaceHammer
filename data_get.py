import tensorflow as tf
import tensorflow.keras.utils
import sqlite3, os, cv2, datetime
import numpy as np
from scipy import interpolate as I

class FileRecord:
    id = None # id in the database
    name = None # file path
    pos = None # position in the file list
    framePos = None # current frame
    t0 = 0 # Start time
    Nff = 1800  # Frames per file. On our dataset it's constant
    cap = None # video capture handler

    def nextFramePos(self):
        self.framePos = self.framePos+1

    def init(self, id, name, t0):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.id=id
        self.name = name
        self.t0 = t0
        self.framePos=0

        self.cap = cv2.VideoCapture(self.name)
        frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frameCount >= self.Nff, "Unexpected frame count {} in '{}', should be 0 .. {}".format(
            frameCount, self.name, self.Nff)
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))

    def reset(self):
        self.pos=-1
        self.id=-1
        self.framePos=-1
        self.name=None
        self.t0=-1

'''
Create inputs and targets from videos and geo-tagged video sequence
Inputs is a set of sliding windows prepared for convolution as in PoseConvGRU
Targets are speed regression values
'''

class tfGarminFrameGen(tensorflow.keras.utils.Sequence):
    file = FileRecord()
    name = "frame_generator" # can be train, valid, etc

    connection = None # database connection handler for GT
    cursor = None

    Vthreshold = 0.5  # km/h, GPS trheshold noise
    num_samples = -1 # number of frames in a whole track (or file)
    
    # feeding into the model
    num_batches: int = -1 # number of batches
    batch_size = 15 # N of temporal frame pairs sequences in the batch
    batch_stride = 4 # temporal stride between batches (in sequences)

    file_ids = [] # indexes of files id/Tstart in the dataset
    batch_x = None # batch_x - batches RAM placeholders
    batch_y = None # batch_y 
    #t0 = -1 # beginning of the track, epoch
    train_image_dim = (640, 480)

    Htxt = 50 # Garmin text cropping line height
    Wframe = 1080 # Garmin frame WxH
    Hframe = 1920
    CHframe = 3 # Num channels per X

    # Caching and data check
    fid_name = {} # file id to name dict to avoid sql query
    Tmin = None 
    Tmax = None

    '''
                        INITIALIZE (generator method)
    Use file or track id to load data sequence
    '''
    def __init__(self, fn_idx, name, track_id=None, file_id=None):
        self.name = name
        # db operations
        #---------------------------------------------------------------
        self.index_file = fn_idx
        self.db_open(fn_idx)
        # load ground truth data from the db
        if track_id is not None:
            self.preload_track(track_id)
        else:
            self.preload_file(file_id)
        # cache sql queries to prevent sqlite threading conflict
        #---------------------------------------------------------------
        for fid,_ in self.file_ids:
            self.file_name(fid)
        #
        self.num_samples = len(self.file_ids)*self.file.Nff
        self.num_batches = int(
            (self.num_samples-self.batch_size)/self.batch_stride)
        # initialise dimensions
        #---------------------------------------------------------------
        self.batch_x = np.zeros((self.batch_size, 
            self.train_image_dim[1], self.train_image_dim[0],
            self.CHframe*2), dtype='float16')
        self.batch_y = np.zeros((self.batch_size), dtype='float16')

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
        return cv2.resize(img[0:sz[0]-dw[0], 0:sz[1]-dw[1]],target_dim)

    '''
    position to a particular file
    '''
    def select_file(self, file_id: int):
        # if we're already positioned in the file, return
        if file_id==self.file.id:
            return
        self.file.init(file_id,self.file_name(file_id), self.file_ids[self.file.pos][1])

        print("Loading {} '{}' {}x{} {}fps, {} frames".format(self.name,
            self.file.name, self.file.W, self.file.H, self.file.FPS, 
            self.file.Nff))

    '''
    Get next frame in the sequence
    '''
    def get_frame(self, frame_idx):

        '''
        Read next frame
        '''
        assert frame_idx >= 0 & frame_idx <= self.file.Nff, "Illegal frame idx"
        # set to a proper frame
        actual_pos = self.file.cap.get(cv2.CAP_PROP_POS_FRAMES)
        if actual_pos != frame_idx:
            self.file.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = self.file.cap.read()
        assert ret, "Broken video '{}'".format(self.fn)
        return self.garmin_crop(img,self.train_image_dim)

    def move_on(self):
        if self.file.framePos+1 >= self.file.Nff:
            # trigger switching to a next file
            self.file.pos = self.file.pos+1
            if self.file.pos>=len(self.file_ids):
                # it's a last file, we need to stop
                self.file.reset()
                return False
            else:
                self.select_file(self.file_ids[self.file.pos][0][0])
        else:
            self.file.nextFramePos()
        # msec of a current frame
        self.Tlocaltime = self.file.t0 + \
            int(self.file.cap.get(cv2.CAP_PROP_POS_MSEC))
        return True
    
    '''
    Get batch_x and batch_y for training (generator method)
    '''
    def __getitem__(self, batch_idx: int):
        #print("Getting {} batch {}".format(self.name,batch_idx))
        assert batch_idx < self.num_batches, "incorrect batch number"
        frame1 = None
        frame2 = None
        # first frame number
        self.file.pos = int(batch_idx*self.batch_stride/self.file.Nff)
        self.select_file(self.file_ids[self.file.pos][0])
        # position in a current video file
        self.file.framePos = int(batch_idx*self.batch_stride) % self.file.Nff
        fEndReached=False
        #test1=None
        for batch_pos in range(self.batch_size):
            if frame2 is None:
                frame1 = self.get_frame(self.file.framePos)
            else:
                frame1 = frame2
            if self.move_on(): 
                frame2 = self.get_frame(self.file.framePos)
                self.batch_x[batch_pos] = tf.concat([frame1, frame2], axis=2)
                self.batch_y[batch_pos] = self.speed(self.Tlocaltime)
            else:
                # We've reached the end, just repeating the last frame
                assert not fEndReached, "We should not stack more than one end frame"
                self.batch_x[batch_pos] = tf.concat([frame1, frame1], axis=2)
                self.batch_y[batch_pos] = self.speed(self.Tlocaltime)
                fEndReached=True
        return self.batch_x, self.batch_y

    '''
    Add text to opencv image for debug
    '''
    def put_text(self,img,txt,clr=(0,0,255),pos=(10,50),thick=2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        cv2.putText(img, txt, pos, font, fontScale, clr, thick, cv2.LINE_AA)
        return img

    '''
    Database operations to provide ground truth
    '''
    def preload_track(self, track_id: int):
        self.cursor.execute(
            '''
            SELECT timestamp/1000,CAST(speed AS real)
            FROM Locations 
            WHERE track_id=(?)
            ORDER BY timestamp
            ''',
            (track_id,)
        )
        self.load_speed_labels(self.cursor.fetchall())
        self.cursor.execute(
            '''
            SELECT DISTINCT(file_id),MIN(timestamp)/1000 FROM Locations WHERE track_id=(?)
            GROUP BY file_id
            HAVING COUNT(file_id)>=60
            ORDER BY timestamp
            ''', (track_id,)
        )
        self.file_ids = self.cursor.fetchall()

    #
    def preload_file(self, file_id):
        self.cursor.execute(
            '''
            SELECT timestamp/1000,CAST(speed AS real)
            FROM Locations 
            WHERE file_id=(?)
            ORDER BY timestamp
            ''',
            (file_id,)
        )        
        self.load_speed_labels(self.cursor.fetchall())
        self.cursor.execute(
            '''
            SELECT COUNT(*),MIN(timestamp)/1000 FROM Locations WHERE file_id=(?)
            ''', (file_id,)
        )
        self.file_ids=self.cursor.fetchall()

    #
    def load_speed_labels(self,d):
        t_v=np.array(d)
        T = t_v[:,0]
        self.Tmin = np.min(T)
        self.Tmax = np.max(T)
        gps_speed = t_v[:,1]
        self.Vinterpoated = I.splrep(T, gps_speed, s=0)
        # average pooling
        Npool=1000
        N = gps_speed.shape[0]
        if Npool < N:
            stride=int(np.gcd(N,Npool))
        else:
            stride=1
        Vdown=np.mean(gps_speed.reshape(-1,stride),axis=1)
        Tdown = np.linspace(np.min(T), np.max(T), Vdown.shape[0])

        import matplotlib.pyplot as plt
        plt.plot(T,gps_speed,'o')
        plt.plot(Tdown, Vdown,'-x')
        plt.show()
        pass

    #
    def speed(self, Tmsec: int):
        # use interpolated gps speed
        assert Tmsec>=self.Tmin and Tmsec<=self.Tmax, "Wrong time point"
        v = abs(I.splev(Tmsec, self.Vinterpoated, der=0))
        assert not np.isnan(v), "Incorrect value interpolation"
        b = (v > self.Vthreshold).astype(int)
        return v*b

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
        self.cursor = self.connection.cursor()
        self.cursor.execute("SELECT load_extension('mod_spatialite')")

    '''
    Keras applies multi-threaded training so this method may potentially 
    be invoked from various threads.
    '''
    def file_name(self,file_id: int):
        if file_id in self.fid_name:
            return self.fid_name[file_id];
        self.cursor.execute(
            '''
            SELECT d.path || "/" || f.name
            FROM Files as f, Folders as d
            WHERE f.hex_digest IS NOT NULL AND f.path_id=d.id AND f.id={}
            '''.format(file_id)
            );
        self.fid_name[file_id] = os.path.normpath(
            self.cursor.fetchall()[0][0]
            );
        return self.fid_name[file_id]

    def get_file_id_by_pattern(self,pat):
        self.cursor.execute(
            '''
            SELECT f.id,  d.path || "/" || f.name as p FROM Files as f, Folders as d 
            WHERE d.path || "/" || f.name LIKE '{}'
            '''.format(pat)
        )
        return self.cursor.fetchall()[0]

    # dtor
    def __del__(self):
        self.connection.close()
