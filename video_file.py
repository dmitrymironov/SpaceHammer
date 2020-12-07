import numpy as np
import sqlite3
import os
import cv2

class DashamDatasetLoader:
    db=os.environ['HOME']+'/.dashcam.software/dashcam.index'
    connection = None
    FPS=0

    def next(self):
        cursor=self.connection.cursor()
        # get file path
        cursor.execute(
            '''
            SELECT f.id,d.path || "/" || f.name 
            FROM Files as f, Folders as d 
            WHERE f.hex_digest IS NOT NULL AND f.path_id=d.id
            LIMIT 1 
            '''
            )
        id, file_name = cursor.fetchall()[0]
        # get lat,lon and speed curve
        cursor.execute(
            '''
            SELECT X(Coordinate),Y(Coordinate),playback_sec,timestamp/1000000
            FROM Locations 
            WHERE file_id=(?)
            ''', 
            (id,)
            )
        return file_name, np.array(cursor.fetchall())

    # ctor
    def __init__(self):
        print("Loading database from '" + self.db + "'")
        assert os.path.isfile(self.db), "Database file is not readable"
        try:
            self.connection = sqlite3.connect(self.db)        
        except sqlite3.Error:
            assert False, sqlite3.Error
        # load spatial extensions (GEOS based wkt etc)
        self.connection.enable_load_extension(True)
        self.connection.cursor().execute("SELECT load_extension('mod_spatialite')")

    # dtor
    def __del__(self):
        self.connection.close()

#
# Use OpenCV to grab frames and build a time sequence
#
class FrameGenerator:
    file_name = ''
    cap = None
    frameCount = 0
    pos_msec = 0

    def __init__(self,fn):
        self.file_name = fn
        self.cap = cv2.VideoCapture(self.file_name)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("Loaded '{}' {}x{} {}fps video".format(fn,self.W,self.H,self.FPS))

    def __del__(self):
        del self.cap

    # read next frame
    def next(self):
        ret, img = self.cap.read()
        if ret and img is not None:
            self.pos_msec = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            return True, img
        return False, None

    def play(self, rate=1./30.):
        while True:
            ret, img = self.next()
            if ret is not True:
                break
            cv2.imshow(self.file_name, img)
            # define q as the exit button
            if cv2.waitKey(int(1000./(rate*self.FPS))) & 0xFF == ord('q'):
                break

def main():
    os.system('clear') # clear the terminal on linux
    #   
    # Get the file and load it's spatial and temporal ground truth
    loader = DashamDatasetLoader()
    file_name, trajectory=loader.next()
    del loader # free memory
    #
    # Use OpenCV to load frame sequence and video temporal charateristics
    framer = FrameGenerator(file_name)
    framer.play()
    del framer

if __name__ == "__main__":
    main()
