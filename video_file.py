import numpy as np
import sqlite3
import os
import cv2

#
# Kinetic model of a vehicle handling oversampling
#

class KineticModel:
    coords = None
    R = 6372800  # Earth radius

    #
    # Geodesial distance
    #
    def haversine(self, lat1, lon1, lat2, lon2):

        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        a = np.sin(dphi/2)**2 + \
            np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2

        return 2*self.R*np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
    # https://www.movable-type.co.uk/scripts/latlong.html
    def short_distance(self, lat1, lon1, lat2, lon2):
        # flat earthers
        t1=(np.pi/2)-np.radians(lat1)
        t2=(np.pi/2)-np.radians(lat2)
        #return (self.R * np.sqrt(t1*t1+t2*t2-2*t1*t2*np.cos(np.radians(lon2-lon1)))).astype(int)
        return self.R * np.sqrt(t1*t1+t2*t2-2*t1*t2*np.cos(np.radians(lon2-lon1)))

    def __init__(self, trajectory):
        lat = trajectory[:, 0]
        lon = trajectory[:, 1]
        lat1 = np.append(lat, 0.0)
        lon1 = np.append(lon, 0.0)
        lat2 = np.insert(lat, 0, 0.0)  # all but the last element
        lon2 = np.insert(lon, 0, 0.0)

        self.dist = self.short_distance(lat1, lon1, lat2, lon2)
        self.dist[0] = 0 # we can not approximate vector in the first point

        '''
        # Cross-refrerencing GPS speed vs recorded
        approxSpeed = self.short_distance(lat1, lon1, lat2, lon2)*3600./1000.
        approxSpeed[0] = -1
        gps_speed = trajectory[:, 3]
        for i in range(len(lat)):
            print("{:.6f}\t{:.6f}\t{:.2f}\t{:.2f}".format(lat[i],lon[i],approxSpeed[i],gps_speed[i]))
        '''

        self.T = trajectory[:, 2]
        t0 = self.T[0]
        self.T -= t0
        self.T *= 1000
        #print(self.T)

#
# DashamDatasetLoader
#
class DashamDatasetLoader:
    db = os.environ['HOME']+'/.dashcam.software/dashcam.index'
    connection = None
    FPS = 0
    kinetic_model = None

    def next(self):
        cursor = self.connection.cursor()
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
            SELECT X(Coordinate),Y(Coordinate),timestamp/1000000,CAST(speed AS real)
            FROM Locations 
            WHERE file_id=(?)
            ''',
            (id,)
        )
        self.kinetic_model = KineticModel(np.array(cursor.fetchall()))
        # return file_name, np.array(cursor.fetchall())
        return file_name

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

    def __init__(self, fn):
        self.file_name = fn
        self.cap = cv2.VideoCapture(self.file_name)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        print("Loaded '{}' {}x{} {}fps video".format(
            fn, self.W, self.H, self.FPS))

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
    os.system('clear')  # clear the terminal on linux

    #
    # Get the file and load it's spatial and temporal ground truth
    loader = DashamDatasetLoader()
    file_name  = loader.next()
    #return 0
    #
    # Use OpenCV to load frame sequence and video temporal charateristics
    framer = FrameGenerator(file_name)
    framer.play(1.0)
    del framer


if __name__ == "__main__":
    main()
