import numpy as np
import sqlite3
import os
import cv2
from numpy.polynomial import Polynomial as P

#
# Kinetic model of a vehicle handling oversampling
# TODO: Going full precision geodesical calculation is suboptimal for
# TODO: sub-50 meters distances, need fast approximation model
#


def sigmoid(z, b=0.):
    return 1/(1 + np.exp(b-z))


class KineticModel:
    R = 6372800  # Earth radius
    # interpolated polynomial coefficients
    interpolation = []

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

    def __init__(self, trajectory):
        lat = trajectory[:, 0]
        lon = trajectory[:, 1]
        fakeZero = 1e-20  # stability epsilon

        #
        # Construct temporal interpolation
        #

        self.T = trajectory[:, 2]
        t0 = self.T[0]
        self.T -= t0
        self.T *= 1000

        # Cross-refrerencing GPS speed vs recorded
        gps_speed = trajectory[:, 3]

        # Polynomial degree 10 is a severe overfitting but works well
        FEC = 285799889
        self.interpolation = {}
        self.interpolation['lat'] = P.fit(self.T, lat, 10)
        self.interpolation['lon'] = P.fit(self.T, lon, 10)
        self.interpolation['dlat'] = FEC*self.interpolation['lat'].deriv()
        self.interpolation['dlon'] = FEC*self.interpolation['lon'].deriv()

        '''
        import matplotlib.pyplot as plt
        ml = np.min(lon)
        lon-=ml
        plt.plot(self.T,lon,'o')
        xvals = np.linspace(0,59000,100)
        yvals = self.interpolation['lon'](xvals) - ml
        plt.plot(xvals,yvals,'-x')
        plt.show()
        '''

        '''
        # MSE print((np.square(gps_speed - self.speed(self.T))).mean())
        print("lat      \tlon      \tdist\tspeed\tangle")
        for T in range(self.T[0].astype(int), 1000+self.T[-1].astype(int), 1000):
            Plat, Plon = self.p(T)
            idx = int(T / 1000)
            d = self.dist(T, T+1000)
            print("{:.6f}\t{:.6f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
                Plat, Plon, d, self.speed(T)-self.dist(T,T-1000)*18./5., self.bearing(T-500, T, T+200)))
        '''

    def bearing(self, t1, t2, t3):
        lat1, lon1 = self.p(t1)
        lat2, lon2 = self.p(t2)
        lat3, lon3 = self.p(t3)
        x1 = lat2-lat1
        y1 = lon2-lon1
        x2 = lat3-lat2
        y2 = lon3-lon2
        dot = x1*x2+y1*y2
        det = x1*y2-y1*x2
        return np.rad2deg(np.arctan2(det, dot))

    def speed(self, Tmsec):
        dlat = self.interpolation['dlat'](Tmsec)
        dlon = self.interpolation['dlon'](Tmsec)
        return np.sqrt(dlat**2+dlon**2)

    # distance between two moments (msec)
    def dist(self, t1, t2):
        lat1, lon1 = self.p(t1)
        lat2, lon2 = self.p(t2)
        return self.haversine(lat1, lon1, lat2, lon2)

    # interpolated smooth coordinate at time point
    def p(self, Tmsec):
        return self.interpolation['lat'](Tmsec), self.interpolation['lon'](Tmsec), self.speed(Tmsec)

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
        # AND f.name LIKE "%6038%" AND d.path NOT LIKE '%Monument%'
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

    '''
    # quick test case
    data = np.zeros((3,4))
    data[0, :] = [45.375624, -122.761118, 1000, 30]
    data[1, :] = [45.375579, -122.767429, 1000, 20]
    data[2, :] = [45.375679, -122.767419, 1000, 10]
    m = KineticModel(data)
    return 0
    '''
    #
    # Get the file and load it's spatial and temporal ground truth
    loader = DashamDatasetLoader()
    file_name = loader.next()
    return 0
    #
    # Use OpenCV to load frame sequence and video temporal charateristics
    framer = FrameGenerator(file_name)
    framer.play(1.0)
    del framer


if __name__ == "__main__":
    main()
