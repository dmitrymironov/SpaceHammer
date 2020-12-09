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

    '''
    # geodesical sphere precision

    # 
    # Earth-center touching radian angle of the triangle (c1,c2,Earth Center)
    #
    
    def ec_angle(self, c1, c2):
        lat1, lon1 = c1
        lat2, lon2 = c2
        sin_lat = 2*self.R*np.sin(np.radians(lat2-lat1)/2)
        sin_lon = 2*self.R*np.sin(np.radians(lon2-lon1)/2)
        s3 = np.sqrt(sin_lat*sin_lat+sin_lon*sin_lon)
        arg = np.clip(s3/(2.*self.R),-1,1)
        return 2*np.arcsin(arg)
    
    #
    # bearing (degrees, internal angle, no sign)
    #

    def bearing(self,C,A,B):      

        # https://en.wikipedia.org/wiki/Solution_of_triangles#Solving_spherical_triangles
        a  = self.ec_angle(B, C)
        b  = self.ec_angle(C, A)
        c  = self.ec_angle(A, B)

        arg = np.clip((np.cos(a)-np.cos(b)*np.cos(c))/(1e-50+np.sin(b)*np.sin(c)),-1,1)

        # alpha
        return np.rad2deg(np.pi-np.arccos(arg))
    '''

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
        # print(self.T)

        # a,b,c coefs for x and y.
        # Polynomial degree 10 is a severe overfitting but works well
        self.interpolation = {}
        self.interpolation['lat'] = P.fit(self.T, lat, 10)
        self.interpolation['lon'] = P.fit(self.T, lon, 10)

        '''
        import matplotlib.pyplot as plt
        ml = np.min(lon)
        lon-=ml
        plt.plot(self.T,lon,'o')
        xvals = np.linspace(0,59000,100)
        yvals = self.interpolation['y'](xvals) - ml
        plt.plot(xvals,yvals,'-x')
        plt.show()
        '''

        # TODO: lots of array duplication, q&d for debugging
        lat1 = np.append(lat, fakeZero)
        lon1 = np.append(lon, fakeZero)
        lat2 = np.insert(lat, 0, fakeZero)  # all but the last element
        lon2 = np.insert(lon, 0, fakeZero)

        # calculate next vector distance
        self.dist = self.haversine(lat1, lon1, lat2, lon2)
        self.dist[0] = 0  # we can not approximate vector in the first point

        # calculate next vector bearing, generate 3 vectors with shift 1
        lat1 = np.append(lat, [fakeZero, fakeZero])  # 2 zeros in the end
        lon1 = np.append(lon, [fakeZero, fakeZero])
        # angle point
        lat2 = np.insert(lat, 0, fakeZero)  # zero in the beginning and end
        lon2 = np.insert(lon, 0, fakeZero)
        lat2 = np.append(lat2, fakeZero)  # 2 zeros in the end
        lon2 = np.append(lon2, fakeZero)
        # closing point
        lat3 = np.insert(lat, 0, fakeZero)
        lat3 = np.insert(lat3, 0, fakeZero)
        lon3 = np.insert(lon, 0, fakeZero)
        lon3 = np.insert(lon3, 0, fakeZero)

        '''
        # Sign-less precision way to do it
        self.bearing = self.bearing((lat1,lon1),(lat2,lon2),(lat3,lon3))
        '''

        # shape tweak
        self.bearing[1] = 0
        self.bearing = self.bearing[:-1]

        # Speed estimation
        approxSpeed = self.haversine(lat1, lon1, lat2, lon2)*3600./1000.
        approxSpeed[0] = 0
        approxSpeed = approxSpeed[:-1]
        gps_speed = trajectory[:, 3]

        # apply minimum speed threshold
        # Angle fluctuations are terrible on low speed due to GPS noise
        minSpeedThreshold = 10.0
        self.bearing = (approxSpeed >= minSpeedThreshold).astype(
            int)*self.bearing

        assert self.bearing.shape == self.dist.shape
        '''
        # Cross-refrerencing GPS speed vs recorded
        print("lat      \tlon      \tdist\tspeed\tangle")
        for i in range(len(lat)):
            print("{:.6f}\t{:.6f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
                lat[i], lon[i], self.dist[i], gps_speed[i], self.bearing[i]))
        '''

    def bearing(self,t1,t2,t3):
        lat1,lon1=self.p(t1)
        lat2,lon2=self.p(t2)
        lat3,lon3=self.p(t3)
        # Flat earth cheating
        # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
        x1 = lat2-lat1
        y1 = lon2-lon1
        x2 = lat3-lat2
        y2 = lon3-lon2
        dot = x1*x2+y1*y2
        det = x1*y2-y1*x2
        return np.rad2deg(np.arctan2(det, dot))

    def speed(self, Tmsec):
        dlat = self.interpolation['lat'].deriv(Tmsec)
        dlon = self.interpolation['lon'].deriv(Tmsec)
        return np.sqrt(dlat**2+dlon**2)

    # distance between two moments (msec)
    def dist(self, t1, t2):
        lat1, lon1 = self.p(t1)
        lat2, lon2 = self.p(t2)
        return self.haversine(lat1, lon1, lat2, lon2)

    # interpolated smooth coordinate at time point
    def p(self, Tmsec):
        return self.interpolation['lat'](Tmsec), self.interpolation['lon'](Tmsec)

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
            AND f.name LIKE "%6038%" AND d.path NOT LIKE '%Monument%'
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
