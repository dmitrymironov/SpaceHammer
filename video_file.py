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
        self.interpolation['lat'] = P.fit(self.T, lat, 20)
        self.interpolation['lon'] = P.fit(self.T, lon, 20)
        self.interpolation['dlat'] = FEC*self.interpolation['lat'].deriv()
        self.interpolation['dlon'] = FEC*self.interpolation['lon'].deriv()

        # Sample dy/dx aka trajectory rotation angle with descrete step
        # significant enough to exceed averge GPS noise. 
        # Considering it's done on a high-degree smoothing polinomial 
        # that has derivative in any point we likely don't care that much
        # still averaging on a bigger vector gives less angles fluctuation
        step = 100 # 1/10th of a second
        Twindow=10
        Nsteps = int((self.T[-1].astype(int)+np.max([Twindow,step])-self.T[0].astype(int))/step)
        angle = np.zeros(Nsteps)
        T_angle_sampling = np.zeros(Nsteps)
        for idx in range(Nsteps):
            T=self.T[0].astype(int)+step*idx
            T_angle_sampling[idx]=T
            # (self.speed(T)>5).astype(int)*
            angle[idx] = (self.speed(T) > 5).astype(int) * \
                self.bearing(T-Twindow, T, T+Twindow)
        self.interpolation['angle']=P.fit(T_angle_sampling,angle,30)

        # Angle plot
        if False:
            import matplotlib.pyplot as plt
            plt.plot(T_angle_sampling, angle, 'o')
            xvals = np.linspace(0, 58000, 100)
            yvals = self.interpolation['angle'](xvals)
            plt.plot(xvals, yvals, '-x')
            plt.show()

        # Lat/Lon
        if False:
            import matplotlib.pyplot as plt
            mlon = np.min(lon)
            lon -= mlon
            plt.plot(self.T,lon,'o')
            xvals = np.linspace(0,58000,100)
            yvals = self.interpolation['lon'](xvals) - mlon
            plt.plot(xvals,yvals,'-x')
            plt.show()
            mlat = np.min(lat)
            lat -= mlat
            plt.plot(self.T, lat, 'o')
            xvals = np.linspace(0, 58000, 100)
            yvals = self.interpolation['lat'](xvals) - mlat
            plt.plot(xvals, yvals, '-x')
            plt.show()


        # XML
        if True:
            f = open('/home/dmi/Desktop/dbg.kml','w')
            f.write('''<?xml version="1.0" encoding="UTF-8"?>
                    <kml xmlns="http://www.opengis.net/kml/2.2">
                    <Document>
                    <Placemark>
                    <name>/mnt/video/Folders/31-July-2020/DCIM/105UNSVD/GRMN0099.MP4</name>
                    <Description></Description>
                    <Style id="yellowLineGreenPoly">
                    <LineStyle>
                    <color>ff0000ff</color>
                    <width>10</width>
                    </LineStyle>
                    <PolyStyle>
                    <color> 00ff0000 </color>
                    </PolyStyle>
                    </Style>
                    <LineString>
                    <extrude>1</extrude>
                    <tessellate>1</tessellate>
                    <altitudeMode>clampToGround</altitudeMode>
                    <coordinates>''')

            step=100
            for T in range(self.T[0].astype(int), step+self.T[-1].astype(int), step):
                Plat, Plon = self.p(T)
                f.write(Plat,Plon,0.0)
                
            f.write('''</coordinates>
                    </LineString>
                    </Placemark>
                    </Document>
                    </kml>
                    ''')
            f.close()

        # MSE print((np.square(gps_speed - self.speed(self.T))).mean())
        #'''
        print("lat      \tlon      \tdist\tspeed\tVgps\tangle\tT")
        step=1000
        for T in range(self.T[0].astype(int), step+self.T[-1].astype(int), step):
            Plat, Plon = self.p(T)
            idx = int(T / step)
            d = self.dist(T, T+step)
            print("{:.6f}\t{:.6f}\t{:.2f}\t{:.2f}\t{:.2f}\t{: .2f}\t{}".format(
                Plat, Plon, d, self.speed(T), gps_speed[idx], self.angle(T),T
                ))
        #'''

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
        return self.interpolation['lat'](Tmsec), \
            self.interpolation['lon'](Tmsec)

    def angle(self,Tmsec):
        return self.interpolation['angle'](Tmsec)

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

# define class variable 
loader = DashamDatasetLoader()

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
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        # B G R 
        red = (0,0,255)
        blue = (255, 0, 0)
        thickness = 2

        while True:
            ret, img = self.next()
            if ret is not True:
                break
            lat,lon=loader.kinetic_model.p(self.pos_msec)
            V = loader.kinetic_model.speed(self.pos_msec)
            A = loader.kinetic_model.angle(self.pos_msec)
            txt1 = "{:.6f} {:.6f}".format(lat, lon)
            txt2 = "{:3.2f} km/h   {:3.2f} deg".format(V, A)
            cv2.putText(img, txt1, (670, 1000), font,
                        fontScale, red, thickness, cv2.LINE_AA)
            cv2.putText(img, txt2, (1070, 1030), font,
                        fontScale, blue, thickness, cv2.LINE_AA)
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
    file_name = loader.next()

    #
    # Use OpenCV to load frame sequence and video temporal charateristics
    framer = FrameGenerator(file_name)
    framer.play(1.0)
    del framer


if __name__ == "__main__":
    main()
