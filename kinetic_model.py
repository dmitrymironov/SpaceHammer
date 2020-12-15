#from numpy.polynomial import Polynomial as P
from scipy import interpolate as I
import numpy as np

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
    # Quick gradient approximation constant
    FEC = 285799889
    gps_speed = None
    Vthreshold = 2  # km/h, ignore GPS noise

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

        self.interpolation = {}

        # Cross-refrerencing GPS speed vs recorded
        self.gps_speed = trajectory[:, 3]
        self.interpolation['speed'] = I.splrep(self.T, self.gps_speed, s=0)

        self.interpolation['lat'] = I.splrep(self.T, lat, s=0)
        self.interpolation['lon'] = I.splrep(self.T, lon, s=0)

        # Sample dy/dx aka trajectory rotation angle with descrete step
        # significant enough to exceed averge GPS noise.
        # Considering it's done on a high-degree smoothing polinomial
        # that has derivative in any point we likely don't care that much
        # still averaging on a bigger vector gives less angles fluctuation
        step = 100  # 1/10th of a second
        Twindow = 500
        Nsteps = int(
            (self.T[-1].astype(int)+np.max([Twindow, step])-self.T[0].astype(int))/step)
        angle = np.zeros(Nsteps)
        T_angle_sampling = np.zeros(Nsteps)
        for idx in range(Nsteps):
            T = self.T[0].astype(int)+step*idx
            T_angle_sampling[idx] = T
            angle[idx] = int(self.speed(T)>self.Vthreshold) * \
                self.bearing(T-Twindow, T, T+Twindow)
        self.interpolation['angle'] = I.splrep(T_angle_sampling, angle, s=0)

        # Lat/Lon show
        if True:
            import matplotlib.pyplot as plt
            xvals = np.linspace(0, 58000, 100)
            fig, axs = plt.subplots(5)
            fig.suptitle('Booblik')
            mlon = np.min(lon)
            Jlon = lon - mlon
            axs[0].plot(self.T, Jlon, 'o')
            yvals = I.splev(xvals,self.interpolation['lon'],der=0) - mlon
            axs[0].plot(xvals, yvals, '-x')
            mlat = np.min(lat)
            Jlat = lat - mlat
            axs[1].plot(self.T, Jlat, 'o')
            yvals = I.splev(xvals, self.interpolation['lat'], der=0) - mlat
            axs[1].plot(xvals, yvals, '-x')
            # gps speed and interpolated speed
            axs[2].plot(self.T, self.gps_speed, 'o')
            axs[2].plot(self.T, self.speed(self.T), '-x')
            # 2D path
            axs[3].plot(lat, lon, '-o')
            x, y = self.p(xvals)
            axs[3].plot(x, y, '-x')
            # Angle plot
            plt.plot(T_angle_sampling, angle, 'o')
            yvals = I.splev(xvals,self.interpolation['angle'],der=0)
            axs[4].plot(xvals, yvals, '-x')
            # showtime
            plt.show()

        # XML
        if False:
            f = open('/home/dmi/Desktop/dbg.kml', 'w')
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

            step = 100
            for T in range(self.T[0].astype(int), step+self.T[-1].astype(int), step):
                Plat, Plon = self.p(T)
                f.write("{},{},{}\n".format(Plon, Plat, 0.0))

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
        step = 1000
        for T in range(self.T[0].astype(int), step+self.T[-1].astype(int), step):
            Plat, Plon = self.p(T)
            idx = int(T / step)
            d = self.dist(T, T+step)
            print("{:.5f}\t{:.5f}\t{:.2f}\t{:.2f}\t{:.2f}\t{: .2f}\t{}".format(
                Plat, Plon, d, self.speed(T), self.gps_speed[idx], self.angle(T), T
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
        # use interpolated gps speed
        v=abs(I.splev(Tmsec, self.interpolation['speed'], der=0))
        b=(v>self.Vthreshold).astype(int)
        return v*b

        '''
        Looks like gps speed is calculated using accelerometer

        dlat = self.FEC * I.splev(Tmsec, self.interpolation['lat'], der=1)
        dlon = self.FEC * I.splev(Tmsec, self.interpolation['lon'], der=1)
        return np.sqrt(dlat**2+dlon**2)
        '''

    # distance between two moments (msec)
    def dist(self, t1, t2):
        lat1, lon1 = self.p(t1)
        lat2, lon2 = self.p(t2)
        return self.haversine(lat1, lon1, lat2, lon2)

    # interpolated smooth coordinate at time point
    def p(self, Tmsec):
        return \
            I.splev(Tmsec, self.interpolation['lat'], der=0), \
            I.splev(Tmsec,self.interpolation['lon'],der=0)

    def angle(self, Tmsec):
        b = (self.speed(Tmsec) > self.Vthreshold).astype(int)
        return b*I.splev(Tmsec, self.interpolation['angle'], der=0)
