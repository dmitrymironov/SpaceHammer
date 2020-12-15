from numpy.polynomial import Polynomial as P
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

        # Sample dy/dx aka trajectory rotation angle with descrete step
        # significant enough to exceed averge GPS noise.
        # Considering it's done on a high-degree smoothing polinomial
        # that has derivative in any point we likely don't care that much
        # still averaging on a bigger vector gives less angles fluctuation
        step = 100  # 1/10th of a second
        Twindow = 200
        Nsteps = int(
            (self.T[-1].astype(int)+np.max([Twindow, step])-self.T[0].astype(int))/step)
        angle = np.zeros(Nsteps)
        T_angle_sampling = np.zeros(Nsteps)
        for idx in range(Nsteps):
            T = self.T[0].astype(int)+step*idx
            T_angle_sampling[idx] = T
            angle[idx] = self.bearing(T-Twindow, T, T+Twindow)
        self.interpolation['angle'] = P.fit(T_angle_sampling, angle, 10)

        # Lat/Lon show
        if True:
            import matplotlib.pyplot as plt
            xvals = np.linspace(0, 58000, 100)
            fig, axs = plt.subplots(4)
            fig.suptitle('Booblik')
            mlon = np.min(lon)
            Jlon = lon - mlon
            axs[0].plot(self.T, Jlon, 'o')
            yvals = self.interpolation['lon'](xvals) - mlon
            axs[0].plot(xvals, yvals, '-x')
            mlat = np.min(lat)
            Jlat = lat - mlat
            axs[1].plot(self.T, Jlat, 'o')
            yvals = self.interpolation['lat'](xvals) - mlat
            axs[1].plot(xvals, yvals, '-x')
            # 2D path
            axs[2].plot(lat, lon, '-o')
            x, y = self.p(xvals)
            axs[2].plot(x, y, '-x')
            # Angle plot
            plt.plot(T_angle_sampling, angle, 'o')
            yvals = self.interpolation['angle'](xvals)
            axs[3].plot(xvals, yvals, '-x')
            # showtime
            plt.show()

        # XML
        if True:
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
            print("{:.6f}\t{:.6f}\t{:.2f}\t{:.2f}\t{:.2f}\t{: .2f}\t{}".format(
                Plat, Plon, d, self.speed(T), gps_speed[idx], self.angle(T), T
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

    def angle(self, Tmsec):
        return self.interpolation['angle'](Tmsec)
