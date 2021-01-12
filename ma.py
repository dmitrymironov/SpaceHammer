import sqlite3
import os
import platform
import cv2
import datetime
import numpy as np
from scipy import interpolate as I
from rdp import rdp
from pykalman import KalmanFilter
import matplotlib.pyplot as plt

'''
Create inputs and targets from videos and geo-tagged video sequence
Inputs is a set of sliding windows prepared for convolution as in PoseConvGRU
Targets are speed regression values
'''


class tfTemporalCompressor:
    name = "frame_generator"  # can be train, valid, etc

    connection = None  # database connection handler for GT
    cursor = None

    num_samples = -1  # number of frames in a whole track (or file)

    # feeding into the model
    num_batches: int = -1  # number of batches
    batch_size = 15  # N of temporal frame pairs sequences in the batch
    batch_stride = 4  # temporal stride between batches (in sequences)

    batch_x = None  # batch_x - batches RAM placeholders
    batch_y = None  # batch_y

    '''
                        INITIALIZE (generator method)
    '''

    def __init__(self, fn_idx, name):
        self.name = name
        # db operations
        #---------------------------------------------------------------
        self.index_file = fn_idx
        self.db_open(fn_idx)

    def get_tracks(self):
        self.cursor.execute(
            '''
            SELECT track_id,COUNT(*) as N FROM Locations
            GROUP BY track_id
            ORDER BY N DESC
            '''
        )
        return self.cursor.fetchall()

    def get_track(self, id):
        self.cursor.execute(
            '''
            SELECT timestamp/1000,X(coordinate),Y(coordinate),CAST(speed AS decimal)
            FROM Locations
            WHERE track_id={}
            ORDER BY timestamp
            '''.format(id)
        )
        data = np.asarray(self.cursor.fetchall())
        t = data[:, 0]
        t0 = t[0]
        return (t-t0)/1000, data[:, 1:]
    #

    def db_open(self, index_file):
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

    # dtor
    def __del__(self):
        self.connection.close()

# https://www.kaggle.com/aussie84/train-fare-trends-using-kalman-filter-1d
def Kalman1D(observations):
    # To return the smoothed time series data
    observation_covariance = 5
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.01
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    #kf = kf.em(observations, n_iter=5)
    '''
    kf = kf.em(observations, em_vars=[
               'initial_state_mean'])
    '''
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def main():
    os.system('clear')  # clear the terminal on linux
    if platform.system() == "Windows":
        db = r'C:\\msys64\\home\\dmmie\\.dashcam.software\\dashcam.index'
    else:
        db = os.environ['HOME']+'/.dashcam.software/dashcam.index'
    db = os.path.normpath(db)

    print('================================================== TRAIN')

    rdp_tv = 'after_rdp_tv.npy'

    if os.path.exists(rdp_tv):
        measurements = np.load(rdp_tv)
        T=np.load('T.npy')
        V=np.load('V.npy')
    else:
        comp = tfTemporalCompressor(db, "comp")
        track_id = 6

        '''prepare generators'''
        T, y = comp.get_track(track_id)
        V = y[:, 2]
        np.save('T',T)
        np.save('V',V)

        plt.plot(T, V)
        '''
        for factor in [10,100,200,500]:
            rm = running_mean(V, factor)
            plt.plot(T[:rm.shape[0]],rm)]
        '''
        # douglas pecker track simplifier
        inp = np.stack((T,V)).transpose()
        measurements = rdp(inp, epsilon=0.5)
        x = measurements[:, 0]
        y = measurements[:, 1]
        plt.plot(x,y)
        np.save(rdp_tv, measurements)
        plt.show()

    print('Running kalman filter')
    plt.plot(T, V, '-')
    Vk=V.copy()
    for w in find_handsaw(T, Vk,3):
        t = T[w[0]:w[1]]
        v = Kalman1D(Vk[w[0]:w[1]])
        Vk[w[0]:w[1]] = v.reshape(len(v))
    plt.plot(T, Vk, 'r-')

    inp = np.stack((T, Vk)).transpose()
    measurements = rdp(inp, epsilon=0.5)
    x = measurements[:, 0]
    y = measurements[:, 1]
    print('Input samples: {}, output samples: {}'.format(
        T.shape[0], x.shape[0]))
    plt.plot(x, y, 'x-')

    #plt.plot(measurements[:, 0], measurements[:, 1], 'x')
    #plt.plot(filtered_state_means[:,0], filtered_state_means[:,1],'g-')
    #plt.plot(smoothed_state_means[:, 0], smoothed_state_means[:, 1], 'r--')
    #plt.plot(T, Kalman1D(V), 'r--')

    plt.show()

def find_handsaw(x,y,steps=3):
    N=len(x)
    i=0
    windows=[]
    minN=4
    while i < N:
        j=i
        stopFlag=False
        vals=[y[i]]
        vmin = y[i]
        vmax = vmin
        while True:
            j=j+1
            if j>=N:
                stopFlag=True
                break
            val = y[j]
            if val>vmax:
                vmax=val
            elif val<vmin:
                vmin=val
            if not val in vals:
                if len(vals)>=steps or vmax-vmin>=steps:
                    stopFlag=True
                    break
                vals.append(val)
        if stopFlag:
            if j-i > minN:
                windows.append((i,j))
        i=j
    return windows

if __name__ == "__main__":
    main()
