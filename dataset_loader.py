import os
import sqlite3
import kinetic_model
import numpy as np

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
        self.kinetic_model = kinetic_model.KineticModel(
            np.array(cursor.fetchall()))
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
