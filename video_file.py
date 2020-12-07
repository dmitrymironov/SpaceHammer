import numpy as np
import sqlite3
import os

class DashamDatasetLoader:
    db=os.environ['HOME']+'/.dashcam.software/dashcam.index'
    connection = None
    
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
            SELECT X(Coordinate),Y(Coordinate),playback_sec 
            FROM Locations 
            WHERE file_id=(?)
            ''', 
            (id,)
            )
        rows = cursor.fetchall()
        for row in rows:
            print(row)

    # ctor
    def __init__(self):
        print("Loading database from '" + self.db + "'")
        assert os.path.isfile(self.db), "Database file is not readbale"
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

def main():
    loader = DashamDatasetLoader()
    loader.next()

if __name__ == "__main__":
    main()
