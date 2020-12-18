import numpy as np
import sqlite3
import os
import cv2
import kinetic_model
import dataset_loader
# define class variable 
loader = dataset_loader.DashamDatasetLoader()
import frame_generator

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
    framer = frame_generator.FrameGenerator(file_name)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    # B G R
    red = (0, 0, 255)
    blue = (255, 0, 0)
    thickness = 2
    rate = 1.0

    trainingMode = False

    while True:
        ret, img = framer.next()
        if ret is not True:
            break
        # Y
        V = loader.kinetic_model.speed(framer.pos_msec)
        A = loader.kinetic_model.angle(framer.pos_msec)
        # TODO: Debugging output to validate training input
        if not trainingMode:
            lat, lon = loader.kinetic_model.p(framer.pos_msec)
            txt1 = "{:.5f} {:.5f}".format(lat, lon)
            txt2 = "{:3.2f} km/h   {:3.2f} deg".format(V, A)
            cv2.putText(img, txt1, (670, 1000), font,
                        fontScale, red, thickness, cv2.LINE_AA)
            cv2.putText(img, txt2, (1070, 1030), font,
                        fontScale, blue, thickness, cv2.LINE_AA)
            cv2.imshow(framer.file_name, img)
            # define q as the exit button
            if cv2.waitKey(int(1000./(rate*framer.FPS))) & 0xFF == ord('q'):
                break

    del framer


if __name__ == "__main__":
    main()
