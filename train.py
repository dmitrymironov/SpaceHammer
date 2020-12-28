#import training_set_generator
import os
import data_get
import tensorflow as tf
import platform
import numpy as np

def main():
    os.system('clear')  # clear the terminal on linux
    print("Using Tensorflow {}".format(tf.__version__))
    # train1 = training_set_generator.TrainingSetGenerator()
    if platform.system() == "Windows":
        db = r'C:\\msys64\\home\\dmmie\\.dashcam.software\\dashcam.index'
    else:
        db = os.environ['HOME']+'/.dashcam.software/dashcam.index'
    ggen = data_get.tfGarminFrameGen(db,1)
    # traverse generator for debug reasons
    for batch_idx in range(ggen.__len__()):
        x, y = ggen.__getitem__(batch_idx)

    print("Done!")

if __name__ == "__main__":
    main()
