#import training_set_generator
import os
import data_get
import tensorflow as tf
import platform
import numpy as np
from keras.models import Sequential
from tensorflow.keras import layers
import models

def main():
    os.system('clear')  # clear the terminal on linux
    print("Using Tensorflow {}".format(tf.__version__))
    # train1 = training_set_generator.TrainingSetGenerator()
    if platform.system() == "Windows":
        db = r'C:\\msys64\\home\\dmmie\\.dashcam.software\\dashcam.index'
    else:
        db = os.environ['HOME']+'/.dashcam.software/dashcam.index'
    db = os.path.normpath(db)
    
    '''
    Generators
    '''
    train_gen = data_get.tfGarminFrameGen(db, track_id=1)
    validation_gen = data_get.tfGarminFrameGen(db,file_id=5)
    '''
    # to match a particular file
    id, path = validation_gen.get_file_id_by_pattern('%Mt-Adams-11-nov-2020%GRMN0005.MP4')
    path = os.path.normpath(path)
    print('File "{}" id is "{}"'.format(path,id))
    return 
    '''

    '''
    # memory leak test loop
    for iter in range(20):
        for batch_idx in range(validation_gen.__len__()):
            x, y = validation_gen.__getitem__(batch_idx)
    '''

    '''
    Model
    '''
    topModel = Sequential(
        [
        models.FlowNet(height=train_gen.Hframe,width=train_gen.Wframe),
        models.PoseConvGRUNet()
        ]
    )


    model = Sequential(
        [
            layers.TimeDistributed(topModel())
        ]
    )

    model.compile()

    '''
    Train
    '''
    model.fit_generator(
        generator=training_gen,
        validation_data=validation_gen,
        use_multiprocessing=True,
        workers=4
        ) 

    print("Done!")

if __name__ == "__main__":
    main()
