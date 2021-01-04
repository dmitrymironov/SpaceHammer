#import training_set_generator
import os
import data_get
import tensorflow as tf
import platform
import numpy as np
#from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, MaxPool2D, Dense, \
    TimeDistributed, GRU, Reshape, Input, Bidirectional, LSTM, \
        RepeatVector, Wrapper
import keras.optimizers
import models

#-----------------------------------------------------------------------
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export

class TimeDistributedNoBatch(Wrapper):
  
  def __init__(self, layer, **kwargs):
    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `TimeDistributedNoBatch` layer with a '
          '`tf.keras.layers.Layer` instance. You passed: {input}'.format(
              input=layer))
    super(TimeDistributedNoBatch, self).__init__(layer, **kwargs)
    self.supports_masking = False

    # It is safe to use the fast, reshape-based approach with all of our
    # built-in Layers.
    self._always_use_reshape = (
        layer_utils.is_builtin_layer(layer) and
        not getattr(layer, 'stateful', False))

  def _get_shape_tuple(self, init_tuple, tensor, start_idx=0, int_shape=None):
    """Finds non-specific dimensions in the static shapes.

    The static shapes are replaced with the corresponding dynamic shapes of the
    tensor.
    Args:
      init_tuple: a tuple, the first part of the output shape
      tensor: the tensor from which to get the (static and dynamic) shapes
        as the last part of the output shape
      start_idx: int, which indicate the first dimension to take from
        the static shape of the tensor
      int_shape: an alternative static shape to take as the last part
        of the output shape
    Returns:
      The new int_shape with the first part from init_tuple
      and the last part from either `int_shape` (if provided)
      or `tensor.shape`, where every `None` is replaced by
      the corresponding dimension from `tf.shape(tensor)`.
    """
    # replace all None in int_shape by K.shape
    if int_shape is None:
      int_shape = K.int_shape(tensor)[start_idx:]
    if isinstance(int_shape, tensor_shape.TensorShape):
      int_shape = int_shape.as_list()
    if not any(not s for s in int_shape):
      return init_tuple + tuple(int_shape)
    shape = K.shape(tensor)
    int_shape = list(int_shape)
    for i, s in enumerate(int_shape):
      if not s:
        int_shape[i] = shape[start_idx + i]
    return init_tuple + tuple(int_shape)

  def _remove_timesteps(self, dims):
    dims = dims.as_list()
    # dmitry
    #return tensor_shape.TensorShape([dims[0]] + dims[2:])
    return tensor_shape.TensorShape(dims[1:])

  def build(self, input_shape):
    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
    input_dims = nest.flatten(
        nest.map_structure(lambda x: x.ndims, input_shape))
    if any(dim < 3 for dim in input_dims):
      raise ValueError(
          '`TimeDistributed` Layer should be passed an `input_shape ` '
          'with at least 3 dimensions, received: ' + str(input_shape))
    # Don't enforce the batch or time dimension.
    self.input_spec = nest.map_structure(
        lambda x: InputSpec(shape=[None] + x.as_list()[1:]), input_shape)
    child_input_shape = nest.map_structure(self._remove_timesteps, input_shape)
    child_input_shape = tf_utils.convert_shapes(child_input_shape)
    super(TimeDistributedNoBatch, self).build(tuple(child_input_shape))
    self.built = True

  def compute_output_shape(self, input_shape):
    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)

    child_input_shape = nest.map_structure(self._remove_timesteps, input_shape)
    child_output_shape = self.layer.compute_output_shape(child_input_shape)
    child_output_shape = tf_utils.convert_shapes(
        child_output_shape, to_tuples=False)
    timesteps = tf_utils.convert_shapes(input_shape)
    timesteps = nest.flatten(timesteps)[1]

    def insert_timesteps(dims):
      dims = dims.as_list()
      return tensor_shape.TensorShape([dims[0], timesteps] + dims[1:])

    return nest.map_structure(insert_timesteps, child_output_shape)

  def call(self, inputs, training=None, mask=None):
    kwargs = {}
    if generic_utils.has_arg(self.layer.call, 'training'):
      kwargs['training'] = training

    input_shape = nest.map_structure(
        lambda x: tensor_shape.TensorShape(K.int_shape(x)), inputs)
    batch_size = tf_utils.convert_shapes(input_shape)
    batch_size = nest.flatten(batch_size)[0]
    if batch_size and not self._always_use_reshape:
      inputs, row_lengths = K.convert_inputs_if_ragged(inputs)
      is_ragged_input = row_lengths is not None
      input_length = tf_utils.convert_shapes(input_shape)
      input_length = nest.flatten(input_length)[1]

      # batch size matters, use rnn-based implementation
      def step(x, _):
        output = self.layer(x, **kwargs)
        return output, []

      _, outputs, _ = K.rnn(
          step,
          inputs,
          initial_states=[],
          input_length=row_lengths[0] if is_ragged_input else input_length,
          mask=mask,
          unroll=False)
      # pylint: disable=g-long-lambda
      y = nest.map_structure(
          lambda output: K.maybe_convert_to_ragged(is_ragged_input, output,
                                                   row_lengths), outputs)
    else:
      # No batch size specified, therefore the layer will be able
      # to process batches of any size.
      # We can go with reshape-based implementation for performance.
      is_ragged_input = nest.map_structure(
          lambda x: isinstance(x, ragged_tensor.RaggedTensor), inputs)
      is_ragged_input = nest.flatten(is_ragged_input)
      if all(is_ragged_input):
        input_values = nest.map_structure(lambda x: x.values, inputs)
        input_row_lenghts = nest.map_structure(
            lambda x: x.nested_row_lengths()[0], inputs)
        y = self.layer(input_values, **kwargs)
        y = nest.map_structure(ragged_tensor.RaggedTensor.from_row_lengths, y,
                               input_row_lenghts)
      elif any(is_ragged_input):
        raise ValueError('All inputs has to be either ragged or not, '
                         'but not mixed. You passed: {}'.format(inputs))
      else:
        input_length = tf_utils.convert_shapes(input_shape)
        input_length = nest.flatten(input_length)[1]
        if not input_length:
          input_length = nest.map_structure(lambda x: array_ops.shape(x)[1],
                                            inputs)
          input_length = generic_utils.to_list(nest.flatten(input_length))[0]

        inner_input_shape = nest.map_structure(
            lambda x: self._get_shape_tuple((-1,), x, 2), inputs)
        # Shape: (num_samples * timesteps, ...). And track the
        # transformation in self._input_map.
        inputs = nest.map_structure_up_to(inputs, array_ops.reshape, inputs,
                                          inner_input_shape)
        # (num_samples * timesteps, ...)
        if generic_utils.has_arg(self.layer.call, 'mask') and mask is not None:
          inner_mask_shape = self._get_shape_tuple((-1,), mask, 2)
          kwargs['mask'] = K.reshape(mask, inner_mask_shape)

        y = self.layer(inputs, **kwargs)

        # Shape: (num_samples, timesteps, ...)
        output_shape = self.compute_output_shape(input_shape)
        # pylint: disable=g-long-lambda
        output_shape = nest.map_structure(
            lambda tensor, int_shape: self._get_shape_tuple(
                (-1, input_length), tensor, 1, int_shape[2:]), y, output_shape)
        y = nest.map_structure_up_to(y, array_ops.reshape, y, output_shape)
        if not context.executing_eagerly():
          # Set the static shape for the result since it might be lost during
          # array_ops reshape, eg, some `None` dim in the result could be
          # inferred.
          nest.map_structure_up_to(
              y, lambda tensor, shape: tensor.set_shape(shape), y,
              self.compute_output_shape(input_shape))

    return y

#-----------------------------------------------------------------------


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
    Addressing CUDA/cuDNN driver issues on Windows
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    print('================================================== TRAIN')
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
    opt = keras.optimizers.Adam(learning_rate=0.01)
    flownet = models.FlowNet()
    # data_get.tfGarminFrameGen.batch_size,
    inputs = Input(shape=(480, 640, 6))
    x = flownet(inputs) # flownet is a temporal model
    x = Reshape(((-1, 5 * 1 * 1024)))(x)
    x = Bidirectional(LSTM(3, return_sequences=True))(x)
    x = LSTM(3)(x)
    x = Dense(4096)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.1)(x)
    outputs = Dense(1)(x) # temporal dimension X V

    '''
    pgu = models.PoseConvGRUNet()
    outputs = pgu(x)
    '''

    model = keras.Model(inputs=inputs, outputs=outputs, name="egomotion")

    model.compile(loss='mean_squared_error', optimizer=opt)

    model.summary()
    #keras.utils.plot_model(model, show_shapes=True)
    #model.build(input_shape=(train_gen.seq_size,480, 640, 6))
    #print(model.summary())

    '''
    Train
    '''
    model.fit(train_gen,validation_data=validation_gen)

    '''
    #debug
    use_multiprocessing=True,
    workers=4
    ) 
    '''

    print("Done!")

if __name__ == "__main__":
    main()
