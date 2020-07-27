"""
This is the 'old' version of the CycleGAN model that works without labels.
To use it, simply add '-useLabels=False' to the cmd parameters.
"""

import data_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #log levels: 0=all, 1=Info+, 2=Warnings+, 3=Only Errors
import tensorflow as tf
tf.keras.backend.clear_session()

# Toggle this variable to use either the standalone keras or the keras built into tensorflow2
# The implementaion was initially built and tested with the standalone version of keras
use_tf_keras = False #True
if not(use_tf_keras) :
    import keras as k
    import keras.objectives
    from keras.layers import Conv2D, Conv2DTranspose
    from keras.layers import LeakyReLU
    from keras.layers import Activation
    from keras.utils import plot_model
    from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
else :
    import tensorflow.keras as k
    import tensorflow.keras.losses
    from tensorflow.keras.layers import Conv2D, Conv2DTranspose
    from tensorflow.keras.layers import LeakyReLU, Activation
    from tensorflow.keras.utils import plot_model
    from tensorflow_addons.layers import InstanceNormalization

import matplotlib.pyplot as plt
import numpy as np
from func_utils import timeit
import time
from datetime import date
import sys
import signal
import gc
import random
#from scipy.spatial import distance_matrix

"""
Global variables.
"""
BATCHSIZE = 4
DEBUG = False
PRINT_MODEL_SUMMARY = False
PLOT_IMAGE_TRANSLATION_AFTER_EPOCH = True
target_directory = ""
USE_MINI_BATCHES = True
MINI_BATCH_SIZE = 6
INITIAL_LEARNING_RATE = 0.0002
LEARNING_RATE_DECAY_FACTOR = 0.5
LEARNING_RATE_DECAY_FREQUENCY = 50 # It is recommended to set it to a value that is a clear divisor of USE_DELAYED_DECAY, e.g. 2 and 100
USE_DECAYING_LR = True
USE_DELAYED_DECAY = 0 # Specify the number of epochs to delay the first update. An update is performed once this counter is met. E.g. for 100, the first update is performed after epoch 100, independent of other parameters
MAX_DISCRIMINATOR_TRAINING_ITERATIONS = 4

# Can be set to True to use the composite models for generating images. The results are not feasible for image prediction as two inputs are required.
# However, this flag helps to visualize the current training progress of a CycleGAN, e.g. for debugging
USE_COMPOSITE_PREDICTION = False 
BOOL_MATCH_IMAGES = True

"""
The DYNAMIC_UPDATE_STRATEGY specifies how the dynamic discriminator training speed increase or decrease is applied:
    0: Every dynamicUpdateIterations iterations, it is checked how much the discriminator and generator losses directly differ
    1: Every dynamicUpdateIterations iterations, it is checked how much both discriminator and generator losses have changed in percent since the last update
"""
DYNAMIC_UPDATE_STRATEGY = 0

"""
The STRATEGY variable indicates which approach is taken for test purposes:
    0: 'old' strategy with .hdf5 model files and no tensorflow gpu sharing
    1: using .h5 models and .h5 weights with mirroredStrategy() and model.compile()
    2: using .h5 models and .h5 weights with mirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()) and model.compile()
    3: using .h5 models and .h5 weights with mirroredStrategy() but no model.compile()
    4: using .hdf5 models and no separate weights with mirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()) but no model.compile()
    5: using .hdf5 models and no separate weights with mirroredStrategy() and model.compile()
    6: using .hdf5 models and no separate weights with mirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()) and model.compile()
    7: using .hdf5 models and no separate weights with mirroredStrategy() but no model.compile()
    8: using .hdf5 models and no separate weights with mirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()) but no model.compile()
"""
STRATEGY = 0
# NOT YET IMPLEMENTED in train(): If True, attempt to distribute the images on the second GPU
DISTRIBUTE_IMAGES = False
"""
End of global variables.
"""

# In order to parallelize predictions, use a tensorflow session with the ability to start multiple threads in keras' backend
# Afterwards, each keras.model.predict() call will use this session
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  try:
    tf.config.experimental.set_visible_devices(gpus, 'GPU') #gpus[0], 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU(s),", len(logical_gpus), "Logical GPU(s)\n\n")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

"""
A custom loss function that calculates the distance between a sketch and its associated UI screenshot.

Parameters (automatically added by Keras' backend when used):
    - y_true: the actual y-values
    - y_pred: the predicted y-values
Returns:
    - The distance between the inputs as a scalar number
"""
def custom_loss_function(y_true, y_pred) :
    #print(y_true, y_pred)
    print("Shape of y_true="+str(y_true.get_shape()))
    print("Shape of y_pred="+str(y_pred.get_shape()))
    loss = (y_true - y_pred) ** 2
    print("Loss = "+str(loss))
    return loss
    #return sum(sum(distance_matrix(y_true, y_pred)))

# This is appearantly necessary in version >=2.0.0, as otherwise even a custom object import does not suffice
# For a version < 2.0.0, use keras.losses.custom_loss instead
if ('tf' in k.__version__) :
    tensorflow.keras.losses.custom_loss = custom_loss_function
else :
    keras.objectives.custom_loss = custom_loss_function

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
#gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
#gpu_options.per_process_gpu_memory_fraction = 0.4
run_options = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
#session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) #intra_op_parallelism_threads=15, inter_op_parallelism_threads=15, 
#k.backend.set_session(session)
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

"""
For test purposes, modify the STRATEGY variable
"""
def set_strategy(index) :
    global STRATEGY
    STRATEGY = index
    print("Changed STRATEGY to "+str(STRATEGY))
    
"""
Set the target directory for storing all related files, i.e. Logs, Plots and Models
"""
def set_target_directory(targetDirectory) :
    global target_directory
    target_directory = targetDirectory
    print("Changed target directory to "+target_directory)
    
"""
Set the flag for using the decaying learning rate
"""
def set_use_decaying_lr(bool_use_decaying_lr) :
    global USE_DECAYING_LR
    USE_DECAYING_LR = bool_use_decaying_lr
    print("Set the usage of decaying learning to rate to "+str(USE_DECAYING_LR))

"""
Set the initial learning rate
"""
def set_initial_lr(lr_start) :
    global INITIAL_LEARNING_RATE
    INITIAL_LEARNING_RATE = lr_start
    print("Set the initial learning rate to "+str(INITIAL_LEARNING_RATE))
     
"""
Set when the learning rate decay may be applied. E.g. if set to 100, the first decay update takes place after the 100th epoch.
Additionally, if LEARNING_RATE_DECAY_FREQUENCY is set to e.g. 1, the learning rate is updated every epoch after the 100th epoch.
"""
def set_delayed_decay(delayed_decay) :
    global USE_DELAYED_DECAY
    USE_DELAYED_DECAY = delayed_decay
    print("Applying delayed learning rate decay after epoch "+str(USE_DELAYED_DECAY))
    
"""
Set how often the learning rate should be updated. An update can happen only after completing an epoch, thus this
parameter specifies how many epochs need to be completed before an updated is applied
"""
def set_learning_rate_decay_frequency(lr_decay_frequency) :
    global LEARNING_RATE_DECAY_FREQUENCY
    LEARNING_RATE_DECAY_FREQUENCY = lr_decay_frequency
    print("Updating learning rate after every "+str(LEARNING_RATE_DECAY_FREQUENCY)+" epoch(s).")
    
"""
Set the learning rate decay factor
"""
def set_lr_decay_factor(lr_decay_factor) :
    global LEARNING_RATE_DECAY_FACTOR
    if (lr_decay_factor > 0) :
        LEARNING_RATE_DECAY_FACTOR = lr_decay_factor
        print("Changed learning rate decay factor to "+str(LEARNING_RATE_DECAY_FACTOR))
    else :
        print("Specified learning rate decay factor is invalid (<= 0)! Learning rate decay remains unchanged at "+str(LEARNING_RATE_DECAY_FACTOR))
    
"""
Specify whether to use the composite models for predictions
"""
def set_use_composite_prediction(bool_value) :
    global USE_COMPOSITE_PREDICTION
    USE_COMPOSITE_PREDICTION = bool_value
    print("Set usage of composite model prediction to "+str(USE_COMPOSITE_PREDICTION))
    
"""    
Specify the number of maximum discriminator training iterations
"""  
def set_max_discriminator_training_iterations(n_max) :
    global MAX_DISCRIMINATOR_TRAINING_ITERATIONS
    MAX_DISCRIMINATOR_TRAINING_ITERATIONS = n_max
    print("Set maximum discriminator training iterations to every  "+str(MAX_DISCRIMINATOR_TRAINING_ITERATIONS)+" iteration(s).")
    
"""
Set whether to keep the matching between sketches and UIs during shuffles
"""
def set_bool_match_images(bool_value) :
    global BOOL_MATCH_IMAGES
    BOOL_MATCH_IMAGES = bool_value
    print("Set usage of matched images to "+str(BOOL_MATCH_IMAGES))

"""
For test purposes, modify the DISTRIBUTE_IMAGES variable
"""
def set_distribute_images(bool_value) :
    global DISTRIBUTE_IMAGES
    DISTRIBUTE_IMAGES = bool_value
    print("Changed DISTRIBUTE_IMAGES to "+str(DISTRIBUTE_IMAGES))

"""
Clear the garbage collector and free all memory allocated by Keras
This prevents memory errors that can occur when using the Keras model.predict() / model.fit() methods
"""
def reset_keras() :
    print("Resetting keras backend...")
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    print("Garbage collector found "+str(gc.collect())+" trash objects.")

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

"""
If it is desired to apply e.g. @timeit on a function, call this method instead with a condition, e.g. DEBUG.
Depending on the condition, either the method is executed directly or the desired decorator is applied.
"""
def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            # Return the function unchanged, not decorated.
            return func
        return dec(func)
    return decorator

"""
This CycleGAN implementation is inspired by:
    https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
    
Other useful websites:
    https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628
    https://towardsdatascience.com/style-transfer-with-gans-on-hd-images-88e8efcf3716
    https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/
"""
   
"""
Writes a model to a .hdf5 file in the subdirectory 'Models' (which is created if it not yet exists)
containing the model itself and all of its weights
    
Parameters:
    - the keras model to store
    - the desired file name without a data type, so e.g. just 'ModelA'; .json and .h5 are added automatically
    
Returns:
    - nothing. The files are written on disk and can be loaded using e.g. load_model()
        
"""
def save_model(model, fileName) :
    if (len(target_directory) > 0) :
        path = os.path.join(os.getcwd(), target_directory)
        if not (os.path.exists(path)) :
            os.mkdir(path)
        path = os.path.join(path, 'Models')
    else :
        path = os.path.join(os.getcwd(), 'Models')
    if not (os.path.exists(path)) :
        os.mkdir(path)
    
    if (STRATEGY == 0 or (STRATEGY >= 5 and STRATEGY <= 8)) :
        fileName = os.path.join(path, fileName+".hdf5")
        model.save(fileName)
    else :
        modelName = os.path.join(path, fileName+".h5")
        weightsName = os.path.join(path, fileName+"_weights.h5")
        model.save(modelName)
        model.save_weights(weightsName)
    
"""
Loads a model from a .hdf5 file.

Parameters:
    - modelName, without a data type, so e.g. just 'ModelA' instead of 'ModelA.hdf5'
    
Returns:
    - the loaded model including its weights OR None if the model does not exist in the 'Models' subdirectory
"""
@conditional_decorator(timeit, DEBUG)
def load_model(modelName) :
    if (len(target_directory) > 0) :
        path = os.path.join(os.getcwd(), target_directory)
        if not (os.path.exists(path)) :
            os.mkdir(path)
        path = os.path.join(path, 'Models')
    else :
        path = os.path.join(os.getcwd(), 'Models')
    if not (os.path.exists(path)) :
        print("Specified model does not exist in the directory: "+path)
        return None
    
    if (STRATEGY == 0 or (STRATEGY >= 5 and STRATEGY <= 8)) :
        fileName = os.path.join(path, modelName+".hdf5")
        loaded_model = k.models.load_model(fileName, custom_objects={'InstanceNormalization': InstanceNormalization, 'custom_loss_function': custom_loss_function})
    else :
        fileName = os.path.join(path, modelName+".h5")
        loaded_model = k.models.load_model(fileName, custom_objects={'InstanceNormalization': InstanceNormalization, 'custom_loss_function': custom_loss_function})
    return loaded_model

# Blogs proposed to split the large image down into small 64x64 samples and translate them individually
# Afterwards, the images can be put back together, though the borders between two separate subimages
# need to be fixed afterwards, e.g. by training another GAN on groups of 4 subimages to smooth the edges
# A possibility for 64x64 subimages is to crop 1920x1080 images down to 1920x1024 (30*16 samples)
@conditional_decorator(timeit, DEBUG)
def define_discriminator(input_shape=(64,64,1), plotName="", loss_function='mse') :
    if (loss_function == 'custom') :
        print("Define Discriminator: Using custom loss function.")
        loss_function = custom_loss_function
    # Initialize weights
    init = k.initializers.RandomNormal(stddev=0.2)
    # Input the source image
    if ('tf' in k.__version__) :
        input_image = k.Input(shape=input_shape)
    else :
        input_image = k.models.Input(shape=input_shape)
    
    # For 256x256(x3) pixel images, 64 filters are used in the convolution
    discriminator = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(input_image) #int(input_shape[0] / 4)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)
    
    # 128 filters in the convolution
    discriminator = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(discriminator)
    discriminator = InstanceNormalization(axis=-1)(discriminator)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)
    
    if (input_shape[0] != 64) :
        # 256 filters in the convolution
        discriminator = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(discriminator)
        discriminator = InstanceNormalization(axis=-1)(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        
        # 512 filters in the convolution
        discriminator = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(discriminator)
        discriminator = InstanceNormalization(axis=-1)(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
        
        # Second last layer
        discriminator = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(discriminator)
        discriminator = InstanceNormalization(axis=-1)(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
    else :
        # 256 filters in the convolution
        discriminator = Conv2D(256, (4,4), padding='same', kernel_initializer=init)(discriminator)
        discriminator = InstanceNormalization(axis=-1)(discriminator)
        discriminator = LeakyReLU(alpha=0.2)(discriminator)
    
    # Output of the last layer
    patch_output = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(discriminator)
    
    # Define the actual model
    model = k.models.Model(input_image, patch_output)
    
    # Compile the model
    # IMPORTANT: If 'metrics=['xyz']' is used, the gan produces a list of outputs, containing [loss, metric x, metric y etc.]
    model.compile(loss=loss_function, optimizer=k.optimizers.Adam(lr=INITIAL_LEARNING_RATE, beta_1=0.5), loss_weights=[0.5])#, options=run_options) #metrics=['accuracy'], lr=0.0002
    if (len(plotName) > 0) :
        if (len(target_directory) > 0) :
            path = os.path.join(os.getcwd(), target_directory)
            if not (os.path.exists(path)) :
                os.mkdir(path)
            path = os.path.join(path, 'Plots')
        else :
            path = os.path.join(os.getcwd(), 'Plots')
        if not (os.path.exists(path)) :
            os.mkdir(path)
        path = os.path.join(path, plotName)
        plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
    
    if (PRINT_MODEL_SUMMARY) :
        model.summary()
    
    return model

# Generate a block of a residual network
# These residual networks are used in a CycleGAN to actually do the transformation into another style
# and are placed behind convolutional layers that encode the input image. Afterwards, upsampling convolutional layers
# have to be used to actually generate the output image
# It is recommended in "Unpaired Image-To-Image Translation using Cycle-consistent Adversarial Networks" by Zhu, Park, Isola and Efros (2017)
# to use a 6-block residual network for 128x128 images and a 9-block residual network for 256x256 images
#@conditional_decorator(timeit, DEBUG)
def residual_network_block(n_filters, input_layer) :
    # weight initialization
    if ('tf' in k.__version__) :
        init = tf.compat.v1.keras.initializers.RandomNormal(stddev=0.02)
    else :
        init = k.initializers.RandomNormal(stddev=0.02)
    # first layer convolutional layer
    generator = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)
    # second convolutional layer
    generator = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    # concatenate merge channel-wise with input layer
    generator = k.layers.Concatenate()([generator, input_layer])
    return generator
    
# Define the standalone generator model
# It is recommended to use n_resnet=9 for 256x256 images and n_resnet=6 for 128x128 images, but n_resnet=6 also works for 256x256 images
@conditional_decorator(timeit, DEBUG)
def define_generator(image_shape=(256,256,3), n_resnet=6, plotName=""):
    # Weight initialization
    init = k.initializers.RandomNormal(stddev=0.02)
    # Image input
    if ('tf' in k.__version__) :
        in_image = k.Input(shape=image_shape)
    else :
        in_image = k.models.Input(shape=image_shape)
    
    # 7x7 Convolution-InstanceNorm-ReLU layer with 64 filters and stride 1
    generator = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)
    
	# Downsamplingeneratorlayer with 128 filters
    generator = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)
    
    # Downsamplingeneratorlayer with 256 filters
    generator = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)
    
    # Residual network for images of size 256x256
    for _ in range(n_resnet):
        generator = residual_network_block(256, generator)
        
    # Upsamplingeneratorlayer with 128 filters
    generator = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)
    
    # Upsamplingeneratorlayer with 64 filters
    generator = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    generator = Activation('relu')(generator)
    
    # 7x7 Convolution-InstanceNorm-ReLU layer with 3 filters and stride 1
    generator = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(generator)
    generator = InstanceNormalization(axis=-1)(generator)
    out_image = Activation('tanh')(generator)
    
    # Define the actual model
    model = k.models.Model(in_image, out_image)
    if (len(plotName) > 0) :
        if (len(target_directory) > 0) :
            path = os.path.join(os.getcwd(), target_directory)
            if not (os.path.exists(path)) :
                os.mkdir(path)
            path = os.path.join(path, 'Plots')
        else :
            path = os.path.join(os.getcwd(), 'Plots')
        if not (os.path.exists(path)) :
            os.mkdir(path)
        path = os.path.join(path, plotName)
        plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
        
    if (PRINT_MODEL_SUMMARY) :
        model.summary()
        
    return model

# Define a composite model for updating generators through adversarial and cycle loss
@conditional_decorator(timeit, DEBUG)
def define_composite_model(generator_model_1, discriminator_model, generator_model_2, 
                           image_shape, plotName="", loss_function='mse', bool_trainable_discriminator=False, bool_trainable_generator2=False):
    if (loss_function == 'custom') :
        loss_function = custom_loss_function
    # Allow the generator model to be trainable
    generator_model_1.trainable = True
    # Forbid the discriminator to be trainable
    discriminator_model.trainable = bool_trainable_discriminator #False
    # Forbid the other generator model to be trainable
    generator_model_2.trainable = bool_trainable_generator2 #False
    
    # In tf2, a warning appears that no trainable weights exist and the discriminator losses don't go down...
    if ('tf' in k.__version__) :
        discriminator_model.trainable = True
    
    # Define the discriminator element
    if ('tf' in k.__version__) :
        input_gen = k.Input(shape=image_shape)
    else :
        input_gen = k.models.Input(shape=image_shape)
    gen1_out = generator_model_1(input_gen)
    output_d = discriminator_model(gen1_out)
    
    # Define the identity element
    if ('tf' in k.__version__) :
        input_id = k.Input(shape=image_shape)
    else :
        input_id = k.models.Input(shape=image_shape)
    output_id = generator_model_1(input_id)
    
    # Define the forward cycle
    output_f = generator_model_2(gen1_out)
    
    # Define the backward cycle
    gen2_out = generator_model_2(input_id)
    output_b = generator_model_1(gen2_out)
    
    # Define the model graph
    model = k.models.Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    
    # Define the optimization algorithm configuration
    # The most common recommendation in literatur is the Adam optimizer for all types of GANs
    opt = k.optimizers.Adam(lr=INITIAL_LEARNING_RATE, beta_1=0.5) # lr=0.0002
    
    # Compile the model with weighting of least squares loss and L1 loss
    model.compile(loss=[loss_function, 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)#, options=run_options)
    if (len(plotName) > 0) :
        if (len(target_directory) > 0) :
            path = os.path.join(os.getcwd(), target_directory)
            if not (os.path.exists(path)) :
                os.mkdir(path)
            path = os.path.join(path, 'Plots')
        else :
            path = os.path.join(os.getcwd(), 'Plots')
        if not (os.path.exists(path)) :
            os.mkdir(path)
        path = os.path.join(path, plotName)
        plot_model(model, to_file=path, show_shapes=True, show_layer_names=True)
        
    if (PRINT_MODEL_SUMMARY) :
        model.summary()
        
    return model

"""
Generate a batch of real samples.
Returns: a list of true images (in their original shape) and a list of corresponding true labels (1)

Parameters:
    images: A list of images OR paths where the images can be found; use e.g. data_utils.get_image_paths(properties.UNIQUE_UIS_DIR)
    ui_mapping: A set of two lists that indicates which sketch corresponds to which ui; use e.g. data_utils.get_matching_uis(...)
    patch_size: Size of patches of images. Used to create class labels in a vector with e.g. shape (batchsize, 256x256, '1')
    batch_size: This hyperparameter is specified in this class and in the main.py class. Specifies the batch size.
    flip_labels: This parameter allows to flip the labels so that the real images are classified as 'fake' images
    use_random_noise: This parameter specifies whether to apply random noise to the generated real samples (recommended)
"""
#@conditional_decorator(timeit, DEBUG)
def generate_real_samples(images, patch_size=256, batch_size=BATCHSIZE, flip_labels=False, use_random_noise=True) :
    
    # If the paths are used as a parameter, load the images first
    # However, this might take anything between 10-35 seconds
    if (isinstance(images, str)) :
        images = data_utils.load_image(images)
    elif (isinstance(images[0], str)) :
        images = data_utils.load_images_randomized(images, batch_size)

    images = np.asarray([data_utils.normalize_image(images[i]) for i in range(len(images))])
    X = np.asarray(images)
    
    # Add some random noise so that the discriminator cannot learn that real images only have color values of n/255 or 0..255
    # If batchsize > 1, X has a shape of e.g. (batchsize, 256, 256, 3)
    if (use_random_noise) :
        if (len(X.shape) > 3) :
            #print("Adding random noise to "+str(len(X))+" images.")
            X = np.asarray([X[i] + np.random.normal(size=X[i].shape, loc=0.0, scale=0.1) for i in range(len(X))])
        # Otherwise we just add noise to a single image
        else :
            X = np.asarray(X + np.random.normal(size=X.shape, loc=0.0, scale=0.1))
        X = np.clip(X, 0, 1)
    
    # Generate flipped 'real' class labels (meaning '0')
    # This is useful to slow down the training speed of the discriminator; 
    # however, it should not be called too often (e.g. every 2nd to 3rd training epoch)
    if (flip_labels) :
        y = np.zeros((len(images), patch_size, patch_size, 1))
    # Generate 'real' class labels (meaning '1')
    else :
        y = np.ones((len(images), patch_size, patch_size, 1))#(batch_size, patch_size, patch_size, 1))
    return X, y

"""
Generates a set of fake images - a matrix filled with zeros of shape (num_images, patch_size x patch_size, 1)

Parameters:
    generator_model: The path to load the model OR the model used to predict / generate images
    images: A set of images to base the predictions on
    patch_size: Size of images patches, e.g. 256x256 pixels.
    
Returns:
    A list of fake images and their corresponding fake labels (0)
"""
@conditional_decorator(timeit, DEBUG)
def generate_fake_samples(generator_model, images, patch_size, flip_labels=False):
    if (isinstance(images, str)) :
        images = data_utils.load_image(images)
    elif (isinstance(images[0], str)) :
        images = data_utils.load_images(images)
    
    # Normalize images to [-1,..1] because the generator uses tanh as activation
    # If [0..255]
    if (np.any(images > 1)) :
        images = (images - 127.5) / 127.5 #np.asarray([data_utils.normalize_image(images[i]) for i in range(len(images)), low=-1])
    # If already [-1..1], it's alright
    elif (np.any(images < 0)) :
        pass
    # Else normalize from [0..1] to [-1..1]
    else :
        images = (images * 2) - 1
        
    if (isinstance(generator_model, str)) :
        # If the model shall be loaded from a file
        model = load_model(generator_model)
        X = model.predict_on_batch(images)

    else :
        # If the model is handed over as a parameter            
        X = generator_model.predict_on_batch(images)
        
    # Denormalize the image from [-1,1] to [0,1]
    X = (X + 1) / 2
    
    # Generate flipped 'fake' class labels (meaning '1')
    # This is useful to slow down the training speed of the discriminator; 
    # however, it should not be called too often (e.g. every 2nd to 3rd training epoch)
    if (flip_labels) :
        y = np.ones((len(images), patch_size, patch_size, 1))
    # Generate 'fake' class labels (meaning '0')
    else :
        y = np.zeros((len(images), patch_size, patch_size, 1))
	
    return X, y

"""
An alternative to the 'generate_fake_samples()' method which uses the composite model instead of the generator model.
The overall functionality remains unchanged.
However, the composite model requires to provide images from both domains
"""
@conditional_decorator(timeit, DEBUG)
def generate_fake_samples_composite(c_model, imagesA, imagesB, patch_size, flip_labels=False) :
    if (isinstance(imagesA, str)) :
        imagesA = data_utils.load_image(imagesA)
    elif (isinstance(imagesA[0], str)) :
        imagesA = data_utils.load_images(imagesA)
    
    # Normalize images to [-1,..1] because the generator uses tanh as activation
    # If [0..255]
    if (np.any(imagesA > 1)) :
        imagesA = (imagesA - 127.5) / 127.5 #np.asarray([data_utils.normalize_image(images[i]) for i in range(len(images)), low=-1])
    # If already [-1..1], it's alright
    elif (np.any(imagesA < 0)) :
        pass
    # Else normalize from [0..1] to [-1..1]
    else :
        imagesA = (imagesA * 2) - 1
        
    if (isinstance(imagesB, str)) :
        imagesB = data_utils.load_image(imagesB)
    elif (isinstance(imagesB[0], str)) :
        imagesB = data_utils.load_images(imagesB)
    
    # Normalize images to [-1,..1] because the generator uses tanh as activation
    # If [0..255]
    if (np.any(imagesB > 1)) :
        imagesB = (imagesB - 127.5) / 127.5 #np.asarray([data_utils.normalize_image(images[i]) for i in range(len(images)), low=-1])
    # If already [-1..1], it's alright
    elif (np.any(imagesB < 0)) :
        pass
    # Else normalize from [0..1] to [-1..1]
    else :
        imagesB = (imagesB * 2) - 1
        
    # If only a single image is used, ensure that it is still in the format of [1, 256, 256, 3]
    if (len(imagesA.shape) == 3) :
        imagesA = np.reshape(imagesA, (1, 256, 256, 3))
        
    if (len(imagesB.shape) == 3) :
        imagesB = np.reshape(imagesB, (1, 256, 256, 3))
        
    # Predict the images, then discard everything but the result of the forward pass
    if (isinstance(c_model, str)) :
        # If the model shall be loaded from a file
        model = load_model(c_model)
        X = model.predict_on_batch([imagesA, imagesB])[3] #Produces outputs: 16x16 discriminator activations, Identity Loss, Forward Cycle Loss, Backward Cycle Loss

    else :
        # If the model is handed over as a parameter            
        X = c_model.predict_on_batch([imagesA, imagesB])[3]
        
    # Denormalize the image from [-1,1] to [0,1]
    X = (X + 1) / 2
    
    # Generate flipped 'fake' class labels (meaning '1')
    # This is useful to slow down the training speed of the discriminator; 
    # however, it should not be called too often (e.g. every 2nd to 3rd training epoch)
    if (flip_labels) :
        y = np.ones((len(imagesA), patch_size, patch_size, 1))
    # Generate 'fake' class labels (meaning '0')
    else :
        y = np.zeros((len(imagesA), patch_size, patch_size, 1))
	
    return X, y
	
"""
Update the image pool of fake images used for training.

Parameters:
    pool: the current / previous pool of fake images to select image from for training
    images: a set of images to select new 'fake' images from; 
            the number of images should be at least twice as large as max_pool_size
            If another datatype is handed over, the results will have the submitted datatype in a list instead of images,
            but the method does work for other datatypes as well
    max_pool_size:  maximum size of the image pool; larger pool size makes it less likely to get the same image twice 
                    Additionally, if the current pool size is smaller than max_pool_size, the remaining images are loaded
                    directly into the pool instead of being able to be selected immediately
                    Thus for e.g. an empty pool, the first max_pool_size images are automatically selected
    
Returns: a set of images that are at random:
    - chosen directly from the set of images as soon as the image pool is full
    - chosen iid. from the image pool
    
"""
# update image pool for fake images
@conditional_decorator(timeit, DEBUG)
def update_image_pool(pool, images, max_pool_size=50):
    selected = list()
    for image in images:
        
        # If the pool has a smaller size than desired, fill the pool with new images
        # Images in the pool are not guaranteed to be part of the final set of images
        # The chance depends on the total number of images
        if len(pool) < max_pool_size:
            pool.append(image)
            selected.append(image)
        # Alt1: it is chosen at random for each new image whether to add it to the
        # final list of selected images
        elif np.random.random() < 0.5:
            selected.append(image)
        # Alt2: If the image is not instantly chosen as part of the new set of images
        # a random image of the pool is chosen instead, allowing to reuse images
        # Additionally, the randomly chosen pool image is replaced with the current image, allowing delayed pool updates
        else:
            ix = np.random.randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    # For some odd reason, using np.asarray() in tensorflow2 is extremely slow, but np.array() works
    if ('tf' in k.__version__) :
        return np.array([np.array(s) for s in selected])
    else :
        return np.asarray(selected)

"""
Load and prepare image patches on a given set of images or paths to the images.
The length of the results can be influenced by using an index to start from and a batchsize
to specify how many images shall be loaded from the given start index.

Note that each image is split into patches afterwards, thus the resulting number of patches
depends on the size of the loaded images - in the given datasets, it's either 6 or 28 patches.

Parameters:
    trainSketches, trainUIs: lists of images OR paths to images of training sketches / corresponding training UIs
    index: the index to start from (explicit call is e.g. trainUIs[index:index+batchsize])
    batchsize: the number of images to load, given the start point, e.g. batchsize=5 loads 5 images
    patchsize: If required, the patchsize can be altered. It's set to 256 by default and should only be altered if
               the GAN models are altered as well, e.g. the generator model should use 6 instead of 9 residual networks
               when using size 128
    
Returns:
    Two lists of image patches; first one for sketches and the second one for uis. Their lengths depend on the size
    of the loaded images! E.g. loading 6 images of size 540x960 results in two lists with 36 patches (6 images * 6 patches)
"""
def load_batch(trainSketches, trainUIs, index, batchsize, patchsize=256) :
    if (index >= len(trainSketches) or index >= len(trainUIs)) :
        index = 0
    if (isinstance(trainSketches, str)) :
        trainSketches = np.reshape(data_utils.load_image(trainSketches), (1,256,256,3))
    elif (isinstance(trainSketches[0], str)) :
        trainSketches = data_utils.load_images(trainSketches[index:index+batchsize])
    else :
        trainSketches = trainSketches[index:index+batchsize]
        
    if (isinstance(trainUIs, str)) :
        trainUIs = np.reshape(data_utils.load_image(trainUIs), (1,256,256,3))
        return trainSketches, trainUIs
    elif (isinstance(trainUIs[0], str)) :
        trainUIs = data_utils.load_images(trainUIs[index:index+batchsize])
    else :
        trainUIs = trainUIs[index:index+batchsize]
    
    if (isinstance(trainSketches, np.ndarray)) :
        #print("Images are not split into patches because they already have shape (256, 256, 3).")
        return trainSketches, trainUIs
    else :
        return data_utils.get_image_patches_from_image_list(trainSketches, 256), data_utils.get_image_patches_from_image_list(trainUIs, 256)

# train cyclegan models
"""
Train the different models, that is the two discriminator models and the two composite models to train the generator models.

Parameters:
    - trainSketches: the training set of sketches
    - trainUIs: the training set of UIs
    - batchsize: hyperparameter of how many images to use per training iteration; appearantly OOM occurs for batchsize > 5
    - n_steps: hyperparameter of how many training iterations are performed;
               Use n_steps <= 0 to use an automated number of steps (100 epochs * number of batches per epoch)
    - discriminatorTrainingIterations: hyperparameter of how often the discriminator is trained in comparison to the generator. 2 means every 2nd iteration, 3 every 3rd etc.
    - num_iterations_until_flip: hyperparameter of how often real labels are used until labels are flipped for one training iteration. 
                                 If set to 2 (min), flipped labels are used in every 2nd iteration, 3 for every 3rd iteration etc. Set to >n_steps to effectively not use it for training. 
    - dynamicUpdateIterations: Specifies how often the discriminator and generator losses are compared to decide whether to increase or decrease discriminator training speed
    - n_start: If it is intended to continue to train models, set n_start to the desired epoch index to continue the training. E.g. n_start=50 creates a folder Epoch_050.
    - n_epochs: Specify the number of training epochs; you may also use a combination, e.g. n_start=50 and n_epochs=70 to train for only 20 epochs, starting with index 50
    - bool_new_discriminator: If set to true, the discriminator and thus composite models will be recreated. However, pretrained generator models are loaded if available.
    - discriminator_update_stop_epoch: If set to a value > 0, the discriminator will not be updated anymore upon finishing this epoch index; also supports lists as input
    - discriminator_update_continue_epoch: If set to a value > 0, the discriminator will be able to update again, continuing with its old frequency; supports lists as input
                    As an example, setting either of these two to 10 means that the discriminator is not trained anymore / trained again after completing epoch 10.
    
Old parameters: 
    d_model_A: the discriminator model for domain A, e.g. sketches
    d_model_B: the discriminator model for domain B, e.g. UIs
    
    g_model_AtoB: the generator model to translate from domain A to domain B, e.g. sketches to UIs
    g_model_BtoA: the generator model to translate from domain B to domain A, e.g. backwards translation
    
    c_model_AtoB: the composite model to update generator models by adversarial and cycle loss
    c_model_BtoA: the composite model to update generator models by adversarial and cycle loss
    
    trainA: training data of domain A, e.g. hand-drawn sketches of UI designs
    trainB: training data of domain B, e.g. unique uis from the Rico dataset
"""
#@profile
def train(inputTrainSketches, inputTrainUIs, batchsize=BATCHSIZE, n_steps=100, discriminatorTrainingIterations=2, num_iterations_until_flip=3, 
          dynamicUpdateIterations=60, n_start=0, n_epochs=150, bool_new_discriminator=False, discriminator_update_stop_epoch=-1, discriminator_update_continue_epoch=-1,
          image_size=256): #d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, 
    # Ensure that when interrupted by e.g. CTRL+C, models are saved
    def signal_handler(sig, frame) :
        print("\nThe code execution has been interrupted. Saving the models...\n")
        save_model(d_model_A, "d_model_A")
        save_model(d_model_B, "d_model_B")
        if not (USE_COMPOSITE_PREDICTION) :
            save_model(g_model_AtoB, "g_model_AtoB")
            save_model(g_model_BtoA, "g_model_BtoA")
        save_model(c_model_AtoB, "c_model_AtoB")
        save_model(c_model_BtoA, "c_model_BtoA")
        with open(path_timestamps, "a+") as file :
                file.write("Keyboard Interrupt: Saved models\n")
        print("Models have been saved. Stopping the execution now.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Define properties of the training run
    n_epochs, n_batch = n_epochs, 16
    # Determine the output square shape of the discriminator
    n_patch = 16 
    
    discriminatorTrainingIterations = discriminatorTrainingIterations
    trainSketches, trainUIs = inputTrainSketches, inputTrainUIs
    if (isinstance(trainSketches, str)) :
        exampleA, exampleB = trainSketches, trainUIs
        print(exampleA, exampleB)
    else :
        exampleA, exampleB = trainSketches[0], trainUIs[0]
    del inputTrainSketches
    del inputTrainUIs
    reset_keras()

    # Prepare image pools for fake images
    poolA, poolB = list(), list()
    # Calculate the number of batches per training epoch
    if (isinstance(trainSketches, tuple)) :
        trainSketches = list(trainSketches)
    if (isinstance(trainUIs, tuple)) :
        trainUIs = list(trainUIs)
    if (isinstance(trainSketches, list)) :
        num_batches_per_epoch = int(len(trainSketches) / batchsize)
    elif (isinstance(trainSketches, np.ndarray)) :
        if (len(trainSketches.shape) > 3) :
            num_batches_per_epoch = int(trainSketches.shape[0] / batchsize)
        else :
            num_batches_per_epoch = 1
    else :
        num_batches_per_epoch = 1

    # Set the index of the current batch. This value is increased artificially if training should be continued instead of being completely restarted.
    current_batch_index = 0     
    if (n_start > 1) :
        current_epoch_index = n_start
    else :
        current_epoch_index = 1
    
    # Use n_steps <= 0 to use an automated calculation of steps
    if (n_steps <= 0) :
        n_steps = n_epochs * num_batches_per_epoch
        print("Set n_steps = "+str(n_steps)+" (= "+str(n_epochs)+" epochs * "+str(num_batches_per_epoch)+" batches)")
	
    # If the training is continued, the already passed epochs are skipped
    if (n_start > 1) :
        n_steps -= n_start * num_batches_per_epoch

    # Results are written to a file, hence the directory and file are created if they not yet exist
    if (len(target_directory) > 0) :
        path = os.path.join(os.getcwd(), target_directory)
        if not (os.path.exists(path)) :
            os.mkdir(path)
        path = os.path.join(path, 'Logs')
    else :
        path = os.path.join(os.getcwd(), "Logs")
    path_timestamps = os.path.join(path, "Timestamps.txt")
    
    if not (os.path.exists(path)) :
        os.mkdir(path)
        print("Created directory 'Logs': "+path)
    path = os.path.join(path, "Results.txt")

    # The log file will also contain the date when the execution was started;
    # further runs will append in the same way below existing records, but never overwrite.
    current_time = date.today()
    current_time = current_time.strftime("%d/%m/%Y")
    
    if not (os.path.exists(path)) :
        with open(path, "w+") as file :
            if (USE_DECAYING_LR) :
                file.write(str(current_time)+" - Results of each iteration (parameters: batchsize="+str(batchsize)+", n_steps="+str(n_steps)
                           +", discriminatorTrainingIterations="+str(discriminatorTrainingIterations)+", num_iterations_until_flip="
                           +str(num_iterations_until_flip)+", lr_start="+str(INITIAL_LEARNING_RATE)+", lr_decay_factor="
                           +str(LEARNING_RATE_DECAY_FACTOR)+", lr_update_frequency="+str(LEARNING_RATE_DECAY_FREQUENCY)
                           +", lr_decay_delay="+str(USE_DELAYED_DECAY)+"):\n")
            else :
                file.write(str(current_time)+" - Results of each iteration (parameters: batchsize="+str(batchsize)+", n_steps="+str(n_steps)
                           +", discriminatorTrainingIterations="+str(discriminatorTrainingIterations)+", num_iterations_until_flip="
                           +str(num_iterations_until_flip)+", lr_start="+str(INITIAL_LEARNING_RATE)+"):\n")
        print("Created 'Results.txt' in directory 'Logs' to write results to a file.")
    else :
        with open(path, "a+") as file :
            file.write(str(current_time)+" - Results of each iteration (parameters: n_steps="+str(n_steps)
                       +", discriminatorTrainingIterations="+str(discriminatorTrainingIterations)+", num_iterations_until_flip="
                       +str(num_iterations_until_flip)+", lr_start="+str(INITIAL_LEARNING_RATE)+", lr_decay_factor="
                       +str(LEARNING_RATE_DECAY_FACTOR)+", lr_update_frequency="+str(LEARNING_RATE_DECAY_FREQUENCY)
                       +", lr_decay_delay="+str(USE_DELAYED_DECAY)+"):\n")
    
    # Prepare the model directory path
    if (len(target_directory) > 0) :
        path_models = os.path.join(os.getcwd(), target_directory)
        path_models = os.path.join(path, 'Models')
    else :
        path_models = os.path.join(os.getcwd(), 'Models')
    # If it is intended to train a pretrained generator model with a new discriminator, set this flag
    # Of course, this can only be used if the composite models are not used for prediction.
    if (bool_new_discriminator) :
        if (USE_COMPOSITE_PREDICTION) :
            print("Train(): Cannot continue training the generator model if use_composite_prediction=True!\n"
                  +"Please train the general model with use_composite_prediction=False before calling train() with bool_new_discriminator=True!")
            sys.exit(1)
        image_shape = (image_size,image_size,3)
        # Usually the flag should only be set if a generator model has already been trained in the specified target directory
        if (os.path.exists(os.path.join(path_models, "g_model_AtoB_trained_final.hdf5"))) :
             g_model_AtoB = load_model("g_model_AtoB_trained_final")
             g_model_BtoA = load_model("g_model_BtoA_trained_final")
        # If no 'trained_final' version exists, the last stored model is loaded instead
        elif (os.path.exists(os.path.join(path_models, "g_model_AtoB.hdf5"))) :
             g_model_AtoB = load_model("g_model_AtoB")
             g_model_BtoA = load_model("g_model_BtoA")
        # If the flag was set without any existing pretrained model, the models are completely recreated
        else :   
             g_model_AtoB = define_generator(image_shape, n_resnet=6)
             g_model_BtoA = define_generator(image_shape, n_resnet=6)
             
        # With the set flag, the discriminator models are recreated. As the composite models are a combination of three models,
        # it is necessary to recreate them as well.
        d_model_A = define_discriminator(image_shape, loss_function='mse')
        d_model_B = define_discriminator(image_shape, loss_function='mse')
        c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape, loss_function='mse')
        c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape, loss_function='mse')
        
    # Load models; if the composite models are used for prediction, the generator models are not required
    else :
        if (os.path.exists(os.path.join(path_models, "d_model_A_trained_final.hdf5")) and n_start > 0) :
            d_model_A = load_model("d_model_A_trained_final")
            d_model_B = load_model("d_model_B_trained_final")
            if not (USE_COMPOSITE_PREDICTION) :
                g_model_AtoB = load_model("g_model_AtoB_trained_final")
                g_model_BtoA = load_model("g_model_BtoA_trained_final")
            c_model_AtoB = load_model("c_model_AtoB_trained_final")
            c_model_BtoA = load_model("c_model_BtoA_trained_final")
        elif (os.path.exists(os.path.join(path_models, "d_model_A.hdf5"))) :
            d_model_A = load_model("d_model_A")
            d_model_B = load_model("d_model_B")
            if not (USE_COMPOSITE_PREDICTION) :
                g_model_AtoB = load_model("g_model_AtoB")
                g_model_BtoA = load_model("g_model_BtoA")
            c_model_AtoB = load_model("c_model_AtoB")
            c_model_BtoA = load_model("c_model_BtoA")
        else :
            print("Train(): Models do not yet exist in the specified target directory.\nCreating new models...")
            image_shape = (image_size,image_size,3)
            # generator: A -> B
            g_model_AtoB = define_generator(image_shape, n_resnet=6)#, plotName='generator_modelAtoB_plot.png')
            # generator: B -> A
            g_model_BtoA = define_generator(image_shape, n_resnet=6)#, plotName='generator_modelBtoA_plot.png')
            # discriminator: A -> [real/fake]
            d_model_A = define_discriminator(image_shape, loss_function='mse')#, plotName='discriminator_modelAtoB_plot.png')
            # discriminator: B -> [real/fake]
            d_model_B = define_discriminator(image_shape, loss_function='mse')#, plotName='discriminator_modelBtoA_plot.png')
            # composite: A -> B -> [real/fake, A]
            c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape, loss_function='mse')#, plotName='composite_modelAtoB_plot.png')
            # composite: B -> A -> [real/fake, B]
            c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape, loss_function='mse')

    # Define initial values for the discriminator losses. These are necessary to store loss values over multiple iterations
    dA_loss1, dA_loss2 = 100, 100
    dB_loss1, dB_loss2 = 100, 100
    
    # Define store variables for the old loss values. Used for the update mechanism of discriminator update frequency
    dA_loss1_old, dB_loss1_old, g_loss1_old = 0, 0, 0
    
    # Prepare the usage of flipped labels
    bool_use_flipped_labels = False
    num_iterations_until_flip = num_iterations_until_flip
    if (num_iterations_until_flip <= 1) :
        num_iterations_until_flip = 2
    counter_num_iterations_until_flip = 0
    
    # Time measurement for the training procedure
    time_before_training = time.process_time()
    time_per_epoch = time.process_time()
    
    # If values are specified to stop the discriminator training for an epoch, the numbers are checked before starting the training 
    # This block must specifically be called if it is intended to stop the discriminator training before the training actually starts
    oldDiscriminatorTrainingIterations = discriminatorTrainingIterations
    if (isinstance(discriminator_update_stop_epoch, list)) :
        if (len(discriminator_update_stop_epoch) > 0) :
            if (int(discriminator_update_stop_epoch[0]) == n_start) :
                discriminatorTrainingIterations = n_steps + 1
                with open(path_timestamps, "a+") as file :
                    file.write("After epoch "+str(current_epoch_index)+": Stopping discriminator update "+str(discriminator_update_stop_epoch))
                discriminator_update_stop_epoch = discriminator_update_stop_epoch[1:len(discriminator_update_stop_epoch)]
    elif (discriminator_update_stop_epoch == n_start) :
        discriminatorTrainingIterations = n_steps + 1
        with open(path_timestamps, "a+") as file :
            file.write("After epoch "+str(current_epoch_index)+": Stopping discriminator update "+str(discriminator_update_stop_epoch))
        
    # Additionally, it is checked and applied if specified whether to continue the discriminator training. By default, the discriminator is always trained
    if (isinstance(discriminator_update_continue_epoch, list)) :
        if (len(discriminator_update_continue_epoch) > 0) :
            if (int(discriminator_update_continue_epoch[0]) == n_start) :
                discriminatorTrainingIterations = oldDiscriminatorTrainingIterations
                with open(path_timestamps, "a+") as file :
                    file.write("After epoch "+str(current_epoch_index)+": Continuing discriminator update "+str(discriminator_update_continue_epoch))
                discriminator_update_continue_epoch = discriminator_update_continue_epoch[1:len(discriminator_update_continue_epoch)]
    elif (discriminator_update_continue_epoch == n_start) :
        discriminatorTrainingIterations = oldDiscriminatorTrainingIterations   
        with open(path_timestamps, "a+") as file :
            file.write("After epoch "+str(current_epoch_index)+": Continuing discriminator update "+str(discriminator_update_continue_epoch))

    # For test purposes of 'end of epoch' code, use this line to skip most of the training
    #current_batch_index = num_batches_per_epoch - 2

    """
    This is the actual training routine. In short, it performs the following tasks:
        - Load new batch
        - Generate real images
        - Generate fake images
        - Train composite_model_BtoA (and thus Generator BtoA)
        - Train discriminator A
        - Train composite_model_AtoB (and thus Generator AtoB)
        - Train discriminator B
        - If applicable, check losses and modify discriminator update frequency
        - Repeat the above steps until all batches of the current epoch are processed
        - After an epoch, store the models, create up to 10 plots to visualize the current training progress and update parameters.
        - Repeat the above steps until all epochs are completed
        - After the training, store all models to disk and create another up to 10 plots to visualize the final training progress.
    """
    for i in range(n_start, n_start + n_steps):
        # The taken time for each iteration is measured
        time_iteration_start = time.process_time()
        
        # At the beginning of each iteration, load the next batch until the whole training dataset has been processed once
        if (current_batch_index < num_batches_per_epoch) :
            trainA, trainB = load_batch(trainSketches, trainUIs, current_batch_index * batchsize, batchsize=batchsize)
            current_batch_index += 1
        else :
            # After each epoch:
            current_batch_index = 0
            
            # Shuffle the training data
            if (isinstance(trainSketches, list)) :
                if (BOOL_MATCH_IMAGES) :
                    shuffleTrainingSet = list(zip(trainSketches, trainUIs))
                    random.shuffle(shuffleTrainingSet)
                    trainSketches, trainUIs = zip(*shuffleTrainingSet)
                    del shuffleTrainingSet
                else :
                    random.shuffle(trainSketches)
                    random.shuffle(trainUIs)
            elif (isinstance(trainSketches, np.ndarray)) :
                np.random.shuffle(trainSketches)
                if (isinstance(trainUIs, list)) :
                    random.shuffle(trainUIs)
                elif (isinstance(trainUIs, np.ndarray)) :
                    np.random.shuffle(trainUIs)

            # Load a new batch and adapt the current_batch_index
            trainA, trainB = load_batch(trainSketches, trainUIs, current_batch_index * batchsize, batchsize=batchsize)
            current_batch_index += 1
            
            # If specified, check if this epoch is specified as a point to stop the discriminator training
            if (isinstance(discriminator_update_stop_epoch, list)) :
                if (len(discriminator_update_stop_epoch) > 0) :
                    if (int(discriminator_update_stop_epoch[0]) == current_epoch_index) :
                        oldDiscriminatorTrainingIterations = discriminatorTrainingIterations
                        discriminatorTrainingIterations = n_steps + 1
                        with open(path_timestamps, "a+") as file :
                            file.write("After epoch "+str(current_epoch_index)+": Stopping discriminator update "+str(discriminator_update_stop_epoch))
                        discriminator_update_stop_epoch = discriminator_update_stop_epoch[1:len(discriminator_update_stop_epoch)]
            elif (discriminator_update_stop_epoch == current_epoch_index) :
                oldDiscriminatorTrainingIterations = discriminatorTrainingIterations
                discriminatorTrainingIterations = n_steps + 1
                with open(path_timestamps, "a+") as file :
                    file.write("After epoch "+str(current_epoch_index)+": Stopping discriminator update "+str(discriminator_update_stop_epoch))
                
            # If specified, check if this epoch is specified as a point to continue the discriminator training
            if (isinstance(discriminator_update_continue_epoch, list)) :
                if (len(discriminator_update_continue_epoch) > 0) :
                    if (int(discriminator_update_continue_epoch[0]) == current_epoch_index) :
                        discriminatorTrainingIterations = oldDiscriminatorTrainingIterations
                        with open(path_timestamps, "a+") as file :
                            file.write("After epoch "+str(current_epoch_index)+": Continuing discriminator update "+str(discriminator_update_continue_epoch))
                        discriminator_update_continue_epoch = discriminator_update_continue_epoch[1:len(discriminator_update_continue_epoch)]
            elif (discriminator_update_continue_epoch == current_epoch_index) :
                discriminatorTrainingIterations = oldDiscriminatorTrainingIterations
                with open(path_timestamps, "a+") as file :
                    file.write("After epoch "+str(current_epoch_index)+": Continuing discriminator update "+str(discriminator_update_continue_epoch))
            
            if (batchsize > 1 or (batchsize == 1 and current_epoch_index % 25 == 0)) :
			
                # Save models after an epoch to ensure that the current models are used for the translation examples
                save_model(d_model_A, "d_model_A")
                save_model(d_model_B, "d_model_B")
                if not (USE_COMPOSITE_PREDICTION) :
                    save_model(g_model_AtoB, "g_model_AtoB")
                    save_model(g_model_BtoA, "g_model_BtoA")
                save_model(c_model_AtoB, "c_model_AtoB")
                save_model(c_model_BtoA, "c_model_BtoA")
                with open(path_timestamps, "a+") as file :
                    file.write("End of epoch "+str(current_epoch_index)+" (Iteration "+str(i+1)+"): Saved models\n")
                
                # After iterating over the whole dataset, plot 11 example image translations (one is always fixed plus 10 randomly selected samples)
                # These examples are stored in individual subdirectories per epoch in "TargetDirectory/Plots/Epochs_Plots"
                if (PLOT_IMAGE_TRANSLATION_AFTER_EPOCH) :
                    if (current_epoch_index <= 25 or current_epoch_index % 25 == 0) :
                        current_epoch = str(current_epoch_index)
                        if (len(current_epoch) == 1) :
                            current_epoch = "00"+current_epoch
                        elif (len(current_epoch) == 2) :
                            current_epoch = "0"+current_epoch
                        if (batchsize==1) :
                            plot_image_translation([exampleA], [exampleB], title="Epoch_"+current_epoch+"_Translation_Example_")
                        else :
                            plot_image_translation([exampleA]+list(trainSketches[0:10]), [exampleB]+list(trainUIs[0:10]), title="Epoch_"+current_epoch+"_Translation_Example_")
                    
                # If a decaying learning rate is used, it gets updated in this section
                if (USE_DECAYING_LR) :
                    # If this parameter is set, the learning rate is updated in the specified frequency after finishing the first USE_DELAYED_DECAY epochs
                    # Additionally, the updated is applied as soon as the specified delay counter is hit, so that e.g.
                    # delay = 100 and frequency = 3 updates in iterations 100, 102, 105, 108 etc.
                    if (USE_DELAYED_DECAY > 0) :
                        if ((current_epoch_index > USE_DELAYED_DECAY and current_epoch_index % USE_DELAYED_DECAY + LEARNING_RATE_DECAY_FREQUENCY == 0) or
                            (current_epoch_index == USE_DELAYED_DECAY)):
                            # Update the learning rate of discriminator and composite model; the generator is trained via the composite model, thus does not need to be updated
                            current_lr = k.backend.get_value(d_model_A.optimizer.lr)
                            k.backend.set_value(d_model_A.optimizer.lr, current_lr * LEARNING_RATE_DECAY_FACTOR)
                            current_lr = k.backend.get_value(d_model_B.optimizer.lr)
                            k.backend.set_value(d_model_B.optimizer.lr, current_lr * LEARNING_RATE_DECAY_FACTOR)
                            
                            current_lr = k.backend.get_value(c_model_AtoB.optimizer.lr)
                            k.backend.set_value(c_model_AtoB.optimizer.lr, current_lr * LEARNING_RATE_DECAY_FACTOR)
                            current_lr = k.backend.get_value(c_model_BtoA.optimizer.lr)
                            k.backend.set_value(c_model_BtoA.optimizer.lr, current_lr * LEARNING_RATE_DECAY_FACTOR)
                    else :
                        if (current_epoch_index > 0 and current_epoch_index % LEARNING_RATE_DECAY_FREQUENCY == 0) :
                            # Update the learning rate of discriminator and composite model; the generator is trained via the composite model, thus does not need to be updated
                            current_lr = k.backend.get_value(d_model_A.optimizer.lr)
                            k.backend.set_value(d_model_A.optimizer.lr, current_lr * LEARNING_RATE_DECAY_FACTOR)
                            current_lr = k.backend.get_value(d_model_B.optimizer.lr)
                            k.backend.set_value(d_model_B.optimizer.lr, current_lr * LEARNING_RATE_DECAY_FACTOR)
                            
                            current_lr = k.backend.get_value(c_model_AtoB.optimizer.lr)
                            k.backend.set_value(c_model_AtoB.optimizer.lr, current_lr * LEARNING_RATE_DECAY_FACTOR)
                            current_lr = k.backend.get_value(c_model_BtoA.optimizer.lr)
                            k.backend.set_value(c_model_BtoA.optimizer.lr, current_lr * LEARNING_RATE_DECAY_FACTOR)
                    
                timestamp_summary_string = "Total time taken for epoch "+str(current_epoch_index)+" = " + str(time.process_time() - time_per_epoch) + " seconds.\n"
                with open(path_timestamps, "a+") as file :
                    file.write(timestamp_summary_string)
                time_per_epoch = time.process_time()
            current_epoch_index += 1

            # To keep the necessary VRAM in check, delete everything that is not necessary anymore and clear the garbage collector
            reset_keras()
        
        # Set the bool to false in case flipped labels were used last iteration
        bool_use_flipped_labels = False
    
        # Create some real samples
        # For the first 50 epochs, random noise is added to disguise that the real images originate from n/255
        # The outputs are clipped to [0..1], so that the random noise cannot 'break' an image out of its typical bounds
        if (current_epoch_index <= 50) :
            X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch, use_random_noise=True)
            X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch, use_random_noise=True)
        else :
            X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch, use_random_noise=False)
            X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch, use_random_noise=False)

        # Generate some fake samples based on the real images
        if (USE_COMPOSITE_PREDICTION) :
            X_fakeA, y_fakeA = generate_fake_samples_composite(c_model_BtoA, X_realB, X_realA, n_patch)  
            X_fakeB, y_fakeB = generate_fake_samples_composite(c_model_AtoB, X_realA, X_realB, n_patch)
        else :
            X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)  
            X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
        
        # Update the pool of fake images. This randomizes the training set even further and allows duplicates during training
        X_fakeA = update_image_pool(poolA, X_fakeA, max_pool_size=batchsize/2)
        X_fakeB = update_image_pool(poolB, X_fakeB, max_pool_size=batchsize/2)
        
        # Since there are some memory-related issues with Keras, the model.predict() methods and large batch sizes for these model sizes,
        # large batches may be split down to small minibatches that can be definitely be handled without raising any memory errors.
        if (USE_MINI_BATCHES) :
            g_losses_2, g_losses_1 = list(), list()
            dA_losses_2, dA_losses_1 = list(), list()
            dB_losses_2, dB_losses_1 = list(), list()
            
            num_images = len(X_realA)
            
            # If the train_on_batch() memory leak occurs, reset_keras() can be called earlier by modifying reset_keras_early_cap_*
            # Setting the cap to e.g. 4 means that after processing 4 minibatches in a single loop, keras is reset
            # Alternatively / Additionally, the reset_keras() can also be called between each generator update
            reset_keras_early_generator = 0
            reset_keras_early_discriminator = 0
            reset_keras_early_cap_generator = 20
            reset_keras_early_cap_discriminator = 20
            reset_keras_between_generator_updates = False
            
            # When using minibatches, e.g. because the GPU cannot handle large batchsizes, the actual batch is split into several MINI_BATCH_SIZE sized chunks
            # which are then processed individually
            for j in range(0, num_images, MINI_BATCH_SIZE) :
                mini_X_realA = X_realA[j:j+MINI_BATCH_SIZE]
                mini_X_realB = X_realB[j:j+MINI_BATCH_SIZE]
                mini_y_realA = y_realA[j:j+MINI_BATCH_SIZE]
                mini_y_realB = y_realB[j:j+MINI_BATCH_SIZE]
                
                # Update generator B->A via adversarial and cycle loss
                g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([mini_X_realB, mini_X_realA], [mini_y_realA, mini_X_realA, mini_X_realB, mini_X_realA])
                g_losses_2.append(g_loss2)
                
                reset_keras_early_generator += 1
                if (reset_keras_early_generator >= reset_keras_early_cap_generator) :
                    reset_keras()
                    reset_keras_early_generator = 0
            del mini_X_realA, mini_X_realB, mini_y_realA, mini_y_realB
            
            if (i % discriminatorTrainingIterations == 0) :
                counter_num_iterations_until_flip += 1
                if (counter_num_iterations_until_flip >= num_iterations_until_flip) :
                    # Flip the usage of flipped labels
                    # This value is kept also for training discriminator B
                    bool_use_flipped_labels = True
                    counter_num_iterations_until_flip = 0
            
                for j in range(0, num_images, MINI_BATCH_SIZE) :
                    mini_X_realA = X_realA[j:j+MINI_BATCH_SIZE]
                    mini_y_realA = y_realA[j:j+MINI_BATCH_SIZE]
                    
                    mini_X_fakeA = X_fakeA[j:j+MINI_BATCH_SIZE]
                    mini_y_fakeA = y_fakeA[j:j+MINI_BATCH_SIZE]
                    
                    if (bool_use_flipped_labels) :
                        dA_loss1 = d_model_A.train_on_batch(mini_X_realA, mini_y_fakeA)
                        dA_loss2 = d_model_A.train_on_batch(mini_X_fakeA, mini_y_realA)   
                    else :
                        dA_loss1 = d_model_A.train_on_batch(mini_X_realA, mini_y_realA)
                        dA_loss2 = d_model_A.train_on_batch(mini_X_fakeA, mini_y_fakeA)
                    dA_losses_1.append(dA_loss1)
                    dA_losses_2.append(dA_loss2)
                    
                    reset_keras_early_discriminator += 1
                    if (reset_keras_early_discriminator >= reset_keras_early_cap_discriminator) :
                        reset_keras()
                        reset_keras_early_discriminator = 0
                del mini_X_realA, mini_X_fakeA, mini_y_realA, mini_y_fakeA
            
            if (reset_keras_between_generator_updates) :
                reset_keras()
                
            reset_keras_early_generator = 0
            reset_keras_early_discriminator = 0
            
            for j in range(0, num_images, MINI_BATCH_SIZE) :
                mini_X_realA = X_realA[j:j+MINI_BATCH_SIZE]
                mini_X_realB = X_realB[j:j+MINI_BATCH_SIZE]
                mini_y_realA = y_realA[j:j+MINI_BATCH_SIZE]
                mini_y_realB = y_realB[j:j+MINI_BATCH_SIZE]
                
                g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([mini_X_realA, mini_X_realB], [mini_y_realB, mini_X_realB, mini_X_realA, mini_X_realB])
                g_losses_1.append(g_loss1)
                #print("g_loss1: Processed mini batch "+str(j+1)+" of "+str(num_images))
                
                reset_keras_early_generator += 1
                if (reset_keras_early_generator >= reset_keras_early_cap_generator) :
                    reset_keras()
                    reset_keras_early_generator = 0
            del mini_X_realA, mini_X_realB, mini_y_realA, mini_y_realB
               
            if (i % discriminatorTrainingIterations == 0) :
                for j in range(0, num_images, MINI_BATCH_SIZE) :
                    mini_X_realB = X_realB[j:j+MINI_BATCH_SIZE]
                    mini_y_realB = y_realB[j:j+MINI_BATCH_SIZE]
                    
                    mini_X_fakeB = X_fakeB[j:j+MINI_BATCH_SIZE]
                    mini_y_fakeB = y_fakeB[j:j+MINI_BATCH_SIZE]
                
                    if (bool_use_flipped_labels) :
                        dB_loss1 = d_model_B.train_on_batch(mini_X_realB, mini_y_fakeB)
                        dB_loss2 = d_model_B.train_on_batch(mini_X_fakeB, mini_y_realB)    
                    else :
                        dB_loss1 = d_model_B.train_on_batch(mini_X_realB, mini_y_realB)
                        dB_loss2 = d_model_B.train_on_batch(mini_X_fakeB, mini_y_fakeB)
                    dB_losses_1.append(dB_loss1)
                    dB_losses_2.append(dB_loss2)
                    
                    reset_keras_early_discriminator += 1
                    if (reset_keras_early_discriminator >= reset_keras_early_cap_discriminator) :
                        reset_keras()
                        reset_keras_early_discriminator = 0
                del mini_X_realB, mini_X_fakeB, mini_y_realB, mini_y_fakeB
                    
            # Average over all obtained losses as a representative for this iteration
            g_loss2 = sum(g_losses_2) / len(g_losses_2)
            g_loss1 = sum(g_losses_1) / len(g_losses_1)
            if (len(dA_losses_2) > 0) :
                dA_loss2 = sum(dA_losses_2) / len(dA_losses_2)
                dA_loss1 = sum(dA_losses_1) / len(dA_losses_1)
                dB_loss2 = sum(dB_losses_2) / len(dB_losses_2)
                dB_loss1 = sum(dB_losses_1) / len(dB_losses_1)
            
        else :
            # Update generator B->A via adversarial and cycle loss        
            g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		
            # Update discriminator for A -> [real/fake] every 'discriminatorTrainingIterations' iteration after the first one
            if (i % discriminatorTrainingIterations == 0) :
                counter_num_iterations_until_flip += 1
                if (counter_num_iterations_until_flip >= num_iterations_until_flip) :
                    # Flip the usage of flipped labels
                    # This value is kept also for training discriminator B
                    bool_use_flipped_labels = True
                    counter_num_iterations_until_flip = 0
                
                if (bool_use_flipped_labels) :
                    dA_loss1 = d_model_A.train_on_batch(X_realA, y_fakeA)
                    dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_realA)   
                else :
                    dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
                    dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		
            # Update generator A->B via adversarial and cycle loss
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		
            # Update discriminator for B -> [real/fake] every 'discriminatorTrainingIterations' iteration after the first one
            if (i % discriminatorTrainingIterations == 0) :
                if (bool_use_flipped_labels) :
                    dB_loss1 = d_model_B.train_on_batch(X_realB, y_fakeB)
                    dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_realB)    
                else :
                    dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
                    dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)                

        # Dynamically update the number of generator updates between discriminator updates
        # A multiple of 60 is recommended because it can be divided by 2, 3, 4, 5 and 6. This might change for larger batch sizes.
        if (i % dynamicUpdateIterations == 0) :
            if (DYNAMIC_UPDATE_STRATEGY == 0) :
                # If one of the two discriminators is training too fast, slow them down (if possible; for now limited to 10 generator updates in between)
                if (dA_loss1 * 1.5 < g_loss1 or dB_loss1 * 1.5 < g_loss1) :
                    if (discriminatorTrainingIterations < MAX_DISCRIMINATOR_TRAINING_ITERATIONS) :
                        discriminatorTrainingIterations += 1
                        with open(path_timestamps, "a+") as file :
                            file.write("Iteration "+str(i+1)+": Increased discriminatorTrainingIterations to "+str(discriminatorTrainingIterations)+".\n")
                # If the generator is training a lot faster than a discriminator, speed up the discriminator training (if possible)
                elif (dA_loss1 > g_loss1 * 1.5 or dB_loss1 > g_loss1 * 1.5) :
                    if (discriminatorTrainingIterations > 1) :
                        discriminatorTrainingIterations -= 1
                        with open(path_timestamps, "a+") as file :
                            file.write("Iteration "+str(i+1)+": Decreased discriminatorTrainingIterations to "+str(discriminatorTrainingIterations)+".\n")
                """
                # As the interesting direction is sketches -> UIs, the dynamic update is based on that direction for now. However, the other
                # direction can be included as well by just un-commenting this block
                elif (dA_loss2 * 1.5 < g_loss2 or dB_loss2 * 1.5 < g_loss2) :
                    if (discriminatorTrainingIterations < 10) :
                        discriminatorTrainingIterations += 1
                        with open(path_timestamps, "a+") as file :
                            file.write("Iteration "+str(i+1)+": Increased discriminatorTrainingIterations to "+str(discriminatorTrainingIterations)+".\n")
                # If the generator is training a lot faster than a discriminator, speed up the discriminator training (if possible)
                elif (dA_loss2 > g_loss2 * 1.5 or dB_loss2 > g_loss2 * 1.5) :
                    if (discriminatorTrainingIterations > 1) :
                        discriminatorTrainingIterations -= 1
                        with open(path_timestamps, "a+") as file :
                            file.write("Iteration "+str(i+1)+": Decreased discriminatorTrainingIterations to "+str(discriminatorTrainingIterations)+".\n")
                """
            elif (DYNAMIC_UPDATE_STRATEGY == 1) :
                # Ensure that the current loss values are larger than 0 to avoid exceptions
                if (dA_loss1 > 0.1 and dB_loss1 > 0.1 and g_loss1 > 0.1) :
                    # if the gradient is of the discriminator dropped much more than the generator gradient, slow down the discriminator 
                    if (dA_loss1_old / dA_loss1 - 0.25 > g_loss1_old / g_loss1 or dB_loss1_old / dB_loss1 - 0.25 > g_loss1_old / g_loss1) :
                        discriminatorTrainingIterations += 1
                        with open(path_timestamps, "a+") as file :
                            file.write("Iteration "+str(i+1)+": Increased discriminatorTrainingIterations to "+str(discriminatorTrainingIterations)+".\n")
                    # Otherwise, if the generator gradient improved much more than the discriminator one, speed up the discriminator
                    elif (dA_loss1_old / dA_loss1 < g_loss1_old / g_loss1 - 0.25 or dB_loss1_old / dB_loss1 < g_loss1_old / g_loss1 - 0.25) :
                        discriminatorTrainingIterations -= 1
                        with open(path_timestamps, "a+") as file :
                            file.write("Iteration "+str(i+1)+": Decreased discriminatorTrainingIterations to "+str(discriminatorTrainingIterations)+".\n")
                else :
                    if ((dA_loss1 <= 0.1 or dB_loss1 <= 0.1) and g_loss1 > 0.1) :
                        # If the generator training was really slow, probably because the discriminator is very good already, give the generator more training time
                        if (g_loss1_old / g_loss1 > 0.9) :
                            discriminatorTrainingIterations += 1
                            with open(path_timestamps, "a+") as file :
                                file.write("Iteration "+str(i+1)+": Increased discriminatorTrainingIterations to "+str(discriminatorTrainingIterations)+".\n")
                    elif ((dA_loss1 > 0.1 or dB_loss1 > 0.1) and g_loss1 <= 0.1) :
                        # Otherwise, since the generator is already really strong, increase the discriminator's update frequency
                        discriminatorTrainingIterations -= 1
                        with open(path_timestamps, "a+") as file :
                            file.write("Iteration "+str(i+1)+": Decreased discriminatorTrainingIterations to "+str(discriminatorTrainingIterations)+".\n")
                dA_loss1_old = dA_loss1
                dB_loss1_old = dB_loss1
                g_loss1_old = g_loss1

        # summarize performance
        print('>%d/%d: dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, n_steps, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
        print("Total time taken for iteration "+str(i+1)+" = " + str(time.process_time() - time_iteration_start) + " seconds.")
        
        summary_string = "%d/%d: dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]\n" % (i+1, n_steps, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2)
        with open(path, "a+") as file :
            file.write(summary_string)

        # Resetting the keras backend is necessary to avoid a
        # ResourceExhaustedException due to model.predict()
        # cf. e.g. https://github.com/tensorflow/tensorflow/issues/33030
        reset_keras()

    # After completion of the training, show the required training time, save models and print final translation examples
    print("Total time taken for training the GAN = " + str(time.process_time() - time_before_training) + " seconds; number of epochs = "+str(current_epoch_index)+".\nSaving the models...")
    with open(path, "a+") as file :
            file.write("Total time taken for training the GAN = " + str(time.process_time() - time_before_training) + " seconds; number of epochs = "+str(current_epoch_index)+".")
            
    if not (USE_COMPOSITE_PREDICTION) :
        save_model(g_model_AtoB, "g_model_AtoB_trained_final")
        save_model(g_model_BtoA, "g_model_BtoA_trained_final")
    save_model(d_model_A, "d_model_A_trained_final")
    save_model(d_model_B, "d_model_B_trained_final")
    save_model(c_model_AtoB, "c_model_AtoB_trained_final")
    save_model(c_model_BtoA, "c_model_BtoA_trained_final")
    
    if (PLOT_IMAGE_TRANSLATION_AFTER_EPOCH) :
        current_epoch = str(current_epoch_index)
        if (len(current_epoch) == 1) :
            current_epoch = "00"+current_epoch
        elif (len(current_epoch) == 2) :
            current_epoch = "0"+current_epoch
        if (isinstance(trainSketches, str)) :
            plot_image_translation([exampleA], [exampleB], title="Epoch_"+current_epoch+"_Translation_Example_")
        else :
            plot_image_translation([exampleA]+list(trainSketches[0:10]), [exampleB]+list(trainUIs[0:10]), title="Epoch_"+current_epoch+"_Translation_Example_", use_final_models=True)

"""
Translates an image or a list of images from one domain into another domain.
Technically, this method is almost equivalent to generate_fake_samples(), except that
no labels are created. Also, this method name is slightly more appropriate for evaluations.

Parameters:
    image: the source image that should be translated, e.g. a hand-drawn sketch; requires either an image or its path or a list of either.
    generator_model: a trained generator model to translate the image into the desired domain; requires either the model or its file-name
                     the default translation cycle is A (sketches) to B (UIs) and vice versa
                     
Returns:
    A list containing the image(s) translated to the new domain. 
    This is because keras' predict() method always returns a list, even for a single element.
"""
def translate_image(image, generator_model) :
    if (isinstance(generator_model, str)) :
        generator_model = load_model(generator_model)
        
    # This check only works on actual lists
    # If the data was submitted as a numpy.array, it is not detected
    if (isinstance(image, list)) :
        if (isinstance(image[0], str)) :
            image = data_utils.load_images(image)
    else :
        if (isinstance(image, str)) :
            image = data_utils.load_image(image)
            
        # In case the list of images was an numpy.array, everything is fine, but this block is executed
        # However if the input was a single image, it has be reshaped to e.g. (1, 256, 256, 3) as 
        # keras' predict() method requires a 4-dimensional input
        shape = image.shape
        if (len(shape) == 3) :
            image = np.reshape(image, (1, shape[0], shape[1], shape[2]))
            
    # Normalize images to [-1,..1] because the generator uses tanh as activation
    # If [0..255]
    if (np.any(image > 1)) :
        image = (image - 127.5) / 127.5 #np.asarray([data_utils.normalize_image(images[i]) for i in range(len(images)), low=-1])
    # If already [-1..1], it's alright
    elif (np.any(image < 0)) :
        pass
    # Else normalize from [0..1] to [-1..1]
    else :
        image = (image * 2) - 1

    X = generator_model.predict(image)
    # Normalize image to [0..1]
    X = (X + 1) / 2
        
    return X

"""
An alternative to the 'translate_image()' method which uses the composite model instead of the generator model
The overall structure and way of functionality remains the same
For a later stage of actual prediction, it should suffice to provide only the hand-drawn sketches for domain A and noise for domain B 
As an example, imageB = np.reshape(np.random.rand(256 * 256 * 3), (256, 256, 3))
"""
def translate_image_composite(imageA, imageB, c_model) :
    if (isinstance(imageA, str)) :
        if (os.path.isdir(imageA)) :
            for root, dirs, files in os.walk(imageA) :
                for file in files :
                    print("Found file "+str(file))
                    imageA = file
    if (isinstance(imageB, str)) :
        if (os.path.isdir(imageB)) :
            for root, dirs, files in os.walk(imageB) :
                for file in files :
                    print("Found file "+str(file))
                    imageB = file

    #print(imageA, imageB)
    if (isinstance(imageA, list)) :
        if (isinstance(imageA[0], str)) :
            imageA = data_utils.load_images(imageA)
            
    elif (isinstance(imageA, str)) :
        imageA = data_utils.load_image(imageA)
    
    imageA = np.asarray(imageA)
    # Normalize images to [-1,..1] because the generator uses tanh as activation
    # If [0..255]
    if (np.any(imageA > 1)) :
        imageA = (imageA - 127.5) / 127.5 #np.asarray([data_utils.normalize_image(images[i]) for i in range(len(images)), low=-1])
    # If already [-1..1], it's alright
    elif (np.any(imageA < 0)) :
        pass
    # Else normalize from [0..1] to [-1..1]
    else :
        imageA = (imageA * 2) - 1
        
    if (isinstance(imageB, list)) :
        if(isinstance(imageB[0], str)) :
            imageB = data_utils.load_images(imageB)
        else :
            imageB = data_utils.load_image(imageB)    
            
    elif (isinstance(imageB, str)) :
        imageB = data_utils.load_image(imageB)

    # Normalize images to [-1,..1] because the generator uses tanh as activation
    # If [0..255]
    if (np.any(imageB > 1)) :
        imageB = (imageB - 127.5) / 127.5 #np.asarray([data_utils.normalize_image(images[i]) for i in range(len(images)), low=-1])
    # If already [-1..1], it's alright
    elif (np.any(imageB < 0)) :
        pass
    # Else normalize from [0..1] to [-1..1]
    else :
        imageB = (imageB * 2) - 1
        
    # If only a single image is used, ensure that it is still in the format of [1, 256, 256, 3]
    if (len(imageA.shape) == 3) :
        imageA = np.reshape(imageA, (1, 256, 256, 3))
        
    if (len(imageB.shape) == 3) :
        imageB = np.reshape(imageB, (1, 256, 256, 3))
        
    # Predict the images
    if (isinstance(c_model, str)) :
        # If the model shall be loaded from a file
        model = load_model(c_model)
        X = model.predict_on_batch([imageA, imageB]) 

    else :
        # If the model is handed over as a parameter            
        X = c_model.predict_on_batch([imageA, imageB]) 
        
    # Keep the results of forward and backward pass
    X1, X2, X3 = X[1], X[2], X[3]
    # Denormalize the images from [-1,1] to [0,1]
    X1 = (X1 + 1) / 2
    X2 = (X2 + 1) / 2
    X3 = (X3 + 1) / 2
	
    return X1, X2, X3

"""
Plots and stores the plot of test translations.
Parameters:
    image_src, imageA1, imageA2, imageA3: The source image used for translation and the three model outputs provided by translate_image_composite()
    image_target: The noisy image used for calculations (out of curiosity)
    imageB1, imageB2, imageB3: Although probably not necessary, these images are obtained by reversing the parameter order in the translate_image_composite() call
    index: Optional parameter. Use it to add a number to the file name, e.g. if multiple images are translated in a row.
Returns:
    Nothing. All results are written to disc. 
"""
def plot_test_translation_results(image_src, imageA1, imageA2, imageA3, image_target, imageB1, imageB2, imageB3, path="", index=-1) :
    plt.suptitle("Image Translation - Above: (Image, Noise), Below: (Noise, Image)")
    # AtoB - this should be the correct image order
    plt.subplot(2, 4, 1)
    plt.gca().set_title('Source Image')
    plt.axis('off')
    plt.imshow(image_src)
    
    plt.subplot(2, 4, 2)
    plt.axis('off')
    plt.gca().set_title('Identity')
    plt.imshow(imageA1)
    
    plt.subplot(2, 4, 3)
    plt.axis('off')
    plt.gca().set_title('Forward Cycle')
    plt.imshow(imageA2)
    
    plt.subplot(2, 4, 4)
    plt.axis('off')
    plt.gca().set_title('Backward Cycle')
    plt.imshow(imageA3)
    
    # BtoA - out of curiosity, what does the inverse order look like?
    plt.subplot(2, 4, 5)
    plt.gca().set_title('Noise')
    plt.axis('off')
    plt.imshow(image_target)
    
    plt.subplot(2, 4, 6)
    plt.axis('off')
    plt.gca().set_title('Identity')
    plt.imshow(imageB1)
    
    plt.subplot(2, 4, 7)
    plt.axis('off')
    plt.gca().set_title('Forward Cycle')
    plt.imshow(imageB2)
    
    plt.subplot(2, 4, 8)
    plt.axis('off')
    plt.gca().set_title('Backward Cycle')
    plt.imshow(imageB3)
	
    if (path=="") :
        path = os.path.join(os.getcwd(), "Plots")
        if not (os.path.exists(path)) :
            os.mkdir(path)
        print("No savepath specified. Using the default directory "+path+" as savepath.")
    if (index == -1) :
        # Using the default title
        path = os.path.join(path, "Test_Image_Translation.pdf")
    else :
        path = os.path.join(path, "Test_Image_Translation-"+str(index)+".pdf")
    plt.savefig(path)

"""
Perform a test translation on test images.
Parameters:
    images: A single image of a list of images that serve as test images for the translation
    c_model: The composite model to use for translation. May be either the already loaded model or the filename.
    direction: Optional. If set, the results directory will be named with this as an appendix, e.g. 'AtoB' leads to 'Test_Translations_AtoB'
               If the generator models are used for plotting, the c_model parameter does not matter if this parameter is set.
Returns:
    Nothing. Calls plot_test_translation_results() which creates plots and writes the results to disc.
"""
def test_translation(images, c_model, direction="") :
    if (isinstance(images, list)) :
        if (isinstance(images[0], str)) :
            images = data_utils.load_images(images)
            # If the result is a list of shape [x, 1, 256, 256, 3], it is reshaped to [x, 256, 256, 3]
            if (len(images.shape) > 4) :
                images = [np.reshape(np.asarray(image), (images.shape[2], images.shape[3], 3)) for image in images]
    elif (isinstance(images, str)) :
        images = data_utils.load_image(images)
        if (len(images.shape) > 3) :
            images = [np.reshape(np.asarray(images), (images.shape[1], images.shape[2], 3))]
    elif (isinstance(images, np.ndarray)) :
	    # In case the input is stored as a numpy array of shape (num_images,), load images
        if (len(images.shape) == 1) :
            images = data_utils.load_images(list(images))
        if (len(images.shape) > 4) :
            images = [np.reshape(np.asarray(image), (images.shape[2], images.shape[3], 3)) for image in images]
    
    if (len(target_directory) > 0) :
        path = os.path.join(os.getcwd(), target_directory)
        path = os.path.join(path, "Plots")
        if not (os.path.exists(path)) :
            os.mkdir(path)
    else :
        path = os.path.join(os.getcwd(), "Plots")
        if not (os.path.exists(path)) :
            os.mkdir(path)
    if (direction == "") :
        path = os.path.join(path, "Test_Translations")
    else :
        path = os.path.join(path, "Test_Translations_"+direction)
    if not (os.path.exists(path)) :
        os.mkdir(path)
    
    import warnings
    warnings.filterwarnings("ignore")
    
    image_index = 0
    
    if (USE_COMPOSITE_PREDICTION) :
        if (isinstance(c_model, str)) :
            c_model = load_model(c_model)
        noise = np.reshape(np.random.rand(256 * 256 * 3), (256, 256, 3))
        
        for image in images :
            imageA1, imageA2, imageA3 = translate_image_composite(image, noise, c_model)
            imageB1, imageB2, imageB3 = translate_image_composite(noise, image, c_model)
            
            if (len(imageA1.shape) > 3) :
                imageA1 = np.reshape(imageA1, (imageA1.shape[1], imageA1.shape[2], 3))
                imageA2 = np.reshape(imageA2, (imageA2.shape[1], imageA2.shape[2], 3))
                imageA3 = np.reshape(imageA3, (imageA3.shape[1], imageA3.shape[2], 3))
                imageB1 = np.reshape(imageB1, (imageB1.shape[1], imageB1.shape[2], 3))
                imageB2 = np.reshape(imageB2, (imageB2.shape[1], imageB2.shape[2], 3))
                imageB3 = np.reshape(imageB3, (imageB3.shape[1], imageB3.shape[2], 3))

            plot_test_translation_results(image, imageA1, imageA2, imageA3, noise, imageB1, imageB2, imageB3, path=path, index=image_index)
            image_index += 1
            
    else :
        if (direction == "BtoA") :
            try :
                g_model = load_model("g_model_BtoA_trained_final")
                g_model2 = load_model("g_model_AtoB_trained_final")
            except OSError :
                g_model = load_model("g_model_BtoA")
                g_model2 = load_model("g_model_AtoB")
        else :
            try :
                g_model = load_model("g_model_AtoB_trained_final")
                g_model2 = load_model("g_model_BtoA_trained_final")
            except OSError :
                g_model = load_model("g_model_AtoB")
                g_model2 = load_model("g_model_BtoA")
        
        for image in images :
            imageA1 = translate_image(image, g_model)
            imageA2 = translate_image(image, g_model2)
            if (len(imageA2.shape) > 3) :
                imageA2 = np.reshape(imageA2, (imageA2.shape[1], imageA2.shape[2], 3))
            imageA2 = translate_image(imageA2, g_model)
        
            if (len(imageA1.shape) > 3) :
                imageA1 = np.reshape(imageA1, (imageA1.shape[1], imageA1.shape[2], 3))
            if (len(imageA2.shape) > 3) :
                imageA2 = np.reshape(imageA2, (imageA2.shape[1], imageA2.shape[2], 3))
        
            num_plots = 3
                    
            plt.subplot(1, num_plots, 1)
            plt.gca().set_title('Source Image')
            plt.axis('off')
            plt.imshow(image)
            
            plt.subplot(1, num_plots, 2)
            plt.axis('off')
            plt.gca().set_title('Proposal 1')
            plt.imshow(imageA1)
            
            plt.subplot(1, num_plots, 3)
            plt.axis('off')
            plt.gca().set_title('Proposal 2')
            plt.imshow(imageA2)
            
            savepath_pdf = os.path.join(path, "Test_Image_Translation-"+str(image_index)+".pdf")
            savepath_png = os.path.join(path, "Test_Image_Translation-"+str(image_index)+".png")
            plt.savefig(savepath_pdf)
            plt.savefig(savepath_png)
            plt.clf()
            
            image_index += 1
           
    print("Finished test translation(s). Results were stored in "+path)

"""
Computes translation plots and stores them to disk.

Parameters:
    - paths_sketches: The paths to load sketches from or already loaded sketches as numpy arrays
    - paths_uis: The corresponding UIs. Even if the UIs are not matched to the sketches, index i of both lists will be shown in plot i.
    - title: Optionally, a title may be specified. If not, a default name is used (overwriting existing plots!)
    - printDebugInfo: If desired, some debug information could be printed.
    - use_final_models: If desired, this method may use the fully trained models instead of intermediate models (as called after each epoch)
    
Returns:
    - Nothing. All results are written to disk.
"""
def plot_image_translation(paths_sketches, paths_uis, title="", printDebugInfo=False, use_final_models=False) :
    if (printDebugInfo) :
        print("Using "+paths_sketches+" and "+paths_uis+" as images for an image translation.")
    if (USE_COMPOSITE_PREDICTION) :
        if (use_final_models) :
            c_model_AtoB = load_model("c_model_AtoB_trained_final")
            c_model_BtoA = load_model("c_model_BtoA_trained_final")
        else :
            c_model_AtoB = load_model("c_model_AtoB")
            c_model_BtoA = load_model("c_model_BtoA")
    else :
        if (use_final_models) :
            g_model_AtoB = load_model("g_model_AtoB_trained_final")
            g_model_BtoA = load_model("g_model_BtoA_trained_final")
        else :
            g_model_AtoB = load_model("g_model_AtoB")
            g_model_BtoA = load_model("g_model_BtoA")

    delimiter = "/"
    if (sys.platform == "win32") :
        delimiter = "\\"

    for i in range(len(paths_sketches)) :
        path_sketch = paths_sketches[i]
        path_ui = paths_uis[i]

        if (USE_COMPOSITE_PREDICTION) :
            print(path_sketch, path_ui)
            imageAtoB1, imageAtoB2, imageAtoB3 = translate_image_composite(path_sketch, path_ui, c_model_AtoB)
            imageBtoA1, imageBtoA2, imageBtoA3 = translate_image_composite(path_ui, path_sketch, c_model_BtoA)
            
        else :
            imageAtoB1 = translate_image(path_sketch, g_model_AtoB)
            imageBtoA1 = translate_image(imageAtoB1, g_model_BtoA)
            
            imageBtoA2 = translate_image(path_ui, g_model_BtoA)
            imageAtoB2 = translate_image(imageBtoA2, g_model_AtoB)
            
            # Visualization of the forward cycle
            imageAtoB3 = translate_image(path_sketch, g_model_BtoA)
            imageBtoA3 = translate_image(imageAtoB3, g_model_AtoB)
            
            imageBtoA4 = translate_image(path_ui, g_model_AtoB)
            imageAtoB4 = translate_image(imageBtoA4, g_model_BtoA)
                
            if (len(imageAtoB4.shape) > 3) :
                imageAtoB4 = imageAtoB4.reshape(imageAtoB4.shape[1], imageAtoB4.shape[2], imageAtoB4.shape[3])
            if (len(imageBtoA4.shape) > 3) :
                imageBtoA4 = imageBtoA4.reshape(imageBtoA4.shape[1], imageBtoA4.shape[2], imageBtoA4.shape[3]) 
    
        if (len(imageAtoB1.shape) > 3) :
            imageAtoB1 = imageAtoB1.reshape(imageAtoB1.shape[1], imageAtoB1.shape[2], imageAtoB1.shape[3])
        if (len(imageAtoB2.shape) > 3) :
            imageAtoB2 = imageAtoB2.reshape(imageAtoB2.shape[1], imageAtoB2.shape[2], imageAtoB2.shape[3])
        
        if (len(imageBtoA1.shape) > 3) :
            imageBtoA1 = imageBtoA1.reshape(imageBtoA1.shape[1], imageBtoA1.shape[2], imageBtoA1.shape[3]) 
        if (len(imageBtoA2.shape) > 3) :
            imageBtoA2 = imageBtoA2.reshape(imageBtoA2.shape[1], imageBtoA2.shape[2], imageBtoA2.shape[3]) 
            
        if (len(imageAtoB3.shape) > 3) :
            imageAtoB3 = imageAtoB3.reshape(imageAtoB3.shape[1], imageAtoB3.shape[2], imageAtoB3.shape[3])
        if (len(imageBtoA3.shape) > 3) :
            imageBtoA3 = imageBtoA3.reshape(imageBtoA3.shape[1], imageBtoA3.shape[2], imageBtoA3.shape[3]) 

        if (isinstance(path_sketch, str)) :
            titleA = path_sketch[path_sketch.rfind(delimiter)+1:]
        else :
            titleA = "Input Image"
        if (isinstance(path_ui, str)) :
            titleB = path_ui[path_ui.rfind(delimiter)+1:]
        else :
            titleB = "Target Image"
    
        if (USE_COMPOSITE_PREDICTION) :
            plt.suptitle("Sketch: "+titleA+"; UI: "+titleB)
            # AtoB
            plt.subplot(2, 4, 1)
            plt.gca().set_title('Sketch')
            plt.axis('off')
            plt.imshow(data_utils.load_image(path_sketch))
            
            plt.subplot(2, 4, 2)
            plt.axis('off')
            plt.gca().set_title('Identity')
            plt.imshow(imageAtoB1)
            
            plt.subplot(2, 4, 3)
            plt.axis('off')
            plt.gca().set_title('Forward Cycle')
            plt.imshow(imageAtoB2)
            
            plt.subplot(2, 4, 4)
            plt.axis('off')
            plt.gca().set_title('Backward Cycle')
            plt.imshow(imageAtoB3)
            
            # BtoA
            plt.subplot(2, 4, 5)
            plt.gca().set_title('UI')
            plt.axis('off')
            plt.imshow(data_utils.load_image(path_ui))
            
            plt.subplot(2, 4, 6)
            plt.axis('off')
            plt.gca().set_title('Identity')
            plt.imshow(imageBtoA1)
            
            plt.subplot(2, 4, 7)
            plt.axis('off')
            plt.gca().set_title('Forward Cycle')
            plt.imshow(imageBtoA2)
            
            plt.subplot(2, 4, 8)
            plt.axis('off')
            plt.gca().set_title('Backward Cycle')
            plt.imshow(imageBtoA3)
            
        else :
            # Row 1
            plt.suptitle("Sketch: "+titleA+"; UI: "+titleB)
            plt.subplot(3, 4, 1)
            plt.gca().set_title('Sketch')
            plt.axis('off')
            if (isinstance(path_sketch, str)) :
                plt.imshow(data_utils.load_image(path_sketch))
            else :
                plt.imshow(path_sketch)
            
            plt.subplot(3, 4, 3)
            plt.gca().set_title('Original UI')
            plt.axis('off')
            if (isinstance(path_ui, str)) :
                plt.imshow(data_utils.load_image(path_ui))
            else :
                plt.imshow(path_ui)
        
            # Row 2
            plt.subplot(3, 4, 5)
            plt.axis('off')
            plt.gca().set_title('Forward')
            plt.imshow(imageAtoB1)
        
            plt.subplot(3, 4, 6)
            plt.axis('off')
            plt.gca().set_title('Backward')
            plt.imshow(imageBtoA1)
            
            plt.subplot(3, 4, 7)
            plt.axis('off')
            plt.gca().set_title('Forward')
            plt.imshow(imageBtoA2)
        
            plt.subplot(3, 4, 8)
            plt.gca().set_title('Backward')
            plt.axis('off')
            plt.imshow(imageAtoB2)
            
            # Row 3
            plt.subplot(3, 4, 9)
            plt.axis('off')
            plt.gca().set_title('Identity')
            plt.imshow(imageAtoB3)
        
            plt.subplot(3, 4, 10)
            plt.axis('off')
            plt.gca().set_title('Forward')
            plt.imshow(imageBtoA3)
        
            plt.subplot(3, 4, 11)
            plt.axis('off')
            plt.gca().set_title('Identity')
            plt.imshow(imageBtoA4)
        
            plt.subplot(3, 4, 12)
            plt.gca().set_title('Forward')
            plt.axis('off')
            plt.imshow(imageAtoB4)

        if (len(target_directory) > 0) :
            path = os.path.join(os.getcwd(), target_directory)
            if not (os.path.exists(path)) :
                os.mkdir(path)
            path = os.path.join(path, 'Plots')
        else :
            path = os.path.join(os.getcwd(), "Plots")
        if not (os.path.exists(path)) :
            print("Created directory "+path+" to store loss evaluation plot.")
            os.mkdir(path)
    
        if ("Epoch_" in title or "epoch_" in title) :
            path = os.path.join(path, "Epochs_Plots")
            if not (os.path.exists(path)) :
                print("Created directory "+path+" to store epoch plots.")
                os.mkdir(path)
            path = os.path.join(path, title[0:9])
            if not (os.path.exists(path)) :
                os.mkdir(path)
	
        if (title == "") :
            print("No title was specified, using the default plot title.")
            path = os.path.join(path, "Example_Image_Translation.pdf")
        else :
            if (".pdf" in title) :
                path = os.path.join(path, title)
            else :
                path = os.path.join(path, title+str(i)+".pdf")
        
        plt.savefig(path)
        plt.clf()
        if (printDebugInfo) :
            print("Saved a image translation plot to: "+path)