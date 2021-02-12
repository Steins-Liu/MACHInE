#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Activation, Dense, Dropout, MaxPooling3D, Conv3D, Convolution2D, MaxPooling2D, SpatialDropout2D, Lambda, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AlphaDropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras.layers import ELU
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, Callback, History
from tensorflow.keras import metrics
from model_utils.functions import mse_loss
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

#################IMAGE TRANSFORMER######################
"""
## Implement patch creation as a layer
"""
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

"""
## Implement the patch encoding layer
The `PatchEncoder` layer will linearly transform a patch by projecting it into a
vector of size `projection_dim`. In addition, it adds a learnable position
embedding to the projected vector.
"""
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

"""
## Implement multilayer perceptron (MLP)
"""
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def build_transformer(w, h, d, s):
    ## Configure the hyperparameters
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 100
    image_size = 162  # We'll resize input images to this size
    patch_size = 6  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
    projection_dim * 2,
    projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 5
    mlp_head_units = [256, 256]  # Size of the dense layers of the final classifier
    image_inputs = Input(shape=(h, w, d), name = 'image_input')
    sensor_inputs = Input(shape=(1,), name= 'sensor_input')
    # Create patches.
    patches = Patches(patch_size)(image_inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)
    # Classify outputs.
    angle_out = Dense(1, activation='linear', name= 'angle_out')(features)
    throttle_out = Dense(1, activation='linear', name= 'throttle_out')(features)
    model = Model(inputs=[image_inputs, sensor_inputs], outputs=[angle_out, throttle_out])
    optimizer = optimizers.Adam(lr = 0.0001)	
    model.compile(loss= {'angle_out': 'mean_squared_error', 
    			 'throttle_out': 'mean_squared_error'},
                  optimizer=optimizer,
                   metrics={'angle_out': ['mse'], 'throttle_out': ['mse']}, loss_weights=[1, 1])
    model.summary()
    return model

#################LRCN with Sensor Input#################

def build_lrcn_sensor(w, h, d, s):
    image_inputs = Input(shape=(s, h, w, d), name = 'image_input')
    sensor_inputs = Input(shape=(1,), name= 'sensor_input')
    x = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(image_inputs)
    x = TimeDistributed(Convolution2D(filters=32, kernel_size=(3, 3), 
        strides=(2,2),data_format='channels_last', input_shape=(h, w, d), activation='relu', name = 'conv1'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Convolution2D(filters=32, kernel_size=(3, 3),
        strides=(2,2),data_format='channels_last', input_shape=(h, w, d), activation='relu', name = 'conv2'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Convolution2D(filters=64, kernel_size=(3, 3),
        strides=(2,2),data_format='channels_last', input_shape=(h, w, d), activation='relu', name = 'conv3'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Convolution2D(filters=64, kernel_size=(3, 3),
        strides=(1,1),data_format='channels_last', input_shape=(h, w, d), activation='relu', name = 'conv4'))(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Convolution2D(filters=64, kernel_size=(3, 3),
        strides=(1,1),data_format='channels_last', input_shape=(h, w, d), activation='relu', name = 'conv5'))(x)
    x = BatchNormalization()(x)
    # Fully connected layer
    x = TimeDistributed(Flatten())(x)
    x1 = LSTM(512, return_sequences=False, stateful=False)(x)
    a = Dense(256, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense1")(x1)
    a = BatchNormalization()(a)
    a = Dense(128, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense2")(a)
    a = BatchNormalization()(a)
    a = Dense(64, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense3")(a)
    a = BatchNormalization()(a)
    x2 = x1 #Concatenate()([x1, sensor_inputs])
    b = Dense(256, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense1")(x2)
    b = BatchNormalization()(b)
    b = Dense(128, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense2")(b)
    b = BatchNormalization()(b)
    b = Dense(64, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense3")(b)
    b = BatchNormalization()(b)
    angle_out = Dense(1, activation='linear', name= 'angle_out')(a)
    throttle_out = Dense(1, activation='linear', name= 'throttle_out')(b)
    model = Model(inputs=[image_inputs, sensor_inputs], outputs=[angle_out, throttle_out])
    optimizer = optimizers.Adam(lr = 0.0001)	
    model.compile(loss= {'angle_out': 'mean_squared_error', 
    			 'throttle_out': 'mean_squared_error'},
                  optimizer=optimizer,
                   metrics={'angle_out': ['mse'], 'throttle_out': ['mse']}, loss_weights=[1, 1])
    model.summary()
    return model

#################3D CNN with Sensor Input#################

def build_3d_sensor(w, h, d, s):
    image_inputs = Input(shape=(s, h, w, d), name = 'image_input')
    sensor_inputs = Input(shape=(1,), name= 'sensor_input')
    x = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0))(image_inputs)
    x = Conv3D(
        filters=32, kernel_size=(3,3,3), strides=(1,2,2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name = 'conv1')(x)
    x = BatchNormalization()(x)
    x = Conv3D(
        filters=32, kernel_size=(3,3,3), strides=(1,2,2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name = 'conv2')(x)
    x = BatchNormalization()(x)
    x = Conv3D(
        filters=64, kernel_size=(3,3,3), strides=(1,2,2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name = 'conv3')(x)
    x = BatchNormalization()(x)
    x = Conv3D(
        filters=64, kernel_size=(3,3,3), strides=(1,2,2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name = 'conv4')(x)
    x = BatchNormalization()(x)
    x = Conv3D(
        filters=64, kernel_size=(3,3,3), strides=(1,2,2),
        data_format='channels_last', padding='same',
        input_shape=(s, h, w, d), activation='relu', name = 'conv5')(x)
    x = BatchNormalization()(x)
    # Fully connected layer
    x = TimeDistributed(Flatten())(x)
    x1 = LSTM(512, return_sequences=False, stateful=False)(x)
    a = Dense(256, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense1")(x1)
    a = BatchNormalization()(a)
    a = Dense(128, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense2")(a)
    a = BatchNormalization()(a)
    a = Dense(64, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "angledense3")(a)
    a = BatchNormalization()(a)
    x2 = x1 #Concatenate()([x1, sensor_inputs])
    b = Dense(256, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense1")(x2)
    b = BatchNormalization()(b)
    b = Dense(128, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense2")(b)
    b = BatchNormalization()(b)
    b = Dense(64, activation= 'elu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.0), name = "throttledense3")(b)
    b = BatchNormalization()(b)
    angle_out = Dense(1, activation='linear', name= 'angle_out')(a)
    throttle_out = Dense(1, activation='linear', name= 'throttle_out')(b)
    model = Model(inputs=[image_inputs, sensor_inputs], outputs=[angle_out, throttle_out])
    optimizer = optimizers.adam(lr = 0.0001)	
    model.compile(loss= {'angle_out': 'mean_squared_error', 
    			 'throttle_out': 'mean_squared_error'},
                  optimizer=optimizer,
                   metrics={'angle_out': ['mse'], 'throttle_out': ['mse']}, loss_weights=[1, 1])
    model.summary()
    return model


