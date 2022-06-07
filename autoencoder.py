import numpy as np
import keras

from variables import WIDTH, HEIGHT, NUM_COLORS, NUM_FEATURES, DENSITY


def autoencoder():
    #                                                           NEURAL NETWORK MODEL

    # condense, 2 models in 1 model (encoder and decoder)

    #                                                             ENCODER
    encoder_input = keras.Input(shape=(HEIGHT, WIDTH, NUM_COLORS), name='img')
    left = keras.layers.Flatten()(encoder_input)
    # 64 -> 196 -> because 49*40 = 1960 /10 = 196 to compress it; 196/1960 = 0,1 -> reduced 10%
    encoder_output = keras.layers.Dense(DENSITY, activation="relu")(left)  # why 64, relu = rectify

    encoder = keras.Model(encoder_input, encoder_output, name='encoder')

    #                                                           DECODER
    decoder_input = keras.layers.Dense(DENSITY, activation="relu")(encoder_output)  # 196/5880 = 0,33%
    left = keras.layers.Dense(NUM_FEATURES, activation="relu")(decoder_input)  # n features (40x49)*3 (channel 3 (rgb)) -> 5880
    decoder_output = keras.layers.Reshape((HEIGHT, WIDTH, NUM_COLORS))(left)

    # optimizer
    opt = keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)

    auto_encoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
    auto_encoder.summary()
    '''
    Model: "autoencoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     img (InputLayer)            [(None, 49, 40, 3)]       0         
    
     flatten (Flatten)           (None, 5880)              0            FROM 5880 FEATURES IT WILL GO TO 196 FEATURES
    
     dense (Dense)               (None, 196)               1152676      
    
     dense_1 (Dense)             (None, 196)               38612     
    
     dense_2 (Dense)             (None, 5880)              1158360   
    
     reshape (Reshape)           (None, 49, 40, 3)         0         
    
    =================================================================
    Total params: 2,349,648
    Trainable params: 2,349,648
    Non-trainable params: 0
    _________________________________________________________________
    '''

    auto_encoder.compile(opt, loss="mse")

    return auto_encoder, encoder
