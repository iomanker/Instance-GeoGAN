import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.losses import *
from tensorflow.keras import activations

class autoencoder_model(Model):
    def __init__(self, num_filters,outputChannels,img_size,data_format,model_name):
        super(autoencoder_model, self).__init__()
        # channels_last
        
        # formula = (W-Kernel+2P)/S + 1
        self.ZeroPadding2D1 = ZeroPadding2D(3)
        self.Conv1 = Conv2D(filters=num_filters,kernel_size=7,strides=1,
                            input_shape=img_size,data_format=data_format,
                           name="Conv1_"+model_name) # use_bias
        self.norm1 = BatchNormalization(name="Conv1B_"+model_name)
        self.Conv2 = Conv2D(filters=num_filters*2,kernel_size=3,strides=2,padding='same',
                           name="Conv2_"+model_name)
        self.norm2 = BatchNormalization(name="Conv2B_"+model_name)
        # 64x64
        self.Conv3 = Conv2D(filters=num_filters*4,kernel_size=3,strides=2,padding='same',
                           name="Conv3_"+model_name)
        self.norm3 = BatchNormalization(name="Conv3B_"+model_name)
        # 32x32
        self.Conv4 = Conv2D(filters=num_filters*8,kernel_size=3,strides=2,padding='same',
                           name="Conv4_"+model_name)
        self.norm4 = BatchNormalization(name="Conv4B_"+model_name)
        # 16x16
        
        # formula = (W-1)*S-2P+Kernel
        self.Dcon1 = Conv2DTranspose(filters=num_filters*4,kernel_size=4,strides=2,padding='same',
                                    input_shape=(16,16,num_filters*8*2),name="Dcon1_"+model_name)
        self.normd1 = BatchNormalization(name="Dcon1B_"+model_name)
        # 32x32
        self.Dcon2 = Conv2DTranspose(filters=num_filters*2,kernel_size=4,strides=2,padding='same',
                                    name="Dcon2_"+model_name)
        self.normd2 = BatchNormalization(name="Dcon2B_"+model_name)
        # 64x64
        self.Dcon3 = Conv2DTranspose(filters=num_filters,kernel_size=4,strides=2,padding='same',
                                     name="Dcon3_"+model_name)
        self.normd3 = BatchNormalization(name="Dcon3B_"+model_name)
        # 128x128
        # need self.ZeroPadding2D = ZeroPadding2D(3)
        self.ZeroPadding2D2 = ZeroPadding2D(3)
        self.Dcon4 = Conv2D(filters=outputChannels,kernel_size=7,strides=1,
                           name="Dcon4_"+model_name)
        
        self.Conv1_out = None
        self.Conv2_out = None
        self.Conv3_out = None
        
    def call(self, inputTensor): 
        # Whole Network
        EncoderOutput = self.Encoder(inputTensor)
        DecoderOutput = self.Decoder(EncoderOutput)
        return DecoderOutput
    def Encoder(self, inputTensor):
        # a half of network 
        inputTensor = self.ZeroPadding2D1(inputTensor)
        Conv1_out = self.Conv1(inputTensor)
        Conv1_out = self.norm1(Conv1_out)
        self.Conv1_out = activations.relu(Conv1_out)
        
        Conv2_out = self.Conv2(self.Conv1_out)
        Conv2_out = self.norm2(Conv2_out)
        self.Conv2_out = activations.relu(Conv2_out)
        
        Conv3_out = self.Conv3(self.Conv2_out)
        Conv3_out = self.norm3(Conv3_out)
        self.Conv3_out = activations.relu(Conv3_out)
        
        Conv4_out = self.Conv4(self.Conv3_out)
        Conv4_out = self.norm4(Conv4_out)
        Conv4_out = activations.relu(Conv4_out)
        
        return Conv4_out
    def Decoder(self, inputTensor):
        # a half of network 
        Dcon1_out = self.Dcon1(inputTensor)
        Dcon1_out = self.normd1(Dcon1_out)
        Dcon1_out += self.Conv3_out
        Dcon1_out = activations.relu(Dcon1_out)
        
        Dcon2_out = self.Dcon2(Dcon1_out)
        Dcon2_out = self.normd2(Dcon2_out)
        Dcon2_out += self.Conv2_out
        Dcon2_out = activations.relu(Dcon2_out)
        
        Dcon3_out = self.Dcon3(Dcon2_out)
        Dcon3_out = self.normd3(Dcon3_out)
        Dcon3_out += self.Conv1_out
        Dcon3_out = activations.relu(Dcon3_out)
        
        Dcon4_out = self.ZeroPadding2D2(Dcon3_out)
        Dcon4_out = self.Dcon4(Dcon4_out)
        return Dcon4_out