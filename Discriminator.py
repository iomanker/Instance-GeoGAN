import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.losses import *
from tensorflow.keras import activations

class SinglePatchDisc(Model):
                #       n_layers,input_nc,        ndf,        nc
    def __init__(self,num_layers,img_size,dis_filters,attributes):
        super(SinglePatchDisc,self).__init__()
        self.d_layers = []
        self.d_layers.append(tf.keras.Sequential([ZeroPadding2D(1),
                                           Conv2D(filters=dis_filters,kernel_size=3,strides=2,padding='valid',
                                           input_shape=img_size),
                                           LeakyReLU(alpha=0.2)]))
        nf = dis_filters
        for n in range(1,num_layers):
            nf_prev = nf
            nf = min(nf*2,512)
            self.d_layers.append(tf.keras.Sequential([ZeroPadding2D(1),
                                                      Conv2D(nf,kernel_size=3,strides=2,padding='valid'),
                                                      BatchNormalization(),
                                                      LeakyReLU(alpha=0.2)]))
        nf_prev = nf
        nf = min(nf*2,512)
        self.dilate_layer = []
        for dilate_rate in [2,4,6]:
            self.dilate_layer.append(tf.keras.Sequential([ZeroPadding2D(dilate_rate),
                                                     Conv2D(filters=nf,kernel_size=3,dilation_rate=dilate_rate,padding='valid')]))
        self.dilate_concat = Conv2D(filters=nf,kernel_size=1)
        self.disc_layer = tf.keras.Sequential([ZeroPadding2D(1),
                                               Conv2D(filters=1,kernel_size=3,strides=1,padding='valid')])
        self.attr_layer = tf.keras.Sequential([ZeroPadding2D(1),
                                               Conv2D(filters=attributes,kernel_size=3,strides=1,padding='valid'),
                                               Activation('sigmoid')])
        self.dilateConcat = Concatenate(axis=3)
    def call(self, img):
        out = img
        feat = []
        for d_layer in self.d_layers:
            out = d_layer(out)
            feat.append(out)
            # if self.dilated:
        dilate_outs = []
        for idx,_ in enumerate([2,4,6]):
            layer = self.dilate_layer[idx](out)
            dilate_outs.append(layer)
        out = self.dilate_concat(self.dilateConcat([dilate_outs[0],dilate_outs[1],dilate_outs[2]]))
        disc_out = self.disc_layer(out)
        attr_out = self.attr_layer(out)
        return feat,disc_out,attr_out
class AttrbuteMultiscalePatchDisc(Model):
    def __init__(self,num_scale=3,img_size=(128,128,3),num_layers=3,dis_filters=32,attributes=1):
        super(AttrbuteMultiscalePatchDisc,self).__init__()
        self.disc = []
        self.num_scale = num_scale
        for rank in range(num_scale):
            self.disc.append(SinglePatchDisc(num_layers,img_size,dis_filters,attributes))
        self.downsampled = tf.keras.Sequential([ZeroPadding2D(1),
                                                AveragePooling2D(pool_size=3,strides=2,padding='valid')])
    def call(self,inputs):
        features, discrims = [],[]
        attr_outs = []
        image_downsampled = inputs
        for i in range(self.num_scale):
            # print("image_shape: {}".format(image_downsampled.shape))
            feat,out,attr_out = self.disc[i](image_downsampled)
            features.append(feat)
            discrims.append(out)
            attr_outs.append(attr_out)
            image_downsampled = self.downsampled(image_downsampled)
        return features,discrims,attr_outs