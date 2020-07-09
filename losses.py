import tensorflow as tf
import numpy as np
from STNFunction import *
from tensorflow.keras.losses import *

class flow_loss():
    def __init__(self):
        super(flow_loss, self).__init__()
        self.mse = MeanSquaredError()
    def landmark_loss(self,By_landmark,Ax_landmark,By_flow):
        By_landmark = tf.expand_dims(By_landmark,1)
        Ax_landmark = tf.expand_dims(Ax_landmark,1)
        sampled_flow = stn_bilinear_sampler(By_flow,By_landmark[:,:,:,0],By_landmark[:,:,:,1])
        target_flow = (By_landmark-Ax_landmark)
        # print("By_flow: {}".format(By_flow.shape))
        # print("Ax_landmark: {}".format(Ax_landmark.shape))
        # print("By_landmark: {}".format(By_landmark.shape))
        # print("sampled_flow: {}".format(sampled_flow.shape))
        # print("target_flow: {}".format(target_flow.shape))
        # By_flow: (10, 128, 128, 2)
        # Ax_landmark: (10, 1, 68, 2)
        # By_landmark: (10, 1, 68, 2)
        # sampled_flow: (10, 1, 68, 2)
        # target_flow: (10, 1, 68, 2)
        loss = 256*self.mse(sampled_flow, target_flow)
        return loss
    def totalVariation_loss(self,flow):
        #         (10, 128, 128, 2)
        x_diff = flow[:,:-1,:-1,:] - flow[:,:-1,1:,:]
        y_diff = flow[:,:-1,:-1,:] - flow[:,1:,:-1,:]
        sq_diff = tf.clip_by_value(x_diff*x_diff+y_diff*y_diff, clip_value_min=1e-3,clip_value_max=1000000)
        #tv = tf.image.total_variation(flow)
        return tf.reduce_mean(sq_diff)
    # return landmark_loss, totalVariation_loss
class lsgan_loss():
    def __init__(self):
        super(lsgan_loss, self).__init__()
        self.mse = MeanSquaredError()
    def loss_func(self,output,is_real):
        if is_real:
            loss = self.mse(tf.ones_like(output),output)
        else:
            loss = self.mse(tf.zeros_like(output),output)
        return loss
    def discriminator_loss(self,real_output, fake_output, is_real):
        real_loss = self.mse(tf.ones_like(real_output),real_output)
        fake_loss = 0.5*self.mse(tf.zeros_like(fake_output),fake_output)
        return real_loss + fake_loss
    def generator_loss(self,fake_output):
        return self.mse(tf.ones_like(fake_output),fake_output)
    # return discriminator_loss, generator_loss
def cls_loss(predict_output,answer):
    bce = BinaryCrossentropy()
    return bce(predict_output,answer)
def recon_loss(a,b):
    mae = MeanAbsoluteError()
    return mae(a,b)