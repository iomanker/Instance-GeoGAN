import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import cv2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.losses import *
from tensorflow.keras import activations
from CelebA_image_load import *
from losses import *
from Generator import *
from Discriminator import *

BATCH_SIZE = 8 
IMG_SIZE = (128,128,3)
EPOCHS = 100
NUMIMAGES = 1000
BUFFER_SIZE = 20
lambda_cls = 0.5
lambda_flow = 1
lambda_mask = 0.1
lambda_landmark = 10
lambda_reco = 5

def train_step(Ax,By,landmark_Ax,landmark_By,epoch):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        g_items = generator(Ax,By,epoch,True)
        # 1. TV_reg loss
        G_flow_loss = 0.0
        for flow in g_items['flows']:
            G_flow_loss += flow_loss_func.totalVariation_loss(flow)
        G_flow_loss = lambda_flow * tf.cast(G_flow_loss,'float32')
        # 2. landmark loss (By_flow)
        G_land_loss = flow_loss_func.landmark_loss(landmark_By,landmark_Ax,g_items['flows'][0])
        G_land_loss += flow_loss_func.landmark_loss(landmark_Ax,landmark_By,g_items['flows'][1])
        G_land_loss = lambda_landmark * tf.cast(G_land_loss,'float32')
        # 3. recon loss
        G_rcon_loss = recon_loss(By,g_items['fakeBx_to_By']) + recon_loss(Ax,g_items['fakeAy_to_Ax'])
        G_rcon_loss = lambda_reco * tf.cast(G_rcon_loss,'float32')
        
        # 4. GAN loss
        D_fake_loss = 0.0
        D_real_loss = 0.0
        GAN_loss = 0.0
        G_cls_loss = 0.0
        D_cls_loss = 0.0
        # -- For Generator
        for itemname in ['fake_Ay','fakeAy_to_Ax','fakeBx_to_By','fake_Bx']:
            item = g_items[itemname]
            _, pred, attr_pred = discriminator(item)
            for pred_i in pred:
                D_fake_loss += lsgan_loss_func.loss_func(pred_i, False)
                GAN_loss += lsgan_loss_func.loss_func(pred_i, True)
            
            for attr in attr_pred:
                if itemname in 'fake_Ay' or itemname in 'By':
                    attr_label = tf.ones_like(attr)
                else:
                    attr_label = tf.zeros_like(attr)
                G_cls_loss += lambda_cls * tf.cast(cls_loss(attr,attr_label),'float32')
        D_fake_loss = 0.5 * D_fake_loss
        # -- For Real Data
        for (img,hasAttr) in [(Ax,False),(By,True)]:
            _, pred, attr_pred = discriminator(img)
            for pred_i in pred:
                D_real_loss += lsgan_loss_func.loss_func(pred_i, True)
            for attr in attr_pred:
                if hasAttr:
                    label = tf.ones_like(attr)
                else:
                    label = tf.zeros_like(attr)
                D_cls_loss += lambda_cls * tf.cast(cls_loss(attr,label),'float32')
        # -- CAST
        D_fake_loss = tf.cast(D_fake_loss,'float32')
        D_real_loss = tf.cast(D_real_loss,'float32')
        GAN_loss = tf.cast(GAN_loss,'float32')
        
        DIS_loss = D_fake_loss+D_real_loss
        flow_loss = G_flow_loss+G_land_loss
        
        G_loss = G_rcon_loss+G_cls_loss+GAN_loss+flow_loss
        D_loss = DIS_loss+D_cls_loss
    gradients_of_generator = g_tape.gradient(G_loss,generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    gradients_of_discriminator = d_tape.gradient(D_loss,discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,discriminator.trainable_variables))
    return_items = {}
    return_items["G_flow_loss"] = G_flow_loss
    return_items["G_land_loss"] = G_land_loss
    return_items["G_rcon_loss"] = G_rcon_loss
    return_items["G_cls_loss"] = G_cls_loss
    return_items["GAN_loss"] = GAN_loss
    return_items["D_fake_loss"] = D_fake_loss
    return_items["D_real_loss"] = D_real_loss
    return_items["D_cls_loss"] = D_cls_loss
    return_items["G_loss"] = G_loss
    return_items["D_loss"] = D_loss
    return return_items

def save_images(imgs,epoch,batch_size,filename):
    fig = plt.figure(figsize=(batch_size,len(imgs))) 
    gs = matplotlib.gridspec.GridSpec(batch_size,len(imgs))
    for i in range(len(imgs)):
        imgs[i] = np.clip((imgs[i]+1)/2,0,1)
    for i in range(batch_size):
        for j in range(len(imgs)):
            ax = plt.subplot(gs[i,j])
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.imshow(imgs[j][i])
    plt.savefig('{}_at_epoch_{}.png'.format(filename,epoch), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    Ax_filenames_ds,By_filenames_ds,landmark_dict = load_csvdata_weneed()
    (Ax,Ax_landmark),(By,By_landmark) = get_raw_image(Ax_filenames_ds,By_filenames_ds,landmark_dict)

    Ax = Ax[:NUMIMAGES]
    Ax_landmark = Ax_landmark[:NUMIMAGES]
    By = By[:NUMIMAGES]
    By_landmark = By_landmark[:NUMIMAGES]
    Ax_ds = tf.data.Dataset.from_tensor_slices((Ax, Ax_landmark))
    Ax_ds = Ax_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    By_ds = tf.data.Dataset.from_tensor_slices((By, By_landmark))
    By_ds = By_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    discriminator = AttrbuteMultiscalePatchDisc()
    generator = generator_model(IMG_SIZE,BATCH_SIZE)

    flow_loss_func = flow_loss()
    lsgan_loss_func = lsgan_loss()
    d_LrDecay = tf.keras.optimizers.schedules.ExponentialDecay(0.002,decay_steps=1800,
                                                            decay_rate=0.95,
                                                            staircase=True)
    generator_optimizer = tf.keras.optimizers.Adam(0.002)
    discriminator_optimizer = tf.keras.optimizers.Adam(d_LrDecay)

    lossnames = ["G_flow_loss","G_land_loss","G_rcon_loss","G_cls_loss","GAN_loss",
                "D_fake_loss","D_real_loss","D_cls_loss","G_loss","D_loss"]
    metrics_list = []
    for itemname in lossnames:
        metrics_list.append(tf.keras.metrics.Mean(itemname, dtype=tf.float32))

    for epoch in range(1,200+1):
        for (one_Ax, one_Ax_landmark), (one_By, one_By_landmark) in zip(Ax_ds, By_ds):
            # train_flownet_step(one_Ax,one_By,one_Ax_landmark,one_By_landmark,tf.cast(epoch,'float32'))
            train_items = train_step(one_Ax,one_By,one_Ax_landmark,one_By_landmark,tf.cast(epoch,'float32'))
            for (idx, itemname) in enumerate(lossnames):
                metrics_list[idx](train_items[itemname])
        print("epoch: {}".format(epoch))
        for idx,itemname in enumerate(lossnames):
            print("    {}: {:.4f}".format(itemname,metrics_list[idx].result()))
        #print("epoch: {}, G_loss: {:.4f}, D_loss: {:.4f}, flow_loss: {:.3}".format(epoch, 
        #                                                         metrics_list[9].result(),metrics_list[10].result(),
        #                                                         metrics_list[0].result()+metrics_list[1].result()))
        for metric in metrics_list:
            metric.reset_states()
        if epoch % 5 == 0:
            for (one_Ax, one_Ax_landmark), (one_By, one_By_landmark) in zip(Ax_ds, By_ds):
                g_items = generator(one_Ax,one_By,tf.cast(epoch,'float32'),False)
                save_images([one_Ax,one_By,g_items['By_warpped'].numpy(),
                            g_items['masks'][2][:,:,:,0].numpy(),
                            g_items['fake_Ay'].numpy(),g_items['residual_Ay'].numpy(),
                            g_items['fakeAy_to_Ax'].numpy(),
                            g_items['masks'][1][:,:,:,0].numpy(),
                            g_items['fake_Bx'].numpy(),g_items['fakeBx_to_By'].numpy()],
                            epoch,BATCH_SIZE,'GeoGAN')
                break