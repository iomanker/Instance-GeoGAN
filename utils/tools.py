import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def draw_landmark_face(raw_img,raw_landmark):
    img = np.clip((raw_img+1)/2,0,1)
    X = np.clip((raw_landmark[:,0]+1)/2*128,0,128)
    Y = np.clip((raw_landmark[:,1]+1)/2*128,0,128)
    implot = plt.imshow(img)
    plt.scatter(X,Y,c='r',s=5)
    plt.show()

def show_images(imgs,batch_size):
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