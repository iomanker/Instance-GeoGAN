import numpy as np

def landmark_resize_with_pad(landmark,
                             img,
                             target_height,target_width):
    # 先縮放再加間隔
    source_height = float(img.shape[0])
    source_width = float(img.shape[1])
    target_height = float(target_height)
    target_width = float(target_width)
    ratio = max(source_width / target_width, source_height / target_height)
    resized_height_float = float(source_height) / ratio
    resized_width_float = float(source_width) / ratio
    # resized_height = int(resized_height_float)
    # resized_width = int(resized_width_float)

    padding_height = (float(target_height) - resized_height_float) / 2
    padding_width = (float(target_width) - resized_width_float) / 2
    f_padding_height = float(padding_height)
    f_padding_width = float(padding_width)
    p_height = max(0, int(f_padding_height))
    p_width = max(0, int(f_padding_width))

    landmark = np.float32(landmark) / ratio
    landmark[:,0] = (landmark[:,0] + p_width) / target_width
    landmark[:,1] = (landmark[:,1] + p_height) / target_height
    return landmark

def img_to_float(img, target_size):
    scale = target_size / 2
    return (np.float32(img)-scale)/scale
def img_to_uint8(img):
    return np.uint8(img*127.5+128).clip(0, 255)

def save_images(imgs,epoch,batch_size,filename):
    fig = plt.figure(figsize=(batch_size,len(imgs))) 
    gs = matplotlib.gridspec.GridSpec(batch_size,len(imgs))#(batch_size, len(imgs))
    # gs.update()
    #, width_ratios=[1 for i in range(len(imgs))],
    #     wspace=0.0, hspace=0.0, top=0.05, bottom=0.05, left=0.1, right=0.2)
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

def draw_landmark_points(raw_img,raw_landmark):
    img = np.clip((raw_img+1)/2,0,1)
    # X = raw_landmark[:,0]
    # Y = raw_landmark[:,1]
    X = np.clip((raw_landmark[:,0]*32+32),0,64)
    Y = np.clip((raw_landmark[:,1]*127.5)+128,0,128)
    implot = plt.imshow(img)
    plt.scatter(X,Y,c='r',s=5)
    plt.show()