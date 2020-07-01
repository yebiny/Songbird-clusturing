import numpy as np
import matplotlib.pyplot as plt

c_list=['r','salmon','chocolate',
        'darkorange','gold','yellow',
        'yellowgreen','lightgreen','green',
        'turquoise','c','deepskyblue','royalblue',
        'slateblue','darkviolet','m','deeppink']
c_base = 'greenyellow'

def convert_shape(data, mode=1):
    if mode == 1:
        reshape = data.reshape(data.shape[0], data.shape[1], 1)
    else:
        reshape = data.reshape(data.shape[0], data.shape[1])

    return reshape

def plot_recimg(idx, x_data, x_rec):

    org_img = convert_shape(x_data[idx], -1)
    rec_img = convert_shape(x_rec[idx], -1)

    fig = plt.figure(figsize=(4,8))
    plt.subplot(1,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(org_img, cmap = 'gray')
    plt.subplot(1,2,2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(rec_img, cmap = 'gray')
    plt.show()

def plot_scatter(data, y_pred, gidx=None, size=10, save = None):
    
    plt.figure(figsize=(size,size))
    ax = plt.subplot()
    
    x = data[:,0]
    y = data[:,1]
    
    xmin = min(x)-0.5
    xmax = max(x)+0.5
    ymin = min(y)-0.5
    ymax = max(y)+0.5
    print(xmin, xmax, ymin, ymax)
    plt.axis((xmin, xmax, ymin, ymax))
    
    if gidx != None:
        if (type(gidx)==int):
            cond = (y_pred == gidx)
            g = data[cond]
            gx = g[:, 0]
            gy = g[:, 1]
            plt.scatter(gx, gy, color = c_list[gidx], marker='o', label = str(gidx))
            plt.legend(prop={'size': size*2})
        else:
            for i in range(max(y_pred)+1):
                cond = (y_pred == i)
                g = data[cond]
                gx = g[:,0]
                gy = g[:,1]
                plt.scatter(gx, gy, color = c_base, marker='o', label=str(i))

    else:    
        for i in range(max(y_pred)+1):
            cond = (y_pred == i)
            g = data[cond]
            gx = g[:,0]
            gy = g[:,1]
            plt.scatter(gx, gy, color = c_list[i], marker='o', label=str(i))

        plt.legend(prop={'size': size})
    
    if save != None:
        plt.savefig(save)
    plt.show()
    
def plot_group(x_data, y_data, gidx, save='n'):
    
    x, y = 5, 10
    fig = plt.figure(figsize=(y,x*2))
    
    cond = y_data == gidx
    g_data = x_data[cond]

    for i in range(x):
        for j in range(y):
            idx = i*y+j
            plt.subplot(x, y, idx+1)
            plt.xticks([])
            plt.yticks([])
            img = convert_shape(g_data[idx], -1)
            plt.imshow(img)
    fig.suptitle('Group %s'%(str(gidx)), fontsize=30)
    if save != None:
        plt.savefig(save+'gidx_%s'%(str(gidx)))
    plt.show() 
