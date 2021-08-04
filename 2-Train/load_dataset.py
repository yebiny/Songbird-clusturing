import numpy as np

def padding_x(xset, xlen=9000):
    x_out = []
    x_len = []
    for x in xset:
        y = (x-np.mean(x))/np.std(x)
        add_len = xlen-len(y)
        if add_len<0: continue
        zeros = np.zeros(add_len)
        y = np.concatenate((y, zeros))
        y = np.reshape(y, (xlen,1))
        x_out.append(y)
        x_len.append(len(x))
    x_out = np.array(x_out, dtype='float32')
    return x_out

def process_imgset(imgset):
    imgset = imgset/255
    imgset = np.array(imgset, dtype='float32')
    imgset = np.expand_dims(imgset, 3)
    return imgset

def process_dataset(img_path, val_path, pad_size=9000, test_size=1000):
    imgset = np.load(img_path)
    valset = np.load(val_path, allow_pickle=True)

    imgset = process_imgset(imgset)
    valset = padding_x(valset, xlen=pad_size)
    print(imgset.shape, valset.shape)

    img_train, img_test = imgset[:-test_size], imgset[-test_size:]
    val_train, val_test = valset[:-test_size], valset[-test_size:]

    print(img_train.shape, img_test.shape, val_train.shape, val_test.shape)

    return img_train, img_test, val_train, val_test

def main():
    img_path = '../3-Analysis/bengal/0729/imgset.npy'
    val_path = '../3-Analysis/bengal/0729/valset.npy'
    img_train, img_test, val_train, val_test = process_dataset(img_path, val_path)

if __name__=='__main__':
    main()
