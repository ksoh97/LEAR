import numpy as np
import tensorflow as tf

def data_load():
    """
        - All_train_data -
        images_train.shape = (60000, 28, 28)
        labels_train.shape = (60000,)

        - All_test data -
        images_test.shape = (10000, 28, 28)
        labels_test.shape = (10000,)
    """
    data_train, data_test = tf.keras.datasets.mnist.load_data()
    images_train, labels_train = data_train
    images_test, labels_test = data_test

    return images_train, labels_train, images_test, labels_test

def permutation(train_lbl, test_lbl):
    all_train_idx = np.random.RandomState(seed=970304).permutation(np.argwhere(train_lbl >= 0))
    all_valid_idx = all_train_idx[:int(all_train_idx.shape[0] * 0.1)]
    all_train_idx = all_train_idx[int(all_train_idx.shape[0] * 0.1):]
    all_test_idx = np.random.RandomState(seed=970304).permutation(np.argwhere(test_lbl >= 0))

    return all_train_idx, all_valid_idx, all_test_idx

def seperate_data(idx, all_dat, all_lbl, CENTER=True):
    """
    Change the data dimension for the convolution
    dat.shape = (batch_size, 28, 28, 1)
    """
    dat = np.squeeze(all_dat[idx].astype("float32"))

    if CENTER:
        for batch in range(len(idx)):
            dat[batch] = (dat[batch] - np.min(dat[batch])) / (np.max(dat[batch] - np.min(dat[batch])))
        dat = np.expand_dims(dat, axis=-1)
    else:
        padding = 2
        npad = ((padding, padding), (padding, padding))
        emp = np.empty(shape=(dat.shape[0], dat.shape[1], dat.shape[2]))

        for cnt, dat in enumerate(dat):
            tmp = np.pad(dat, npad, "constant")
            emp[cnt] = tf.image.random_crop(tmp, emp[cnt].shape)

        for batch in range(len(idx)):
            emp[batch] = (emp[batch] - np.min(emp[batch])) / (np.max(emp[batch] - np.min(emp[batch])))
        dat = np.expand_dims(emp, axis=-1)

    lbl = all_lbl[idx]
    lbl = np.eye(10)[lbl.squeeze()]
    lbl = np.asarray(lbl, "float32")

    return dat, lbl

def code_creator(size, MINI=True):
    if MINI == True:  # Per mini-batch
        target_c = np.empty((size, 10))
        for i in range(size):
            code = np.random.choice(10, 10, replace=False)
            code = np.where(code < 9., 0., code)
            code = np.where(code != 0., 1., code)
            target_c[i] = code
    return target_c

# TODO: Change the target condition map from a target to predicted label, not from target to reverse target
# def codemap(target_c):
#     one = np.argmax(target_c, axis=-1)
#     c1, c2, c3 = np.zeros((len(target_c), 14, 14, 10)), np.zeros((len(target_c), 7, 7, 10)), np.zeros(
#         (len(target_c), 4, 4, 10))
#     for i in range(len(target_c)):
#         c1[i, :, :, one[i]], c2[i, :, :, one[i]], c3[i, :, :, one[i]] = 1., 1., 1.
#     return c1, c2, c3

def codemap(target_c):
    c1, c2, c3 = np.zeros((len(target_c), 14, 14, 10)), np.zeros((len(target_c), 7, 7, 10)), np.zeros((len(target_c), 4, 4, 10))
    c4 = np.zeros((len(target_c), 2, 2, 10))
    for batch in range(len(target_c)):
        for classes in range(target_c.shape[-1]):
            c1[batch, ..., classes], c2[batch, ..., classes] = target_c[batch, classes], target_c[batch, classes]
            c3[batch, ..., classes] = target_c[batch, classes]
            c4[batch, ..., classes] = target_c[batch, classes]
    return c1, c2, c3, c4