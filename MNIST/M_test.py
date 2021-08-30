import os
import numpy as np
from MNIST import M_utils as utils, network as net
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

def visualization(save_path, images_test, dec_model, epoch):
    plt_path = save_path + "/plt/"
    # model_path = "/DataCommon/ksoh/Results/MNIST/mode1/MNIST_10cls_encoder_only_fcdrop_smooth_one_sidedx/Model/dec_model_018/variables/variables"
    # model_path = "/DataCommon/ksoh/Results/MNIST/mode1/MNIST_10cls_encoder_only_fcdrop_smooth_one_sidedx/Model/dec_model_013/variables/variables"
    # dec_model.load_weights(model_path)
    if not os.path.exists(plt_path): os.makedirs(plt_path)
    idx = np.array([3, 2, 1, 18, 4, 8, 11, 0, 61, 7])
    fig_dat = np.empty((10, 28, 28)).astype("float32")

    for cnt, dat in enumerate(idx):
        fig_dat[cnt] = (images_test[dat] - np.min(images_test[dat])) / (np.max(images_test[dat] - np.min(images_test[dat])))

    code = np.eye(10)
    for num in range(10):
        if num == 0:
            target_c = code
            total_input = np.expand_dims(fig_dat[num], axis=0)
            for i in range(9): total_input = np.append(total_input, np.expand_dims(fig_dat[num], axis=0), axis=0)
        else:
            target_c = np.concatenate((target_c, code), axis=0)
            for i in range(10): total_input = np.append(total_input, np.expand_dims(fig_dat[num], axis=0), axis=0)

    c1, c2, c3, c4 = utils.codemap(target_c=target_c)
    CFmap = dec_model({"dec_in": np.expand_dims(total_input, axis=-1), "c1": c1, "c2": c2, "c3": c3, "c4": c4}, training=False)["dec_out"]
    CFmap = np.squeeze(np.array(CFmap))
    pseudo_image = total_input + CFmap

    fig = plt.figure()
    fig.suptitle("MNIST Counterfactual Map Generation")
    rows, cols = 10, 21
    for i in range(10):
        axs1 = fig.add_subplot(rows, cols, i * cols + 1)
        axs1.imshow(fig_dat[i])
        axs1.axis("off")
        if i == 0:
            axs1.set_title("Input", fontsize=5)

        for j in range(10):
            axs2 = fig.add_subplot(rows, cols, i * cols + 2 + j)
            axs2.imshow(CFmap[i * 10 + j])
            axs2.axis("off")
            axs3 = fig.add_subplot(rows, cols, i * cols + 12 + j)
            axs3.imshow(pseudo_image[i * 10 + j])
            axs3.axis("off")
            if i == 0:
                axs2.set_title("CF %d" % j, fontsize=5)
                axs3.set_title("%d" % j, fontsize=5)

    plt.savefig(plt_path + "epoch%d.png" % epoch)


images_train, labels_train, images_test, labels_test = utils.data_load()
g_network = net.MNIST_network()
cls_model, dec_model = g_network.CFmap_generator()
visualization("/DataCommon/ksoh/Results/MNIST/professor", images_test, dec_model, 1)

def FID_score(images_test, labels_test, dec_model):
    code = np.eye(10)
    min_idx = 800
    real_idx, fake_idx = 0, 0

    for i in range(10):
        dat = np.where(labels_test == i)[0]
        dat = np.random.RandomState(seed=970304).permutation(dat)
        if i == 0:
            real_idx = dat[:(min_idx // 2)]
            fake_idx = dat[(min_idx // 2):min_idx]
        else:
            real_idx = np.append(real_idx, dat[:(min_idx // 2)])
            fake_idx = np.append(fake_idx, dat[(min_idx // 2):min_idx])

    fake_idx = np.random.RandomState(seed=970304).permutation(fake_idx)
    real_dat, fake_dat = images_test[real_idx], images_test[fake_idx]

    tmp_real_dat = np.empty((len(real_idx), 28, 28)).astype("float32")
    tmp_fake_dat = np.empty((len(fake_idx), 28, 28)).astype("float32")

    for i in range(len(real_idx)):
        tmp_real_dat[i] = (real_dat[i] - np.min(real_dat[i])) / (np.max(real_dat[i] - np.min(real_dat[i])))
        tmp_fake_dat[i] = (fake_dat[i] - np.min(fake_dat[i])) / (np.max(fake_dat[i] - np.min(fake_dat[i])))

    ###############################################################################################################

    for num in range(10):
        dat = tmp_real_dat[num * (min_idx // 2):(num + 1) * (min_idx // 2)]
        for i in range(10):
            if i == 0:
                real_dat = dat
            else:
                real_dat = np.concatenate((real_dat, dat), axis=0)
        fake_dat = tmp_fake_dat

        for i in range(len(real_dat)):
            if i == 0:
                target_c = np.expand_dims(code[num], axis=0)
            else:
                target_c = np.concatenate((target_c, np.expand_dims(code[num], axis=0)), axis=0)

        c1, c2, c3 = utils.codemap(target_c=target_c)
        CFmap = dec_model({"dec_in": np.expand_dims(fake_dat, axis=-1), "c1": c1, "c2": c2, "c3": c3}, training=False)["dec_out"]
        CFmap = np.squeeze(np.array(CFmap))
        pseudo_image = fake_dat + CFmap

        total_fid, count = 0, 0
        for i in range(len(pseudo_image)):
            score = calculate_fid(pseudo_image[i], real_dat[i])
            total_fid += score
            count += 1

        print("num%d FID score: %f" % (num, (total_fid / count)))

def calculate_fid(act1, act2):
    mu1, sigma1 = np.mean(act1), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid