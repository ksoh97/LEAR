mode_dict = {"pre-training the classifier": 0, "training the counterfactual map generator": 1}
mode = 1

disc_ch, cfmap_ch = 32, 32
epoch = 100
batch_size = 256

save_path = "/DataCommon/ksoh/Results/MNIST/mode%d/" % mode
cls_weight_path = "/DataCommon/ksoh/Results/MNIST/mode0/MNIST_10cls_encoder_only_fcdrop/cls_model_095/variables/variables"  # 95.56%
enc_weight_path = "/DataCommon/ksoh/Results/MNIST/mode0/MNIST_10cls_encoder_only_fcdrop/enc_model_095/variables/variables"  # 95.56%

if mode == 0:
    file_name = "MNIST_classifier_pretraining"
    lr = 0.0001
    lr_decay = 0.98
else:
    # TODO: Set the file name here
    file_name = "210305_Spectral_predlabel_lsx_l1_10"
    g_step, d_step = 1, 1
    lr_g, lr_d = 0.001, 0.001
    lr_decay = 0.99
    beta_1 = 0.9
    one_sided_label_smoothing = False

# Hyper-param
hyper_param = [1.0, 10.0, 1.0, 1.0, 0.5]
loss_type = {'cls': hyper_param[0], 'norm': hyper_param[1], 'GAN': hyper_param[2], 'cyc': hyper_param[3],
             'dis': hyper_param[4]}