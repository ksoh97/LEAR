mode_dict = {"Learn": 0, "Explain": 1}
mode = 1

disc_ch, cfmap_ch = 32, 32
epoch = 100
batch_size = 256

save_path = "/..."
cls_weight_path = "/..."
enc_weight_path = "/..."

if mode == 0:
    file_name = "Learn"
    lr = 0.0005
    lr_decay = 0.98
else:
    file_name = "Explain"
    g_step, d_step = 1, 1
    lr_g, lr_d = 0.001, 0.001
    lr_decay = 0.99
    beta_1 = 0.9
    one_sided_label_smoothing = 0.1

# Hyper-param
hyper_param = [1.0, 10.0, 1.0, 1.0, 0.5]
loss_type = {'cls': hyper_param[0], 'norm': hyper_param[1], 'GAN': hyper_param[2], 'cyc': hyper_param[3],
             'dis': hyper_param[4]}
