mode_dict = {"Learn": 0, "Explain": 1, "Reinforce": 2, "Iter_Explanation": 3, "Iter_reinforcement": 4}  # Mode description
mode = "Learn"
save_path = "/..."

fold = 5
file_name = mode
save_path = save_path + "/%s" % mode

if mode_dict[mode] == 0 or 2 or 4:
    epoch = 150
    batch_size = 12
    lr, lr_decay = 0.0001, 0.98

    # Weight constants
    if mode_dict[mode] == 4:
        ratio, xga_hyper_param = 4, 0.1
        xga_cls_weight_path = save_path + "/xga_cls_weight/variables/variables"
        xga_enc_weight_path = save_path + "/xga_enc_weight/variables/variables"

    else:
        cls_weight_path = save_path + "/cls_weight/variables/variables"
        enc_weight_path = save_path + "/enc_weight/variables/variables"

elif mode_dict[mode] == 1 or 3:
    epoch = 100
    batch_size = 3
    lr_g, lr_d, lr_decay = 0.01, 0.01, 1

    # Weight constants
    cmg_hyper_param = [1.0, 10.0, 10.0, 1.0, 5., 10.0, 5e-6]
    cmg_loss_type = {'cls': cmg_hyper_param[0], 'norm': cmg_hyper_param[1], 'gen': cmg_hyper_param[2], 'cyc': cmg_hyper_param[3],
                     'dis': cmg_hyper_param[4], 'l2': cmg_hyper_param[5], "TV": cmg_hyper_param[6]}

    if mode_dict[mode] == 1: cmg_weight_path = save_path + "/cmg_weight/variables/variables"
    else: xga_cmg_weight_path = save_path + "/xga_cmg_weight/variables/variables"
