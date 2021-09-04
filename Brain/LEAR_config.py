import os
import GPUtil

# GPU setting in server
GPU = -1
if GPU == -1: devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else: devices = "%d" % GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Mode description
mode_dict = {"Learn": 0, "Explain": 1, "Reinforce": 2, "Iter_Explanation": 3, "Iter_reinforcement": 4}
mode = "Reinforce"

# TODO: Make sure you define your data load/save path here!!
# data_path = "/..."
# longitudinal_data_path = "/..."
# save_path = "/..."

data_path = "/DataRead/ksoh/js_ws_data/"
longitudinal_data_path = "/DataCommon/ksoh/longitudinal/3class"
save_path = "/DataCommon2/ksoh/classification_performance/3CLASS/"

file_name = mode
save_path = save_path + "/%s" % mode

fold = 5
ch = 64
classes = 3

if mode_dict[mode] == 0 or mode_dict[mode] == 2 or mode_dict[mode] == 4:
    epoch = 150
    batch_size = 12
    lr, lr_decay = 0.0001, 0.98
    ratio, xga_hyper_param = 4, 0.1

    # Weight constants
    if mode_dict[mode] == 4:
        xga_cls_weight_path = save_path + "/xga_cls_model/variables/variables"
        xga_enc_weight_path = save_path + "/xga_enc_model/variables/variables"

    elif mode_dict[mode] == 2:
        cmg_weight_path = save_path + "/cmg_model/variables/variables"
        cls_weight_pkl_path = save_path + "cls_model_weights.pkl"

elif mode_dict[mode] == 1 or mode_dict[mode] == 3:
    epoch = 100
    batch_size = 4
    lr_g, lr_d, lr_decay = 0.01, 0.01, 1

    # Weight constants
    cmg_hyper_param = [1.0, 10.0, 10.0, 1.0, 5., 10.0, 5e-6]
    cmg_loss_type = {'cls': cmg_hyper_param[0], 'norm': cmg_hyper_param[1], 'gen': cmg_hyper_param[2], 'cyc': cmg_hyper_param[3],
                     'dis': cmg_hyper_param[4], 'l2': cmg_hyper_param[5], "TV": cmg_hyper_param[6]}

    if mode_dict[mode] == 1:
        cls_weight_path = "/cls_model/variables/variables"
    # else: xga_cmg_weight_path = save_path + "/xga_cmg_model/variables/variables"