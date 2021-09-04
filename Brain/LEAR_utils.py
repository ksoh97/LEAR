import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import LEAR_config as config

class Utils:
    def __init__(self):
        self.data_path = config.data_path
        self.long_path = config.longitudinal_data_path
        self.padding = 5

    def load_adni_data(self):
        """
        class #0: NC(433), class #1: pMCI(251), class #2: sMCI(497), class #3: AD(359)
        :return: NC, pMCI, sMCI, AD
        """
        dat = np.load(self.data_path + "total_dat.npy", mmap_mode="r")
        lbl = np.load(self.data_path + "labels.npy")
        return dat, lbl

    def load_longitudinal_data(self):
        long_nc, long_mci, long_ad = np.load(self.long_path+"/resized_quan_NC.npy"), np.load(self.long_path+"/resized_quan_MCI.npy"), np.load(self.long_path+"/resized_quan_AD.npy")
        long_nc, long_mci, long_ad = np.expand_dims(long_nc, axis=-1), np.expand_dims(long_mci, axis=-1), np.expand_dims(long_ad, axis=-1)
        return long_nc, long_mci, long_ad

    def data_permutation(self, lbl, cv):
        Total_NC_idx, Total_AD_idx = np.squeeze(np.argwhere(lbl == 0)), np.squeeze(np.argwhere(lbl == 3))
        Total_sMCI_idx, Total_pMCI_idx = np.squeeze(np.argwhere(lbl == 2)), np.squeeze(np.argwhere(lbl == 1))
        Total_MCI_idx = np.concatenate((Total_sMCI_idx, Total_pMCI_idx), axis=0)
        amount_NC, amount_MCI, amount_AD = int(len(Total_NC_idx) / 5), int(len(Total_MCI_idx) / 5), int(len(Total_AD_idx) / 5)

        NCvalid_idx = Total_NC_idx[cv * amount_NC:(cv + 1) * amount_NC]
        NCtrain_idx = np.setdiff1d(Total_NC_idx, NCvalid_idx)
        NCtest_idx = NCvalid_idx[:int(len(NCvalid_idx) / 2)]
        NCvalid_idx = np.setdiff1d(NCvalid_idx, NCtest_idx)

        MCIvalid_idx = Total_MCI_idx[cv * amount_MCI:(cv + 1) * amount_MCI]
        MCItrain_idx = np.setdiff1d(Total_MCI_idx, MCIvalid_idx)
        MCItest_idx = MCIvalid_idx[:int(len(MCIvalid_idx) / 2)]
        MCIvalid_idx = np.setdiff1d(MCIvalid_idx, MCItest_idx)

        ADvalid_idx = Total_AD_idx[cv * amount_AD:(cv + 1) * amount_AD]
        ADtrain_idx = np.setdiff1d(Total_AD_idx, ADvalid_idx)
        ADtest_idx = ADvalid_idx[:int(len(ADvalid_idx) / 2)]
        ADvalid_idx = np.setdiff1d(ADvalid_idx, ADtest_idx)

        self.train_all_idx = np.concatenate((NCtrain_idx, MCItrain_idx))
        self.train_all_idx = np.concatenate((self.train_all_idx, ADtrain_idx))
        self.valid_all_idx = np.concatenate((NCvalid_idx, MCIvalid_idx))
        self.valid_all_idx = np.concatenate((self.valid_all_idx, ADvalid_idx))
        self.test_all_idx = np.concatenate((NCtest_idx, MCItest_idx))
        self.test_all_idx = np.concatenate((self.test_all_idx, ADtest_idx))

        return self.train_all_idx, self.valid_all_idx, self.test_all_idx

    def seperate_data(self, data_idx, dat, lbl, CENTER=False):
        dat, lbl = dat[data_idx], lbl[data_idx]
        dat = np.squeeze(dat)
        lbl = np.where(lbl == 2, 1, lbl).astype("int32")
        lbl = np.where(lbl == 3, 2, lbl).astype("int32")
        lbl = np.eye(3)[lbl.squeeze()]
        lbl = lbl.astype('float32')
        if len(data_idx) == 1: dat = np.expand_dims(dat, axis=0)

        # Validation/test
        if CENTER:
            for batch in range(len(data_idx)):
                # Quantile normalization
                Q1, Q3 = np.quantile(dat[batch], 0.1), np.quantile(dat[batch], 0.9)
                dat[batch] = np.where(dat[batch] < Q1, Q1, dat[batch])
                dat[batch] = np.where(dat[batch] > Q3, Q3, dat[batch])

                # Gaussian normalization
                m, std = np.mean(dat[batch]), np.std(dat[batch])
                dat[batch] = (dat[batch] - m) / std
            dat = np.expand_dims(dat, axis=-1)

        # Training
        else:
            npad = ((self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding))
            emp = np.empty(shape=(dat.shape[0], dat.shape[1], dat.shape[2], dat.shape[3]))

            for cnt, dat in enumerate(dat):
                tmp = np.pad(dat, npad, "constant")
                emp[cnt] = tf.image.random_crop(tmp, emp[cnt].shape)

            for batch in range(len(emp)):
                # Quantile normalization
                Q1, Q3 = np.quantile(emp[batch], 0.1), np.quantile(emp[batch], 0.9)
                emp[batch] = np.where(emp[batch] < Q1, Q1, emp[batch])
                emp[batch] = np.where(emp[batch] > Q3, Q3, emp[batch])

                # Gaussian normalization
                m, std = np.mean(emp[batch]), np.std(emp[batch])
                emp[batch] = (emp[batch] - m) / std
            dat = np.expand_dims(emp, axis=-1)

        return dat, lbl

    def code_creator(self, train_dat_size):
        # Original
        for i in range(0, train_dat_size):
            code = np.random.choice(config.classes, config.classes, replace=False).astype("float32")
            code = np.expand_dims(code, axis=0)
            if i == 0:
                target_c = code
            else:
                target_c = np.append(target_c, code, axis=0)
        target_c = target_c[:train_dat_size]
        target_c = np.where(target_c == 2., 0., target_c)
        return target_c

    def codemap(self, condition):
        c1, c2 = np.zeros((len(condition), 48, 57, 48, config.classes)), np.zeros((len(condition), 24, 29, 24, config.classes))
        c3, c4 = np.zeros((len(condition), 12, 15, 12, config.classes)), np.zeros((len(condition), 6, 8, 6, config.classes))
        c5 = np.zeros((len(condition), 3, 4, 3, config.classes))
        for batch in range(len(condition)):
            for classes in range(condition.shape[-1]):
                c1[batch, ..., classes], c2[batch, ..., classes] = condition[batch, classes], condition[batch, classes]
                c3[batch, ..., classes], c4[batch, ..., classes] = condition[batch, classes], condition[batch, classes]
                c5[batch, ..., classes] = condition[batch, classes]
        return c1, c2, c3, c4, c5

    def map_interpolation(self, input, c1, c2, c3, c4, c5, map_generator):
        map = map_generator({"dec_in": input, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5})["dec_out"]
        annotated_map = np.expand_dims(np.squeeze(map), axis=1)
        annotated_map = torch.from_numpy(annotated_map)
        ann_map1 = F.interpolate(input=annotated_map, size=(48, 57, 48), mode="trilinear", align_corners=True)
        ann_map2 = F.interpolate(input=annotated_map, size=(24, 29, 24), mode="trilinear", align_corners=True)
        return np.squeeze(ann_map1), np.squeeze(ann_map2)

    def range_scale(self, input):
        input = np.array(input)
        for cnt, dat in enumerate(input): input[cnt] = (dat - np.min(dat)) / (np.max(dat) - np.min(dat))
        return input