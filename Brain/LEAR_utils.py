class Utils:
    def __init__(self):
        self.project_path = '/home/ksoh/server_stratification/'
        self.data_path = '/DataRead/ksoh/js_ws_data/'

    def load_adni_data(self):
        """
        class #0: NC(433), class #1: pMCI(251), class #2: sMCI(497), class #3: AD(359)
        :return: NC, pMCI, sMCI, AD
        """
        dat = np.load(self.data_path + "total_dat.npy", mmap_mode="r")
        lbl = np.load(self.data_path + "labels.npy")
        return dat, lbl

    def data_permutation(self, lbl, cv):
        Total_NC_idx, Total_AD_idx = np.squeeze(np.argwhere(lbl == 0)), np.squeeze(np.argwhere(lbl == 3))
        Total_sMCI_idx, Total_pMCI_idx = np.squeeze(np.argwhere(lbl == 2)), np.squeeze(np.argwhere(lbl == 1))
        Total_MCI_idx = np.concatenate((Total_sMCI_idx, Total_pMCI_idx), axis=0)
        amount_NC, amount_MCI, amount_AD = int(len(Total_NC_idx) / fold), int(len(Total_MCI_idx) / fold), int(len(Total_AD_idx) / fold)

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

        # Original
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

        else:
            padding = 5
            npad = ((padding, padding), (padding, padding), (padding, padding))
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