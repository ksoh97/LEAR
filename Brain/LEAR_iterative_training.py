import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import LEAR_config as config
import LEAR_networks as network
import LEAR_losses as loss
import LEAR_test as test
import LEAR_utils as utils
import tqdm

l = keras.layers
K = keras.backend

class Trainer:
    def __init__(self):
        self.file_name = config.file_name
        self.fold = config.fold

        self.valid_acc, self.compare_acc, self.valid_loss, self.compare_loss, self.count = 0, 0, 0, 100, 0
        tf.keras.backend.set_image_data_format("channels_last")
        self.valid_save, self.nii_save, self.model_select = False, False, False
        self.build_model()

        self.path = config.save_path
        if not os.path.exists(self.path): os.makedirs(self.path)

    def build_model(self):
        if config.mode == "Iter_Explanation":
            xga_generator, discriminator = network.ResNet18_XGA_Generator(), network.ResNet18_Discriminator()
            xga_resnet = network.ResNet18_XGA()
            generator = network.ResNet18_Generator()

            self.train_discri_vars = []
            self.discriminator_model = discriminator.build_model()
            self.train_discri_vars += self.discriminator_model.trainable_variables

            self.train_vars = []
            self.enc_model, self.cls_model = xga_resnet.build_model()
            self.xga_gen_model = xga_generator.build_model()
            self.gen_model = generator.build_model()
            self.cls_model.load_weights(config.xga_cls_weight_path), self.gen_model.load_weights(config.gen_weight_path)

            # Encoder part initialization and freeze
            self.cls_model.trainable = False
            for enc_layer, gen_layer in zip(self.enc_model.layers, self.xga_gen_model.layers):
                gen_layer.set_weights(enc_layer.get_weights())
                gen_layer.trainable, enc_layer.trainable = False, False

            # Decoder part initialization
            freeze_gen_num, freeze_xga_gen_num = 0, 0
            for cnt, dec_layer in enumerate(self.gen_model.layers):
                if dec_layer.name == "dec_code_conv5_1_conv": freeze_gen_num = cnt
            for cnt, gen_layer in enumerate(self.xga_gen_model.layers):
                if gen_layer.name == "dec_code_conv5_1_conv": freeze_xga_gen_num = cnt
            for b_dec_layer, gen_layer in zip(self.gen_model.layers[freeze_gen_num:], self.xga_gen_model.layers[freeze_xga_gen_num:]):
                gen_layer.set_weights(b_dec_layer.get_weights())

            for variables in self.xga_gen_model.trainable_variables:
                self.train_vars += [variables]

        elif config.mode == "Iter_reinforcement":
            resnet = network.ResNet18()
            xga_resnet = network.ResNet18_XGA()
            self.train_vars = []
            _, self.cls_model =xga_resnet.build_model()
            self.old_cls_model = resnet.build_model()

            self.cls_model.load_weights(config.xga_cls_weight_path)

            for old_l in self.old_cls_model.layers:
                for old_w in old_l.weights:
                    for l in self.cls_model.layers:
                        for w in l.weights:
                            if old_w.name == w.name:
                                l.trainable = False
            self.train_vars += self.cls_model.trainable_variables

            # TODO: Counterfactual map generator setting
            map_generator = network.ResNet18_XGA_Generator()
            self.map_generator = map_generator.build_model()
            self.map_generator.load_weights(config.xga_gen_weight_path)
            for layer in self.map_generator.layers: layer.trainable = False

    def _train_one_batch(self, dat_all, lbl, gen_optim, train_vars, step, cv, c_nc, c_ad):
        nc_c1, nc_c2, nc_c3, nc_c4, nc_c5 = utils.Utils().codemap(c_nc)
        ad_c1, ad_c2, ad_c3, ad_c4, ad_c5 = utils.Utils().codemap(c_ad)
        nc_map1, nc_map2 = utils.Utils().map_interpolation(dat_all, nc_c1, nc_c2, nc_c3, nc_c4, nc_c5, self.map_generator)
        ad_map1, ad_map2 = utils.Utils().map_interpolation(dat_all, ad_c1, ad_c2, ad_c3, ad_c4, ad_c5, self.map_generator)
        map1 = utils.Utils().range_scale((nc_map1 + ad_map1))
        map2 = utils.Utils().range_scale((nc_map2 + ad_map2))

        with tf.GradientTape() as tape:
            res = self.cls_model({"cls_in": dat_all}, training=True)
            train_loss = loss.CE_loss(lbl, res["cls_out"])
            att_loss = loss.annotated_loss(res["m1"], map1)
            att_loss += loss.annotated_loss(res["m2"], map2)
            train_loss += config.xga_hyper_param * att_loss / 2

        grads = tape.gradient(train_loss, train_vars)
        gen_optim.apply_gradients(zip(grads, train_vars))

        if step % 10 == 0:
            with self.train_summary_writer.as_default():
                tf.summary.scalar("%dfold_train_loss" % (cv + 1), train_loss, step=step)

    def cycle_consistency(self, pseudo_image, source):
        c1, c2, c3, c4, c5 = utils.Utils().codemap(condition=source)
        tilde_map = self.xga_gen_model({"gen_in": pseudo_image, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, training=True)["gen_out"]
        return pseudo_image + tilde_map

    def _GAN_train_one_batch(self, dat_all, gen_optim, disc_optim, target_c, train_vars, train_discri_vars, step):
        label = self.cls_model({"cls_in": dat_all}, training=False)["cls_out"]
        t1, t2, t3, t4, t5 = utils.Utils().codemap(condition=target_c)

        # Discriminator step
        with tf.GradientTape() as tape:
            cfmap = self.xga_gen_model({"gen_in": dat_all, "c1": t1, "c2": t2, "c3": t3, "c4": t4, "c5": t5}, training=True)["gen_out"]
            pseudo_image = dat_all + cfmap
            dat_all_like = self.cycle_consistency(pseudo_image, label)

            real = self.discriminator_model({"discri_in": dat_all}, training=True)["discri_out"]
            real_like = self.discriminator_model({"discri_in": dat_all_like}, training=True)["discri_out"]
            fake = self.discriminator_model({"discri_in": pseudo_image}, training=True)["discri_out"]

            real_loss = (loss.MSE_loss(tf.ones_like(real), real) + loss.MSE_loss(tf.ones_like(real_like), real_like))/2
            fake_loss = loss.MSE_loss(tf.zeros_like(fake), fake)
            train_dis_loss = config.cmg_loss_type["dis"] * (real_loss + fake_loss)

        grads = tape.gradient(train_dis_loss, train_discri_vars)
        disc_optim.apply_gradients(zip(grads, train_discri_vars))

        if step % 10 == 0:
            with self.discriminator_summary_writer.as_default():
                tf.summary.scalar("discriminator_loss", train_dis_loss, step=step)

        # Generator step
        with tf.GradientTape() as tape:
            cfmap = self.xga_gen_model({"gen_in": dat_all, "c1": t1, "c2": t2, "c3": t3, "c4": t4, "c5": t5}, training=True)["gen_out"]
            pseudo_image = dat_all + cfmap
            dat_all_like = self.cycle_consistency(pseudo_image, label)

            fake = self.discriminator_model({"discri_in": pseudo_image}, training=True)["discri_out"]
            fake_like = self.discriminator_model({"discri_in": dat_all_like}, training=True)["discri_out"]

            pred = self.cls_model({"cls_in": pseudo_image}, training=False)["cls_out"]

            cyc_loss = loss.cycle_loss(dat_all, dat_all_like)
            l1_loss, l2_loss = loss.L1_norm(effect_map=cfmap), loss.L2_norm(effect_map=cfmap)
            gen_loss = (loss.MSE_loss(tf.ones_like(fake), fake) + loss.MSE_loss(tf.ones_like(fake_like), fake_like))/2
            cls_loss = loss.CE_loss(target_c, pred)
            tv_loss = loss.tv_loss(pseudo_image)

            cyc = config.cmg_loss_type["cyc"] * cyc_loss
            l1 = config.cmg_loss_type["norm"] * l1_loss
            l2 = config.cmg_loss_type["l2"] * l2_loss
            gen = config.cmg_loss_type["gen"] * gen_loss
            cls = config.cmg_loss_type["cls"] * cls_loss
            tv = config.cmg_loss_type["TV"] * tv_loss
            train_gen_loss = cyc + l1 + l2 + gen + cls + tv

        grads = tape.gradient(train_gen_loss, train_vars)
        gen_optim.apply_gradients(zip(grads, train_vars))

        if step % 10 == 0:
            with self.generator_summary_writer.as_default():
                tf.summary.scalar("mode1_G_total_train_loss", train_gen_loss, step=step)
                tf.summary.scalar("generator_cyc_loss", cyc, step=step)
                tf.summary.scalar("generator_l1_loss", l1, step=step)
                tf.summary.scalar("generator_l2_loss", l2, step=step)
                tf.summary.scalar("generator_gen_loss", gen, step=step)
                tf.summary.scalar("generator_cls_loss", cls, step=step)
                tf.summary.scalar("generator_tv_loss", tv, step=step)

    def _valid_logger(self, dat_all, lbl, epoch, cv, c_nc, c_ad):
        nc_c1, nc_c2, nc_c3, nc_c4, nc_c5 = utils.Utils().codemap(c_nc)
        ad_c1, ad_c2, ad_c3, ad_c4, ad_c5 = utils.Utils().codemap(c_ad)
        nc_map1, nc_map2 = utils.Utils().map_interpolation(dat_all, nc_c1, nc_c2, nc_c3, nc_c4, nc_c5, self.map_generator)
        ad_map1, ad_map2 = utils.Utils().map_interpolation(dat_all, ad_c1, ad_c2, ad_c3, ad_c4, ad_c5, self.map_generator)
        map1 = utils.Utils().range_scale((nc_map1 + ad_map1))
        map2 = utils.Utils().range_scale((nc_map2 + ad_map2))

        res = self.cls_model({"cls_in": dat_all}, training=False)
        valid_loss, valid_acc = loss.CE_loss(lbl, res["cls_out"]), test.ACC(lbl, res["cls_out"])
        att_loss = loss.annotated_loss(res["m1"], map1)
        att_loss += loss.annotated_loss(res["m2"], map2)
        valid_loss += config.xga_hyper_param * att_loss / 2

        self.valid_loss += valid_loss
        self.valid_acc += valid_acc
        self.count += 1

        if self.valid_save == True:
            self.valid_acc = self.valid_acc / self.count
            self.valid_loss = self.valid_loss / self.count

            if self.compare_acc <= self.valid_acc:
                self.model_select = True
                self.compare_acc = self.valid_acc

            with self.valid_summary_writer.as_default():
                tf.summary.scalar("%dfold_valid_loss" % (cv + 1), self.valid_loss, step=epoch)
                tf.summary.scalar("%dfold_valid_acc" % (cv + 1), self.valid_acc, step=epoch)
                self.valid_acc, self.valid_loss, self.count = 0, 0, 0
                self.valid_save = False

    def _GAN_valid_logger(self, dat_all, target_c, epoch):
        label = self.cls_model({"cls_in": dat_all}, training=False)["cls_out"]
        t1, t2, t3, t4, t5 = utils.Utils().codemap(condition=target_c)

        cfmap = self.xga_gen_model({"gen_in": dat_all, "c1": t1, "c2": t2, "c3": t3, "c4": t4, "c5": t5}, training=True)["gen_out"]
        pseudo_image = dat_all + cfmap
        dat_all_like = self.cycle_consistency(pseudo_image, label)

        fake = self.discriminator_model({"discri_in": pseudo_image}, training=True)["discri_out"]
        fake_like = self.discriminator_model({"discri_in": dat_all_like}, training=True)["discri_out"]

        pred = self.cls_model({"cls_in": pseudo_image}, training=False)["cls_out"]

        cyc_loss = loss.cycle_loss(dat_all, dat_all_like)
        l1_loss, l2_loss = loss.L1_norm(effect_map=cfmap), loss.L2_norm(effect_map=cfmap)
        gen_loss = (loss.MSE_loss(tf.ones_like(fake), fake) + loss.MSE_loss(tf.ones_like(fake_like), fake_like)) / 2
        cls_loss = loss.CE_loss(target_c, pred)
        tv_loss = loss.tv_loss(pseudo_image)

        cyc = config.cmg_loss_type["cyc"] * cyc_loss
        l1 = config.cmg_loss_type["norm"] * l1_loss
        l2 = config.cmg_loss_type["l2"] * l2_loss
        gen = config.cmg_loss_type["gen"] * gen_loss
        cls = config.cmg_loss_type["cls"] * cls_loss
        tv = config.cmg_loss_type["TV"] * tv_loss

        self.valid_loss += cyc + l1 + l2 + gen + cls + tv
        self.count += 1

        if self.valid_save == True:
            self.valid_loss = self.valid_loss / self.count

            if self.valid_loss <= self.compare_loss:
                self.model_select = True
                self.compare_loss = self.valid_loss

            with self.valid_summary_writer.as_default():
                tf.summary.scalar("valid_loss", self.valid_loss, step=epoch)
                self.valid_loss, self.count = 0, 0
                self.valid_save = False

    def iter_cls_train(self):
        dat, lbl = utils.Utils().load_adni_data()

        for cv in range(0, self.fold):
            self.train_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_train" % (cv + 1))
            self.valid_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_valid" % (cv + 1))
            self.test_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_test" % (cv + 1))
            self.train_all_idx, self.valid_all_idx, self.test_all_idx = utils.Utils().data_permutation(lbl, cv)
            self.build_model()

            lr_schedule = keras.optimizers.schedules.ExponentialDecay(config.lr, decay_steps=len(self.train_all_idx) // config.batch_size, decay_rate=config.lr_decay, staircase=True)
            optim = keras.optimizers.Adam(lr_schedule)
            global_step, self.compare_acc = 0, 0

            for cur_epoch in tqdm.trange(config.epoch, desc="3class_resnet18_%s.py" % self.file_name):
                self.train_all_idx = np.random.permutation(self.train_all_idx)
                target_nc, target_ad = utils.Utils().code_creator(len(self.train_all_idx))

                # training
                for cur_step in tqdm.trange(0, len(self.train_all_idx), config.batch_size, desc="%dfold_%depoch_%s" % (cv + 1, cur_epoch, self.file_name)):
                    cur_idx = self.train_all_idx[cur_step:cur_step + config.batch_size]
                    cur_dat, cur_lbl = utils.Utils().seperate_data(cur_idx, dat, lbl, CENTER=False)

                    condition_nc, condition_ad = target_nc[cur_step:cur_step + config.batch_size], target_ad[cur_step:cur_step + config.batch_size]
                    condition_nc, condition_ad = condition_nc[:len(cur_dat)], condition_ad[:len(cur_dat)]
                    self._train_one_batch(dat_all=cur_dat, lbl=cur_lbl, gen_optim=optim, train_vars=self.train_vars, step=global_step, cv=cv, c_nc=condition_nc, c_ad=condition_ad)
                    global_step += 1

                # validation
                for val_step in tqdm.trange(0, len(self.valid_all_idx), config.batch_size, desc="Validation step: %dfold" % (cv + 1)):
                    val_idx = self.valid_all_idx[val_step:val_step + config.batch_size]
                    val_dat, val_lbl = utils.Utils().seperate_data(val_idx, dat, lbl, CENTER=True)

                    condition_nc, condition_ad = target_nc[cur_step:cur_step + config.batch_size], target_ad[cur_step:cur_step + config.batch_size]
                    condition_nc, condition_ad = condition_nc[:len(cur_dat)], condition_ad[:len(cur_dat)]

                    if val_step + config.batch_size >= len(self.valid_all_idx): self.valid_save = True
                    self._valid_logger(dat_all=val_dat, lbl=val_lbl, epoch=cur_epoch, cv=cv, c_nc=condition_nc, c_ad=condition_ad)

                if self.model_select == True:
                    self.cls_model.save(os.path.join(self.path + '/%dfold_iter_xga_cls_model' % (cv + 1)))
                    self.model_select = False

                # Test
                tot_true, tot_pred = 0, 0
                for tst_step in tqdm.trange(0, len(self.test_all_idx), config.batch_size, desc="Testing step: %dfold" % (cv + 1)):
                    tst_idx = self.test_all_idx[tst_step:tst_step + config.batch_size]
                    tst_dat, tst_lbl = utils.Utils().seperate_data(tst_idx, dat, lbl, CENTER=True)
                    res = self.cls_model({"cls_in": tst_dat}, training=False)["cls_out"]

                    if tst_step == 0: tot_true, tot_pred = tst_lbl, res
                    else:
                        tot_true, tot_pred = np.concatenate((tot_true, tst_lbl), axis=0), np.concatenate((tot_pred, res), axis=0)

                mAUC = test.MAUC(np.concatenate((np.argmax(tot_true, axis=-1).astype("float32").reshape(-1, 1), tot_pred), axis=1), num_classes=config.classes)
                ACC = test.ACC(tot_true, tot_pred)
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("%dfold_test_mAUC" % (cv + 1), mAUC, step=cur_epoch)
                    tf.summary.scalar("%dfold_test_ACC" % (cv + 1), ACC, step=cur_epoch)

    def iter_gan_train(self):
        dat, lbl = utils.Utils().load_adni_data()

        for cv in range(0, self.fold):
            self.discriminator_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_train_critic" % (cv + 1))
            self.generator_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_train_generator" % (cv + 1))
            self.valid_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_valid_generator" % (cv + 1))
            self.train_all_idx, self.valid_all_idx, self.test_all_idx = utils.Utils().data_permutation(lbl, cv)
            self.build_model()

            g_lr_schedule = keras.optimizers.schedules.ExponentialDecay(config.lr_g, decay_steps=len(self.train_all_idx) // config.batch_size, decay_rate=config.lr_decay, staircase=True)
            d_lr_schedule = keras.optimizers.schedules.ExponentialDecay(config.lr_d, decay_steps=len(self.train_all_idx) // config.batch_size, decay_rate=config.lr_decay, staircase=True)
            gen_optim = keras.optimizers.Adam(learning_rate=g_lr_schedule)
            disc_optim = keras.optimizers.Adam(learning_rate=d_lr_schedule)
            global_step = 0

            f = open(self.path + "/%fold_all_ncc_result.txt" % (cv+1), "w")
            f.write("|  1: AD to NC (+)  |  2: AD to MCI (+)  |  3: MCI to NC (+)  |  4: NC to AD (-)  |  5: NC to MCI (-)  |  6: MCI to AD (-)  |\n")
            f.close()

            for cur_epoch in tqdm.trange(config.epoch, desc=config.file_name):
                self.train_all_idx = np.random.permutation(self.train_all_idx)

                # training
                for cur_step in tqdm.trange(0, len(self.train_all_idx), config.batch_size, desc="%dfold_%depoch_%s" % (cv + 1, cur_epoch, self.file_name)):
                    cur_idx = self.train_all_idx[cur_step:cur_step + config.batch_size]
                    cur_dat, cur_lbl = utils.Utils().seperate_data(cur_idx, dat, lbl, CENTER=False)
                    target_idx = utils.Utils().code_creator(len(cur_idx))

                    self._GAN_train_one_batch(dat_all=cur_dat, gen_optim=gen_optim, disc_optim=disc_optim, target_c=target_idx,
                                              train_vars=self.train_vars, train_discri_vars=self.train_discri_vars, step=global_step)
                    global_step += 1

                # validation
                for val_step in tqdm.trange(0, len(self.valid_all_idx), config.batch_size, desc="Validation step: %dfold" % (cv + 1)):
                    val_idx = self.valid_all_idx[val_step:val_step + config.batch_size]
                    val_dat, val_lbl = utils.Utils().seperate_data(val_idx, dat, lbl, CENTER=True)
                    target_idx = utils.Utils().code_creator(len(val_idx))

                    if val_step + config.batch_size >= len(self.valid_all_idx): self.valid_save = True
                    self._GAN_valid_logger(dat_all=val_dat, target_c=target_idx, epoch=cur_epoch)

                if self.model_select == True:
                    self.xga_gen_model.save(os.path.join(self.path + '/%dfold_iter_xga_gen_model' % (cv + 1)))
                    self.model_select = False

                # Test
                a1, a2, a3, a4, a5, a6, n1, n2, n3, n4, n5, n6 = test.ncc_evaluation(self.cls_model, self.xga_gen_model)
                f = open(self.path + "/%fold_all_ncc_result.txt" % (cv+1), "a")
                f.write("Epoch:%03d -> acc1:%.3f | acc2: %.3f | acc3:%.3f | acc4: %.3f | acc5:%.3f | acc6: %.3f |\n" % (cur_epoch, a1, a2, a3, a4, a5, a6))
                f.write("Epoch:%03d -> ncc1:%.3f | ncc2: %.3f | ncc3:%.3f | ncc4: %.3f | ncc5:%.3f | ncc6: %.3f |\n\n" % (cur_epoch, n1, n2, n3, n4, n5, n6))
                f.close()

Tr = Trainer()
if config.mode == "Iter_reinforcement": Tr.iter_cls_train()
elif config.mode == "Iter_Explanation": Tr.iter_gan_train()