import tensorflow.keras as keras
from sklearn import metrics
import numpy as np
import LEAR_utils as utils
import LEAR_layers as layers
K = keras.backend

def ACC(y_true, y_pred):
    return K.mean(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

def MAUC(data, num_classes):
    """
    Calculates the MAUC over a set of multi-class probabilities and
    their labels. This is equation 7 in Hand and Till's 2001 paper.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        data (list): A zipped list (NOT A GENERATOR) of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        num_classes (int): The number of classes in the dataset.
    Returns:
        The MAUC as a floating point value.
    """
    # Find all pairwise comparisons of labels
    import itertools as itertools
    class_pairs = [x for x in itertools.combinations(range(num_classes), 2)]
    # Have to take average of A value with both classes acting as label 0 as this
    # gives different outputs for more than 2 classes
    sum_avals = 0
    for pairing in class_pairs:
        sum_avals += (a_value(data, zero_label=pairing[0], one_label=pairing[1]) + a_value(data, zero_label=pairing[1], one_label=pairing[0])) / 2.0
    return sum_avals * (2 / float(num_classes * (num_classes - 1)))  # Eqn 7


def a_value(probabilities, zero_label=0, one_label=1):
    """
    Approximates the AUC by the method described in Hand and Till 2001,
    equation 3.
    NB: The class labels should be in the set [0,n-1] where n = # of classes.
    The class probability should be at the index of its label in the
    probability list.
    I.e. With 3 classes the labels should be 0, 1, 2. The class probability
    for class '1' will be found in index 1 in the class probability list
    wrapped inside the zipped list with the labels.
    Args:
        probabilities (list): A zipped list of the labels and the
            class probabilities in the form (m = # data instances):
             [(label1, [p(x1c1), p(x1c2), ... p(x1cn)]),
              (label2, [p(x2c1), p(x2c2), ... p(x2cn)])
                             ...
              (labelm, [p(xmc1), p(xmc2), ... (pxmcn)])
             ]
        zero_label (optional, int): The label to use as the class '0'.
            Must be an integer, see above for details.
        one_label (optional, int): The label to use as the class '1'.
            Must be an integer, see above for details.
    Returns:
        The A-value as a floating point.
    """
    # Obtain a list of the probabilities for the specified zero label class
    expanded_points = []
    for instance in probabilities:
        if instance[0] == zero_label or instance[0] == one_label:
            expanded_points.append((instance[0].item(), instance[zero_label + 1].item()))
    sorted_ranks = sorted(expanded_points, key=lambda x: x[1])
    n0, n1, sum_ranks = 0, 0, 0
    # Iterate through ranks and increment counters for overall count and ranks of class 0
    for index, point in enumerate(sorted_ranks):
        if point[0] == zero_label:
            n0 += 1
            sum_ranks += index + 1  # Add 1 as ranks are one-based
        elif point[0] == one_label:
            n1 += 1
        else:
            pass  # Not interested in this class
    # print('Before: n0', n0, 'n1', n1, 'n0*n1', n0*n1)
    if n0 == 0:
        n0 = 1e-10
    elif n1 == 0:
        n1 = 1e-10
    else:
        pass
    # print('After: n0', n0, 'n1', n1, 'n0*n1', n0*n1)
    return (sum_ranks - (n0 * (n0 + 1) / 2.0)) / float(n0 * n1)  # Eqn 3

def evaluation_matrics(y_true, y_pred):
    acc = K.mean(K.equal(y_true, y_pred))
    auc = metrics.roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (fp + tn)
    return acc, auc, sen, spe

def ncc(a, v, zero_norm=False):
    """
    zero_norm = False:
    :return NCC

    zero_norm = True:
    :return ZNCC
    """
    if zero_norm:
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        v = (v - np.mean(v)) / np.std(v)
    else:
        a = (a) / (np.std(a) * len(a))  # obser = layers.flatten()(np.expand_dims(observed_map[idx], axis=0))
        v = (v) / np.std(v)  # pred = layers.flatten()(np.expand_dims(pred_map[idx], axis=0))
    return np.correlate(a, v)

def ncc_evaluation(cls_model, gen_model):
    long_nc, long_mci, long_ad = utils.Utils().load_longitudinal_data()
    nc_cls, mci_cls, ad_cls = cls_model(long_nc)["cls_out"], cls_model(long_mci)["cls_out"], cls_model(long_ad)["cls_out"]

    gt_adTOnc, gt_adTOmci, gt_mciTOnc = (long_nc - long_ad), (long_mci - long_ad), (long_nc - long_mci)
    gt_ncTOad, gt_mciTOad, gt_ncTOmci = (-gt_adTOnc), (-gt_adTOmci), (-gt_mciTOnc)

    nc_lbl = np.zeros(len(long_nc)).astype("int32")
    mci_lbl = np.ones(len(long_ad)).astype("int32")
    ad_lbl = (np.ones(len(long_ad)).astype("int32") + 1)

    # AD to NC
    adTOnc_ncc = 0
    c1, c2, c3, c4, c5 = utils.Utils().codemap(condition=nc_cls)
    cfmap = gen_model({"gen_in": long_ad, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, training=False)["gen_out"]
    pseudo_img = long_ad + cfmap
    pred = np.argmax(cls_model(pseudo_img)["cls_out"], axis=-1)
    acc1 = K.mean(K.equal(nc_lbl, pred))  ##
    gtmap, cfmap = layers.flatten()(np.array(gt_adTOnc).astype("float32")), layers.flatten()(cfmap)
    for i in range(len(cfmap)): adTOnc_ncc += ncc(a=gtmap[i], v=cfmap[i])
    adTOnc_ncc /= len(cfmap)  ##

    # AD to MCI
    adTOmci_ncc = 0
    c1, c2, c3, c4, c5 = utils.Utils().codemap(condition=mci_cls)
    cfmap = gen_model({"gen_in": long_ad, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, training=False)["gen_out"]
    pseudo_img = long_ad + cfmap
    pred = np.argmax(cls_model(pseudo_img)["cls_out"], axis=-1)
    acc2 = K.mean(K.equal(mci_lbl, pred))  ##
    gtmap, cfmap = layers.flatten()(np.array(gt_adTOmci).astype("float32")), layers.flatten()(cfmap)
    for i in range(len(cfmap)): adTOmci_ncc += ncc(a=gtmap[i], v=cfmap[i])
    adTOmci_ncc /= len(cfmap)  ##

    # MCI to NC
    mciTOnc_ncc = 0
    c1, c2, c3, c4, c5 = utils.Utils().codemap(condition=nc_cls)
    cfmap = gen_model({"gen_in": long_mci, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, training=False)["gen_out"]
    pseudo_img = long_mci + cfmap
    pred = np.argmax(cls_model(pseudo_img)["cls_out"], axis=-1)
    acc3 = K.mean(K.equal(nc_lbl, pred))  ##
    gtmap, cfmap = layers.flatten()(np.array(gt_mciTOnc).astype("float32")), layers.flatten()(cfmap)
    for i in range(len(cfmap)): mciTOnc_ncc += ncc(a=gtmap[i], v=cfmap[i])
    mciTOnc_ncc /= len(cfmap)  ##

    ###############################

    # NC to AD
    ncTOad_ncc = 0
    c1, c2, c3, c4, c5 = utils.Utils().codemap(condition=ad_cls)
    cfmap = gen_model({"gen_in": long_nc, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, training=False)["gen_out"]
    pseudo_img = long_nc + cfmap
    pred = np.argmax(cls_model(pseudo_img)["cls_out"], axis=-1)
    acc4 = K.mean(K.equal(ad_lbl, pred))  ##
    gtmap, cfmap = layers.flatten()(np.array(gt_ncTOad).astype("float32")), layers.flatten()(cfmap)
    for i in range(len(cfmap)): ncTOad_ncc += ncc(a=gtmap[i], v=cfmap[i])
    ncTOad_ncc /= len(cfmap)  ##

    # NC to MCI
    ncTOmci_ncc = 0
    c1, c2, c3, c4, c5 = utils.Utils().codemap(condition=mci_cls)
    cfmap = gen_model({"gen_in": long_nc, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, training=False)["gen_out"]
    pseudo_img = long_nc + cfmap
    pred = np.argmax(cls_model(pseudo_img)["cls_out"], axis=-1)
    acc5 = K.mean(K.equal(mci_lbl, pred))  ##
    gtmap, cfmap = layers.flatten()(np.array(gt_ncTOmci).astype("float32")), layers.flatten()(cfmap)
    for i in range(len(cfmap)): ncTOmci_ncc += ncc(a=gtmap[i], v=cfmap[i])
    ncTOmci_ncc /= len(cfmap)  ##

    # MCI to AD
    mciTOad_ncc = 0
    c1, c2, c3, c4, c5 = utils.Utils().codemap(condition=ad_cls)
    cfmap = gen_model({"gen_in": long_mci, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, training=False)["gen_out"]
    pseudo_img = long_nc + cfmap
    pred = np.argmax(cls_model(pseudo_img)["cls_out"], axis=-1)
    acc6 = K.mean(K.equal(ad_lbl, pred))  ##
    gtmap, cfmap = layers.flatten()(np.array(gt_mciTOad).astype("float32")), layers.flatten()(cfmap)
    for i in range(len(cfmap)): mciTOad_ncc += ncc(a=gtmap[i], v=cfmap[i])
    mciTOad_ncc /= len(cfmap)  ##

    return acc1, acc2, acc3, acc4, acc5, acc6, adTOnc_ncc, adTOmci_ncc, mciTOnc_ncc, ncTOad_ncc, ncTOmci_ncc, mciTOad_ncc