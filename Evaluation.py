import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from math import log10, sqrt



def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def NegativeLivelihoodRatio(tnr, fnr):
    # Negative Likelihood Ratio (LR-)
    lrminus = tnr / fnr
    return lrminus


def DOR(lrplus, lrminus):
    # Diagnostic Odds Ratio (DOR)
    dor = lrplus / lrminus
    return dor


def evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(act.shape[0]):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp)[0])
    fp = np.squeeze(sum(Fp)[0])
    tn = np.squeeze(sum(Tn)[0])
    fn = np.squeeze(sum(Fn)[0])

    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100
    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100
    precision = tp / (tp + fp) * 100
    FPR = (fp / (fp + tn)) * 100
    FNR = (fn / (tp + fn)) * 100
    NPV = (tn / (tn + fn)) * 100
    FDR = (fp / (tp + fp)) * 100
    F1_score = ((2 * tp) / (2 * tp + fp + fn)) * 100
    MCC = (((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))) * 100
    # For = (fn / (fn + tn)) * 100
    # pt = np.math.sqrt(FPR) / (np.math.sqrt(sensitivity) + np.math.sqrt(sensitivity))
    # ba = (sensitivity + specificity) / 2
    # fm = np.math.sqrt(sensitivity * precision)
    # bm = sensitivity + specificity - 100
    # mk = precision + NPV - 100
    # PLHR = sensitivity / FPR
    # lrminus = NegativeLivelihoodRatio(specificity, FNR)
    # dor = (tp * tn) / (fp * fn)
    # prevalence = ((tp + fn) / (tp + fp + tn)) * 100
    # TS = (tp / (tp + fn + fp)) * 100
    EVAL = [tp, tn, fp, fn, accuracy, sensitivity, specificity, precision, FPR, FNR, NPV, FDR, F1_score, MCC]
            # , For, pt,ba, fm, bm, mk, PLHR, lrminus, dor, prevalence, TS]
    return EVAL

no_of_dataset = 1
