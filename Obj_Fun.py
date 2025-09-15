import random
import numpy as np
from Global_Vars import Global_Vars
from Model_EfficientNet import Model_AAENet
from RELIEF import reliefF


def objfun_feat(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            for j in range(Feat.shape[0]):
                sol = np.round(Soln[i, :]).astype(np.int16)
                len_spect_feat = Feat.shape[1]
                feature_len = min(len_spect_feat)
                F = Feat[j, :feature_len]
                w = sol[0]
                weighted_feature = (w * F).astype('f')
                Feature = weighted_feature
                score = reliefF(Feature, Tar)
                Fitn[i] = score
        return Fitn

    else:
        sol = np.round(Soln).astype(np.int16)
        for j in range(Feat.shape[0]):
            len_spect_feat = Feat.shape[1]
            len_wave_feat = Feat.shape[1]
            feature_len = min(len_spect_feat).astype(int)  #, len_wave_feat)
            F = Feat[j, :feature_len]
            w = sol[0]
            weighted_feature = (w * F).astype('f')
            Feature = weighted_feature
            score = reliefF(Feature, Tar)
            Fitn = score
            return Fitn



def Obj_fun_CLS(Soln):
    Feat = Global_Vars.Feat
    Target = Global_Vars.Target
    learnperc = Global_Vars.learnperc
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
            Train_Data = Feat[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Feat[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            sol = np.round(Soln[i]).astype('uint8')
            Eval, pred = Model_AAENet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Fitn[i] = (1 / Eval[13]) + Eval[11]

        return Fitn
    else:
        sol = np.round(Soln).astype('uint8')
        learnperc = round(Feat.shape[0] * 0.75)  # Split Training and Testing Datas
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval, pred = Model_AAENet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Fitn = (1 / Eval[13]) + Eval[11]

        return Fitn


def objfun_Segmentation(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            # predict = Model_MaskRCNN(Feat)
            # EVAl = []
            # for img in range(len(predict)):
            #     Eval = net_evaluation(predict[img], Tar[img])
            #     EVAl.append(Eval)
            # mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)

            Fitn[i] = 1 / 10   # (mean_EVAl[0, 4]) # Dice Coefficient
        return Fitn
    else:
        # sol = np.round(Soln).astype(np.int16)
        # predict = Model_MaskRCNN(Feat)
        # EVAl = []
        # for img in range(len(predict)):
        #     Eval = net_evaluation(predict[img], Tar[img])
        #     EVAl.append(Eval)
        # mean_EVAl = np.mean(EVAl, axis=0, keepdims=True)
        Fitn = 1 / 10   # (mean_EVAl[0, 4])  # Dice Coefficient
        return Fitn