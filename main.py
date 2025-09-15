import os
from numpy import matlib
from DMO import DMO
from GSOA import GSOA
from Global_Vars import Global_Vars
from LOA import LOA
from MTBO import MTBO
from MMTBO import MMTBO
from Model_Adaboost import Model_Adaboost
from Model_CNN import Model_CNN_Cls
from Model_CasANN import Model_Cas_ANN
from Model_DCNN import Model_DCNN
from Model_EfficientNet import Model_AAENet
from Model_KNN import Model_KNN
from Model_NN import Model_NN
from Model_Resnet import Model_Resnet
from Model_SVM import Model_SVM
from Obj_Fun import Obj_fun_CLS
from PlotResults import *


def Read_Image(filename):  # Read image files
    images = cv.imread(filename)
    Image = cv.resize(images, (512, 512))
    # Image = cv.resize(image, (1050, 1610))
    return Image


def Read_Dataset_1():
    Directory = './Dataset/Dataset_1/'
    List_dir = os.listdir(Directory)
    Images_1 = []
    Labels_1 = []
    for i in range(len(List_dir)):
        In_Fold = Directory + List_dir[i]
        Img_array = os.listdir(In_Fold)
        for j in range(len(Img_array)):
            out_fold = In_Fold + '/' + Img_array[j]
            list_outfold = os.listdir(out_fold)
            for k in range(len(list_outfold)):
                if k == 0:
                    file_fold = out_fold + '/' + list_outfold[k]
                    list_filefold = os.listdir(file_fold)
                    for m in range(len(list_filefold)):
                        filename = file_fold + '/' + list_filefold[m]
                        Images_1.append(Read_Image(filename))
                else:
                    file_fold = out_fold + '/' + list_outfold[k]
                    list_filefold = os.listdir(file_fold)
                    for m in range(len(list_filefold)):
                        filename = file_fold + '/' + list_filefold[m]
                        Labels_1.append(Read_Image(filename))
    return Images_1, Labels_1


def Read_dataset_2():
    Dir = './Dataset/Dataset_2'
    List_Dir = os.listdir(Dir)
    Images_2 = []
    Labels_2 = []

    for i in range(len(List_Dir)):
        if List_Dir[i] == 'IDC_regular_ps50_idx5':
            continue
        else:
            In_fold = Dir + '/' + List_Dir[i]
            List_Infold = os.listdir(In_fold)

            for j in range(len(List_Infold)):
                if List_Infold[j] == '1':
                    Sub_fold = In_fold + '/' + List_Infold[j]
                    list_Sub_fold = os.listdir(Sub_fold)
                    for k in range(len(list_Sub_fold)):
                        filename = Sub_fold + '/' + list_Sub_fold[k]
                        file_read = Read_Image(filename)
                        Images_2.append(file_read)

                elif List_Infold[j] == '0':
                    Sub_fol = In_fold + '/' + List_Infold[j]
                    list_Sub_fol = os.listdir(Sub_fol)
                    for l in range(len(list_Sub_fol)):
                        filename = Sub_fol + '/' + list_Sub_fol[l]
                        file_rea = Read_Image(filename)
                        Labels_2.append(file_rea)
        return Images_2, Labels_2


# Read Dataset
an = 0
if an == 1:
    Images_1, Labels_1 = Read_Dataset_1()
    Images_2, Labels_2 = Read_dataset_2()
    np.save('Images_1.npy', Images_1)
    np.save('Labels_1.npy', Labels_1)
    np.save('Images_2.npy', Images_2)
    np.save('Labels_2.npy', Labels_2)

no_of_Dataset = 2

# Generate Target
an = 0
if an == 1:
    for n in range(no_of_Dataset):
        Tar = []
        Tot_Label = []
        Ground_Truth = np.load('Labels_' + str(n + 1) + '.npy', allow_pickle=True)

        for i in range(len(Ground_Truth)):
            image = Ground_Truth[i]
            if len(image.shape) == 3:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            if image.dtype != np.uint8:
                image = image.astype('np.uint8')

            # Applying threshold
            threshold = image

            # Apply the Component analysis function
            analysis = cv.connectedComponentsWithStats(threshold, 4, cv.CV_32S)
            (totalLabels, label_ids, values, centroid) = analysis
            Tot_Label.append(totalLabels)
        mean_value = np.mean(Tot_Label)

        for k in range(len(Tot_Label)):
            if Tot_Label[k] > mean_value:
                Tar.append(1)  # Abnormal
            else:
                Tar.append(0)  # Normal
            Tars = (np.asarray(Tar).reshape(-1, 1)).astype('int')
            np.save('Targets_' + str(n + 1) + '.npy', Tars)


# Optimization for classification
an = 0
if an == 1:
    Fit = []
    for n in range(no_of_Dataset):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)   # Load the images
        Target = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)  # Load the images
        Global_Vars.Feat = Images
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # 1 for Hidden Neuron Count in AAENet and 1 for No of epoches in AAENe, 1 for Learning rate in AAENet
        xmin = matlib.repmat([5, 5, 5], Npop, 1)
        xmax = matlib.repmat([255, 50, 255], Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(Chlen):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Obj_fun_CLS
        Max_iter = 50

        print("DMO...")
        [bestfit1, fitness1, bestsol1, time1] = DMO(initsol, fname, xmin, xmax, Max_iter)  # DMO

        print("GSOA...")
        [bestfit2, fitness2, bestsol2, time2] = GSOA(initsol, fname, xmin, xmax, Max_iter)  # GSOA

        print("LOA...")
        [bestfit3, fitness3, bestsol3, time3] = LOA(initsol, fname, xmin, xmax, Max_iter)  # LOA

        print("MTBO...")
        [bestfit4, fitness4, bestsol4, time4] = MTBO(initsol, fname, xmin, xmax, Max_iter)  # MTBO

        print("MMTBO...")
        [bestfit5, fitness5, bestsol5, time5] = MMTBO(initsol, fname, xmin, xmax, Max_iter)  # MMTBO

        Fitness = [fitness1, fitness2, fitness3, fitness4, fitness5]
        Bestsol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        Fit.append(Fitness)
        np.save('BestSol_' + str(n + 1) + '.npy', Bestsol)  # Save the Bestsoluton
    np.save('Fitness.npy', Fit)  # Save the Fittness


# classification for detection
an = 0
if an == 1:
    Data = [82, 70]
    for n in range(no_of_Dataset):
        Image = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)[:Data[n], :]
        Target = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)[:Data[n], :]  # loading step
        Bestsol = np.load('BestSol_' + str(n + 1) + '.npy', allow_pickle=True)  # loading step
        Evaluate = []
        K = 5
        Per = 1 / 5
        Perc = round(Image.shape[0] * Per)
        for i in range(K):
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Image.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Image[train_index, :]
            Train_Target = Target[train_index, :]
            Test_Data = Image[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            Eval = np.zeros((10, 14))
            for j in range(5):  # For 5 algorithms
                sol = np.round(Bestsol[i]).astype('uint8')
                Eval[n, :], preds = Model_AAENet(Train_Data, Train_Target, Test_Data,
                                              Test_Target, sol)  # Model_AAENet with optimization
            Eval[5, :], pred = Model_DCNN(Train_Data, Train_Target, Test_Data, Test_Target)  # D_R2UNet
            Eval[6, :], pred1 = Model_CNN_Cls(Train_Data, Train_Target, Test_Data, Test_Target)  # Model_CNN
            Eval[7, :], pred2 = Model_Resnet(Train_Data, Train_Target, Test_Data, Test_Target)  # Model_RESNET
            Eval[8, :], pred3 = Model_AAENet(Train_Data, Train_Target, Test_Data, Test_Target)  # Model_AAENet with optimization
            Eval[9, :] = Eval[4, :]  # Model Proposed
            Evaluate.append(Eval)
            np.save('Eval_all_Kfold_Detection.npy', Evaluate)  # Save Eval all

# classification for prediction
an = 0
if an == 1:
    for n in range(1, no_of_Dataset):
        Data = [82, 70]
        Image = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)[:Data[n], :]
        Target = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)[:Data[n], :]# loading step

        Evaluate = []
        Hid_Neu = [100, 200, 300, 400, 500]
        for learn in range(len(Hid_Neu)):
            Hid_Neuron = round(Image.shape[0] * 0.75)
            Train_Data = Image[:Hid_Neuron, :]
            Train_Target = Target[:Hid_Neuron, :]
            Test_Data = Image[Hid_Neuron:, :]
            Test_Target = Target[Hid_Neuron:, :]
            Eval = np.zeros((10, 14))
            Eval[5, :], pred = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target)  # Model_SVM
            Eval[6, :], pred1 = Model_KNN(Train_Data, Train_Target, Test_Data, Test_Target)  # Model_KNN
            Eval[7, :], pred2 = Model_Adaboost(Train_Data, Train_Target, Test_Data, Test_Target)  # Model_Adaboost
            Eval[8, :], pred3 = Model_NN(Train_Data, Train_Target, Test_Data, Test_Target)  # ARDNet with optimization
            Eval[9, :], pred4 = Model_Cas_ANN(Image, Target)  # Model Proposed
            Evaluate.append(Eval)
            np.save('Eval_all_Hid_neu_RiskandSurvival.npy', Evaluate)

Plot_ROC_Curve_Detection()
plot_Con_results()
Plot_ROC_Curve_Riskandsurvival()
plot_results_Kfold()
plot_Hidden_Neuron()