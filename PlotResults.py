from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, confusion_matrix
import cv2 as cv

def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])

def plot_Con_results():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'DMO-AAENet', 'GSOA-AAENet', 'LOA-AAENet', 'MTBO-AAENet', 'MMTBO-AAENet']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']

    for i in range(1): # For 2 datasets
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)
    for n in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((5, 5))
        length = np.arange(Fitness.shape[2])
        Conv_Graph = Fitness[n]

        plt.plot(length, Conv_Graph[0, :], color='tab:blue', linewidth=2, label='DMO-AAENet')
        plt.plot(length, Conv_Graph[1, :], color='tab:orange', linewidth=2, label='GSOA-AAENet')
        plt.plot(length, Conv_Graph[2, :], color='tab:green', linewidth=2, label='LOA-AAENet')
        plt.plot(length, Conv_Graph[3, :], color='tab:red', linewidth=2, label='MTBO-AAENet')
        plt.plot(length, Conv_Graph[4, :], color='tab:purple', linewidth=2, label='MMTBO-AAENet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence_%s.png" % (n + 1))
        plt.show()


def Plot_ROC_Curve_Detection():
    lw = 2
    cls = ['D-R2UNet', 'CNN', 'Resnet', 'AAENet', 'MMTBO-AAENet']
    for a in range(1):  # For 2 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')

        colors = cycle(["dodgerblue", "magenta", "cornflowerblue", "green", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Detection_Y_Score' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Detection_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()

def Plot_ROC_Curve_Riskandsurvival():
    lw = 2
    cls = ['SVM', 'KNN', 'Adaboost', 'NN', 'CasANN']
    for a in range(1, 2):  # For 2 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')

        colors = cycle(["dodgerblue", "magenta", "cornflowerblue", "green", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Riskandsurvival_Y_Score' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/RiskandSurvival_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_results_Kfold():  # Table Results
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all_Kfold_Detection.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [3, 6, 9]
    Algorithm = ['TERMS', 'DMO-AAENet', 'GSOA-AAENet', 'LOA-AAENet', 'LOA-AAENet', 'MTBO-AAENet']
    Classifier = ['TERMS', 'D-R2UNet', 'CNN', 'Resnet', 'AAENet', 'MMTBO-AAENet']

    for i in range(1):
        for k in range(eval.shape[1]):
            value = eval[i, k, :, 4:]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], Terms[:3])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[j, :3])
            print('--------------------------------------------------Dataset', i + 1, '  Fold ', k + 1, '-Algorithm Comparison ',
                  '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Terms[:3])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :3])
            print('--------------------------------------------------  Dataset', i + 1, ' Fold ', k + 1, '-Classifier Comparison',
                  '--------------------------------------------------')
            print(Table)


def plot_Hidden_Neuron():
    eval1 = np.load('Eval_all_Hid_neu_RiskandSurvival.npy', allow_pickle=True)
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Hidden_neuron = [50, 100, 150, 200, 250]
    for i in range(eval1.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros((eval1.shape[1], eval1.shape[2]))
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]

            plt.plot(Hidden_neuron, Graph[:, 0], color='tab:blue', linewidth=3, marker="d", markerfacecolor='#000000',
                     markersize=12,
                     label="SVM")
            plt.plot(Hidden_neuron, Graph[:, 1], color='tab:orange', linewidth=3, marker="d", markerfacecolor='#ffffff',
                     markersize=12,
                     label="KNN")
            plt.plot(Hidden_neuron, Graph[:, 2], color='tab:green', linewidth=3, marker="d", markerfacecolor='#d607ed',
                     markersize=12,
                     label="Adaboost")
            plt.plot(Hidden_neuron, Graph[:, 3], color='tab:red', linewidth=3, marker="d", markerfacecolor='#6bed07',
                     markersize=12,
                     label="NN")
            plt.plot(Hidden_neuron, Graph[:, 4], color='tab:purple', linewidth=3, marker="d", markerfacecolor='cyan',
                     markersize=12,
                     label="CasANN")
            plt.xlabel('Hidden Neuron Count')
            plt.xticks(Hidden_neuron, ('50', '100', '150', '200', '250'))
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_line_risk and Survival prediction.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()



if __name__ == '__main__':
    Plot_ROC_Curve_Detection()
    plot_Con_results()
    Plot_ROC_Curve_Riskandsurvival()
    plot_results_Kfold()
    plot_Hidden_Neuron()


