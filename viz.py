"""
This file was used to create the plots in the supplemental material.
"""

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pickle
import torch
import eval

def plot_accuracy_vs_batch(accuracies, batch_nos,
                           labels=None,
                           colors=['g','b','r','c','m','k'],
                           file_name = 'Accuracy_{}.{}'):
    # fig, ax = plt.subplots(figsize=(10, 3))
    fig, ax = plt.subplots(figsize=(10, 3))
    # fig, ax = plt.subplots()
    # ax.plot(batch_nos, accuracies, label=labels, color=colors[0], alpha=0.7)
    for acc, batch_no, label, color in zip(accuracies, batch_nos, labels, colors):
        ax.plot(batch_no, acc, label=label, color=color, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='lower center', bbox_to_anchor=(-0.25, 0.), ncol=1, fontsize=18, frameon=False)
    ax.set_xticks(range(min(batch_nos[0]),max(batch_nos[0])+1,50))
    ax.set_ylim([30.0,100.1])
    # ax.set_ylim([0.0001,100.1])
    plt.ylabel('% Accuracy',fontsize=15)
    plt.xlabel('Number of Classes',fontsize=15)
    plt.tight_layout()
    plt.savefig(file_name.format(f'{batch_nos[0][1]-batch_nos[0][0]}', 'pdf'), bbox_inches='tight')


def run_for_incremental_cls_variations():
    root_name = "/scratch/adhamija/NOWL/Incremental_Learning/InitialClasses-50_TotalClasses-100_NewClassesPerBatch-{}/no_of_exemplars_20/EVM_euclidean_EVMParams-1.0_0.7_0.7.pkl"
    all_plots=[]
    for i,l in zip([f"{_}" for _ in (2,5,10)],[f"{_} Cls/Step" for _ in (2,5,10)]):
    # for i in (0,5,10):
        results_for_all_batches=pickle.load(open(root_name.format(i),'rb'))
        acc_to_plot=[]
        batch_nos_to_plot = []
        for batch_no in sorted(results_for_all_batches.keys()):
            scores_order = np.array(sorted(results_for_all_batches[batch_no]['classes_order']))
            correct = 0.
            total = 0.
            for test_cls in list(set(results_for_all_batches[batch_no].keys()) - {'classes_order'}):
                scores = results_for_all_batches[batch_no][test_cls]
                total+=scores.shape[0]
                max_indx = torch.argmax(scores,dim=1)
                correct+=sum(scores_order[max_indx]==test_cls)
            acc = (correct/total)*100.
            acc_to_plot.append(acc)
            batch_nos_to_plot.append(scores_order.shape[0])
            print(f"Accuracy on Batch {batch_no} : {acc}")
        print(f"Average Accuracy {np.mean(acc_to_plot)}")
        all_plots.append((acc_to_plot, batch_nos_to_plot, l))
    all_acc_to_plot, all_batch_nos_to_plot, all_labels = zip(*all_plots)
    plot_accuracy_vs_batch(all_acc_to_plot, all_batch_nos_to_plot, all_labels, file_name="AccClsVariations_{}.{}")


def run_for_incremental():
    root_name = "/scratch/adhamija/NOWL/Incremental_Learning/InitialClasses-50_TotalClasses-100_NewClassesPerBatch-2/{}/EVM_euclidean_EVMParams-1.0_0.7_0.7.pkl"
    all_plots=[]
    for i,l in zip([f"no_of_exemplars_{_}" for _ in (0,10,20,50,100)]+["all_samples"],[f"{_} Exemplars" for _ in (0,10,20,50,100)]+["Upper Bound"]):
    # for i in (0,5,10):
        results_for_all_batches=pickle.load(open(root_name.format(i),'rb'))
        acc_to_plot=[]
        batch_nos_to_plot = []
        for batch_no in sorted(results_for_all_batches.keys()):
            scores_order = np.array(sorted(results_for_all_batches[batch_no]['classes_order']))
            correct = 0.
            total = 0.
            for test_cls in list(set(results_for_all_batches[batch_no].keys()) - {'classes_order'}):
                scores = results_for_all_batches[batch_no][test_cls]
                total+=scores.shape[0]
                max_indx = torch.argmax(scores,dim=1)
                correct+=sum(scores_order[max_indx]==test_cls)
            acc = (correct/total)*100.
            acc_to_plot.append(acc)
            batch_nos_to_plot.append(scores_order.shape[0])
            print(f"Accuracy on Batch {batch_no} : {acc}")
        print(f"Average Accuracy {np.mean(acc_to_plot)}")
        all_plots.append((acc_to_plot, batch_nos_to_plot, l))
    all_acc_to_plot, all_batch_nos_to_plot, all_labels = zip(*all_plots)
    plot_accuracy_vs_batch(all_acc_to_plot, all_batch_nos_to_plot, all_labels)


def run_for_openWorldLearning():
    # root_name = "/scratch/adhamija/NOWL_SWAV_OPEN/OpenWorld_Learning/InitialClasses-50_TotalClasses-500_NewClassesPerBatch-10/no_of_exemplars_20/EVM_euclidean_EVMParams-1.0_0.7_0.7.pkl"
    root_name = "/scratch/adhamija/NOWL/OpenWorld_Learning/InitialClasses-50_TotalClasses-500_NewClassesPerBatch-10/no_of_exemplars_20/EVM_euclidean_EVMParams-1.0_0.7_0.7.pkl"
    all_plots = []
    for file_name,l in zip([root_name],["MoCov2"]):
    # for file_name,l in zip([root_name],["SwAV"]):
        results_for_all_batches=pickle.load(open(file_name,'rb'))
        CCA = eval.calculate_CCA(results_for_all_batches)
        UDA, OCA, batches = eval.calculate_UDA_OCA(results_for_all_batches, unknowness_threshold=0.5)
        print(f"For Tabeling")
        print(f"{np.mean(UDA)} & {np.mean(OCA)} & {np.mean(CCA)}")
        print(f"{round(np.mean(UDA).astype(np.float64), 2)} & {round(np.mean(OCA).astype(np.float64), 2)} & {round(np.mean(CCA).astype(np.float64), 2)}")
        acc_to_plot = (CCA, UDA, OCA)
        batches_to_plot = (batches, batches, batches)
        # from IPython import embed; embed();
        plot_accuracy_vs_batch(acc_to_plot, batches_to_plot, (f"CCA",f"UDA",f"OCA"), file_name="OpenWorld_{}.{}")


if __name__ == "__main__":
    # run_for_incremental()
    run_for_openWorldLearning()
    # run_for_incremental_cls_variations()
