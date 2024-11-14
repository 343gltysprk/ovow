import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   sklearn.decomposition import PCA
from   sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
from   sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_curve, auc, precision_recall_curve, f1_score
import torch

def plot_PDF(known_score,uknown_score,save_path="Probability_Density.png"):
    data = np.array(known_score+uknown_score)
    labels = np.array([0]*len(known_score) + [1]*len(uknown_score))
    fpr, tpr, _ = roc_curve(labels, data)
    roc_auc = auc(fpr, tpr)
    print(f"AUROC: {roc_auc}")
    known = ['Known']*len(known_score) + ['Unknown']*len(uknown_score)
    df = pd.DataFrame({'Score': data, 'Category': known})
    ax = sns.displot(data = df, x='Score', hue='Category', kde=True, stat="probability",common_norm=False, aspect=21/9)
    sns.move_legend(ax, "upper right")
    plt.xlabel('Score')
    plt.ylabel('Probability Density')
    plt.savefig(save_path,dpi=1200)