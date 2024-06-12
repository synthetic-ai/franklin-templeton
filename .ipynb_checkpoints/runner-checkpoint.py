#from pmlb import fetch_data

#from ydata_synthetic.synthesizers.regular import RegularSynthesizer
#from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import os
from io import BytesIO
import base64


def generate_synthetic_data(data):
    num_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass','education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'native-country', 'target']

    if path.exists("synthesizer_stock.pkl"):
        synth = RegularSynthesizer.load(r"/Users/sahilgupta/Franklin Templeton/synthesizer_stock.pkl")
    else:
        synth = RegularSynthesizer(modelname='fast')
        synth.fit(data=data, num_cols=num_cols, cat_cols=cat_cols)
        synth.save("synthesizer_stock.pkl")
    
    synth_data = synth.sample(len(data))
    return synth_data

def get_original_and_synthetic_data():
    data = pd.read_csv(r"/Users/sahilgupta/Franklin Templeton/original_data.csv", index_col=0)
    #synth_data = generate_synthetic_data(data)
    synth_data = pd.read_csv(r"/Users/sahilgupta/Franklin Templeton/synthetic_data.csv", index_col=0)
    original_data = data.copy()
    synthetic_data = synth_data.copy()
    return [original_data, synthetic_data]

def get_corr_mat(original_data, synthetic_data):
    # Combine datasets for visualization
    combined_data = pd.concat([original_data, synthetic_data])
    labels = ['Original'] * len(original_data) + ['Synthetic'] * len(synthetic_data)    
    orig_corr = original_data.corr()
    synth_corr = synthetic_data.corr()

    k2 = orig_corr.copy()
    # Correlation Matrix
    n = len(original_data.corr())
    for i in range(n):
        for j in range(n):
            k2.iloc[i,j] = abs(orig_corr.iloc[i,j]-synth_corr.iloc[i,j]/orig_corr.iloc[i,j])

    print("\n Generationg Correlation Matrix:(% Difference)")
    return k2

# PCA
def pca_plot(original_data, synthetic_data):
    # Combine datasets for visualization
    combined_data = pd.concat([original_data, synthetic_data])
    labels = ['Original'] * len(original_data) + ['Synthetic'] * len(synthetic_data)
    print("Performing PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_data)

    # Filter PCA results to remove y-values < 0 and x-values < -200000
    pca_filtered = pca_result[(pca_result[:, 1] >= 0) & (pca_result[:, 0] >= -200000)]
    labels_filtered_pca = np.array(labels)[(pca_result[:, 1] >= 0) & (pca_result[:, 0] >= -200000)]

    # Plot PCA results
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_filtered[labels_filtered_pca == 'Synthetic', 0], pca_filtered[labels_filtered_pca == 'Synthetic', 1], 
                label='Synthetic', alpha=0.5, c='red', marker='^', edgecolors='w', s=20)
    plt.scatter(pca_filtered[labels_filtered_pca == 'Original', 0], pca_filtered[labels_filtered_pca == 'Original', 1], 
                label='Original', alpha=0.5, c='black', marker='o', edgecolors='w', s=20)

    plt.legend()
    plt.title('PCA Comparison')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def get_original_vs_synthetic_stats(original_data, synthetic_data):
    o_m = original_data.mean()
    s_m = synthetic_data.mean()
    o_s = original_data.std()
    s_s = synthetic_data.std()

    ls = []
    for k in o_m.keys():
        ls.append([o_m[k], o_s[k], s_m[k], s_s[k]])

    df = pd.DataFrame(ls, columns=['original_mean', 'original_std', 'synthetic_mean', 'synthetic_std'], index=o_m.keys())
    df["mean_diff(%)"] = abs(df['original_mean'] - df['synthetic_mean'])/df['original_mean']
    df["std_diff(%)"] = abs(df['original_std'] - df['synthetic_std'])/df['original_std']

    df = df[["mean_diff(%)", "std_diff(%)"]]
    return df