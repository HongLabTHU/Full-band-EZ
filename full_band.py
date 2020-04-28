# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.signal import spectrogram
from scipy.ndimage import gaussian_filter


def choose_kmeans_k(data, k_range):
    k_sse=[]
    for k in k_range:
        tmp_kmeans=KMeans(n_clusters=k)
        tmp_kmeans.fit(data)
        k_sse.append(tmp_kmeans.inertia_)
    k_sse=np.array(k_sse)
    k_sseDiff=-np.diff(k_sse)
    k_sseDiffMean=np.mean(k_sseDiff)
    best_index=np.where(k_sseDiff<k_sseDiffMean)[0][0]
    return k_range[best_index]


def find_ei_cluster_ratio(pei, labels, ei_elec_num=10):
    top_elec_ind = list(np.argsort(-pei)[:ei_elec_num])
    top_elec_labels = list(labels[top_elec_ind])
    top_elec_count = {}
    top_elec_set = set(top_elec_labels)
    for i in top_elec_set:
        top_elec_count[i] = top_elec_labels.count(i)
    cluster_ind1 = [k for k,v in top_elec_count.items() if v>ei_elec_num/2]
    if len(cluster_ind1):
        return np.array(cluster_ind1)
    else:
        cluster_ind2 = [k for k,v in top_elec_count.items() if v>ei_elec_num/3]
        if len(cluster_ind2):
            return np.array(cluster_ind2)
        else:
            return None


def pad_zero(data, length):
    data_len = len(data)
    if data_len < length:
        tmp_data = np.zeros(length)
        tmp_data[:data_len] = data
        return tmp_data
    return data


def cal_zscore(data):
    dmean = np.mean(data, axis=1)
    dstd = np.std(data, axis=1)
    norm_data = (data - dmean[:, None])/dstd[:, None]
    return norm_data
        

def cal_specs_matrix(raw, sfreq, method='STFT'):
    win_len = 0.2
    overlap = 0.9
    freq_range = 300
    half_width = win_len * sfreq
    ch_num = raw.shape[0]
    if method == 'STFT':
        for i in range(ch_num):
            print(str(i)+'/'+str(ch_num))
            time_signal = raw[i, :].ravel()
            time_signal = pad_zero(time_signal, 2 * half_width)
            f, t, hfo_spec = spectrogram(time_signal, fs=int(sfreq), nperseg=int(half_width),
                                         noverlap=int(overlap * half_width),
                                         nfft=2000, mode='magnitude')
            hfo_new = 20 * np.log10(hfo_spec + 1e-10)
            hfo_new = cal_zscore(hfo_new)
            hfo_new = gaussian_filter(hfo_new, sigma=2)
            hfo_new = hfo_new[:freq_range,:]
            tmp_specs = np.reshape(hfo_new, (-1,))
            if i == 0:
                chan_specs = tmp_specs
            else:
                chan_specs = np.row_stack((chan_specs,tmp_specs))
    f_cut = f[:freq_range]
    return chan_specs, hfo_new.shape, t, f_cut


def norm_specs(specs):
    specs_mean = specs - specs.mean(axis=0)
    specs_norm = specs_mean / specs_mean.std(axis=0)
    return specs_norm


def compute_full_band(raw_data, sfreq, ei):
    ei_elec_num = 10
    print('computing spectrogram')
    raw_specs, spec_shape, t, f = cal_specs_matrix(raw_data, sfreq, 'STFT')
    raw_specs_norm = norm_specs(raw_specs)
    print('dimensionality reducting')
    proj_pca = PCA(n_components=5)
    spec_pca = proj_pca.fit_transform(raw_specs_norm)
    top_elec_ind = np.argsort(-ei)[:ei_elec_num]
    top_elec_pca = np.zeros([ei_elec_num, spec_pca.shape[1]])
    for i in range(ei_elec_num):
        top_elec_pca[i] = spec_pca[top_elec_ind[i]]
    print('clustering')
    k_num = choose_kmeans_k(spec_pca, range(2,8))
    tmp_kmeans=KMeans(n_clusters=k_num)
    tmp_kmeans.fit(spec_pca)
    pre_labels = tmp_kmeans.labels_
    cluster_ind_ratio = find_ei_cluster_ratio(ei, pre_labels)
    
    chosen_cluster_ind = np.where(pre_labels==cluster_ind_ratio)[0]
    return spec_pca, pre_labels, chosen_cluster_ind
