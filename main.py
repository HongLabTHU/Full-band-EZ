# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import scipy
from scipy.signal import spectrogram
from collections import Counter
from scipy.ndimage import gaussian_filter
import h5py

import ei
import full_band


# classes
class show_signal(object):
    def __init__(self, canvas, ax, raw_data, sfreq, ch_names, press_type, pca_data=None):
        self.canvas = canvas
        self.ax = ax
        self.raw_data = raw_data.copy()
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.spec_win_len = 0.2
        self.spec_overlap = 0.9
        self.press_type = press_type
        self.pca_data = pca_data
        self.draw_ind = 0
        self.canvas.mpl_connect('button_press_event', self.button_press_func)

    def draw_specs_single(self):
        raw = self.raw_data
        sfreq = self.sfreq
        half_width = self.spec_win_len * sfreq
        time_signal = raw[self.draw_ind, :]
        # pad zero
        if len(time_signal) < 2*half_width:
            tmp_data = np.zeros(2*half_width)
            tmp_data[:len(time_signal)] = time_signal
            time_signal = tmp_data
        f, t, hfo_spec = spectrogram(time_signal, fs=int(sfreq), nperseg=int(half_width),
                                     noverlap=int(self.spec_overlap * half_width),
                                     nfft=sfreq, mode='magnitude')
        hfo_new = 20 * np.log10(hfo_spec + 1e-10)
        # cal zscore
        dmean = np.mean(hfo_new, axis=1)
        dstd = np.std(hfo_new, axis=1)
        hfo_new2 = (hfo_new-dmean[:,None])/dstd[:,None]
        hfo_new3 = gaussian_filter(hfo_new2, sigma=2)

        # draw original signal
        chan_name = self.ch_names[self.draw_ind]
        fig = plt.figure(chan_name, figsize=(8, 6))
        ax_raw = fig.add_axes([0.2, 0.55, 0.65, 0.35])
        ax_raw.cla()
        ax_raw.plot(time_signal)
        ax_raw.set_title(chan_name + ' signal')
        ax_raw.set_xlim([0, time_signal.shape[0]])
        ax_raw.set_xticks([])
        ax_raw.set_ylabel('signal(uV)')

        # draw spectrogram
        ax_spec = fig.add_axes([0.2, 0.15, 0.65, 0.35])
        ax_spec.cla()
        ax_spec.set_title(chan_name + ' spectrogram')
        surf = ax_spec.pcolormesh(t, f, hfo_new3, cmap='hot', vmax=2, vmin=-1)
        ax_spec.set_ylim((0, 300))
        ax_spec.set_xlabel('time(s)')
        ax_spec.set_ylabel('frequency(HZ)')
        position = fig.add_axes([0.88, 0.15, 0.01, 0.35])
        cb = plt.colorbar(surf, cax=position)
        plt.show()

    def button_press_func(self, e):
        # press point coordinates
        x = e.xdata
        y = e.ydata
        if self.press_type == 1:
            idx_min = int(x)
        elif self.press_type == 2:
            idx_min = int(y)
        elif self.press_type == 3:
            pca_data = self.pca_data
            # nearest point index
            distance = np.sum((np.array(pca_data[:, 0:2]) - np.array([x, y])) ** 2, axis=-1)
            idx_min = np.argmin(distance)
        self.draw_ind = idx_min
        self.draw_specs_single()


# input data function
def input_data(mat_filename):
    mat_data = h5py.File(mat_filename)
    modified_mat_data = np.transpose(mat_data['mat_data'])
    fs = mat_data['fs'][0][0]
    ch_labels = mat_data['channel_labels']
    chans_list = []
    for i in range(ch_labels.shape[0]):
        tmp_ch_name = ''.join([chr(v[0]) for v in mat_data[(ch_labels[i][0])]])
        chans_list.append(tmp_ch_name)
    data = {'modified_mat_data': modified_mat_data, 'fs': fs, 'chans_list': chans_list}
    print('data loaded')
    return data


# filter data function
def filter_data(modified_mat_data, fs, band_low, band_high):
    nyq = fs / 2
    b, a = scipy.signal.butter(5, np.array([band_low / nyq, band_high / nyq]), btype='bandpass')
    new_modified_mat_data = scipy.signal.filtfilt(b, a, modified_mat_data)
    return new_modified_mat_data


# set default target and baseline
def cut_data(modified_mat_data):
    target_data = modified_mat_data
    baseline_data = modified_mat_data[:, :int(modified_mat_data.shape[1]/5)]
    return target_data, baseline_data


# refresh electrodes' name and number
def refresh_electrodes_info(chans_list):
    tmp_chs_names = chans_list
    tmp_chlabel_list = [x[0] if x[1].isdigit() else x[:2] for x in tmp_chs_names]
    chs_counter = Counter(tmp_chlabel_list)
    chs_info = chs_counter.items()
    single_chns = [x for x in chs_info if len(x[0]) == 1]
    double_chns = [x for x in chs_info if len(x[0]) == 2]
    single_chns.sort(key=lambda x: x[0])
    double_chns.sort(key=lambda x: x[0])
    if double_chns != []:
        elecs_info = np.concatenate([np.array(single_chns), np.array(double_chns)], axis=0)
    else:
        elecs_info = np.array(single_chns)
    return elecs_info, chs_info


# hfer plot function
def hfer_plot_func(data):
    modified_mat_data = data['modified_mat_data']
    fs = data['fs']
    chans_list = data['chans_list']
    # hfer computation
    filted_data = filter_data(modified_mat_data, fs, 60, 140)
    target_data, baseline_data = cut_data(filted_data)
    norm_target, norm_base = ei.compute_hfer(target_data, baseline_data, fs)
    # plot hfer
    elecs_info, chs_info = refresh_electrodes_info(chans_list)
    elec_labels = [x[0] for x in elecs_info]
    elec_nums = [int(x[1]) for x in elecs_info]
    hfer_fig = plt.figure('hfer')
    hfer_ax = hfer_fig.add_axes([0.1, 0.1, 0.7, 0.8])
    tmp_x, tmp_y = np.meshgrid(np.arange(norm_target.shape[1] + 1), np.arange(norm_target.shape[0] + 1))
    surf = hfer_ax.pcolormesh(tmp_x, tmp_y, norm_target, cmap=plt.cm.jet, vmax=50, vmin=0)
    hfer_ax.set_xticks(np.arange(0, norm_target.shape[1], 2000))
    hfer_ax.set_xticklabels(np.rint(np.arange(0, norm_target.shape[1],2000) / float(fs)).astype(np.int16), fontsize=6)
    hfer_ax.set_xlabel('time(s)')
    hfer_ax.set_ylabel('channels')
    color_cums = np.cumsum(elec_nums)
    tmp_color_cums = np.concatenate([np.array([0]), color_cums])
    y_ticks = [tmp_color_cums[i - 1] for i in range(1, len(tmp_color_cums))]
    hfer_ax.set_yticks(y_ticks)
    hfer_ax.set_yticklabels(elec_labels)
    # colorbar
    color_bar_ax = hfer_fig.add_axes([0.85, 0.1, 0.02, 0.8])
    plt.colorbar(surf, cax=color_bar_ax, orientation='vertical')

    # press function
    show_signal_handle = show_signal(hfer_fig.canvas, hfer_ax, modified_mat_data, fs, chans_list, 2)
    plt.show()


# ei plot function
def ei_plot_func(data):
    modified_mat_data = data['modified_mat_data']
    fs = data['fs']
    chans_list = data['chans_list']
    # calculate ei
    filted_data = filter_data(modified_mat_data, fs, 60, 140)
    target_data, baseline_data = cut_data(filted_data)
    norm_target, norm_base = ei.compute_hfer(target_data, baseline_data, fs)
    ei_ei, ei_hfer, ei_onset_rank = ei.compute_ei_index(norm_target, norm_base, fs)
    ei_thresh = np.mean(ei_ei) + np.std(ei_ei)

    # plot ei
    ei_ei_fig = plt.figure('ei')
    ei_ei_ax = ei_ei_fig.add_subplot(111)
    elecs_info, chs_info = refresh_electrodes_info(chans_list)
    elec_labels = [x[0] for x in elecs_info]
    elec_nums = [int(x[1]) for x in elecs_info]
    color_rgb = cm.jet(np.arange(len(elec_nums)) / float(len(elec_nums)))
    color_cums = np.cumsum(elec_nums)
    ei_ei_ax.bar(range(color_cums[0]), ei_ei[:color_cums[0]], color=color_rgb[0])
    for j in range(1, len(elec_nums)):
        ei_ei_ax.bar(range(color_cums[j - 1], color_cums[j]), ei_ei[color_cums[j - 1]:color_cums[j]],
                     color=color_rgb[j])
    tmp_color_cums = np.concatenate([np.array([0]), color_cums])
    x_ticks = [(tmp_color_cums[i - 1] + tmp_color_cums[i]) / 2.0 for i in range(1, len(tmp_color_cums))]
    ei_ei_ax.set_xticks(x_ticks)
    ei_ei_ax.set_xticklabels(elec_labels)
    ei_ei_ax.plot(np.arange(len(ei_ei)), ei_thresh * np.ones(len(ei_ei)), 'r--')
    ei_ei_ax.spines['top'].set_visible(False)
    ei_ei_ax.spines['right'].set_visible(False)

    # press function
    show_signal_handle = show_signal(ei_ei_fig.canvas, ei_ei_ax, modified_mat_data, fs, chans_list, 1)
    plt.show()

    # plot ei top 10 channels
    ei_top10_fig = plt.figure('ei top 10')
    ei_top10_ax = ei_top10_fig.add_subplot(111)
    x = np.linspace(0, len(ei_ei) - 1, len(ei_ei))
    ei_top10_ax.bar(x, ei_ei, color=[0.8, 0.8, 0.8])
    sorted_ei = np.sort(ei_ei)
    sorted_ei_arg = np.argsort(ei_ei)
    ei_top10_ax.bar(sorted_ei_arg[-10:], sorted_ei[-10:], color='orange')
    elecs_info, chs_info = refresh_electrodes_info(chans_list)
    elec_labels = [x[0] for x in elecs_info]
    elec_nums = [int(x[1]) for x in elecs_info]
    color_cums = np.cumsum(elec_nums)
    tmp_color_cums = np.concatenate([np.array([0]), color_cums])
    x_ticks = [(tmp_color_cums[i - 1] + tmp_color_cums[i]) / 2.0 for i in range(1, len(tmp_color_cums))]
    ei_top10_ax.set_xticks(x_ticks)
    ei_top10_ax.set_xticklabels(elec_labels)
    ei_top10_ax.spines['top'].set_visible(False)
    ei_top10_ax.spines['right'].set_visible(False)

    # press function
    show_signal_handle = show_signal(ei_top10_fig.canvas, ei_ei_ax, modified_mat_data, fs, chans_list, 1)
    plt.show()


# full band plot function
def full_band_plot_func(data):
    modified_mat_data = data['modified_mat_data']
    fs = data['fs']
    chans_list = data['chans_list']
    # ei computation
    filted_data = filter_data(modified_mat_data, fs, 60, 140)
    target_data, baseline_data = cut_data(filted_data)
    norm_target, norm_base = ei.compute_hfer(target_data, baseline_data, fs)
    ei_ei, ei_hfer, ei_onset_rank = ei.compute_ei_index(norm_target, norm_base, fs)
    spec_pca, fullband_labels, fullband_ind = full_band.compute_full_band(modified_mat_data, fs, ei_ei)
    chs_labels = np.array(chans_list)[fullband_ind]
    print('electrodes:', chs_labels)

    # plot full band result
    fullband_fig = plt.figure('full_band')
    fullband_ax = fullband_fig.add_subplot(111)
    fullband_ax.scatter(spec_pca[:, 0], spec_pca[:, 1], alpha=0.8, c=fullband_labels)
    for ind in fullband_ind:
        fullband_ax.text(spec_pca[ind, 0], spec_pca[ind, 1], chans_list[ind], fontsize=8, color='k')

    # press function
    show_signal_handle = show_signal(fullband_fig.canvas, fullband_ax, modified_mat_data, fs, chans_list, 3, spec_pca)
    plt.show()

    # highlight epileptogenic cluster and HFEI top 10 channels
    # orange represents HFEI top 10 channels, yellow and the text represent epileptogenic cluster
    fullband_hl_fig = plt.figure('full_band_highlight')
    fullband_hl_ax = fullband_hl_fig.add_subplot(111)
    ei_top_ind = np.argsort(-ei_ei)[:10]
    for i in range(len(fullband_labels)):
        if i in ei_top_ind:
            fullband_hl_ax.scatter(spec_pca[i, 0], spec_pca[i, 1], alpha=0.8, c='orange')
        elif i in fullband_ind:
            fullband_hl_ax.scatter(spec_pca[i, 0], spec_pca[i, 1], alpha=0.8, c='yellow')
        else:
            fullband_hl_ax.scatter(spec_pca[i, 0], spec_pca[i, 1], alpha=0.8, c='gray')
    for ind in fullband_ind:
        fullband_hl_ax.text(spec_pca[ind, 0], spec_pca[ind, 1], chans_list[ind], fontsize=8, color='k')

    # press function
    show_signal_handle = show_signal(fullband_hl_fig.canvas, fullband_ax, modified_mat_data, fs, chans_list, 3,
                                     spec_pca)
    plt.show()


if __name__ == '__main__':

    # input data
    mat_filename = './data/S1.mat'
    data = input_data(mat_filename)

    # plot hfer result
    hfer_plot_func(data)

    # plot ei result
    ei_plot_func(data)

    # plot full band result
    full_band_plot_func(data)


