# encoding=utf-8
import sys

from PyQt5.QtWidgets import QApplication,  QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QLineEdit, QDesktopWidget, QGridLayout, QFileDialog,  QListWidget, QLabel
from PyQt5.QtCore import Qt, QThread
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as QtCore

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_axes([0.05, 0.1, 0.9, 0.8])
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):
        self.axes.cla()
        self.draw()


class figure_thread(QThread):
    def __init__(self, parent=None):
        super(figure_thread, self).__init__(parent=parent)
        self.ei = parent.ei_ei

    def run(self):
        pass


class fullband_computation_thread(QThread):
    fullband_done_sig = QtCore.pyqtSignal(object)

    def __init__(self, parent=None, raw_signal=None, ei=None, fs=2000):
        super(fullband_computation_thread, self).__init__(parent=parent)
        self.raw_signal = raw_signal
        self.fs = fs
        self.ei = ei

    def run(self):
        spec_pca, fullband_labels, fullband_ind = full_band.compute_full_band(self.raw_signal, self.fs, self.ei)
        fullband_res = [spec_pca, fullband_labels, fullband_ind]
        self.fullband_done_sig.emit(fullband_res)


# main class
class Brainquake(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    # init function
    def initUI(self):
        # variables
        # self.patient_name=None
        # main window
        self.setWindowTitle('Brainquake')
        self.resize(1500, 700)
        self.center()
        self.setStyleSheet("background-color:#bcd0d8;")
        self.gridlayout = QGridLayout()
        # canvas
        self.canvas = PlotCanvas(self, width=10, height=8)
        self.gridlayout.addWidget(self.canvas, 0, 0, 24, 25)
        self.canvas.fig.canvas.mpl_connect('button_press_event', self.canvas_press_button)
        self.canvas.fig.canvas.mpl_connect('scroll_event', self.disp_scroll_mouse)

        # input data
        self.lineedit_patient_name = QLineEdit(self)
        self.lineedit_patient_name.setText('patient name')
        self.lineedit_patient_name.setToolTip('please input the patient name')
        self.lineedit_patient_name.setStyleSheet(
            "QLineEdit{border-style:none;border-radius:5px;padding:5px;background-color:#ffffff}QLineEdit:focus{border:2px solid gray;}")
        self.gridlayout.addWidget(self.lineedit_patient_name, 1, 26, 1, 1)

        self.upload_label = QLabel(self)
        self.upload_label.setText('')
        self.gridlayout.addWidget(self.upload_label, 1, 27, 1, 1)

        self.lineedit_doctor_name = QLineEdit(self)
        self.lineedit_doctor_name.setText('doctor')
        self.lineedit_doctor_name.setToolTip('please input the doctor name')
        self.lineedit_doctor_name.setStyleSheet(
            "QLineEdit{border-radius:5px;padding:5px;background-color:#ffffff}QLineEdit:focus{border:2px solid gray;}")
        self.gridlayout.addWidget(self.lineedit_doctor_name, 2, 26, 1, 1)

        self.lineedit_hospital_name = QLineEdit(self)
        self.lineedit_hospital_name.setText('hospital')
        self.lineedit_hospital_name.setToolTip('please input the hospital name')
        self.lineedit_hospital_name.setStyleSheet(
            "QLineEdit{border-radius:5px;padding:5px;background-color:#ffffff}QLineEdit:focus{border:2px solid gray;}")
        self.gridlayout.addWidget(self.lineedit_hospital_name, 2, 27, 1, 1)
        self.button_inputedf = QPushButton('input data', self)
        self.button_inputedf.setToolTip('click to input data')
        self.button_inputedf.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.button_inputedf, 3, 26, 1, 2)
        self.button_inputedf.clicked.connect(self.dialog_inputdata)
        # display data
        self.reset_data_display = QPushButton('reset', self)
        self.reset_data_display.setToolTip('reset data')
        self.reset_data_display.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.reset_data_display, 5, 26, 1, 2)
        self.reset_data_display.clicked.connect(self.reset_data_display_func)
        # win up down
        self.dis_down = QPushButton('win down', self)
        self.dis_down.setToolTip('roll window down')
        self.dis_down.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.dis_down, 6, 26)
        self.dis_down.clicked.connect(self.disp_win_down_func)  # change value & one common display func

        self.dis_up = QPushButton('win up', self)
        self.dis_up.setToolTip('roll window up')
        self.dis_up.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.dis_up, 6, 27)
        self.dis_up.clicked.connect(self.disp_win_up_func)  # change value & one common display func
        # channels num
        self.dis_more_chans = QPushButton('chans+', self)
        self.dis_more_chans.setToolTip('more channels')
        self.dis_more_chans.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.dis_more_chans, 7, 26)
        self.dis_more_chans.clicked.connect(self.disp_more_chans_func)  # change value & one common display func

        self.dis_less_chans = QPushButton('chans-', self)
        self.dis_less_chans.setToolTip('less channels')
        self.dis_less_chans.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.dis_less_chans, 7, 27)
        self.dis_less_chans.clicked.connect(self.disp_less_chans_func)  # change value & one common display func
        # wave mag
        self.dis_add_mag = QPushButton('wave+', self)
        self.dis_add_mag.setToolTip('wave magnitude up')
        self.dis_add_mag.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.dis_add_mag, 8, 26)
        self.dis_add_mag.clicked.connect(self.disp_add_mag_func)  # change value & one common display func

        self.dis_drop_mag = QPushButton('wave-', self)
        self.dis_drop_mag.setToolTip('wave magnitude down')
        self.dis_drop_mag.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.dis_drop_mag, 8, 27)
        self.dis_drop_mag.clicked.connect(self.disp_drop_mag_func)  # change value & one common display func
        # win left right
        self.dis_left = QPushButton('left', self)
        self.dis_left.setToolTip('roll window left')
        self.dis_left.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.dis_left, 9, 26)
        self.dis_left.clicked.connect(self.disp_win_left_func)  # change value & one common display func

        self.dis_right = QPushButton('right', self)
        self.dis_right.setToolTip('roll window right')
        self.dis_right.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.dis_right, 9, 27)
        self.dis_right.clicked.connect(self.disp_win_right_func)  # change value & one common display func
        # time scale
        self.dis_shrink_time = QPushButton('shrink', self)
        self.dis_shrink_time.setToolTip('shrink time scale')
        self.dis_shrink_time.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.dis_shrink_time, 10, 26)
        self.dis_shrink_time.clicked.connect(self.disp_shrink_time_func)  # change value & one common display func

        self.dis_expand_time = QPushButton('expand', self)
        self.dis_expand_time.setToolTip('expand time scale')
        self.dis_expand_time.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.dis_expand_time, 10, 27)
        self.dis_expand_time.clicked.connect(self.disp_expand_time_func)  # change value & one common display func
        # filter data
        self.disp_filter_low = QLineEdit(self)
        self.disp_filter_low.setText('60')
        self.disp_filter_low.setToolTip('filter low boundary')
        self.disp_filter_low.setStyleSheet(
            "QLineEdit{border-radius:5px;padding:5px;background-color:#ffffff}QLineEdit:focus{border:2px solid gray;}")
        self.gridlayout.addWidget(self.disp_filter_low, 12, 26)

        self.disp_filter_high = QLineEdit(self)
        self.disp_filter_high.setText('140')
        self.disp_filter_high.setToolTip('filter high boudary')
        self.disp_filter_high.setStyleSheet(
            "QLineEdit{border-radius:5px;padding:5px;background-color:#ffffff}QLineEdit:focus{border:2px solid gray;}")
        self.gridlayout.addWidget(self.disp_filter_high, 12, 27)

        self.filter_button = QPushButton('filter', self)
        self.filter_button.setToolTip('filter the data')
        self.filter_button.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.filter_button, 13, 26, 1, 2)
        self.filter_button.clicked.connect(self.filter_data)
        # del channels
        self.chans_list = QListWidget(self)
        self.chans_list.setToolTip('choose chans to delete')
        self.chans_list.setStyleSheet("border-radius:5px;padding:5px;background-color:#ffffff;")
        self.chans_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff);
        self.chans_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.gridlayout.addWidget(self.chans_list, 15, 26, 3, 2)

        self.chans_del_button = QPushButton(self)
        self.chans_del_button.setText('delete chans')
        self.chans_del_button.setToolTip('delete channels')
        self.chans_del_button.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.chans_del_button, 18, 26, 1, 2)
        self.chans_del_button.clicked.connect(self.delete_chans)
        # baseline time and target time selection
        self.baseline_button = QPushButton(self)
        self.baseline_button.setText('baseline')
        self.baseline_button.setToolTip('choose baseline time')
        self.baseline_button.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.baseline_button, 20, 26, 1, 1)
        self.baseline_button.clicked.connect(self.choose_baseline)

        self.target_button = QPushButton(self)
        self.target_button.setText('target')
        self.target_button.setToolTip('choose target time')
        self.target_button.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.target_button, 20, 27, 1, 1)
        self.target_button.clicked.connect(self.choose_target)
        # ei
        self.ei_button = QPushButton(self)
        self.ei_button.setText('ei')
        self.ei_button.setToolTip('compute ei')
        self.ei_button.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.ei_button, 21, 26, 1, 1)
        self.ei_button.clicked.connect(self.ei_computation_func)
        self.ei_button.setEnabled(False)

        # hfer
        self.hfer_button = QPushButton(self)
        self.hfer_button.setText('hfer')
        self.hfer_button.setToolTip('compute hfer')
        self.hfer_button.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.hfer_button, 21, 27, 1, 1)
        self.hfer_button.clicked.connect(self.hfer_computation_func)
        self.hfer_button.setEnabled(False)

        # fullband
        self.fullband_button = QPushButton(self)
        self.fullband_button.setText('full band')
        self.fullband_button.setToolTip('compute full band characteristic')
        self.fullband_button.setStyleSheet(
            "QPushButton{border-radius:5px;padding:5px;color:#ffffff;background-color:#087fb2;}QPushButton:hover{background-color:#045577;}")
        self.gridlayout.addWidget(self.fullband_button, 22, 26, 1, 1)
        self.fullband_button.clicked.connect(self.fullband_computation_func)
        self.fullband_button.setEnabled(False)

        # show main window
        self.setLayout(self.gridlayout)
        self.show()

    # set functions
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # input data
    def dialog_inputdata(self):
        self.mat_filename, b = QFileDialog.getOpenFileName(self, 'open file', '/home')
        if self.mat_filename:
            # load data
            self.patient_name = self.lineedit_patient_name.text()
            self.doctor_name = self.lineedit_doctor_name.text()
            self.hospital_name = self.lineedit_hospital_name.text()
            self.submit_name = self.patient_name + '_' + self.doctor_name + '_' + self.hospital_name
            self.edf_data = h5py.File(self.mat_filename)
            self.modified_edf_data = np.transpose(self.edf_data['mat_data'])
            tmp_fs = self.edf_data['fs']
            self.fs = tmp_fs[0][0]
            self.band_low = 1.0
            self.band_high = 500
            self.upload_label.setText('')
            self.edf_time_max = self.modified_edf_data.shape[1]/self.fs

            QMessageBox.information(self, '', 'data loaded')
            # init display params
            self.init_display_params()
            self.disp_refresh()

    # init display
    def init_display_params(self):
        self.disp_chans_num = 5
        self.disp_chans_start = 0
        self.disp_wave_mul = 10
        self.disp_time_win = 5
        self.disp_time_start = 0

        self.baseline_pos = np.array([0.0, int(self.edf_time_max / 5)])
        self.target_pos = np.array([0.0, self.edf_time_max])
        self.baseline_mouse = 0
        self.target_mouse = 0
        self.ei_target_start = self.target_pos[0]
        self.ei_target_end = self.target_pos[1]
        self.chans_list.clear()
        ch_labels = self.edf_data['channel_labels']
        tmp_chans_list = []
        print(ch_labels.shape)
        for i in range(ch_labels.shape[0]):
            tmp_ch_name = ''.join([chr(v[0]) for v in self.edf_data[(ch_labels[i][0])]])
            tmp_chans_list.append(tmp_ch_name)
        self.chans_list.addItems(tmp_chans_list)
        self.disp_ch_names = tmp_chans_list
        self.modified_edf_data = np.transpose(self.edf_data['mat_data'])

        self.edf_time = self.modified_edf_data.shape[1]/self.fs
        self.edf_nchans = len(self.chans_list)
        self.edf_line_colors = np.array([cm.jet(x) for x in np.random.rand(self.edf_nchans)])
        self.edf_dmin = self.modified_edf_data[:, :].min()
        self.edf_dmax = self.modified_edf_data[:, :].max()
        self.disp_press = 0.7
        self.dr = (self.edf_dmax - self.edf_dmin) * self.disp_press
        self.y0 = self.edf_dmin
        self.y1 = (self.disp_chans_num - 1) * self.dr + self.edf_dmax

    # refresh display
    def disp_refresh(self):
        self.canvas.axes.cla()
        self.canvas.axes.set_ylim(self.y0, self.y1)
        segs = []
        ticklocs = []
        self.disp_start = int(self.disp_time_start*self.fs)
        self.disp_end = int((self.disp_time_start + self.disp_time_win)*self.fs)
        for i in range(self.disp_chans_start, self.disp_chans_start + self.disp_chans_num):
            tmp_data = self.modified_edf_data[i, self.disp_start:self.disp_end]
            tmp_time = np.linspace(self.disp_start/self.fs, self.disp_end/self.fs, self.disp_end-self.disp_start)
            tmp_data = tmp_data * self.disp_wave_mul
            segs.append(np.hstack((tmp_time[:, np.newaxis], tmp_data[:, np.newaxis])))
            ticklocs.append((i - self.disp_chans_start) * self.dr)
        offsets = np.zeros((self.disp_chans_num, 2), dtype=float)
        offsets[:, 1] = ticklocs
        colors = self.edf_line_colors[self.disp_chans_start:self.disp_chans_start + self.disp_chans_num]
        lines = LineCollection(segs, offsets=offsets, transOffset=None)  # ,colors=colors,transOffset=None)
        disp_chan_names = self.disp_ch_names[
                          self.disp_chans_start:(self.disp_chans_start + self.disp_chans_num)]
        self.canvas.axes.set_xlim(segs[0][0, 0], segs[0][-1, 0])
        self.canvas.axes.add_collection(lines)

        self.canvas.axes.set_yticks(ticklocs)
        self.canvas.axes.set_yticklabels(disp_chan_names)
        self.canvas.axes.set_ylim(self.edf_dmin, (self.disp_chans_num - 1) * self.dr + self.edf_dmax)
        self.canvas.axes.set_xlabel('time(s)')
        self.canvas.draw()

    # disp button slot functions
    def reset_data_display_func(self):
        self.init_display_params()
        self.disp_refresh()
        self.ei_button.setEnabled(False)
        self.hfer_button.setEnabled(False)

    def disp_win_down_func(self):
        self.disp_chans_start -= self.disp_chans_num
        if self.disp_chans_start <= 0:
            self.disp_chans_start = 0
        self.disp_refresh()

    def disp_win_up_func(self):
        self.disp_chans_start += self.disp_chans_num
        if self.disp_chans_start + self.disp_chans_num >= self.edf_nchans:
            self.disp_chans_start = self.edf_nchans - self.disp_chans_num
        self.disp_refresh()

    def disp_more_chans_func(self):
        self.disp_chans_num *= 2
        if self.disp_chans_num >= self.edf_nchans:
            self.disp_chans_num = self.edf_nchans
        self.disp_refresh()

    def disp_less_chans_func(self):
        self.disp_chans_num = int(self.disp_chans_num / 2.0)
        if self.disp_chans_num <= 1:
            self.disp_chans_num = 1
        self.disp_refresh()

    def disp_add_mag_func(self):
        self.disp_wave_mul *= 1.5
        print(self.disp_wave_mul)
        self.disp_refresh()

    def disp_drop_mag_func(self):
        self.disp_wave_mul *= 0.75
        print(self.disp_wave_mul)
        self.disp_refresh()

    def disp_win_left_func(self):
        self.disp_time_start -= 0.2 * self.disp_time_win
        if self.disp_time_start <= 0:
            self.disp_time_start = 0
        self.disp_refresh()

    def disp_win_right_func(self):
        self.disp_time_start += 0.2 * self.disp_time_win
        if self.disp_time_start + self.disp_time_win >= self.edf_time:
            self.disp_time_start = self.edf_time - self.disp_time_win
        self.disp_refresh()

    def disp_shrink_time_func(self):
        self.disp_time_win += 0.5
        if self.disp_time_win >= self.edf_time:
            self.disp_time_win = self.edf_time
        self.disp_refresh()

    def disp_expand_time_func(self):
        self.disp_time_win -= 0.5
        if self.disp_time_win <= 0.5:
            self.disp_time_win = 0.5
        self.disp_refresh()

    def disp_scroll_mouse(self, e):
        if e.button == 'up':
            self.disp_win_left_func()
        elif e.button == 'down':
            self.disp_win_right_func()
            # ei functions

    # filter & del chans
    def filter_data(self):
        self.band_low = float(self.disp_filter_low.text())
        self.band_high = float(self.disp_filter_high.text())
        nyq = self.fs/2
        b, a = scipy.signal.butter(5, np.array([self.band_low/nyq, self.band_high/nyq]), btype = 'bandpass')
        self.modified_edf_data = scipy.signal.filtfilt(b,a,self.modified_edf_data)
        self.disp_refresh()
        self.ei_button.setEnabled(True)
        self.hfer_button.setEnabled(True)

    def delete_chans(self):
        deleted_chans = self.chans_list.selectedItems()
        deleted_list = [i.text() for i in deleted_chans]
        deleted_ind_list = []
        for deleted_name in deleted_list:
            deleted_ind_list.append(self.disp_ch_names.index(deleted_name))
        np.delete(self.modified_edf_data,deleted_ind_list,axis=0)
        for d_chan in deleted_list:
            self.disp_ch_names.remove(d_chan)
        self.chans_list.clear()
        self.chans_list.addItems(self.disp_ch_names)
        self.disp_refresh()

    # select base time & target time
    def choose_baseline(self):
        self.baseline_mouse = 1
        self.baseline_count = 0

    def choose_target(self):
        self.target_mouse = 1
        self.target_count = 0

    def canvas_press_button(self, e):
        if self.baseline_mouse == 1:
            self.baseline_pos[self.baseline_count] = e.xdata
            print(e.xdata)
            self.canvas.axes.axvline(e.xdata)
            self.canvas.draw()
            self.baseline_count += 1
            if self.baseline_count == 2:
                self.baseline_mouse = 0
                print('baseline time', self.baseline_pos)
                reply = QMessageBox.question(self, 'confirm', 'confirm baseline?', QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    pass
                else:
                    self.baseline_pos = np.array([0.0, 1.0])
                    self.disp_refresh()
        elif self.target_mouse == 1:
            self.target_pos[self.target_count] = e.xdata
            self.canvas.axes.axvline(e.xdata)
            self.canvas.draw()
            self.target_count += 1
            if self.target_count == 2:
                self.target_mouse = 0
                print('target time', self.target_pos)
                reply = QMessageBox.question(self, 'confim', 'confirm target time?', QMessageBox.Yes | QMessageBox.No,
                                             QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.disp_time_start = self.target_pos[0]
                    self.disp_time_win = self.target_pos[1] - self.target_pos[0]
                    self.disp_refresh()
                else:
                    self.target_pos = np.array([0.0, self.edf_time_max])
                    self.disp_refresh()
                    self.canvas.axes.axvline(self.baseline_pos[0])
                    self.canvas.axes.axvline(self.baseline_pos[1])
                    self.canvas.draw()
        else:
            pass

    # ei computation
    def ei_computation_func(self):
        # local
        self.ei_base_start = int(self.baseline_pos[0]*self.fs)
        self.ei_base_end = int(self.baseline_pos[1]*self.fs)
        self.ei_target_start = int(self.target_pos[0]*self.fs)
        self.ei_target_end = int(self.target_pos[1]*self.fs)

        self.ei_baseline_data = self.modified_edf_data.copy()[:, self.ei_base_start:self.ei_base_end]
        self.ei_target_data = self.modified_edf_data.copy()[:, self.ei_target_start:self.ei_target_end]
        self.ei_norm_target, self.ei_norm_base = ei.compute_hfer(self.ei_target_data, self.ei_baseline_data, self.fs)
        self.ei_ei, self.ei_hfer, self.ei_onset_rank = ei.compute_ei_index(self.ei_norm_target, self.ei_norm_base,
                                                                           self.fs)
        print('finish ei computation')
        self.fullband_button.setEnabled(True)
        self.ei_plot_func()

    # hfer computation
    def hfer_computation_func(self):
        self.hfer_base_start = int(self.baseline_pos[0]*self.fs)
        self.hfer_base_end = int(self.baseline_pos[1]*self.fs)
        self.hfer_target_start = int(self.target_pos[0]*self.fs)
        self.hfer_target_end = int(self.target_pos[1]*self.fs)
        self.hfer_baseline = self.modified_edf_data[:, self.hfer_base_start:self.hfer_base_end]
        self.hfer_target = self.modified_edf_data[:, self.hfer_target_start:self.hfer_target_end]
        self.norm_target, self.norm_base = ei.compute_hfer(self.hfer_target, self.hfer_baseline, self.fs)
        hfer_fig = plt.figure('hfer')
        # hfer
        hfer_ax = hfer_fig.add_axes([0.1, 0.1, 0.7, 0.8])
        tmp_x, tmp_y = np.meshgrid(np.linspace(self.hfer_target_start, self.hfer_target_end, self.norm_target.shape[1]),
                                   np.arange(self.norm_target.shape[0] + 1))
        surf = hfer_ax.pcolormesh(tmp_x, tmp_y, self.norm_target, cmap=plt.cm.hot, vmax=50, vmin=0)
        if 'ei_channel_onset' in dir(self):
            hfer_ax.plot(self.hfer_target_start + self.ei_channel_onset, np.arange(len(self.ei_channel_onset)) + 0.5,
                         'ko')
        hfer_ax.set_xticks(np.arange(self.hfer_target_start, self.hfer_target_start + self.norm_target.shape[1], 2000))
        hfer_ax.set_xticklabels(np.rint(np.arange(self.hfer_target_start, self.hfer_target_start + self.norm_target.shape[1],
                                           2000) / float(self.fs)).astype(np.int16))
        hfer_ax.set_xlabel('time(s)')
        hfer_ax.set_ylabel('channels')
        hfer_fig.canvas.mpl_connect('button_press_event', self.hfer_press_func)
        # colorbar
        color_bar_ax = hfer_fig.add_axes([0.85, 0.1, 0.02, 0.8])
        plt.colorbar(surf, cax=color_bar_ax, orientation='vertical')
        plt.show()

    # press hfer to show original signal and spectrogram
    def hfer_press_func(self, e):
        chosen_elec_index = int(e.ydata)  # int(round(e.ydata))

        # compute spectrogram
        elec_name = self.disp_ch_names[chosen_elec_index]
        raw_data_indx = self.disp_ch_names.index(elec_name)
        tmp_origin_edf_data = np.transpose(self.edf_data['mat_data'])
        tmp_data = tmp_origin_edf_data[raw_data_indx, self.hfer_target_start:self.hfer_target_end]
        tmp_time_target = np.linspace(self.hfer_target_start/self.fs,self.hfer_target_end/self.fs,
                                     int((self.hfer_target_end-self.hfer_target_start)))

        fig = plt.figure('signal')
        ax1 = fig.add_axes([0.2, 0.6, 0.6, 0.3])
        ax1.cla()
        ax1.set_title(elec_name + ' signal')
        ax1.plot(tmp_time_target, tmp_data)
        ax1.set_xlabel('time(s)')
        ax1.set_ylabel('signal(V)')
        ax1.set_xlim(tmp_time_target[0], tmp_time_target[-1])
        ax1_ymax = np.abs(tmp_data).max()
        ax1.set_ylim([-ax1_ymax, ax1_ymax])
        # ax2
        ax2 = fig.add_axes([0.2, 0.15, 0.6, 0.3])
        ax2.cla()
        ax2.set_title(elec_name + ' spectrogram')
        f, t, sxx = spectrogram(x=tmp_data, fs=int(self.fs), nperseg=int(0.5 * self.fs),
                                noverlap=int(0.9 * 0.5 * self.fs), nfft=1024, mode='magnitude')
        sxx = (sxx - np.mean(sxx, axis=1, keepdims=True)) / np.std(sxx, axis=1, keepdims=True)
        sxx = gaussian_filter(sxx, sigma=2)
        spec_time = np.linspace(t[0] + tmp_time_target[0], t[-1] + tmp_time_target[0], sxx.shape[1])
        spec_f_max = 300
        spec_f_nums = int(len(f) * spec_f_max / f.max())
        spec_f = np.linspace(0, spec_f_max, spec_f_nums)
        spec_sxx = sxx[:spec_f_nums, :]

        spec_time, spec_f = np.meshgrid(spec_time, spec_f)
        surf = ax2.pcolormesh(spec_time, spec_f, spec_sxx, cmap=plt.cm.hot, vmax=2, vmin=-0.8)

        ax2.set_xlabel('time(s)')
        ax2.set_ylabel('frequency(hz)')
        ax2.set_ylim((0, spec_f_max))
        ax2.set_xlim(tmp_time_target[0], tmp_time_target[-1])
        position = fig.add_axes([0.85, 0.15, 0.02, 0.3])
        cb = plt.colorbar(surf, cax=position)
        plt.show()

    # ei plot functions
    def refresh_electrodes_info(self):
        tmp_chs_names = self.disp_ch_names
        tmp_chlabel_list = [x[0] if x[1].isdigit() else x[:2] for x in tmp_chs_names]
        chs_counter = Counter(tmp_chlabel_list)
        self.chs_info = chs_counter.items()
        single_chns = [x for x in self.chs_info if len(x[0]) == 1]
        double_chns = [x for x in self.chs_info if len(x[0]) == 2]
        single_chns.sort(key=lambda x: x[0])
        double_chns.sort(key=lambda x: x[0])
        print(single_chns)
        print(double_chns)
        if double_chns != []:
            self.elecs_info = np.concatenate([np.array(single_chns), np.array(double_chns)], axis=0)
        else:
            self.elecs_info = np.array(single_chns)

    def ei_plot_func(self):
        print('1')
        self.refresh_electrodes_info()
        self.elec_labels = [x[0] for x in self.elecs_info]
        self.elec_nums = [int(x[1]) for x in self.elecs_info]
        ei_mu = np.mean(self.ei_ei)
        ei_std = np.std(self.ei_ei)
        self.ei_thresh = ei_mu + ei_std
        print('2')

        color_rgb = cm.jet(np.arange(len(self.elec_nums)) / float(len(self.elec_nums)))
        color_cums = np.cumsum(self.elec_nums)
        ei_ei_fig = plt.figure('ei')
        ei_ei_ax = ei_ei_fig.add_subplot(111)
        print('3')
        ei_ei_fig.canvas.mpl_connect('button_press_event', self.ei_press_func)
        ei_hfer_fig = plt.figure('hfer')
        ei_hfer_ax = ei_hfer_fig.add_subplot(111)
        ei_onset_rank_fig = plt.figure('onset')
        ei_onset_rank_ax = ei_onset_rank_fig.add_subplot(111)
        ei_data = np.stack([self.ei_ei, self.ei_hfer, self.ei_onset_rank], axis=0)
        ei_axes = [ei_ei_ax, ei_hfer_ax, ei_onset_rank_ax]
        print('4')
        for i in range(3):
            ei_axes[i].bar(range(color_cums[0]), ei_data[i][:color_cums[0]], color=color_rgb[0])
            for j in range(1, len(self.elec_nums)):
                ei_axes[i].bar(range(color_cums[j - 1], color_cums[j]), ei_data[i][color_cums[j - 1]:color_cums[j]],
                               color=color_rgb[j])
        tmp_color_cums = np.concatenate([np.array([0]), color_cums])
        x_ticks = [(tmp_color_cums[i - 1] + tmp_color_cums[i]) / 2.0 for i in range(1, len(tmp_color_cums))]
        for ax in ei_axes:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(self.elec_labels)
        ei_ei_ax.plot(np.arange(len(self.ei_ei)), self.ei_thresh * np.ones(len(self.ei_ei)), 'r--')

        # hfer_plot
        hfer_onset_fig = plt.figure('hfer_onset')
        hfer_onset_ax = hfer_onset_fig.add_subplot(111)
        thresh_index = np.where(self.ei_ei >= self.ei_thresh)[0]
        onset_index = self.ei_onset_rank[thresh_index]
        hfer_index = self.ei_hfer[thresh_index]
        chs_labels = np.array(self.disp_ch_names)[thresh_index]
        scatter_c = np.array([[0.5, 0.5, 0.5, 1.0] for _ in range(len(self.ei_ei))])
        scatter_c[thresh_index] = np.array([1.0, 0.0, 0.0, 1.0])
        hfer_onset_ax.scatter(self.ei_onset_rank, self.ei_hfer, c=scatter_c, alpha=0.4, marker='o')
        for i in range(len(chs_labels)):
            hfer_onset_ax.text(onset_index[i], hfer_index[i], chs_labels[i])
        print('ei >= thresh electrodes:', chs_labels)
        hfer_onset_ax.set_xlabel('TC')
        hfer_onset_ax.set_ylabel('EC')
        plt.show()

    def ei_press_func(self, e):
        chosen_elec_index = int(round(e.xdata))
        # compute spectrum
        elec_name = self.disp_ch_names[chosen_elec_index]
        raw_data_indx = self.disp_ch_names.index(elec_name)
        tmp_origin_edf_data = np.transpose(self.edf_data['mat_data'])
        tmp_data = tmp_origin_edf_data[raw_data_indx, self.ei_target_start:self.ei_target_end]
        tmp_time_target = np.linspace(self.ei_target_start/self.fs, self.ei_target_end/self.fs,
                                      int((self.ei_target_end - self.ei_target_start)))

        fig = plt.figure('signal')
        ax1 = fig.add_axes([0.2, 0.6, 0.6, 0.3])
        ax1.cla()
        ax1.set_title(elec_name + ' signal')
        ax1.plot(tmp_time_target, tmp_data)
        ax1.set_xlabel('time(s)')
        ax1.set_ylabel('signal(V)')
        ax1.set_xlim(tmp_time_target[0], tmp_time_target[-1])
        ax1_ymax = np.abs(tmp_data).max()
        ax1.set_ylim([-ax1_ymax, ax1_ymax])
        # ax2
        ax2 = fig.add_axes([0.2, 0.15, 0.6, 0.3])
        ax2.cla()
        ax2.set_title(elec_name + ' spectrogram')
        f, t, sxx = spectrogram(x=tmp_data, fs=int(self.fs), nperseg=int(0.5 * self.fs),
                                noverlap=int(0.9 * 0.5 * self.fs), nfft=1024, mode='magnitude')
        sxx = (sxx - np.mean(sxx, axis=1, keepdims=True)) / np.std(sxx, axis=1, keepdims=True)
        sxx = gaussian_filter(sxx, sigma=2)
        spec_time = np.linspace(t[0] + tmp_time_target[0], t[-1] + tmp_time_target[0], sxx.shape[1])
        spec_f_max = 300
        spec_f_nums = int(len(f) * spec_f_max / f.max())
        spec_f = np.linspace(0, spec_f_max, spec_f_nums)
        spec_sxx = sxx[:spec_f_nums, :]

        spec_time, spec_f = np.meshgrid(spec_time, spec_f)
        surf = ax2.pcolormesh(spec_time, spec_f, spec_sxx, cmap=plt.cm.hot, vmax=2, vmin=-0.8)

        ax2.set_xlabel('time(s)')
        ax2.set_ylabel('frequency(hz)')
        ax2.set_ylim((0, spec_f_max))
        ax2.set_xlim(tmp_time_target[0], tmp_time_target[-1])
        position = fig.add_axes([0.85, 0.15, 0.02, 0.3])
        cb = plt.colorbar(surf, cax=position)
        plt.show()

    # full band computation
    def fullband_computation_func(self):
        self.fullband_button.setEnabled(False)

        self.fullband_base_start = int(self.baseline_pos[0] * self.fs)
        self.fullband_base_end = int(self.baseline_pos[1] * self.fs)
        self.fullband_target_start = int(self.target_pos[0] * self.fs)
        self.fullband_target_end = int(self.target_pos[1] * self.fs)
        tmp_origin_edf_data = np.transpose(self.edf_data['mat_data'])
        self.fullband_target = tmp_origin_edf_data[:, self.fullband_target_start:self.fullband_target_end]

        QMessageBox.information(self, '', 'full band computation starting, please wait')
        self.fullband_thread = fullband_computation_thread(parent=self, raw_signal=self.fullband_target, ei=self.ei_ei,
                                                           fs=self.fs)
        self.fullband_thread.fullband_done_sig.connect(self.fullband_plot_func)
        self.fullband_thread.start()

    # full band plot function
    def fullband_plot_func(self, fullband_res):
        QMessageBox.information(self, '', 'fullband computation done')
        self.fullband_button.setEnabled(True)

        self.spec_pca = fullband_res[0]
        self.fullband_labels = fullband_res[1]
        self.fullband_ind = fullband_res[2]

        chs_labels = np.array(self.disp_ch_names)[self.fullband_ind]
        print('electrodes:', chs_labels)

        fullband_fig = plt.figure('full_band')
        fullband_ax = fullband_fig.add_subplot(111)
        fullband_fig.canvas.mpl_connect('button_press_event', self.fullband_press_func)
        fullband_ax.scatter(self.spec_pca[:, 0], self.spec_pca[:, 1], alpha=0.8, c=self.fullband_labels)

        for ind in self.fullband_ind:
            fullband_ax.text(self.spec_pca[ind, 0], self.spec_pca[ind, 1], self.disp_ch_names[ind],
                             fontsize=8, color='k')
        plt.show()

    def fullband_press_func(self, e):

        pos_x = e.xdata
        pos_y = e.ydata
        distance = np.sum((np.array(self.spec_pca[:, 0:2]) - np.array([pos_x, pos_y])) ** 2, axis=-1)
        chosen_elec_index = np.argmin(distance)

        elec_name = self.disp_ch_names[chosen_elec_index]
        raw_data_indx = self.disp_ch_names.index(elec_name)
        tmp_origin_edf_data = np.transpose(self.edf_data['mat_data'])
        tmp_data = tmp_origin_edf_data[raw_data_indx, self.fullband_target_start:self.fullband_target_end]
        tmp_time_target = np.linspace(self.fullband_target_start / self.fs, self.fullband_target_end / self.fs,
                                      int((self.fullband_target_end - self.fullband_target_start)))

        fig = plt.figure('signal')
        ax1 = fig.add_axes([0.2, 0.6, 0.6, 0.3])
        ax1.cla()
        ax1.set_title(elec_name + ' signal')
        ax1.plot(tmp_time_target, tmp_data)
        ax1.set_xlabel('time(s)')
        ax1.set_ylabel('signal(V)')
        ax1.set_xlim(tmp_time_target[0], tmp_time_target[-1])
        ax1_ymax = np.abs(tmp_data).max()
        ax1.set_ylim([-ax1_ymax, ax1_ymax])
        # ax2
        ax2 = fig.add_axes([0.2, 0.15, 0.6, 0.3])
        ax2.cla()
        ax2.set_title(elec_name + ' spectrogram')
        f, t, sxx = spectrogram(x=tmp_data, fs=int(self.fs), nperseg=int(0.5 * self.fs),
                                noverlap=int(0.9 * 0.5 * self.fs), nfft=1024, mode='magnitude')
        sxx = (sxx - np.mean(sxx, axis=1, keepdims=True)) / np.std(sxx, axis=1, keepdims=True)
        sxx = gaussian_filter(sxx, sigma=2)
        spec_time = np.linspace(t[0] + tmp_time_target[0], t[-1] + tmp_time_target[0], sxx.shape[1])
        spec_f_max = 300
        spec_f_nums = int(len(f) * spec_f_max / f.max())
        spec_f = np.linspace(0, spec_f_max, spec_f_nums)
        spec_sxx = sxx[:spec_f_nums, :]

        spec_time, spec_f = np.meshgrid(spec_time, spec_f)
        surf = ax2.pcolormesh(spec_time, spec_f, spec_sxx, cmap=plt.cm.hot, vmax=2, vmin=-0.8)

        ax2.set_xlabel('time(s)')
        ax2.set_ylabel('frequency(hz)')
        ax2.set_ylim((0, spec_f_max))
        ax2.set_xlim(tmp_time_target[0], tmp_time_target[-1])
        position = fig.add_axes([0.85, 0.15, 0.02, 0.3])
        cb = plt.colorbar(surf, cax=position)
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Brainquake()
    sys.exit(app.exec_())

