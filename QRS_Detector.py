import torch
import numpy as np
import detector
from matplotlib import pyplot as plt
from scipy.misc import electrocardiogram
import neurokit2 as nk
from wfdb import processing


def wgn(x, snr):
    Ps = np.sum(abs(x) ** 2) / len(x)
    Pn = Ps / (10 ** ((snr / 10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise


def load_qp_ecg(path):
    ecg = np.loadtxt(path, delimiter=',')[:, 1]
    return ecg


def detectbyLSTM(ecg, dt):
    peaks, probs = dt.find_peaks(ecg)
    filtered_peaks = dt.remove_close(peaks=peaks, peak_probs=probs, threshold_ms=350)
    return filtered_peaks


def detectbyTraditional(ecg, method='kalidas'):
    ecg = nk.ecg_clean(ecg, sampling_rate=250)
    peaks, r_peak = nk.ecg_peaks(ecg, sampling_rate=250)
    peaks = np.asarray(peaks['ECG_R_Peaks']).nonzero()[0]
    peaks = processing.correct_peaks(ecg, peak_inds=peaks, search_radius=5, smooth_window_size=20, peak_dir='up')
    return peaks


if __name__ == '__main__':
    model_path = '/data/lixin/exp/QRS_detection/model/6_1_21/QRS_best_score_at_epoch149.pth'
    dt = detector.ECG_detector(sampling_rate=250, gpu_ids=[0, 1, 2, 3], model_path=model_path, stride=100,
                               window_size=1000,
                               threshold=0.05)
    print('QRS detector is online.')

    # ecg = electrocardiogram()
    ecg = load_qp_ecg('/data/lixin/qingpu/81633663c8d1dc6f525c813ac394f58.txt')
    # ecg = processing.normalize_bound(ecg, lb=-1, ub=1)
    # ecg = wgn(ecg, 0.1)


    LSTM_peaks = detectbyLSTM(ecg, dt)
    kalidas_peaks = detectbyTraditional(ecg, 'kalidas')
    tompkins_peaks = detectbyTraditional(ecg, 'pamtompkins1985 ')

    fig, axs = plt.subplots(3, 1, figsize=(15, 20))
    begin = 39921

    ymax, ymin = ecg[begin:begin + 1500].max(), ecg[begin:begin + 1500].min()

    axs[0].plot(ecg[:])
    axs[0].scatter(x=LSTM_peaks[:], y=ecg[LSTM_peaks[:]], color='red')
    axs[0].set_title('LSTM prediction')
    axs[0].set_xlim(xmin=begin, xmax=begin + 1500)
    # axs[0].set_aspect(1)
    axs[0].set_ylim(ymin=0.95 * ymin, ymax=1.05 * ymax)

    axs[1].plot(ecg[:])
    axs[1].scatter(x=kalidas_peaks[:], y=ecg[kalidas_peaks[:]], color='red')
    axs[1].set_title('kalidas prediction')
    axs[1].set_xlim(xmin=begin, xmax=begin + 1500)
    # axs[1].set_aspect(1)

    axs[1].set_ylim(ymin=0.95 * ymin, ymax=1.05 * ymax)

    axs[2].plot(ecg[:])
    axs[2].scatter(x=tompkins_peaks[:], y=ecg[tompkins_peaks[:]], color='red')
    axs[2].set_title('tompkins prediction')
    axs[2].set_xlim(xmin=begin, xmax=begin + 1500)
    # axs[1].set_aspect(1)

    axs[2].set_ylim(ymin=0.95 * ymin, ymax=1.05 * ymax)

    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.show()
