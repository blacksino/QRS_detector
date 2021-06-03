import torch
from torch.utils.data import Dataset
from wfdb.io import get_record_list
from wfdb import rdsamp
from Dataset import data_utils
import numpy as np
from scipy.signal import resample_poly


class NoisyR(Dataset):
    def __init__(self, args, db_name='mitdb', channel=0, from_scratch=True):
        super(NoisyR, self).__init__()
        self.baseline_wander = []
        self.muscle_artifact = []
        self.mitdb_numerical_label = []
        self.args = args
        self.db_name = db_name
        self.channel = channel
        self.signals = []
        self.beats_index = []
        self.beats_types = []
        if from_scratch:
            self.signals, self.beats_index, self.beats_types = self.load_from_scratch()
        else:
            self.signals, self.beats_index, self.beats_types = torch.load(f'{args.db_path}/data.db').values()
            print('data has been loaded.')
        # self.dataset_info()
        self.ma, self.bw = self.preprocessing()

    def load_from_scratch(self):
        mitdb_records = get_record_list(self.db_name)
        mitdb_signals, mitdb_beats, mitdb_beat_types = data_utils.data_from_records(mitdb_records, channel=self.channel,
                                                                                    db='mitdb')
        data = dict(signals=mitdb_signals, beats_idx=mitdb_beats, beats_types=mitdb_beat_types)
        torch.save(data, f'{self.args.db_path}/data.db')
        print("data has been saved.")
        return mitdb_signals, mitdb_beats, mitdb_beat_types

    def __len__(self):
        return 4096

    def dataset_info(self):
        all_symbols = []
        for symbols in self.mitdb_beat_types:
            all_symbols.append(symbols)
        all_symbols = [item for sublist in all_symbols for item in sublist]
        all_symbols = np.asarray(all_symbols)
        u, c = np.unique(all_symbols, return_counts=True)

        # Meanings for different heart beat codings
        label_meanings = {
            "N": "Normal beat",
            "L": "Left bundle branch block beat",
            "R": "Right bundle branch block beat",
            "V": "Premature ventricular contraction",
            "/": "Paced beat",
            "A": "Atrial premature beat",
            "f": "Fusion of paced and normal beat",
            "F": "Fusion of ventricular and normal beat",
            "j": "Nodal (junctional) escape beat",
            "a": "Aberrated atrial premature beat",
            "E": "Ventricular escape beat",
            "J": "Nodal (junctional) premature beat",
            "Q": "Unclassifiable beat",
            "e": "Atrial escape beat",
            "S": "Supraventricular premature or ectopic"
        }

        # Print number of instances in each beat type
        label_counts = [(label, count) for label, count in zip(u.tolist(), c.tolist())]
        label_counts.sort(key=lambda tup: tup[1], reverse=True)
        for label in label_counts:
            print(label_meanings[label[0]], "-" * (40 - len(label_meanings[label[0]])), label[1])

    def preprocessing(self):
        """

        step1:re-arrange labels,1 for Normal,-1 for others
        step2:fit label to peaks
        step3:prepare some noises
        """
        print('convert common labels to numerical labels ')
        for each_sample in self.beats_types:
            numerical_label = [1 if rhythm == 'N' else -1
                               for rhythm in each_sample]
            self.mitdb_numerical_label.append(np.asarray(numerical_label))

        print('done.')

        print('start fix labels.')
        self.mitdb_numerical_label = data_utils.fix_labels(self.signals, self.beats_index, self.mitdb_numerical_label)
        print('done.')

        print('make some noise.')
        self.baseline_wander = rdsamp('bw', pn_dir='nstdb')
        self.muscle_artifact = rdsamp('ma', pn_dir='nstdb')

        ma = np.concatenate((self.muscle_artifact[0][:, 0], self.muscle_artifact[0][:, 1]))
        bw = np.concatenate((self.baseline_wander[0][:, 0], self.baseline_wander[0][:, 1]))

        ma = resample_poly(ma, up=250, down=self.muscle_artifact[1]['fs'])
        bw = resample_poly(bw, up=250, down=self.baseline_wander[1]['fs'])
        print('done.')

        return ma, bw

    def __getitem__(self, index):

        while True:
            index = np.random.randint(0,len(self.signals))
            signal = self.signals[index]
            R_beat = self.beats_index[index]
            R_label = self.mitdb_numerical_label[index]

            begin = np.random.randint(signal.shape[0] - self.args.win_size)
            end = begin + self.args.win_size
            R_beat_in_win = R_beat[(R_beat >= begin + 3) & (R_beat <= end - 3)] - begin
            if len(R_beat_in_win) > 0:
                R_label_in_win = R_label[(R_beat >= begin + 3) & (R_beat <= end - 3)]
                if np.all(R_label_in_win == 1):
                    window_labels = np.zeros(self.args.win_size)
                    np.put(window_labels, R_beat_in_win, R_label_in_win)

                    np.put(window_labels, R_beat_in_win + 1, R_label_in_win)
                    np.put(window_labels, R_beat_in_win + 2, R_label_in_win)
                    np.put(window_labels, R_beat_in_win - 1, R_label_in_win)
                    np.put(window_labels, R_beat_in_win - 2, R_label_in_win)

                    data_win = data_utils.normalize_bound(signal[begin:end], lb=-1, ub=1)
                    data_win = data_win + data_utils.get_noise(self.ma, self.bw, self.args.win_size)

                    data_win = data_utils.normalize_bound(data_win, lb=-1, ub=1)

                    data_win = np.asarray(data_win)
                    window_labels = np.asarray(window_labels)

                    return data_win, window_labels
