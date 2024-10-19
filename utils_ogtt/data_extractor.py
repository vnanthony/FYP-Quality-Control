import os
import uuid
from collections import deque
from itertools import combinations

import dill as pickle
import numpy as np
import pandas as pd
import pyPPG.biomarkers as BM
import pyPPG.fiducials as FP
import pyPPG.ppg_sqi as SQI
import pyPPG.preproc as PP
from utils_ogtt.feature_generator import FeatureGenerator
from pyPPG import PPG, Biomarkers, Fiducials
from pyPPG.datahandling import load_data, plot_fiducials, save_data
from scipy import signal


class DataExtractor:
    '''
    A class used to extract data from PPG files

    Attributes
    -----------------
    params : dict
        a dictionary of parameters for data extraction
    attr_path : str
        the path to the attribute file
    ppg_path : str
        the path to the PPG files
    df_attr : pd.DataFrame
        a data frame of the attributes
    df_attr_seg : pd.DataFrame
        a data frame of the segmented attributes

    Methods
    -----------------
    get_df_attr()
        Read the attribute file and filter by subject id
    read_single_ppg_file(record_id)
        Read a single PPG file and return a data frame
    align_signals(df_single_ppg, slots=['c', 'd', 'e'])
        Align the timestamps and values of different signal slots
    segment_signals(df_single_ppg)
        Segment the PPG signals into equal-length chunks
    get_df_attr_seg()
        Segment the signals from the paths in self.df_attr and allocate the paths of the segments into self.df_attr_seg
    bandpass_filter(sig, fs, low_cut=None, high_cut=None, filter_order=2)
        Apply a bandpass filter to a signal
    notch_filter(sig, fs, w0, Q)
        Apply a notch filter to a signal
    filter_signal()
        Filter the PPG signals from the segments in self.df_attr_seg and save the filtered signals to the cache path
    get_single_segment_quality(ppg_segment_id, cache_path, fs, plot=False)
        Compute the signal quality index (SQI) for a single PPG segment file
    get_signal_quality()
        Compute the SQI for all the PPG segments in self.df_attr_seg and add the SQI values to the data frame
    generate_features()
        Generate features for the PPG segments in self.df_attr_seg using the pyPPG package

    Sample code
    -----------------
    params = {

        'data_path': '/mnt/g/My Drive/donutech/Research/ideation_lester/ogtt',
        # a folder for storing the cached files
        'cache_path': '/mnt/d/ogtt_cache',

        # ============================================================================
        # the subject ID to be included in the study
        'subject_id': [1, 2, 3],
        'subjects_exam': [],

        # segment duration in seconds
        'segment_duration': 40,


        # filtering frequencies
        'bandpass': (0.5, 6),
        # 'notch': 50.0,

        # 'run_name': 'first_trial',
        # 'experiment_name': 'donut',
    }

    data_extractor = DataExtractor(params=params)

    data_extractor.get_df_attr()
    data_extractor.get_df_attr_seg()
    data_extractor.filter_signal()
    data_extractor.get_signal_quality()
    data_extractor.generate_features()

    '''



    def __init__(self, params):
        '''
        Parameters
        -----------------
        params : dict
            A dictionary of parameters for data extraction
        '''

        self.params = params
        self.attr_path = os.path.join(self.params.get('data_path'), 'attr.csv')
        self.ppg_path = os.path.join(self.params.get('data_path'), 'ppg')
        os.makedirs(self.params.get('cache_path'), exist_ok=True)

    def get_df_attr(self):
        '''
        Read the attribute file and filter by subject id

        Sets the df_attr attribute to a data frame of the attributes
        '''

        self.df_attr = pd.read_csv(self.attr_path)
        self.df_attr = self.df_attr.loc[self.df_attr['subject_id'].isin(self.params.get('subject_id'))]
        self.df_attr = self.df_attr.reset_index(drop=True)

    def read_single_ppg_file(self, record_id):
        '''
        Read a single PPG file and return a data frame

        Parameters
        -----------------
        record_id : str
            The record id of the PPG file

        Returns
        -----------------
        df_single_ppg : pd.DataFrame
            A data frame of the PPG signals
        '''

        file_path = os.path.join(self.ppg_path, record_id, 'CSV')
        for file_name in os.listdir(file_path):
            if 'Stream' in file_name:
                target_file_name = file_name
                break
        df_single_ppg = pd.read_csv(os.path.join(file_path, target_file_name), skiprows=[0, 1, 3, 4], on_bad_lines='skip')

        # remove empty columns
        df_single_ppg = df_single_ppg.loc[:, ~df_single_ppg.isna().all()]

        # rename columns
        df_single_ppg = df_single_ppg.rename(
            columns={
            'SLOT C (500.00Hz)': 'c_ts',
            'Unnamed: 2': 'c_val',

            'SLOT D (500.00Hz)': 'd_ts',
            'Unnamed: 5': 'd_val',

            'SLOT E (500.00Hz)': 'e_ts',
            'Unnamed: 8': 'e_val',
        })

        return df_single_ppg

    @staticmethod
    def align_signals(df_single_ppg, slots=['c', 'd', 'e']):
        '''
        Align the timestamps and value of different signal slots

        Parameters
        -----------------
        df_single_ppg : pd.DataFrame
            A data frame of the PPG signals
        slots : list, optional
            A list of the signal slots to align (default is ['c', 'd', 'e'])

        Returns
        -----------------
        df_single_ppg : pd.DataFrame
            A data frame of the aligned PPG signals
        '''

        combs = combinations(slots, 2)

        # for each slot combination
        for comb in combs:
            slot_0 = comb[0]
            slot_1 = comb[1]

            # the timestamp differences
            ts_diff = df_single_ppg[f'{slot_1}_ts'] - df_single_ppg[f'{slot_0}_ts']
            max_abs_ts_diff = max(abs(ts_diff))

            if ts_diff.dropna()[0] > 0:
                direction = 'pos'
            else:
                direction = 'neg'


            # the timestamps of the two slots are already aligned
            if max_abs_ts_diff == 0:
                continue
            else:
                while max_abs_ts_diff != 0:
                    # determine the shifting direction
                    if direction == 'pos':
                        shift_period = 1
                    else:
                        shift_period = -1

                    # shift the timestamp and values of the slot boms
                    df_single_ppg[[f'{slot_1}_ts', f'{slot_1}_val']] = df_single_ppg[[f'{slot_1}_ts', f'{slot_1}_val']].shift(shift_period)

                    # the timestamp differences
                    ts_diff = df_single_ppg[f'{slot_1}_ts'] - df_single_ppg[f'{slot_0}_ts']
                    max_abs_ts_diff = max(abs(ts_diff))

        df_single_ppg = df_single_ppg.dropna()

        return df_single_ppg


    def segment_signals(self, df_single_ppg):
        '''
        Segment the PPG signals into equal-length chunks

        Parameters
        -----------------
        df_single_ppg : pd.DataFrame
            A data frame of the PPG signals

        Returns
        -----------------
        segments : deque
            A deque of data frames, each containing a segment of the PPG signals
        '''

        segment_len = self.params.get('segment_duration') * self.params.get('fs')
        n_segments = df_single_ppg.shape[0] // segment_len
        segments = deque()

        for segment_idx in range(n_segments):
            start = segment_idx * segment_len
            end = start + segment_len - 1

            df_segment = df_single_ppg.loc[start:end, :].reset_index(drop=True)
            segments.append(df_segment)

        return segments


    def get_df_attr_seg(self):
        '''
        Segment the signals from the paths in self.df_attr and allocate the paths of the segments into self.df_attr_seg

        Set the df_attr_seg attribute to a data frame of the segmented attributes and the corresponding PPG segment files
        '''

        self.df_attr_seg = pd.DataFrame()

        # for each record
        for idx, row in self.df_attr.iterrows():

            print(f'[Segmentation] Processing df_attr records {idx + 1}/{self.df_attr.shape[0]}')

            # common attributes for before and after
            common_attr = [
                'subject_id', 'record_date', 'gender', 'age',
                'height', 'weight', 'last_meal_time', 'ethnicity',
                'diabetes', 'smoking', 'alcohol', 'notes',
            ] #'name', 

            # ========== before glucose intake ==========
            new_row_before = row[common_attr].copy()
            new_row_before['state'] = 'before'
            new_row_before['heart_rate'] = row['heart_rate_before']
            new_row_before['blood_glucose'] = row['blood_glucose_before']

            # read the signal files from the paths stored in self.df_attr
            df_ppg_before = self.read_single_ppg_file(row['ppg_file_before'])
            df_ppg_before = self.align_signals(df_ppg_before)
            segments = self.segment_signals(df_ppg_before)

            for idx, df_segment in enumerate(segments):
                new_row_before_seg = new_row_before.copy()
                new_row_before_seg['segment_idx'] = idx

                # save the segment to the 'cache_path' in self.params
                csv_name = uuid.uuid4()
                save_path = os.path.join(self.params.get('cache_path'), 'ppg_segments')
                os.makedirs(save_path, exist_ok=True)
                save_file_path = os.path.join(save_path, f'{csv_name}.csv')
                df_segment.to_csv(save_file_path, index=False)

                new_row_before_seg['ppg_segment_id'] = csv_name

                self.df_attr_seg = pd.concat([self.df_attr_seg, pd.DataFrame([new_row_before_seg])])

            # ========== after glucose intake ==========
            new_row_after = row[common_attr].copy()
            new_row_after['state'] = 'after'
            new_row_after['heart_rate'] = row['heart_rate_after']
            new_row_after['blood_glucose'] = row['blood_glucose_after']

            # read the signal files from the paths stored in self.df_attr
            df_ppg_after = self.read_single_ppg_file(row['ppg_file_after'])
            df_ppg_after = self.align_signals(df_ppg_after)
            segments = self.segment_signals(df_ppg_after)

            for idx, segment in enumerate(segments):
                new_row_after_seg = new_row_after.copy()
                new_row_after_seg['segment_idx'] = idx

                # save the segment to the 'cache_path' in self.params
                csv_name = uuid.uuid4()
                save_path = os.path.join(self.params.get('cache_path'), 'ppg_segments')
                os.makedirs(save_path, exist_ok=True)
                save_file_path = os.path.join(save_path, f'{csv_name}.csv')
                df_segment.to_csv(save_file_path, index=False)

                new_row_after_seg['ppg_segment_id'] = csv_name

                self.df_attr_seg = pd.concat([self.df_attr_seg, pd.DataFrame([new_row_after_seg])])

        self.df_attr_seg = self.df_attr_seg.reset_index(drop=True)

    @staticmethod
    def bandpass_filter(sig, fs, low_cut=None, high_cut=None, filter_order=2):
        '''
        Apply a bandpass filter to a signal

        Parameters
        --------------------
        sig : array_like
            The input signal to be filtered
        fs : float
            The sampling frequency of the digital system
        low_cut : float, optional
            The low cut frequency of the filter (default is None)
        high_cut : float, optional
            The high cut frequency of the filter (default is None)
        filter_order : int, optional
            The order of the filter (default is 2)

        Returns
        --------------------
        sig : array_like
            The output signal after filtering
        '''

        if high_cut:
            b, a = signal.butter(filter_order, high_cut, fs=fs, btype='lowpass', output='ba')
            sig = signal.filtfilt(b, a, sig)
        if low_cut:
            b, a = signal.butter(filter_order, low_cut, fs=fs, btype='highpass', output='ba')
            sig = signal.filtfilt(b, a, sig)

        return sig

    @staticmethod
    def notch_filter(sig, fs, w0, Q):
        '''
        Apply a notch filter to a signal

        Parameters
        --------------------
        sig : array_like
            The input signal to be filtered
        fs : float
            The sampling frequency of the digital system
        w0 : float
            The frequency to remove from the signal
        Q : float
            The quality factor of the filter

        Returns
        --------------------
        sig : array_like
            The output signal after filtering
        '''

        b, a = signal.iirnotch(w0, Q, fs)
        sig = signal.lfilter(b, a, sig)

        return sig


    def filter_signal(self):
        '''
        Filter the PPG signals from the segments in self.df_attr_seg and save the filtered signals to the cache path

        Modify the PPG segment files in the cache path by adding filtered values for each signal slot
        '''

        # for each segment
        for idx, row in self.df_attr_seg.iterrows():

            print(f'[Filtering] Processing df_attr_seg records {idx + 1}/{self.df_attr_seg.shape[0]}')
            cache_path = self.params['cache_path']
            ppg_segment_id = str(row['ppg_segment_id'])
            save_path = os.path.join(cache_path, 'ppg_segments')
            os.makedirs(save_path, exist_ok=True)
            save_file_path = os.path.join(save_path, f'{ppg_segment_id}.csv')
            df_segment = pd.read_csv(save_file_path)

            # for each signal slot
            for val in ['c_val', 'd_val', 'e_val']:

                sig = df_segment[val].values

                if self.params.get('bandpass'):
                    sig= self.bandpass_filter(sig, self.params.get('fs'), low_cut=self.params['bandpass'][0], high_cut=self.params['bandpass'][1])
                if self.params.get('notch'):
                    sig= self.notch_filter(sig, self.params.get('fs'), self.params['notch'], 30.0)

                df_segment[f'{val}_filtered'] = sig

            df_segment.to_csv(save_file_path, index=False)

    @staticmethod
    def get_single_segment_quality(ppg_segment_id, cache_path, fs, slots=['c', 'd', 'e'], plot=False):
        '''
        Compute the signal quality index (SQI) for a single PPG segment file

        Parameters
        --------------------
        ppg_segment_id : str
            The name of the PPG segment file
        cache_path : str
            The path to the cache directory
        fs : float
            The sampling frequency of the PPG signals
        slots : list, optional
            The list of signal slots to compute the SQI for (default is ['c', 'd', 'e'])
        plot : bool, optional
            Whether to plot the fiducial points of the PPG signals (default is False)

        Returns
        --------------------
        sqis : dict
            A dictionary of the SQI values for each signal slot
        figs : dict
            A dictionary of the matplotlib figures for each signal slot (if plot is True)
        '''

        segment_file_path = os.path.join(cache_path, 'ppg_segments', f'{ppg_segment_id}.csv')
        df_segment = pd.read_csv(segment_file_path)

        temp_path = os.path.join(cache_path, 'temp')
        s_path = os.path.join(cache_path, 's')
        fp_path = os.path.join(cache_path, 'fp')

        os.makedirs(temp_path, exist_ok=True)
        os.makedirs(s_path, exist_ok=True)
        os.makedirs(fp_path, exist_ok=True)

        sqis = {}
        figs = {}

        for slot in slots:
            # save the single slot signal to temp.csv
            df_segment[f'{slot}_val_filtered'].to_csv(os.path.join(temp_path, 'temp.csv'), index=False)

            # load the signal to the pyPPG format
            signal = load_data(data_path=os.path.join(temp_path, 'temp.csv'), fs=fs)

            # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"
            signal.sm_wins = {
                'ppg':50,
                'vpg':10,
                'apg':10,
                'jpg':10,
            }

            # get signal derivatives
            prep = PP.Preprocess(sm_wins=signal.sm_wins)
            signal.ppg, signal.vpg, signal.apg, signal.jpg = prep.get_signals(s=signal)

            # Initialise the correction for fiducial points
            corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
            correction = pd.DataFrame()
            correction.loc[0, corr_on] = True
            signal.correction = correction

            # Create a PPG class
            s = PPG(signal)

            # save the PPG class
            s_file_path = os.path.join(s_path, f'{ppg_segment_id}_{slot}')
            with open(s_file_path, 'wb') as f:
                    pickle.dump(s, f)

            # identify fiducial points
            fpex = FP.FpCollection(s=s)
            fiducials = fpex.get_fiducials(s=s)

            # Create a fiducials class
            fp = Fiducials(fp=fiducials)

            # save the fiducial points
            fp_file_path = os.path.join(fp_path, f'{ppg_segment_id}_{slot}')
            with open(fp_file_path, 'wb') as f:
                    pickle.dump(fp, f)

            fig = None
            if plot:
                # Plot fiducial points
                canvas = plot_fiducials(s, fp, legend_fontsize=12, show_fig=False, savefig=False)
                # # edit the fig and ax
                fig, ax = canvas.figure, canvas.figure.axes
                fig.set_size_inches(20, 6)
                fig.suptitle(f'Slot {slot}', fontsize=20)
                figs[slot] = fig

            # Get PPG SQI
            ppgSQI = round(np.mean(SQI.get_ppgSQI(ppg=s.ppg, fs=s.fs, annotation=fp.sp)) * 100, 2)
            sqis[slot] = ppgSQI

        return sqis, figs

    def get_signal_quality(self):
        '''
        Compute the signal quality index (SQI) for all the PPG segments in self.df_attr_seg and adds the SQI values to the data frame

        Modify the self.df_attr_seg attribute by adding SQI columns for each signal slot and the mean SQI
        '''

        # use a copied df to do iterations
        df_attr_seg_buffer = self.df_attr_seg.copy()

        # for each segment
        for idx, row in df_attr_seg_buffer.iterrows():

            print(f'[Signal quality analysis] Processing df_attr_seg records {idx + 1}/{self.df_attr_seg.shape[0]}')

            ppg_segment_id = str(row["ppg_segment_id"])
            sqis, figs = self.get_single_segment_quality(ppg_segment_id, self.params.get('cache_path'), self.params.get('fs'), plot=False)

            # add the qua;ity indices
            for slot in self.params.get('slots').keys():
                self.df_attr_seg.loc[idx, f'{slot}_sqi'] = sqis[slot]

        self.df_attr_seg['sqi_mean'] = np.mean(self.df_attr_seg[[f'{slot}_sqi' for slot in self.params.get('slots').keys()]], axis=1)

    def screen_signal(self):
        if self.params.get('sqi_thresh'):
            mask = (self.df_attr_seg['sqi_mean'] >= self.params.get('sqi_thresh'))
            self.df_attr_seg = self.df_attr_seg.loc[mask, :].reset_index(drop=True)

    def generate_features(self):
        '''
        Generate features for the PPG segments in self.df_attr_seg using the pyPPG package

        Modify the self.df_attr_seg attribute by adding feature columns for each signal slot
        '''

        # use a copied df to do iterations
        df_attr_seg_buffer = self.df_attr_seg.copy()

        # for each segment
        for idx, row in df_attr_seg_buffer.iterrows():
            print(f'[Feature generation] Processing df_attr_seg records {idx + 1}/{self.df_attr_seg.shape[0]}')

            ppg_segment_id = str(row['ppg_segment_id'])
            ppg_segment = pd.read_csv(os.path.join(self.params.get('cache_path'), 'ppg_segments', f'{ppg_segment_id}.csv'))

            for slot in self.params.get('slots').keys():

                # load s
                s_file_path = os.path.join(self.params.get('cache_path'), 's', f'{ppg_segment_id}_{slot}')
                with open(s_file_path, 'rb') as f:
                    s = pickle.load(f)

                # load fp
                fp_file_path = os.path.join(self.params.get('cache_path'), 'fp', f'{ppg_segment_id}_{slot}')
                with open(fp_file_path, 'rb') as f:
                    fp = pickle.load(f)

                feature_generator = FeatureGenerator(ppg_segment, slot)
                feature_generator.get_pyppg_biomarkers(s, fp)

                for feature_name, feature_val in feature_generator.features.items():
                    self.df_attr_seg.loc[idx, feature_name] = feature_val

    def operate(self):
        self.get_df_attr()
        self.get_df_attr_seg()
        self.filter_signal()
        self.get_signal_quality()
        self.screen_signal()
        self.generate_features()