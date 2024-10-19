import os

import pandas as pd

from pyPPG import PPG, Fiducials, Biomarkers
from pyPPG.datahandling import load_data, plot_fiducials, save_data
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM

class FeatureGenerator:
    '''
    A class to generate features from a photoplethysmography (PPG) segment.

    Attributes
    -----------------
    ppg_segment : pd.DataFrame
        A dataframe containing the PPG signal and timestamps
    slot : str
        A string indicating the slot name for the features. Default is 'c'
    features : dict
        A dictionary to store the extracted features as key-value pairs

    Methods
    -----------------
    get_pyppg_biomarkers(s, fp)
        Extract biomarkers from the PPG segment using the pyPPG package
    '''

    def __init__(self, ppg_segment, slot='c'):
        '''
        Parameters
        -----------------
        ppg_segment : pd.DataFrame
            A dataframe containing the PPG signal and timestamps
        slot : str, optional
            A string indicating the slot name for the features (default is 'c')
        '''

        self.ppg_segment = ppg_segment
        self.slot = slot
        self.features = {}

    def get_pyppg_biomarkers(self, s, fp):
        '''
        Extract biomarkers from the PPG segment using the pyPPG package.

        The extracted features are stored in the features attribute.

        Parameters
        -----------------
        s : int
            The sampling frequency of the PPG signal in Hz
        fp : float
            The frequency of the pulse in Hz
        '''

        # Init the biomarkers package
        bmex = BM.BmCollection(s=s, fp=fp)

        # Extract biomarkers
        bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()

        for feature_type, df_feature in bm_stats.items():
            for col in df_feature.columns:
                self.features[f'{self.slot}_{col}_median'] = df_feature.loc['median', col]