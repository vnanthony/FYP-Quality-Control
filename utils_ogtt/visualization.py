import os

import pandas as pd
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, params, df_attr_seg):
        # required parameters: 'cache_path', 'slots', 'fs'
        self.params = params
        self.df_attr_seg = df_attr_seg

    def plot_ppg(self, pgg_segment_id, filtered=True):

        ppg_segment_path = os.path.join(self.params.get('cache_path'), 'ppg_segments', f'{pgg_segment_id}.csv')
        df_ppg_segment = pd.read_csv(ppg_segment_path)

        fig, ax = plt.subplots(figsize=(20, 5))

        for slot in self.params.get('slots').keys():

            timestamp = df_ppg_segment[f'{slot}_ts']
            time = timestamp / self.params.get('fs')
            light_color = self.params.get('slots')[slot]
            alpha = 0.8

            if filtered:
                ax.plot(time, df_ppg_segment[f'{slot}_val_filtered'], alpha=alpha, label=light_color)
            else:
                ax.plot(time, df_ppg_segment[f'{slot}_val'], alpha=alpha, label=light_color)

        ax.legend()

        ax.set_title('PPG Signal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Magnitude')

        return fig