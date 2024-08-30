import __init__
import numpy as np
import geopandas as gpd
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import matplotlib.pyplot as plt
from __init__ import EarthQuakeVisualizer, FilterOutput
from __init__.etas import Etas, EtasParams
import pandas as pd
import json


def window_text_to_list(text):
    windows = text.split(',')
    return [int(windows[0][1:]), int(windows[1][:-1])]


def make_2d(items, n_columns: int, dtype):
    n_items = len(items)
    n_rows = (n_items // n_columns) + min(1, n_items % n_columns)
    output = np.empty(shape=(n_rows, n_columns), dtype=dtype)
    ith = 0
    for row in range(n_rows):
        for col in range(n_columns):
            if ith < n_items:
                output[row, col] = items[ith]
                ith += 1

    return output


def plot_table(etas_models: np.ndarray, windows: list[tuple[int, int]],
               n_columns: int, n_years=1, figsize=(33, 33)):
    from __init__.etas import Etas
    """

    :param etas_models:
    :param windows:
    :param n_years:
    :param figsize:
    :return:
    """
    etas_models_2d = make_2d(items=etas_models, n_columns=n_columns, dtype=Etas)
    n_rows, n_cols = etas_models_2d.shape
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    ith_window = 0
    for row in range(n_rows):
        for col in range(n_cols):
            etas = etas_models_2d[row, col]
            if etas is None:
                ith_window += 1
                continue
            predicted_cum_numbers = [etas.integrated_lambda(X=etas.params.get(), time_until=t) for t in
                                     etas.history.times]
            # tmp = [transformed_time(log_param_fitted, Tmax, Mc, Ti, Mi, t) for t in Ti]
            #             fig, ax = plt.subplots(figsize=(10, 10))
            x = np.arange(1, len(predicted_cum_numbers))
            y = predicted_cum_numbers[1:]
            #             ax.set_xlabel("Observed Cumulative # of Earthquakes")
            #             ax.set_ylabel("Predicted Cumulative # of Earthquakes")
            if n_rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]
            ax.plot(x, y, label='fitted ETAS model')
            ax.plot(x, x, label='real')

            x_min, x_max = 0, len(y)
            y_min, y_max = 0, y[-1]
            x_mid = (x_max - x_min) / 2
            y_mid = (y_max - y_min) / 2

            if n_years == 1:
                of_year = windows[ith_window][0]
                ax.text(x_mid, y_mid, f'year {of_year}', ha='center', va='center',
                        fontsize=25, color='gray', alpha=0.5)
            else:
                from_year = windows[ith_window][0]
                to_year = windows[ith_window][1] - 1
                ax.text(x_mid, y_mid, f'year {from_year} - {to_year}', ha='center', va='center',
                        fontsize=25, color='gray', alpha=0.5)
            ith_window += 1
    return fig, axes


class TimeWindowETAS:

    def __init__(self, etas_dict, years_1, years_2, years_5, years_10, custom_windows=None):
        self.params = ['u', 'K0', 'alpha', 'p', 'c']
        self.etas_dict = etas_dict
        self.years_1 = years_1
        self.years_2 = years_2
        self.years_5 = years_5
        self.years_10 = years_10
        self.u = None
        self.u_2 = None
        self.u_5 = None
        self.u_10 = None

        self.u_std = None
        self.u_std_2 = None
        self.u_std_5 = None
        self.u_std_10 = None

        self.u_20_23 = None
        self.u_20_23_std = None

        self.u_all = None
        self.u_all_std = None

        self.K0 = None
        self.K0_2 = None
        self.K0_5 = None
        self.K0_10 = None

        self.K0_std = None
        self.K0_std_2 = None
        self.K0_std_5 = None
        self.K0_std_10 = None

        self.K0_20_23 = None
        self.K0_20_23_std = None

        self.K0_all = None
        self.K0_all_std = None

        self.p = None
        self.p_2 = None
        self.p_5 = None
        self.p_10 = None

        self.p_std = None
        self.p_std_2 = None
        self.p_std_5 = None
        self.p_std_10 = None

        self.p_20_23 = None
        self.p_20_23_std = None

        self.p_all = None
        self.p_all_std = None

        self.alpha = None
        self.alpha_2 = None
        self.alpha_5 = None
        self.alpha_10 = None

        self.alpha_std = None
        self.alpha_std_2 = None
        self.alpha_std_5 = None
        self.alpha_std_10 = None

        self.alpha_20_23 = None
        self.alpha_20_23_std = None

        self.alpha_all = None
        self.alpha_all_std = None

        self.c_std = None
        self.c_std_2 = None
        self.c_std_5 = None
        self.c_std_10 = None

        self.c_20_23 = None
        self.c_20_23_std = None

        self.c_all = None
        self.c_all_std = None

        self.custom_windows = custom_windows
        self.custom_markers = ['x', '^']
        for param in self.params:
            self.set_values(param)

    def __getitem__(self, key):
        # Check if the key exists as an attribute in the class
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"'{key}' not found")

    def __setitem__(self, key, val):
        # Check if the key exists as an attribute in the class
        if hasattr(self, key):
            setattr(self, key, val)
        else:
            print('manually set attribute, ', key)
            setattr(self, key, val)

    def get_df(self):
        columns = ['year_start', 'n_years',
                   'u', 'K0', 'alpha', 'p', 'c',
                   'u_std', 'K0_std', 'alpha_std',
                   'p_std', 'c_std']

        data = {col: [] for col in columns}

        for length, start_years in [(1, self.years_1), (2, self.years_2),
                                    (5, self.years_5), (10, self.years_10)]:
            for start_year in start_years:
                etas = self.etas_dict[str([start_year, start_year + length])]
                etas_params = etas.params
                data['year_start'].append(start_year)
                data['n_years'].append(length)
                for param_name in self.params:
                    data[f'{param_name}'].append(etas_params[param_name])
                    data[f'{param_name}_std'].append(etas_params[f'{param_name}_std'])
        if self.custom_windows is not None:
            for start_year, end_year in self.custom_windows:
                length = end_year - start_year
                etas = self.etas_dict[str([start_year, end_year])]
                etas_params = etas.params
                data['year_start'].append(start_year)
                data['n_years'].append(length)
                for param_name in self.params:
                    data[f'{param_name}'].append(etas_params[param_name])
                    data[f'{param_name}_std'].append(etas_params[f'{param_name}_std'])
        return pd.DataFrame(data=data)

    def set_values(self, name: str):

        years_1_vals = np.array([self.etas_dict[str([year, year + 1])].params[name] for year in self.years_1])
        years_1_std = np.array([self.etas_dict[str([year, year + 1])].params[f'{name}_std'] for year in self.years_1])

        years_2_vals = np.array([self.etas_dict[str([year, year + 2])].params[name] for year in self.years_2])
        years_2_std = np.array([self.etas_dict[str([year, year + 2])].params[f'{name}_std'] for year in self.years_2])

        years_5_vals = np.array([self.etas_dict[str([year, year + 5])].params[name] for year in self.years_5])
        years_5_std = np.array([self.etas_dict[str([year, year + 5])].params[f'{name}_std'] for year in self.years_5])

        years_10_vals = np.array([self.etas_dict[str([year, year + 10])].params[name] for year in self.years_10])
        years_10_std = np.array([self.etas_dict[str([year, year + 10])].params[f'{name}_std'] for year in self.years_10])

        self.__setattr__(name, years_1_vals)
        self.__setattr__(f'{name}_std', years_1_std)

        self.__setattr__(f'{name}_2', years_2_vals)
        self.__setattr__(f'{name}_std_2', years_2_std)

        self.__setattr__(f'{name}_5', years_5_vals)
        self.__setattr__(f'{name}_std_5', years_5_std)

        self.__setattr__(f'{name}_10', years_10_vals)
        self.__setattr__(f'{name}_std_10', years_10_std)

        if self.custom_windows:
            for start_year, end_year in self.custom_windows:
                self.__setattr__(f'{name}_{str(start_year)[2:]}_{str(end_year-1)[2:]}',
                                 self.etas_dict[f'[{start_year}, {end_year}]'].params[name])
                self.__setattr__(f'{name}_{str(start_year)[2:]}_{str(end_year-1)[2:]}_std',
                                 self.etas_dict[f'[{start_year}, {end_year}]'].params[f'{name}_std'])

    def plot_params(self, name: str, figsize=(16, 4)):
        plt.figure(figsize=figsize)

        if self.years_1.shape[0]:
            plt.errorbar(self.years_1,
                         self.__getitem__(name),
                         self.__getitem__(f'{name}_std'),
                         linestyle='None', marker='*', label='1-year windows')
        if self.years_2.shape[0]:
            plt.errorbar(self.years_2+1,
                         self.__getitem__(f'{name}_2'),
                         self.__getitem__(f'{name}_std_2'),
                         linestyle='None', marker='s', xerr=1, label='2-year windows')
        if self.years_5.shape[0]:
            plt.errorbar(self.years_5+2.5,
                         self.__getitem__(f'{name}_5'),
                         self.__getitem__(f'{name}_std_5'),
                         linestyle='None', marker='P', xerr=2.5, label='5-year windows')
        if self.years_10.shape[0]:
            plt.errorbar(self.years_10+5,
                         self.__getitem__(f'{name}_10'),
                         self.__getitem__(f'{name}_std_10'),
                         linestyle='None', marker='D', xerr=5, label='10-year windows')

        if self.custom_windows:
            marker_idx = 0
            for start_year, end_year in self.custom_windows:
                self.__setattr__(f'{name}_{str(start_year)[2:]}_{str(end_year-1)[2:]}',
                                 self.etas_dict[f'[{start_year}, {end_year}]'].params[name])
                self.__setattr__(f'{name}_{str(start_year)[2:]}_{str(end_year-1)[2:]}_std',
                                 self.etas_dict[f'[{start_year}, {end_year}]'].params[f'{name}_std'])
                mid_point = start_year + (end_year - start_year) / 2
                half_length = (end_year - start_year) / 2
                if end_year - start_year == 1:
                    label = f'{start_year} window'
                else:
                    label = f'{start_year}-{end_year - 1} window'
                plt.errorbar(mid_point,
                             self.__getitem__(f'{name}_{str(start_year)[2:]}_{str(end_year-1)[2:]}'),
                             self.__getitem__(f'{name}_{str(start_year)[2:]}_{str(end_year-1)[2:]}_std'),
                             linestyle='None', marker=self.custom_markers[marker_idx], xerr=half_length, label=label)
                marker_idx += 1

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.gca().yaxis.set_ticks_position('both')
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())

        plt.grid()
        plt.xlabel('Year', fontsize=18)
        return plt.gca()
