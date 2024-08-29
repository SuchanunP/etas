import pandas as pd
from IPython.display import display
from typing import Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import math
import geopandas as gpd
from dataclasses import dataclass
import shapely.geometry


@dataclass
class FilterOutput:
    earthquake_df: pd.DataFrame
    density_df: pd.DataFrame
    times_btw: np.array


class EarthQuakeVisualizer:
    def __init__(self, csv_path: Optional[str] = None, shp_path: str = None, columns: Optional[Dict] = None, df=None):
        self.df = pd.read_csv(csv_path) if df is None else df
        if columns:
            self.df = self.df.rename(columns=columns)
        self.df.date = pd.to_datetime(self.df.date)
        self.df = self.df.sort_values(by='date')
        self.shp_path = shp_path

    def filter_by_magnitude(self, magnitudes: Tuple[float], df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        magnitudes = [magnitudes[0], magnitudes[1]]
        if magnitudes[0] is None:
            magnitudes[0] = -np.inf
        if magnitudes[1] is None:
            magnitudes[1] = np.inf

        return df[(df.magnitude >= magnitudes[0]) & (df.magnitude <= magnitudes[1])].copy()

    def filter_by_lats_longs(self, longs: Tuple[float],
                             lats: Tuple[float], df=None):
        if df is None:
            df = self.df
        df = df[(df.latitude >= lats[0]) &
                  (df.latitude <= lats[1]) &
                  (df.longitude >= longs[0]) &
                  (df.longitude <= longs[1])
                  ].copy()
        return df

    def filter_by_polygon(self, polygon_coors: List[float],
                          df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        polygon = shapely.geometry.Polygon(polygon_coors)

        def in_target(latitude, longitude):
            return shapely.geometry.Point(longitude, latitude).within(polygon)

        df['target'] = df.apply(lambda row: in_target(latitude=row.latitude, longitude=row.longitude), axis=1)

        return df[df.target].copy().drop(columns=['target'])

    def filter_by_depths(self, depths: Tuple[float],
                         df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        return df[(df.depth >= depths[0]) &
                  (df.depth <= depths[1])
                  ].copy()

    def compute_times_btw(self, df: pd.DataFrame = None, day: bool = False) -> np.ndarray:
        if df is None:
            df = self.df
        if day:
            times = (df.date.iloc[1:].values - df.date.iloc[0: -1].values).astype('timedelta64[s]') \
                    / np.timedelta64(1, 's') / 60 / 60 / 24
        else:
            times = (df.date.iloc[1:].values - df.date.iloc[0: -1].values).astype('timedelta64[s]')\
                        / np.timedelta64(1, 's')
        return times

    def visualize(self, depths: Tuple[float] = None,
                  lats: Tuple[float] = None,  # (23, 24),
                  longs: Tuple[float] = None,  # (120, 121),
                  polygon: List[float] = None,
                  magnitudes: Tuple[float] = [None, None],  # (5.0, 7.9),
                  day: bool = False):
        if not lats:
            min_lat, max_lat = self.df.latitude.min(), self.df.latitude.max()
            lats = [min_lat, max_lat]
        if not longs:
            min_long, max_long = self.df.longitude.min(), self.df.longitude.max()
            longs = [min_long, max_long]

        if magnitudes is not None:
            df_filter = self.filter_by_magnitude(magnitudes=magnitudes)
        else:
            df_filter = self.df
        if depths is not None:
            df_filter = self.filter_by_depths(depths=depths, df=df_filter)
        df_filter = self.filter_by_lats_longs(df=df_filter, longs=longs, lats=lats)
        # if pol
        if polygon is not None:
            df_filter = self.filter_by_polygon(polygon_coors=polygon, df=df_filter)

        times: np.ndarray = self.compute_times_btw(df=df_filter, day=day)

        df_density: pd.DataFrame = EarthQuakeVisualizer.get_density(times=times)
        if self.shp_path:
            map_shp = gpd.read_file(self.shp_path)

            fig, ax = plt.subplots()
            map_shp.plot(ax=ax, alpha=0.3)
            sc = plt.scatter(df_filter.longitude, df_filter.latitude, c=df_filter.magnitude,
                             marker='.', s=2, cmap='autumn_r', alpha=0.3)  # cmap='viridis_r')
            color_bar = plt.colorbar(sc)
            color_bar.ax.set_title('Magnitude', fontsize=8.5)

            if magnitudes[0] is None and magnitudes[1] is None:
                plt.title('All Magnitude ranges')
            elif magnitudes[1] is None:
                plt.title(f'Magnitude ranges from {magnitudes[0]} and above')
            else:
                plt.title(f'Magnitude ranges from {magnitudes[0]} to {magnitudes[1]}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid()

        plt.figure()
        plt.loglog(df_density.tau_bin, df_density.density, marker='.')
        plt.title('Density of time btw occurrences')
        plt.ylabel('density')
        if day:
            plt.xlabel('time bin (day)')
        else:
            plt.xlabel('time bin (second)')

        plt.figure()

        plt.figure()
        log_log_hist = plt.hist(times, bins=100, log=True, ec='black', linewidth=0.4)
        plt.yscale('log')
        plt.xscale('log')
        if day:
            plt.xlabel('Days')
        else:
            plt.xlabel('Seconds')
        plt.ylabel('# of Earthquakes')
        plt.title('Histogram of Interarrival Times (log-log)')
        plt.grid(True, which='both', axis='y')

        plt.figure()
        semi_log_hist = plt.hist(times, bins=100, log=True, ec='black', linewidth=0.4)
        if day:
            plt.xlabel('Days')
        else:
            plt.xlabel('Seconds')
        plt.ylabel('# of Earthquakes')
        plt.title('Histogram of Interarrival Times (semi-log)')
        plt.grid(True, which='both', axis='y')

        display(df_density)
        plt.figure()
        plt.hist(df_filter.magnitude, bins=16, ec='black', linewidth=0.4)
        plt.yscale('log')
        plt.title('Histogram of Earthquake Magnitudes')
        plt.ylabel('# of Earthquakes')
        plt.xlabel('Magnitude')
        plt.grid(True, which='major', axis='y')

        return FilterOutput(density_df=df_density, earthquake_df=df_filter, times_btw=times)

    @classmethod
    def get_density(cls, times: np.ndarray, bin_c: float = 2.5) -> pd.DataFrame:
        min_times = times
        min_pow = 0
        max_pow = int(math.log(min_times.max(), bin_c)) + 1
        tau_bins = np.array([bin_c**i for i in range(min_pow,
                                                     max_pow + 1)])
        n_pairs = float(min_times.shape[0])

        first_density = (min_times[(min_times < tau_bins[0])].shape[0] / n_pairs) / (tau_bins[0])

        bin_densities = [first_density]

        for tau_index in range(1, len(tau_bins)):
            bin_density = (min_times[(min_times >= tau_bins[tau_index - 1]) & (min_times < tau_bins[tau_index])].shape
                               [0] / n_pairs) / (tau_bins[tau_index] - tau_bins[tau_index - 1])
            bin_densities.append(bin_density)
        return pd.DataFrame({'tau_bin': tau_bins, 'density': bin_densities})
