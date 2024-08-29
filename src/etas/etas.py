from typing import Tuple, Union
from dataclasses import dataclass
from etas.util import round_half_up as round_up
import pandas as pd
import numpy as np
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt


@dataclass
class History:
    magnitudes: np.ndarray
    times: np.ndarray

    @classmethod
    def from_df(cls, df):
        times = np.zeros(df.shape[0])
        for i, t in enumerate(df["date"]):
            times[i] = (t - df["date"].iloc[0]).total_seconds() / 60 / 60 / 24

        return History(times=times, magnitudes=df.magnitude.values)

    def __post_init__(self):
        self.magnitudes = np.array(self.magnitudes)
        self.times = np.array(self.times)


@dataclass
class EtasParams:
    def __init__(self, params: np.ndarray = np.array([1.0, 0.1, 1.1, 0.1, 0.1]), stds=None):
        (alpha, c, p, K0, u) = params
        self.u = u
        self.K0 = K0
        self.alpha = alpha
        self.p = p
        self.c = c

        self.u_std = np.nan
        self.K0_std = np.nan
        self.alpha_std = np.nan
        self.p_std = np.nan
        self.c_std = np.nan
        self.stds = stds
        self.set_std_errors(std_errors=stds)

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"'{key}' not found in EtasParams")

    def set_std_errors(self, std_errors: np.ndarray):
        if std_errors is not None:
            (self.alpha_std, self.c_std, self.p_std, self.K0_std, self.u_std) = std_errors

    def get(self):
        return np.array([self.alpha, self.c, self.p, self.K0, self.u])

    def print(self, precision=3):
        if self.u_std:
            print(f'u = {round_up(self.u, precision)}±{round_up(self.u_std, precision)}')
        else:
            print(f'u = {round_up(self.u, precision)}')
        if self.K0_std:
            print(f'K0 = {round_up(self.K0, precision)}±{round_up(self.K0_std, precision)}')
        else:
            print(f'K0 = {round_up(self.K0, precision)}')
        if self.c_std:
            print(f'c = {round_up(self.c, precision)}±{round_up(self.c_std, precision)}')
        else:
            print(f'c = {round_up(self.c, precision)}')
        if self.alpha_std:
            print(f'alpha = {round_up(self.alpha, precision)}±{round_up(self.alpha_std, precision)}')
        else:
            print(f'alpha = {round_up(self.alpha, precision)}')
        if self.p_std:
            print(f'p = {round_up(self.p, precision)}±{round_up(self.p_std, precision)}')

    def copy(self):
        copied_etas = EtasParams(params=self.get(), stds=self.stds)
        return copied_etas


class Etas:
    def __init__(self, df: pd.DataFrame,
                 params=EtasParams()):
        df.date = pd.to_datetime(df.date)
        df = df.sort_values(by='date')
        self.df = df.copy()
        self.history = History.from_df(self.df)
        self.params = params
        self.opt_result = None

    def fit(self, gtol=0.0001):
        self.opt_result = fmin_bfgs(self.minus_log_likelihood, np.log(self.params.get()),
                                    fprime=self.calculate_grad, gtol=gtol, retall=True, full_output=True)

        self.params = EtasParams(params=np.exp(self.opt_result[0]))
        return self

    def cond_lambda(self, t: Union[int, float], X: np.ndarray, history: History):
        (alpha, c, p, K0, u) = X
        magnitudes = history.magnitudes[history.times < t]
        times = history.times[history.times < t]
        Mz = np.min(history.magnitudes)

        return u + np.sum(np.array([K0 * np.exp(alpha * (Mi - Mz)) / (t - ti + c) ** p
                                    for Mi, ti in zip(magnitudes, times)]))

    def integrated_lambda(self, X: np.ndarray, time_until: int):
        (alpha, c, p, K0, u) = X
        history = self.history
        Mz = np.min(history.magnitudes)

        if abs(p - 1) < 0.001:
            N = u * time_until + ((history.times <= time_until) * K0 * np.exp(alpha * (history.magnitudes - Mz)) * \
                                  ((np.log((history.times <= time_until) * (time_until - history.times) + c) ** 1 - np.log(c) ** 1) \
                                   + (np.log((history.times <= time_until) * (time_until - history.times) + c) ** 2 - np.log(c) ** 2) * (
                                      1 - p) ** 1 / 2 \
                                   + (np.log((history.times <= time_until) * (time_until - history.times) + c) ** 3 - np.log(c) ** 3) * (
                                      1 - p) ** 2 / 6 \
                                   )).sum()
        else:
            N = u * time_until + ((history.times <= time_until) * K0 * np.exp(alpha * (history.magnitudes - Mz)) * (
                    ((history.times <= time_until) * (time_until - history.times) + c) ** (1 - p) - c ** (1 - p)) / (1 - p)).sum()

        return N

    def log_likelihood(self, X: np.ndarray):
        history = self.history
        sum_log_lambdas = np.sum([np.log(self.cond_lambda(t=t, X=X, history=history)) for t in history.times[1:]])
        integrated_lambda = self.integrated_lambda(X=X, time_until=self.history.times[-1])
        return sum_log_lambdas - integrated_lambda

    def minus_log_likelihood(self, log_etas_params: np.ndarray, log_params=True):
        if log_params:
            return -1 * self.log_likelihood(np.exp(log_etas_params))
        else:
            return -1 * self.log_likelihood(log_etas_params)

    def plot(self):
        predicted_cum_numbers = [self.integrated_lambda(X=self.params.get(), time_until=t) for t in self.history.times]
        fig, ax = plt.subplots(figsize=(8, 8))
        x = np.arange(1, len(predicted_cum_numbers))
        y = predicted_cum_numbers[1:]
        ax.set_xlabel("Observed cumulative number of earthquakes")
        ax.set_ylabel("Fitted ETAS cumulative number of earthquakes")
        ax.plot(x, y, '--', label='Fitted ETAS')
        ax.plot(x, x, label='Observed')
        plt.legend()

        return y, ax

    def plot_lambdas(self, time_range: Tuple[int, int] = (0, 100)):
        lambdas = [self.cond_lambda(t=t, X=self.params.get(), history=self.history) for t in self.history.times]
        plt.figure(figsize=(14, 4))

        plt.plot(self.df.date[time_range[0]: time_range[1]], lambdas[time_range[0]: time_range[1]],
                 label=r'$\lambda^*(t)$')
        sc = plt.scatter(self.df.date[time_range[0]: time_range[1]],
                         -1 * np.ones(len(self.history.times[time_range[0]: time_range[1]])), marker='.',
                         c=self.history.magnitudes[time_range[0]: time_range[1]], cmap='autumn_r', alpha=0.3,
                         label='Earthquakes')
        color_bar = plt.colorbar(sc)
        plt.xlabel('Time')
        plt.legend()
        color_bar.ax.set_title('Magnitude', fontsize=8.5)
        return lambdas

    def calculate_grad(self, log_etas_params, log_params=True):
        if log_params:
            (alpha, c, p, K0, u) = np.exp(log_etas_params)
        else:
            alpha, c, p, K0, u = np.exp(log_etas_params)
        grad = np.zeros(5)

        Tmax = self.history.times[-1]
        Mc = np.min(self.history.magnitudes)
        Ti = self.history.times
        Mi = self.history.magnitudes

        for i, t in enumerate(Ti):
            cond_lambda = u + ((t > Ti) * K0 * np.exp(alpha * (Mi - Mc)) / ((t > Ti) * (t - Ti) + c) ** p).sum()
            grad[0] += ((t > Ti) * K0 * np.exp(alpha * (Mi - Mc)) * (Mi - Mc) / ((t > Ti) * (t - Ti) + c) ** p).sum() / cond_lambda
            grad[1] += ((t > Ti) * K0 * np.exp(alpha * (Mi - Mc)) * (-p) / ((t > Ti) * (t - Ti) + c) ** (p + 1)).sum() / cond_lambda
            grad[2] += ((t > Ti) * K0 * np.exp(alpha * (Mi - Mc)) * np.log(1 / ((t > Ti) * (t - Ti) + c)) / (
                        (t > Ti) * (t - Ti) + c) ** p).sum() / cond_lambda
            grad[3] += ((t > Ti) * 1 * np.exp(alpha * (Mi - Mc)) / ((t > Ti) * (t - Ti) + c) ** p).sum() / cond_lambda
            grad[4] += 1 / cond_lambda

        if abs(p - 1) < 0.001:
            grad[0] += - ((Tmax >= Ti) * K0 * np.exp(alpha * (Mi - Mc)) * (Mi - Mc) * \
                          ((np.log((Tmax >= Ti) * (Tmax - Ti) + c) ** 1 - np.log(c) ** 1) \
                           + (np.log((Tmax >= Ti) * (Tmax - Ti) + c) ** 2 - np.log(c) ** 2) * (1 - p) ** 1 / 2 \
                           + (np.log((Tmax >= Ti) * (Tmax - Ti) + c) ** 3 - np.log(c) ** 3) * (1 - p) ** 2 / 6 \
                           )).sum()
            grad[1] += - ((Tmax >= Ti) * K0 * np.exp(alpha * (Mi - Mc)) * (
                        ((Tmax >= Ti) * (Tmax - Ti) + c) ** (0 - p) - c ** (0 - p))).sum()
            grad[2] += - ((Tmax >= Ti) * K0 * np.exp(alpha * (Mi - Mc)) * \
                          (-(np.log((Tmax >= Ti) * (Tmax - Ti) + c) ** 2 - np.log(c) ** 2) / 2 \
                           - (np.log((Tmax >= Ti) * (Tmax - Ti) + c) ** 3 - np.log(c) ** 3) * (1 - p) / 3 \
                           - (np.log((Tmax >= Ti) * (Tmax - Ti) + c) ** 4 - np.log(c) ** 4) * (1 - p) ** 2 / 8 \
                           )).sum()
            grad[3] += - ((Tmax >= Ti) * np.exp(alpha * (Mi - Mc)) * \
                          ((np.log((Tmax >= Ti) * (Tmax - Ti) + c) ** 1 - np.log(c) ** 1) \
                           + (np.log((Tmax >= Ti) * (Tmax - Ti) + c) ** 2 - np.log(c) ** 2) * (1 - p) ** 1 / 2 \
                           + (np.log((Tmax >= Ti) * (Tmax - Ti) + c) ** 3 - np.log(c) ** 3) * (1 - p) ** 2 / 6 \
                           )).sum()
            grad[4] += - Tmax

        else:
            dp = 0.001
            pp0 = (((Tmax >= Ti) * (Tmax - Ti) + c) ** (1 - p + dp / 2) - c ** (1 - p + dp / 2)) / (1 - p + dp / 2)
            pp1 = (((Tmax >= Ti) * (Tmax - Ti) + c) ** (1 - p - dp / 2) - c ** (1 - p - dp / 2)) / (1 - p - dp / 2)
            pp = (pp1 - pp0) / dp

            grad[0] += - ((Tmax >= Ti) * K0 * np.exp(alpha * (Mi - Mc)) * (Mi - Mc) * (
                        ((Tmax >= Ti) * (Tmax - Ti) + c) ** (1 - p) - c ** (1 - p)) / (1 - p)).sum()
            grad[1] += - ((Tmax >= Ti) * K0 * np.exp(alpha * (Mi - Mc)) * 1 * (
                        ((Tmax >= Ti) * (Tmax - Ti) + c) ** (0 - p) - c ** (0 - p))).sum()
            grad[2] += - ((Tmax >= Ti) * K0 * np.exp(alpha * (Mi - Mc)) * pp).sum()
            grad[3] += - ((Tmax >= Ti) * 1 * np.exp(alpha * (Mi - Mc)) * 1 * (
                        ((Tmax >= Ti) * (Tmax - Ti) + c) ** (1 - p) - c ** (1 - p)) / (1 - p)).sum()
            grad[4] += - Tmax

        if log_params:
            return -2 * grad * np.exp(log_etas_params)
        else:
            return -2 * grad
