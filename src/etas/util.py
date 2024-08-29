from decimal import Decimal, ROUND_HALF_UP
import numpy as np


def round_half_up(number, decimal_places):
    return Decimal(number).quantize(Decimal('1e-{0}'.format(decimal_places)), rounding=ROUND_HALF_UP)


def get_windows(start, end, n):
    ans = []
    for i in range((end - start + 1) // n):
        ans.append([start + int(i * n), min(start + int(i * n) + n, end)])

    return ans


def to_mw(magnitude, unit: str):
    # https://nhess.copernicus.org/articles/21/2059/2021/
    unit_lower = unit.lower()
    if unit_lower.startswith('ml'):
        return 1.017 * magnitude - 0.012
    elif unit_lower.startswith('mw'):
        return magnitude
    elif unit_lower.startswith('md'):
        return 1.111 * magnitude - 0.459
    return np.nan


def calculate_std_errors(log_fitted_params, inv_H):
    fitted_params = np.exp(log_fitted_params)
    dia_inv_H = np.diag(inv_H)

    log_std_errors = np.sqrt(dia_inv_H)

    left_bound = np.exp(log_fitted_params - log_std_errors)
    right_bound = np.exp(log_fitted_params + log_std_errors)

    delta_left = fitted_params - left_bound
    delta_right = right_bound - fitted_params

    delta_avg = (delta_left + delta_right) / 2
    return delta_avg


def get_fit_df(df, window, m_c=3.0):
    window_df = df[(df.date.dt.year >= window[0]) & (df.date.dt.year < window[1])].copy()
    window_df = window_df[window_df.magnitude >= m_c].copy()
    return window_df


def fit_etas(df, window, m_c=3.0):
    from etas import Etas
    window_df = get_fit_df(df=df, window=window, m_c=m_c)
    if window_df.shape[0]:
        print(f'windows: {window}\ndf.n_rows: {window_df.shape[0]}\n')

        etas = Etas(df=window_df)
        etas.fit()
    else:
        etas = None
    return etas
