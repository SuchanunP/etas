### Visualization of an earthquake catalog
```
from etas.vis import EarthQuakeVisualizer

# df: pandas dataframe with columns: date, magnitude, latitude, longitude, depth
# shp_path:  a string folder path of shapefile

vis = EarthQuakeVisualizer(df=naf_df, shp_path=shapefile_path)
vis.visualize()
```

### ETAS model fitting

```
from etas.util import fit_etas, calculate_std_errors

# df: pandas dataframe with columns: date, magnitude
# window: a tuple of start year (inclusive) and end year (exlusive)
# m_c: minimum magnitude that will be considered

etas = fit_etas(df=df, window=window, m_c=m_c)
bounds = calculate_std_errors(log_fitted_params=etas.opt_result[0], inv_H=etas.opt_result[3])
etas.params.set_std_errors(std_errors=bounds)
etas.params.print()
```
