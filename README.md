# Build Time Series Forecasting Ensembles

This CentOS-based container running on Python3 has the tools necessary to build an ensemble of time-series forecacasting models.
Inside the `/home` folder, there are sample data and notebooks with examples on building the following models

- `ETS, TBATS` and `AUTO.ARIMA` (Using `R` through `rpy2`)
- `ARIMA, ARIMAX, SARIMAX` (Using `statsmodels`)
- `Prophet` (using Facebook's Python Library)
- `tsfresh` from Blue Yonder for automated feature extraction from time-series data.

# Forecasting tl;dr

These are the steps

- **Explore**
    - Plot the data
    - Clean outliers, Impute missing values if needed

- **Transform**
    - Take the natural log if needed

- **Decompose**
    - Check if the time-series has any **overall trend** or **seasonality**
    - Plot the decomposed series

- **Check for Stationarity** and find $d$
    - Is the series stationary?
    - Run the Augmented Dickey Fuller test,
    - Check ACF & PACF plots to
    - Determine **order of differencing** needed to stationarize the series

- **Check for Autocorrelations** and find $p, q$
    - Examine ACF and PACF plots

- **Fit ARIMA/SARIMAX model over a grid**
    - Use (p, d, q) and set up a grid search
    - Find the best model using
        - AIC/BIC
        - Out of Sample Prediction Error
    - Check your Residuals, they should be ~$N(0, 1)$ and look like white noise

- **Make predictions**

PS: that ARIMA models assume non-seasonal series, so you'll need to de-seasonalize the series before modeling

# Recommended Reading

- [Notes on Regression and TS Analysis - Duke Univ](http://people.duke.edu/~rnau/411home.htm)
- [Rules for identifying ARIMA models](http://people.duke.edu/~rnau/arimrule.htm)
- [Sean Abu's SARIMAX tutorial](http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/)
- [Modern Pandas', handling TS + SARIMAX](https://tomaugspurger.github.io/modern-7-timeseries.html)
- [ML Mastery' ARIMA Tutorial](http://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)
- [Statsmodels Documentation](http://www.statsmodels.org/dev/examples/index.html#statespace)
- [Hyndman Book](https://www.otexts.org/fpp/8)
- [SO auto.arima equivalent in Python](https://stackoverflow.com/questions/22770352/auto-arima-equivalent-for-python/22770973#22770973)
- [DO Tutorial - GridSearch for (p,d,q)](https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3)
