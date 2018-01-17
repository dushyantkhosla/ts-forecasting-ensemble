# Build Time Series Forecasting Ensembles

This CentOS-based container running on Python3 has the tools necessary to build an ensemble of time-series forecacasting models.
Inside the `/home` folder, there are sample data and notebooks with examples on building the following models

- `ETS, TBATS` and `AUTO.ARIMA` (Using `R` through `rpy2`)
- `ARIMA, ARIMAX, SARIMAX` (Using `statsmodels`)
- `Prophet` (using Facebook's Python Library)
- `tsfresh` from Blue Yonder for automated feature extraction from time-series data.

### Pull the image

- `docker pull eadlab/ds-docker-time-series`

### Run the container

```
docker run -it -v (pwd):/home \
               -p 8080:8080 \
               -p 5000:5000 \
               -e GIT_USER_NAME="Dushyant Khosla" \
               -e GIT_USER_MAIL="dushyant.khosla@yahoo.com" \
               eadlab/ds-docker-time-series
```

