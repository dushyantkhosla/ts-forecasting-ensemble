# Build Time Series Forecasting Ensembles

This CentOS-based container running on Python3 has the tools necessary to build an ensemble of time-series forecacasting models.
Inside the `/ts-reference` folder, there are sample data and notebooks with examples on building the following models

- ETS, TBATS and AUTO.ARIMA (Using R)
- ARIMA, ARIMAX, SARIMAX (Using statsmodels)
- Prophet (using Facebook's Python Library)
- `tsfresh` from Blue Yonder for automated feature extraction from time-series data.

---

### Pull the image

- `docker pull eadlab/ds-docker-time-series`

### Run the container

- `docker run -it -p 8080:8080 eadlab/ds-docker-time-series`
