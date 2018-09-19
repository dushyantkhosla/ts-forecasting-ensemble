[TOC]

---

# Forecasting 

## References

- [Statistical Forecasting Notes - Robert Nau (Duke University)](http://people.duke.edu/~rnau/411home.htm)
- [Timeseries - Tom Augspurger (Modern Pandas Blog Series)](https://tomaugspurger.github.io/modern-7-timeseries.html)
- [Applied Time Series Econometrics - Jeffrey Yau (PyData SF, 2016)](https://github.com/SimiY/pydata-sf-2016-arima-tutorial)

## 1. Principles

### 1.1 Stationarity

- A *stationary* time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all *constant over time*.
  - Most statistical forecasting methods are based on the assumption that *the time series can be rendered approximately stationary* through the use of mathematical transformations. Finding the sequence of transformations needed to stationarize a time series often provides important clues in the search for an appropriate forecasting model
  - Then we simply predict that its statistical properties will be the same in the future as they have been in the past.

### 1.2 Trend- and Difference-Stationarity

- Most business and economic time series are far from stationary and exhibit behavior like trends, cycles, random-walking.   
  - If the series has a stable long-run trend and tends to revert to the trend line following a disturbance, it may be possible to stationarize it by <u>de-trending</u> (e.g., by fitting a trend line and subtracting it out prior to fitting a model, or else by including the time index as an independent variable in a regression or ARIMA model).   
    Such a series is said to be **trend-stationary**.    
  - However, sometimes even de-trending is not sufficient to make the series stationary. In this case it may be necessary to transform it into a series of period-to-period and/or season-to-season *differences*.  If the mean, variance, and autocorrelations of the original series are not constant in time, even after detrending, perhaps the statistics of the *changes* in the series between periods or between seasons *will* be constant.   Such a series is said to be **difference-stationary**.

### 1.3 First Difference

- It is the series of **changes** from one period to the next, $Y_{t} - Y_{t-1}$
- If `Diff1(Y)` is stationary and completely random (not autocorrelated), then Y is described by a [random walk](http://people.duke.edu/~rnau/411rand.htm) model: each value is a random step away from the previous value. 
- If `Diff1(Y)` is stationary but *not* completely random (i.e., if its value at period t is autocorrelated with its value at earlier periods) then Exponential Smoothing or ARIMA may be more appropriate.

```latex
If Y is the original series, and 
   y is the differenced series,

No difference     (d=0):     yt = Yt
First difference  (d=1):     yt = Yt - Yt-1
Second difference (d=2):     yt = (Yt - Yt-1) - (Yt-1 - Yt-2) = Yt - 2Yt-1 + Yt-2
```

> **<u>Rule of Thumb</u>**
>
> - $d=0$ assumes that the original series is stationary (mean-reverting). 
> - $d=1$ assumes that the original series has a constant average trend 
>   (e.g. a random walk or SES-type model, with or without growth). 
> - $d=2$ assumes that the original series has a time-varying trend 
>   (e.g. a random trend or LES-type model)

### 1.4 Logarithms

- The properties $ln(e^x) = x$  and $ln(ab) = ln(a) + ln(b)$ are useful in converting

  - natural-logged forecasts and their confidence limits back into real units
  - *multiplicative* relationships to *additive*, 
    hence translating exponential (compound) growth into linear trends

- Also, we have the following benefits

  - **Errors measured in natural-log units ≈ percentage errors**
    The error statistics of a model fitted to natural-logged data can be interpreted as approximate measures of percentage error. If you use least-squares to fit a linear model to *logged* data, you are implicitly minimizing mean squared *percentage* error, and can interpret error statistics as percentages if they're not too large ($\sigma^2 < 0.1$)

  - **Trend measured in natural-log units ≈ percentage growth** 
    Because changes in the natural logarithm are (almost) equal to *percentage* changes in the original series, it follows that the slope of a trend line fitted to logged data is equal to the average *percentage* growth in the original series.

  ---

## 2. Simple Forecasting Methods

### 2.1 The Mean Model

> The mean model assumes that the best predictor of what will happen tomorrow is the average of everything that has happened up until now.

- assumes that the series consists of $i.i.d$ values (each datum independently drawn from the same population)
- assumes that forecasts (future values) will be drawn from the same population
- the forecast that minimizes `MSE` then is simply the _sample mean_, 
  visualized as a horizontal line.

### 2.2 The Random Walk Model

> The random walk model assumes that the best predictor of what will happen tomorrow is what
> happened today, and all previous history can be ignored.

- One of the simplest and most important model in time-seris forecasting

- **Test**
  If the first-difference or diff-log transformation looks like pure noise

- **Assumption**
  *In each period, the variable takes a random step away from its previous value, and the steps sizes are i.i.d.* This is equivalent to saying that the first-difference of the series can be modeled with the mean-model.

- A RW model is said to have **drift** if the distribution of step-sizes has a *non-zero mean*.

- **Predicts**
  *All future values will equal the last observed value.* 
  Thus, the k-step-ahead forecast from period n

  - Without drift, $\hat{Y}_{n+k} = Y_n$

  - With drift,  is: $\hat{Y}_{n+k} = Y_n + k\hat{d}$

    So, the long-term forecasts look like a trend line with slope $\hat{d}$ (the *drift* term) which is calculated as the slope of a line drawn between the first and last data points.

- **Error**
  - For RW with $\hat{d}$, the 1-step ahead forecast standard error is the std-dev of `diff1(Y)`
  - For RW without $\hat{d}$, it is the RMS of the differenced series.
  - For k-steps ahead, the error is multiplied with $\sqrt{k}$ 

- **Reasons for Use**

  - Important implications in estimating the uncertainty in forecasting more that one period ahead.

    > The CI for a k-period-ahead RW forecast is wider than that of a 1-period-ahead forecast by a factor of  $\sqrt{k}$
    > This is called the <u>square root of time rule</u>, 
    > and is the most important property of the RW model.

  - As a benchmark against which to compare more complex models.

  - Many sophisticated time-series models are just fine-tuned versions of the RW model, in which lagged values of Y (auto-regressive) and/or the lagged values of forecast errors (moving-averages) are added to the prediction equation to account for the fact that the steps in a RW may not be iid - the autocorrelation patterns in step sizes are exploited to improve the forecast.

### 2.3 The Geometric Random Walk

- **Assumption**
  $ln(Y)$ walks a random (usually with drift) walk, or that the percent changes are $iid$.
- **Prediction**
  - The forecast equations remain the same as the RW model, replacing $Y$ with $ln(Y)$
  - The drift $\hat{r}$ is measured in log units, basically the same as predicting compound growth with a factor of $(1 + r)$
- **Error**
  The Standard Error of  
  - 1-step ahead forecast = $STDDEV(DIFF(ln(Y)))$
  - k-step ahead = $\sqrt{k}$ * 1-step ahead
- **Predictions**
  Point forecasts are obtained by unlogging, or taking $e^{prediction}$
- **Useful** when the series appears to have general exponential growth with occasional dips
  - Period-over-period changes are larger towards the end of the series
  - Absolute magnitude is larger towards the end

---

## 3. Time Series Modeling Basics

- *Out-of-Sample Validation* is used for model selection, typically holding-out 20% of the data
- Witholding the 20% validation sample from exploratory analysis is a good practice.
- We look at errors in both the 
  (a) Estimation (training) period 
  (b) Validation period
- *Overfitting* may occur when a model uses a large number of parameters on a small data.
- *Forecast Horizon* is the number of time-periods in the future beyond the end of available data for which we use the selected model to make predictions.
- *Confidence Interval for the Forecasts* should be reported as being rougly equal to plus-or-minus two standard deviations of the forecast error. This interval should widen as the horizon increases.

---

## 4. Forecasting with Moving Averages

### 4.0 Simple (equal-weight) Moving Average

- **Properties**
  - superior to the Mean model in adapting to cyclical patterns
  - superior to the RW model in being not-too-sensitive to periodic random shocks
- **Prediction**
  - **Simple Moving Average**
    - take the average of the most recent *m* values (Simple Moving Average, SMA)
    - The choice of *m* is a trade-off between filtering out more noise (large *m*) vs. responding to turning points quickly (small *m*). At large values of m, the plot is smoother, but starts to lag behind turning points noticeably.
    - *RMSE* and *MAPE* can be used to select the right value of *m*
- In **SMA with Trend**, we add a constant term to the SMA equation.
- In **Tapered Moving Average**, we use a weighted average with weights like `(0.5, 1, 1, 1, 0.5)` for a 5-term MA, thus making it more robust against outliers.
- **Drawbacks**
  - confidence intervals for long-horizon forecasts do not widen

### 4.1 Simple Exponential Smoothing

- Also known as *Exponentially Weighted Moving Average* models because they weight the past data in an exponentially decreasing manner.
- SES are the most widely used time-series models in business applications.
- **Forecasting Equation**
  - We define a series *L* that represents the current *level* (or local mean) computed from its own previous values
    $L_t = \alpha Y_t + (1-\alpha) L_{t-1}$
  - Here, $0 < \alpha < 1$ is a smoothing constant 
  - The level at time $t$ is computed by updating the level at $t-1$ with the most recent value of $Y$, applying weights to each respectively
  - As $\alpha$ approaches 0, the series L becomes smoother as the effect of Y is nearly zero.
  - For the no-trend case, we predict $\hat{Y}_{t+1} = L_t = \hat{Y}_{t} + \alpha . e_t$

### 4.2 Linear Expoenential Smoothing

- computes *local estimates* of both **level** $L_t $ and **trend** $T_t $
- we have *two smoothing constants*, one for the level and one for the trend. 
- **Predictions Equations**
  - Level-updating, $L_t = \alpha . Y_t + (1-\alpha)(L_{t-1} + T_{t-1})$
  - Trend-updating, $T_t = \beta(L_t - L_{t-1}) + (1-\beta)T_{t-1}$
  - Forecast, $\hat{Y}_{t+k} = L_t + k.T_t$
- Models with small $\beta$ assume that trend changes slowly over time.
- The smoothing constants can be estimated by minimizing the MSE of 1-step ahead forecasts. 

### 4.3 The Bottom Line

- Many time series that arise in business and economics (as well as engineering
  and the natural sciences) which are inherently non-seasonal and display a pattern of random variations around a local mean value or a local trend line
  that changes slowly with time. 
- The first difference of such a series is negatively autocorrelated at lag 1: a positive change tends to be followed by a (smaller) negative one. 
  - For time series of this type, a smoothing or averaging model is the appropriate forecasting model. 
  - The simple exponential smoothing model is often a good choice, even though it assumes no trend, and in fact it is probably the most widely used model in applied forecasting. 
  - If there is a compelling reason for estimating a non-zero trend for purposes of longer-term forecasting, a constant trend can be implemented via an ARIMA(0,1,1)+constant model or a time-varying trend can be implemented via a linear exponential smoothing model. 
  - However, it is hard to predict changes in trends or the occurrence of turning points merely from the history of the series itself. For such purposes it is important to incorporate information from other sources.

---

# Forecasting with ARIMA Models

Regression models that use lags and differences, random walk models, exponential smoothing models, and seasonal adjustment models are all special cases of a more general class of time series models known as ARIMA models.

## What ARIMA stands for

- A series which needs to be *differenced* to be made stationary is an *integrated* (I) series
- Lags of the stationarized series are called *autoregressive* (AR) terms
- Lags of the forecast errors are called *moving average* (MA) terms

## Steps in building non-seasonal ARIMA Models

![image-20180919154538915](/var/folders/kv/l02c_chn7tqd9cmrvyms7vy4bfc09j/T/abnerworks.Typora/image-20180919154538915.png)

1. Apply transformations such as $ln(Y)$ 
   (Ensure that *local random variations* are consistent over time and symmetric in appearance.)
2. Make *stationary* by taking Differences ($d = 1, 2, ...$)
   (If the transformed $Y$ exhibits linear/non-linear/randomly-varying trend or random-walk behavior)
3. Then, the ARIMA equation for stationary $Y$ becomes
   $\hat{Y}_t$ = constant + Weighted sum of last $p$ values of $Y$ + Weighted sum of last $q$ forecast errors
4. Here, $p$ and $q$ are small integers, and the weights may be positive or negative.
5. In most cases, either $p=0$ or $q=0$ and $p+q <= 3$
6. The constant term may or may not be zero.
7. The lagged values of stationarized $Y$ are called the *Autoregressive Terms*
8. The lagged values of forecast errors are called the *Moving Average Terms*

> $\hat{y}_t = \mu + (\psi_1.y_{t-1} + ... + \psi_p.y_{t-p} ) - (\theta_1.e_{t-1} + ... + \theta_q.e_{t-q})$
>
> Where
>
> - $\mu$ is the constant
> - $\psi_k$ is the AR coefficient at lag $k$
> - $\theta_k$ is the MA coefficient at lag $k$
> - $e_{t-k} = y_{t-k} - \hat{y}_{t-k}$ is the forecast error made in period $t-k$
>
> The resulting model is called an $ARIMA(p,d,q)$ model if we assume $\mu=0$
> Alternatively, we write it as $ARIMA(p,d,q) + constant$

The I stands for *integrated*, because the series needs to be *differenced* in order to be made stationary.

## ACF and PACF Plots

- The ACF plot shows the correlation of the series with its past selves.
- The PACF plot shows the correlation at different lags that is not explained by lower-order lags. This is the same as the regression coefficient for the particular lag.

## Selecting $p,q$

- Look at ACF (Autocorrelation Function) and PACF (Partial ACF) plots.
- Autocorrelation at lag $k$ is written as
  $ACF(y_k) = CORR(y_t, y_{t-k})$
- The Partial Autocorrelation of $y$ at lag $k$ is the amount of correlation between $y$ and $Y_{LAGk}$ that is not explained by lower-ordered lags. For example, the PACF at `lag=2` is the coefficient of $Y_{LAG2}$ in a regression of $y$ on $Y_{LAG1}$ and $Y_{LAG2}$
- **AR and MA Signatures**
  - The $MA(q)$ Signature
    - Noisy plot
    - the ACF plot cuts-off sharply at lag $k$, while 
      the PACF decays slowly.
    - Set $p=0, q=k$
  - The $AR(p)$ Signature
    - Mean-reverting plot
    - The PACF plot cuts-off sharply at lag $k$, while 
      the ACF decays slowly.
    - Set $p=k, q=0$
- If there are spikes at every $j^{th}$ lag, this might indicate seasonality.
- **Examples**
  - Single positive spike at lag 1 in both ACF and PACF, 
    then $p=1, q=0$
  - Single negative spike at lag 1 in both ACF and PACF, 
    then $p=0, q=1$
  - Gradual decay in ACF, spikes at lags 1, 2 in PACF, 
    then $p=2, q=0$ or $AR(2)$
  - Negative Spike in ACF at lag 1, gradual decay in PACF (from below), 
    then $MA(1)$
- **Model Coefficients**
  - After correctly identifying $p,d,q$, we train model to estimate its coefficients
  - The highest order AR or MA coefficient should be significantly different from zero.
    - If not, try reducing the p or q by 1.
- **Residual ACF, PACF plots**
  - If there are spikes in the residual ACF, PACF plots, try increasing the p or q by 1.
  - Spikes in the residual ACF plot at lags 1, 2, or 3 signify a need for a higher value of q
  - Spikes in the residual PACF plot at lags 1, 2, or 3 indicate a need for a higher order of p

## Common non-seasonal ARIMA Models

| Model              | Equals                                                |
| :----------------- | ----------------------------------------------------- |
| ARIMA(0, 0, 0) + c | Mean (constant) model                                 |
| ARIMA(0, 1, 0)     | Random Walk model                                     |
| ARIMA(0, 1, 0) + c | Random Walk (or Geometric) with Drift                 |
| ARIMA(1, 0, 0) + c | AR(1), regression of Y on LAG1                        |
| ARIMA(2, 0, 0) + c | AR(2), regression of Y on LAG1, LAG2                  |
| ARIMA(1, 1, 0) + c | Regression of DIFF(Y) on LAG(DIFF(Y))                 |
| ARIMA(2, 1, 0) + c | Regression of DIFF(Y) on LAG1(DIFF(Y)), LAG2(DIFF(Y)) |
| ARIMA(0, 1, 1)     | Simple Exponential Smoothing                          |
| ARIMA(0, 1, 1) + c | Simple Exponential Smoothing + constant linear trend  |
| ARIMA(1, 1, 2)     | Linear Exponential Smoothing + damped trend           |
| ARIMA(0, 2, 2)     | Generalized Linear Exponential Smoothing              |
|                    |                                                       |

- ARIMA models with 2 orders of nonseasonal differencing, and LES models assume that there is a randomly-varying local trend (instead of a constant local trend) and so their confidence in predicting longer horizons is more uncertain.

## Example: Unit Sales 

- Strong upward trend $\implies$ non-stationary
- Non-exponential growth $\implies$ transformation not needed
- NOTE: Trended time series always have
  - Very large $+ve$ autocorrelations at <u>all</u> lower order lags
  - Single spike at lag=1 in the PACF
  - This signature indicates that we need $d \ge 1$
- The PACF cuts-off sharply, and 
  the ACF is $+ve$  at the first few lags and decaying $\implies$ AR Signature
- Try $ARIMA(1, 1, 0) + \mu$
- Look at the Residual plot + Residual ACF, PACF
  - The residual plot should look like pure noise
  - Local means should not be non-zero for long stretches of time
  - If so, try higher values of p, d.

## Summary

1. Stationarize the series, if necessary, by differencing (& perhaps also logging, deflating, etc.)
2. Study the pattern of autocorrelations and partial autocorrelations to determine if lags of the stationarized series and/or lags of the forecast errors should be included in the forecasting equation
3. Fit the model that is suggested and check its residual diagnostics, particularly the residual ACF and PACF plots, to see if all coefficients are significant and all of the pattern
  has been explained.
4. Patterns that remain in the ACF and PACF may suggest the need for additional AR or MA terms



## Interpretation of values of $p, q$

### AR Terms

- Autoregressive behavior is a tendency of the series to regress towards the mean
- For $p=1$, the coefficient $\psi_1$ determines how fast this regression happens
  - If $\psi \rightarrow 0$, the series returns to the mean quickly
  - If $\psi \rightarrow 1$, the series returns to the mean slowly
- For $p \ge 2$, the sum of the coefficients determine the speed of mean reversion
  - The series may exhibit an oscillatory pattern

### MA Terms

- Moving-average behavior is displayed by a series that undergoes random 'shocks' whose effects are felt in two or more consecutive periods
- For $MA(1), or,  q=1$ the coefficient $\theta_1$ is the fraction of last period's shock that is still felt in the current period
- For $MA(2), or,  q=2$ the coefficient $\theta_2$ is the fraction of the shock from two periods ago that is still felt in the current period



## AR or MA? It depends!

- Whether a series displays AR or MA behavior often depends on the extent to which it has been differenced.
- An “underdifferenced” series has an AR signature (positive autocorrelation)
- After one or more orders of differencing, the autocorrelation will become more negative and an MA signature will emerge
- Don’t go too far: if series already has zero or negative autocorrelation at lag 1, don’t difference again
- For example
  -  With $d=1$, the ACF + PACF might suggest `AR(1)` or `AR(2)`
  - With $d=2$, the ACF + PACF might suggest `MA(1)`
  - 

