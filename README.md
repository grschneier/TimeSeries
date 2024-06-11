# TimeSeries
In machine learning in general, the idea of trying to calculate the future values of any given variable is called predicting. But, if this process is applied to a variable bound to time is called forecasting. This is a project that has as a goal to apply different models to forecasting. In order to do so I worked with the file AEP_hourly.csv from soumilshah1995 (github handle). It contains 10 years of hourly energy consumption from PJM in MegaWatts. PJM is a regional transmission organization (RTO) that coordinated the movement  of electricity in the Eastern part of the US. You can also find this staset in Kaggle. I converted the hourly consumption to monthly consumption to do our forecast. The dataset goes from 2006 to 2018. I used the last 2 years (2017 and 2018) as test.

 I did split the models in FOUR groups:
 1. TimeSeriesOne. I used basic models starting with the naive one and ending with Holts Winter. This one happened to be one of the best score (RMSE) not only for the collection but also the entire set of notebooks. The last 2 are the ones adding seasonality and it showed lowering the RMSE in a significant way.
 2. TimeSeriesTwo. Use ARIMA, SARIMA and Prophet. While the first two are complex in its application the third one (Prophet) is by far the easiest one to apply and one of the most efficient.
 3. TimeSeriesThree. In this third chapter, I tryb to use models used in predictions in forecasting: RandomForest and XGBoost. The best score obtained.
 4. TimeSeriesFour. Let's try some RNN. These are the most intricated models used. Complexity not only with the layers but also with the morphology of the arrays and matrices.

Final Scores:

| Rank| Model           |RMSE<br>(per million)|
| :---| :-------:       | ---------:|
|  1  | XGboost_tt      |	0.48    |
|  2  |	holt_winter	    |0.54       |
|  3  |LSTM	            |0.56       |
|  4  |prophet          |0.57       |
|  5  |SARIMAX	        |0.59      |
|6	  |RandomForest_tt	|0.61|
|7	  |LSTM_stacked	    |0.67|
|8	  |arima	        |0.74|
|9	  |RandomForest_bu	|0.86|
|10	  |XGBoost_bu	    |0.89|
|11	  |Holt_linear	    |0.93|
|12	  |SES	            |0.94|
|13	  |moving_average	|0.94|
|14	  |simple_average	|1.12|
|15	  |naive	        |1.32|

Once done with this part, I also went over a comparison of SARIMA and prophet. It is done in the folder.

We did an analysis over a time series using SARIMA and FBProphet. In all these cases, using the Normalized Root Mean Squared Error as a metric, Prophet outscores SARIMA. We might try to dig a little deeper and add some other thoughts and ideas. We are taking about the conclusions. Before we do so, I want to prevent us to fall into Recency Bias (thinking that our last case is the one that matters), Confirmation bias (this is what I thought therefore this is it) or, in other words, extend the conclusions of this only case to all other situations and time series we can find in the future.

# **Conclusions**:

1. SARIMA is way slower. It takes more time to train SARIMA and then run the winner. FBrophet has no parameters to tune
2. Prophet scores lower (and better) nrmse most of the times. Sarimax is only better in weekly and hourly. It feels like sarimax does not gain from adding more data points to the dataset. After certain amount of rows, it does not get better rather it worsens
3. SARIMA seems to pick up better the small nuances of the time series like we can see on the weekly analysis. Despite this, Prophet scored better.

