# TimeSeries
This is a project to analyze different models to apply to forecasting. In order to do so I obtained the file AEP_hourly.csv from soumilshah1995 (github handle). It contains 10 years of hourly energy consumption from PJM in MegaWatts. PJM is a regional transmission organization (RTO) that coordinated the movement  of electricity in the Eastern part of the US. You can also find this staset in Kaggle. I converted the hourly consumption to monthly consumption to do our forecast. THe dataset goes from 2006 to 2018. I used the last 2 years (2017 and 2018) as test.

 Split models in FOUR options:
 1. TimeSeriesOne. I used basic models starting with the naive one and ending with Holts Winter. This one happends to be obtaining the best score (RMSE) not only for the collection but also the entire set of notebooks. THe last 2 are the ones adding seasonality and it shows.
 2. TimeSeriesTwo. Use ARIMA, SARIMA and Prophet. While the first two are complex in its application the third one (Prophet) is by far the easiest one to apply and one of the most efficient.
 3. TimeSeriesThree. In this third chapter, I tryb to use models used in predictions in forecasting: RandomForest and XGBoost. Good scores obtained.
 4. TimeSeriesFour. Let's try some RNN. THese are the most intrincated models used.

Final Scores:

| Rank| Model           |RMSE per million|
| :---| :-------:       | ---------:|
|  1  | XGboost_tt      |	0.48    |
|  2  |	holt_winter	    |0.54       |
|  3  |LSTM	            |0.56       |
|  4  |prophet          |0.57       |
5	SARIMAX	0.59099874
6	RandomForest_tt	0.6112166331292902
7	LSTM_stacked	0.6726724218947377
8	arima	0.7373484
9	RandomForest_bu	0.8636077926057387
10	XGBoost_bu	0.8879243226583203
11	holt_linear	0.92976832
12	SES	0.94277767
13	moving_average	0.94404761
14	simple_average	1.1221768799999998
15	naive	1.32560856

I am going to use Naive Methods to do forecasts. Then I will try ARIMA, SARIMA and FBProphet.
It is my intention to use also Strutural Time Series and finaly, at least for now, use some RandomeForest and XGboost

