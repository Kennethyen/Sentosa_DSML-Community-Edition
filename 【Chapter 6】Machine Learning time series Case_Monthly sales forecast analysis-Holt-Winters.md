# 1、Holt-Winters algorithm
What is the Holt Winters prediction algorithm?
&emsp;&emsp;The Holt Winters algorithm is a time series prediction method. Time series prediction methods are used to extract and analyze data and statistical data, and characterize the results, in order to more accurately predict the future based on historical data. The Holt Winters prediction algorithm allows users to smooth time series and use this data to predict areas of interest. The exponential smoothing method assigns weights and values that decrease exponentially based on historical data to reduce the weight values of older data. In other words, in prediction, newer historical data has greater weight than older results.
&emsp;&emsp;There are three exponential smoothing methods used in Holt Winters:
&emsp;&emsp;Single exponential smoothing - suitable for predicting data without trends or seasonal patterns, where data levels may vary over time.
&emsp;&emsp;Double exponential smoothing method - used to predict data with trends.
&emsp;&emsp;Triple exponential smoothing method - used to predict data with trends and/or seasonality.
&emsp;&emsp;Holt Winters includes prediction equations and three smoothing equations, which are used to process levels $\ell_{t},$ trend $b_{t}$ And seasonal components$s t$ ，The corresponding smoothing parameters are $\alpha, \ \beta^{*}$ and$\gamma$ 。Usually, $m $is used to represent seasonal cycles, such as quarterly data,For example, quarterly data $m=4$, monthly data $m=12$,
&emsp;&emsp;There are two variants of the Holt Winters method, with the main difference being the handling of seasonal components:
&emsp;&emsp;1. Additive model: Use additive model when seasonal changes are relatively stable.
&emsp;&emsp;2.. Multiplication model: When seasonal changes are proportional to data levels, the multiplication model is applicable.
## (1) additive model 
&emsp;&emsp;In the additive model, seasonal components are represented by absolute values, and seasonal adjustments are made to the data by subtracting seasonal components in the horizontal equation. Within each year, the sum of seasonal components is approximately zero. The component form of the additive model is:
$$\hat{y}_{t+h | t}=\ell_{t}+h b_{t}+s_{t+h-m ( k+1 )} $$
&emsp;&emsp;Contains three smoothing equations, where the horizontal equation is a weighted average that includes seasonally adjusted observations $( y_{t}-s_{t-m} )$ and non seasonal forecast values$( \ell_{t-1}+b_{t-1} )$
$$\ell_{t}=\alpha( y_{t}-s_{t-m} )+( 1-\alpha) ( \ell_{t-1}+b_{t-1} ) $$
&emsp;&emsp;The trend equation is the same as Holt's linear method.
$$b_{t}=\beta^{*} ( \ell_{t}-\ell_{t-1} )+( 1-\beta^{*} ) b_{t-1} $$
&emsp;&emsp;The seasonal equation is based on the current seasonal index $( y_{t}-\ell_{t-1}-b_{t-1} )$ Seasonal index of the same season as the previous year $s_{t-m}$ To smooth out seasonal components。
$$s_{t}=\gamma( y_{t}-\ell_{t-1}-b_{t-1} )+( 1-\gamma) s_{t-m} $$
## (2) Multiplication model
&emsp;&emsp;In the multiplication model, seasonal components are expressed as relative values (percentages) and seasonal adjustments are made by dividing the time series by the seasonal components. Within each year, the seasonal composition is approximately $m_{\circ}$ ,The component form of the multiplication model is:
$$\hat{y}_{t+h | t}=( \ell_{t}+h b_{t} ) s_{t+h-m ( k+1 )} $$
$$\ell_{t}=\alpha{\frac{y_{t}} {s_{t-m}}}+( 1-\alpha) ( \ell_{t-1}+b_{t-1} ) $$
$$b_{t}=\beta^{*} ( \ell_{t}-\ell_{t-1} )+( 1-\beta^{*} ) b_{t-1} $$
$$s_{t}=\gamma{\frac{y_{t}} {( \ell_{t-1}+b_{t-1} )}}+( 1-\gamma) s_{t-m} $$
## (3) Damping trend
&emsp;&emsp;Holt Winters can introduce damping trends in addition and multiplication seasonal models. Damping trend can make the model more robust in predicting future trends, avoiding infinite trend extension, and is suitable for time series data where trends may gradually stabilize. This method combines seasonality and trend smoothing, and controls the persistence of trends through the damping factor 𝜙 (0<𝜙<1), introducing 𝜙 into the trend component to gradually reduce the contribution of future trends. In this way, as the forecast period increases, the influence of the trend will gradually weaken, thus avoiding over extension.
&emsp;&emsp;The prediction equation for seasonal multiplication combined with damping trend is:
$$\hat{y}_{t+h | t}=\left[ \ell_{t}+( \phi+\phi^{2}+\cdots+\phi^{h} ) b_{t} \right] s_{t+h-m ( k+1 )} $$
$$\ell_{t}=\alpha\left( \frac{y_{t}} {s_{t-m}} \right)+\left( 1-\alpha\right) \left( \ell_{t-1}+\phi b_{t-1} \right) $$
$$b_{t}=\beta^{*} \left( \ell_{t}-\ell_{t-1} \right)+( 1-\beta^{*} ) \phi b_{t-1} $$
$$s_{t}=\gamma\left( {\frac{y_{t}} {\ell_{t-1}+\phi b_{t-1}}} \right)+( 1-\gamma) s_{t-m} $$
# 2、Advantages and disadvantages of Holt Winters algorithm
## Advantages 
&emsp;&emsp;1. The Holt Winters method can effectively capture and model seasonal changes in time series, and is suitable for data with periodic fluctuations.
&emsp;&emsp;2. By setting smoothing parameters, the Holt Winters method can dynamically adjust estimates of trends and seasonality to adapt to changes in time series data.
&emsp;&emsp;3. The parameters included in the model (level, trend, seasonality) are easy to interpret and facilitate understanding of the components of the time series.
&emsp;&emsp;4. In terms of short-term forecasting, the Holt Winters method usually provides high accuracy.
## disadvantages
&emsp;&emsp;The selection of smoothing parameters has a significant impact on model performance, and it is usually necessary to optimize these parameters through experience or cross validation, which increases the complexity of model settings.
&emsp;&emsp;In the prediction of long-term time series, the Holt Winters method may produce unrealistic trends, especially in the absence of damping, which may lead to unstable long-term prediction results.
# 3、Comparison of algorithm implementation between Python code and Sentosa_DSML community edition
## (1) Data reading and statistical analysis
1、Implementation in Python code
```python
#导入需要的库
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


file_path = r'.\每月香槟销量.csv'#文件路径
df = pd.read_csv(file_path, header=0)
print("原始数据前5行:")
print(df.head())

>>原始数据前5行:
     Month  Perrin Freres monthly champagne sales millions ?64-?72
0  1964-01                                             2815.0     
1  1964-02                                             2672.0     
2  1964-03                                             2755.0     
3  1964-04                                             2721.0     
4  1964-05                                             2946.0     

df = df.rename(columns={
    'Month': '月份',
    'Perrin Freres monthly champagne sales millions ?64-?72': '香槟销量'
})


print("\n修改列名后的数据前5行:")
print(df.head())

>>修改列名后的数据前5行:
        月份    香槟销量
0  1964-01  2815.0
1  1964-02  2672.0
2  1964-03  2755.0
3  1964-04  2721.0
4  1964-05  2946.0
```

&emsp;&emsp;After completing the data reading, perform statistical analysis on the data, create a distribution chart, and calculate the extreme values, outliers, and other results for each column of data. The code is as follows:

```python
stats_df = pd.DataFrame(columns=[
    '列名', '数据类型', '最大值', '最小值', '平均值', '非空值数量', '空值数量',
    '众数', 'True数量', 'False数量', '标准差', '方差', '中位数', '峰度', '偏度',
    '极值数量', '异常值数量'
])

def detect_extremes_and_outliers(column, extreme_factor=3, outlier_factor=5):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    iqr = q3 - q1
    
    lower_extreme = q1 - extreme_factor * iqr
    upper_extreme = q3 + extreme_factor * iqr
    lower_outlier = q1 - outlier_factor * iqr
    upper_outlier = q3 + outlier_factor * iqr
    
    extremes = column[(column < lower_extreme) | (column > upper_extreme)]
    outliers = column[(column < lower_outlier) | (column > upper_outlier)]
    
    return len(extremes), len(outliers)

for col in df.columns:
    col_data = df[col]

    dtype = col_data.dtype

    max_value = col_data.max() if np.issubdtype(dtype, np.number) else None
    min_value = col_data.min() if np.issubdtype(dtype, np.number) else None
    mean_value = col_data.mean() if np.issubdtype(dtype, np.number) else None
    non_null_count = col_data.count()
    null_count = col_data.isna().sum()
    mode_value = col_data.mode().iloc[0] if not col_data.mode().empty else None
    true_count = col_data[col_data == True].count() if dtype == 'bool' else None
    false_count = col_data[col_data == False].count() if dtype == 'bool' else None
    std_value = col_data.std() if np.issubdtype(dtype, np.number) else None
    var_value = col_data.var() if np.issubdtype(dtype, np.number) else None
    median_value = col_data.median() if np.issubdtype(dtype, np.number) else None
    kurtosis_value = col_data.kurt() if np.issubdtype(dtype, np.number) else None
    skew_value = col_data.skew() if np.issubdtype(dtype, np.number) else None

    extreme_count, outlier_count = detect_extremes_and_outliers(col_data) if np.issubdtype(dtype, np.number) else (None, None)

    new_row = pd.DataFrame({
        '列名': [col],
        '数据类型': [dtype],
        '最大值': [max_value],
        '最小值': [min_value],
        '平均值': [mean_value],
        '非空值数量': [non_null_count],
        '空值数量': [null_count],
        '众数': [mode_value],
        'True数量': [true_count],
        'False数量': [false_count],
        '标准差': [std_value],
        '方差': [var_value],
        '中位数': [median_value],
        '峰度': [kurtosis_value],
        '偏度': [skew_value],
        '极值数量': [extreme_count],
        '异常值数量': [outlier_count]
    })

    stats_df = pd.concat([stats_df, new_row], ignore_index=True)

print(stats_df)

>>     列名     数据类型      最大值     最小值  ...        峰度        偏度  极值数量 异常值数量
0    月份   object      NaN     NaN  ...       NaN       NaN  None  None
1  香槟销量  float64  13916.0  1413.0  ...  2.702889  1.639003     3     0


rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']

output_dir = r'.\holtwinters'#选择路径

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for col in df.columns:
    plt.figure(figsize=(10, 6))
    df[col].dropna().hist(bins=30)
    plt.title(f"{col} - 数据分布图")
    plt.ylabel("频率")

    file_name = f"{col}_数据分布图.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7dbfb3790b834694842cab789b16653b.png#pic_center)
2、Implementation of Sentosa_DSML Community Edition 

&emsp;&emsp;Firstly, perform data reading by using text operators to directly read the data and select the path where the data is located,
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/48ef9f2879384a9397da144d967f4068.png#pic_center)
&emsp;&emsp;At the same time, column names can be modified or deleted in the deletion and renaming configuration of the text operator. Here, the columns will be changed to 'month' and 'champagne sales volume' respectively.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f52d87d9f15e446d99f1584a76e5d005.jpeg#pic_center)
&emsp;&emsp;Click on the application and right-click to preview to view the data.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71eebacc815045eba19c3cd7af2532e2.png)
&emsp;&emsp;Next, the descriptive operator can be used to perform statistical analysis on the data, obtaining the data distribution map, extreme values, outliers, and other results for each column of data. Connect the description operator and set the extremum multiplier to 3 and the outlier multiplier to 5 on the right side.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/96c5d6557f96448196231abffa40a0ec.jpeg#pic_center)
&emsp;&emsp;Right click to execute, the result is as follows:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/af6858694155409f82116dbcec067741.jpeg#pic_center)
## (2) Data preprocessing
1、Implementation in Python code

```python
#数据预处理
for col in df.columns:
    print(f"列名: {col}, 数据类型: {df[col].dtype}")

>>列名: 月份, 数据类型: object
列名: 香槟销量, 数据类型: float64

df = df.dropna()
df['月份'] = pd.to_datetime(df['月份'], format='%Y-%m', errors='coerce')  

df['香槟销量'] = pd.to_numeric(df['香槟销量'], errors='coerce') 
df = df.dropna(subset=['香槟销量'])
df['香槟销量'] = df['香槟销量'].astype(int)

for col in df.columns:
    print(f"列名: {col}, 数据类型: {df[col].dtype}")
    
print(df)
>>列名: 月份, 数据类型: datetime64[ns]
列名: 香槟销量, 数据类型: int32


filtered_df1 = df[df['月份'] <= '1971-09']
print(filtered_df1)
>>            月份  香槟销量
0   1964-01-01  2815
1   1964-02-01  2672
2   1964-03-01  2755
3   1964-04-01  2721
4   1964-05-01  2946

filtered_df2 = df[df['月份'] > '1971-09']
print(filtered_df2)

>>    月份   香槟销量
93  1971-10-01   6981
94  1971-11-01   9851
95  1971-12-01  12670
96  1972-01-01   4348
97  1972-02-01   3564

filtered_df1.set_index('月份', inplace=True)
resampled_df1 = filtered_df1['香槟销量'].resample('MS').bfill()

print(resampled_df1)

>>     月份   香槟销量
1964-01-01    2815
1964-02-01    2672
1964-03-01    2755
1964-04-01    2721
1964-05-01    2946
              ... 
1971-05-01    5010
1971-06-01    4874
1971-07-01    4633
1971-08-01    1659
1971-09-01    5951
```
2、Implementation of Sentosa_DSML Community Edition

&emsp;&emsp;Firstly, the connection format operator modifies the format of the data by changing the month data format from String type to Data type.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a3dff209fb5845f1980d3627d91260ec.jpeg#pic_center)
&emsp;&emsp;Secondly, filter the data and use data less than or equal to 1971-09 as the training and validation datasets, with the condition that data greater than 1971-09 is used for comparison with time-series prediction data. Two filtering operators can be used to implement it, and the attribute "expression" in the table on the right side of the operator is a Spark SQL expression.
&emsp;&emsp;The first filtering operator, with the condition of 'month'<='1971-09',
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d532ac68d94425380ff9242ab8c81f1.jpeg#pic_center)
&emsp;&emsp;The second filtering operator condition is' month '>' 1971-09 ', right-click preview to view the filtered data.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cfe80c630614476ca4ce961254eacbed.jpeg#pic_center)
&emsp;&emsp;Connect the time-series data cleaning operator, preprocess the data used for model training, set the time column as month (the time column must be of data/DataTime type), select the sampling frequency so that the time column data interval is 1 month, and fill the champagne sales column with data in a linear manner.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/63da8f55ff874fd3be22ec43bac0eb7e.jpeg#pic_center)
## (3) Model training and model evaluation
1、Implementation in Python code

```python
#模型定义
model = ExponentialSmoothing(
    resampled_df1, trend='add', seasonal='mul', seasonal_periods=12,damped_trend=True)
fit = model.fit(damping_slope=0.05)

#预测
forecast = fit.predict(
    start=len(resampled_df1), end=len(resampled_df1) + 11
)

residuals = resampled_df1 - fit.fittedvalues
residual_std = np.std(residuals)
upper_bound = forecast + 1.96 * residual_std
lower_bound = forecast - 1.96 * residual_std

results_df = pd.DataFrame({
    '预测值': forecast,
    '上限': upper_bound,
    '下限': lower_bound
})
print(results_df)
>> 月份            预测值            上限            下限
1971-10-01   7143.862498   8341.179324   5946.545672
1971-11-01  10834.141889  12031.458716   9636.825063
1971-12-01  13831.428845  15028.745671  12634.112019
1972-01-01   4054.821228   5252.138054   2857.504402
1972-02-01   3673.653407   4870.970233   2476.336580

#模型评估
y_true = resampled_df1.values
y_pred = fit.fittedvalues.values

def evaluate_model(y_true, y_pred, model_name="Holt-Winters"):
    r_squared = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    print(f"模型评估结果 ({model_name}):")
    print(f"{'-' * 40}")
    print(f"R² (决定系数): {r_squared:.4f}")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"MSE (均方误差): {mse:.4f}")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"{'-' * 40}\n")

    return {
        "R²": r_squared,
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse
    }

evaluation_results = evaluate_model(y_true, y_pred, model_name="Holt-Winters")

>>模型评估结果 (Holt-Winters):
----------------------------------------
R² (决定系数): 0.9342
MAE (平均绝对误差): 451.4248
MSE (均方误差): 402168.8567
RMSE (均方根误差): 634.1678
```
2、Implementation of Sentosa_DSML Community Edition
&emsp;&emsp;After the time series data cleaning operator, connect the HoltWinters operator, which predicts future time data based on the data corresponding to the existing time series. The input data of the operator supports multiple key keys, but it must be time-series data with fixed time column intervals and non empty numerical columns under the same key key. It is recommended to clean the data processed by the operator for time-series data.
&emsp;&emsp;Here, the time column is set as the month column, the data column is set as the champagne sales column, the predicted quantity and periodicity parameters are set to 12, the analysis frequency is month, the model type is Multivariate, and the significance level alpha is set to 0.05.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/07f49ff2da6c4152b723dfb7693c38ab.jpeg#pic_center)
&emsp;&emsp;Model connection time series model evaluation operator, right-click to execute, you can view the evaluation results.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4e49dcc60da84246bed4b58e757e59cb.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7381dc64820e40af9a6ae1cc98f0db9b.jpeg#pic_center)
## (4) Model visualization
1、Implementation in Python code

```python
#可视化
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei'] 

plt.figure(figsize=(12, 6))
plt.plot(resampled_df1, label='实际销量', color='blue')
plt.plot(fit.fittedvalues, label='拟合值', color='orange')
plt.plot(forecast, label='预测销量', color='green')
plt.title('Holt-Winters 方法预测香槟销量')
plt.xlabel('时间')
plt.ylabel('香槟销量')
plt.axvline(x=resampled_df1.index[-1], color='red', linestyle='--', label='预测起始点')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(resampled_df1.index, resampled_df1, label='实际值', color='blue')
plt.plot(results_df.index, results_df['预测值'], label='预测值', color='orange')
plt.fill_between(results_df.index, results_df['下限'], results_df['上限'], color='lightgray', alpha=0.5, label='95% 置信区间')
plt.title('Holt-Winters 预测与置信区间')
plt.xlabel('时间')
plt.ylabel('香槟销量')
plt.legend()
plt.show()

filtered_forecast_df = results_df[results_df.index > pd.Timestamp('1971-09-01')]
print(filtered_forecast_df)
>> 月份        预测值            上限            下限
1971-10-01   7143.862498   8341.179324   5946.545672
1971-11-01  10834.141889  12031.458716   9636.825063
1971-12-01  13831.428845  15028.745671  12634.112019
1972-01-01   4054.821228   5252.138054   2857.504402
1972-02-01   3673.653407   4870.970233   2476.336580


results_df = results_df.drop(columns=['上限', '下限'])
print(results_df)
>> 月份         预测值
1971-10-01   7143.862498
1971-11-01  10834.141889
1971-12-01  13831.428845
1972-01-01   4054.821228
1972-02-01   3673.653407
1972-03-01   4531.419772
1972-04-01   4821.096141

results_df.index.name = '月份'
merged_df = pd.merge(filtered_df2, results_df, left_on='月份', right_index=True, how='left')

print(merged_df)
>>         月份   香槟销量           预测值
93  1971-10-01   6981   7143.862498
94  1971-11-01   9851  10834.141889
95  1971-12-01  12670  13831.428845
96  1972-01-01   4348   4054.821228
97  1972-02-01   3564   3673.653407


scaler = StandardScaler()
merged_df[['香槟销量', '预测值']] = scaler.fit_transform(merged_df[['香槟销量', '预测值']])

plt.figure(figsize=(12, 6))
plt.plot(merged_df['月份'], merged_df['香槟销量'], label='香槟销量', color='blue')
plt.plot(merged_df['月份'], merged_df['预测值'], label='香槟预测销量', color='orange')
plt.title('时序图')
plt.xlabel('时间')
plt.ylabel('香槟销量')
plt.legend()
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/708c4c76ce3a4d2dab15046819d1d5cd.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/508eb4e2576f4f9c84d025bf1a1b706b.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/285f05fbebfa43af9619dfe5235eb2ef.png)

2、Implementation of Sentosa_DSML Community Edition

&emsp;&emsp;In order to compare the original data with the predicted data, first, the HoltWinters model prediction data is filtered using a filtering operator, with the filtering condition being 'month'>'1971-09'.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ab43aaf3489d4998b3761a4174f62df6.jpeg#pic_center)
&emsp;&emsp;Right click preview to view data filtering results.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3ac94d06eaeb4577800d0e8216363be1.jpeg#pic_center)
&emsp;&emsp;Secondly, the connection deletion and renaming operators retain the required time column and prediction result column, while deleting the remaining columns.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7bf593cae7b748ee9cfc827beb8d24e3.jpeg#pic_center)
&emsp;&emsp;After the application is completed, right-click to view the processing results.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4b757129d981476e9997561b69accfc5.jpeg#pic_center)
&emsp;&emsp;Then, the merge operator is used to merge the original data and predicted data, which can be divided into two types: keyword merge and sequential merge. Here, keyword merge is used, and the keyword used for merging is the month column. The merge method is left join.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b91757f505e34dc9a64d151122635414.jpeg#pic_center)
&emsp;&emsp;Right click preview to get the processing result of the merge operator.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/713263ee3268471e91ef9ecb42623285.jpeg#pic_center)
&emsp;&emsp;Connect the sequence diagram operator in the chart analysis again. The "sequence" can select multiple columns. When the sequence is multiple columns, it is necessary to configure "whether each sequence should be displayed separately",
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a33bde0046ed41e59a2d399eb6eddb17.png#pic_center)
&emsp;&emsp;Right clicking on the execution will result in a visual result, while downloading and other operations can be performed in the upper right corner. Moving the mouse can view the data information at the current location, and sliding can adjust the time series interval of the data below.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/872d9969e732432a9086b577586bc400.jpeg#pic_center)
&emsp;&emsp;For the prediction results of the HoltWinters model, directly connect the time series operator for chart analysis, and use a sequence mode to compare the actual and predicted champagne sales values.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/13b09e4337dd4573b3edf104e322288c.png#pic_center)
&emsp;&emsp;Right click to execute the result as shown below:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d5a3308332c54294b8abb6e889c7638b.jpeg#pic_center)
&emsp;&emsp;Use a time series model to analyze the prediction results of the HoltWinters model using a chart, with attribute settings as shown on the right.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3c90daa9c5dd497bb30092470d0dc09a.png#pic_center)
&emsp;&emsp;Right click to execute the result, where the solid point data represents the original true value, the solid line represents the fitted data to the original data, the hollow dashed line represents the predicted data, and the upper and lower dashed lines of the shaded boundary represent the predicted upper and lower limits of the confidence interval, respectively.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4894951d8a643e3b64085dbcf469c94.jpeg#pic_center)
# 4、summarize

&emsp;& emsp; Compared to traditional coding methods, using Sentosa_SSML Community Edition to complete the process of machine learning algorithms is more efficient and automated. Traditional methods require manually writing a large amount of code to handle data cleaning, feature engineering, model training, and evaluation. In Sentosa_SSML Community Edition, these steps can be simplified through visual interfaces, pre built modules, and automated processes, effectively reducing technical barriers. Non professional developers can also develop applications through drag and drop and configuration, reducing dependence on professional developers.
&emsp;& emsp; Sentosa_SSML Community Edition provides an easy to configure operator flow, reducing the time spent writing and debugging code, and improving the efficiency of model development and deployment. Due to the clearer structure of the application, maintenance and updates become easier, and the platform typically provides version control and update features, making continuous improvement of the application more convenient.

Sentosa Data Science and Machine Learning Platform (Sentosa_DSML) is a one-stop AI development, deployment, and application platform with full intellectual property rights owned by Liwei Intelligent Connectivity. It supports both no-code "drag-and-drop" and notebook interactive development, aiming to assist customers in developing, evaluating, and deploying AI algorithm models through low-code methods. Combined with a comprehensive data asset management model and ready-to-use deployment support, it empowers enterprises, cities, universities, research institutes, and other client groups to achieve AI inclusivity and simplify complexity.

The Sentosa_DSML product consists of one main platform and three functional platforms: the Data Cube Platform (Sentosa_DC) as the main management platform, and the three functional platforms including the Machine Learning Platform (Sentosa_ML), Deep Learning Platform (Sentosa_DL), and Knowledge Graph Platform (Sentosa_KG). With this product, Liwei Intelligent Connectivity has been selected as one of the "First Batch of National 5A-Grade Artificial Intelligence Enterprises" and has led important topics in the Ministry of Science and Technology's 2030 AI Project, while serving multiple "Double First-Class" universities and research institutes in China.

To give back to society and promote the realization of AI inclusivity for all, we are committed to lowering the barriers to AI practice and making the benefits of AI accessible to everyone to create a smarter future together. To provide learning, exchange, and practical application opportunities in machine learning technology for teachers, students, scholars, researchers, and developers, we have launched a lightweight and completely free Sentosa_DSML Community Edition software. This software includes most of the functions of the Machine Learning Platform (Sentosa_ML) within the Sentosa Data Science and Machine Learning Platform (Sentosa_DSML). It features one-click lightweight installation, permanent free use, video tutorial services, and community forum exchanges. It also supports "drag-and-drop" development, aiming to help customers solve practical pain points in learning, production, and life through a no-code approach.

This software is an AI-based data analysis tool that possesses capabilities such as mathematical statistics and analysis, data processing and cleaning, machine learning modeling and prediction, as well as visual chart drawing. It empowers various industries in their digital transformation and boasts a wide range of applications, with examples including the following fields:
1.Finance: It facilitates credit scoring, fraud detection, risk assessment, and market trend prediction, enabling financial institutions to make more informed decisions and enhance their risk management capabilities.
2.Healthcare: In the medical field, it aids in disease diagnosis, patient prognosis, and personalized treatment recommendations by analyzing patient data.
3.Retail: By analyzing consumer behavior and purchase history, the tool helps retailers understand customer preferences, optimize inventory management, and personalize marketing strategies.
4.Manufacturing: It enhances production efficiency and quality control by predicting maintenance needs, optimizing production processes, and detecting potential faults in real-time.
5.Transportation: The tool can optimize traffic flow, predict traffic congestion, and improve transportation safety by analyzing transportation data.
6.Telecommunications: In the telecommunications industry, it aids in network optimization, customer behavior analysis, and fraud detection to enhance service quality and user experience.
7.Energy: By analyzing energy consumption patterns, the software helps utilities optimize energy distribution, reduce waste, and improve sustainability.
8.Education: It supports personalized learning by analyzing student performance data, identifying learning gaps, and recommending tailored learning resources.
9.Agriculture: The tool can monitor crop growth, predict harvest yields, and detect pests and diseases, enabling farmers to make more informed decisions and improve crop productivity.
10.Government and Public Services: It aids in policy formulation, resource allocation, and crisis management by analyzing public data and predicting social trends.

Welcome to the official website of the Sentosa_DSML Community Edition at https://sentosa.znv.com/. Download and experience it for free. Additionally, we have technical discussion blogs and application case shares on platforms such as Bilibili, CSDN, Zhihu, and cnBlog. Data analysis enthusiasts are welcome to join us for discussions and exchanges.

Sentosa_DSML Community Edition: Reinventing the New Era of Data Analysis. Unlock the deep value of data with a simple touch through visual drag-and-drop features. Elevate data mining and analysis to the realm of art, unleash your thinking potential, and focus on insights for the future.

Official Download Site: https://sentosa.znv.com/
Official Community Forum: http://sentosaml.znv.com/
GitHub:https://github.com/Kennethyen/Sentosa_DSML
Bilibili: https://space.bilibili.com/3546633820179281
CSDN: https://blog.csdn.net/qq_45586013?spm=1000.2115.3001.5343
Zhihu: https://www.zhihu.com/people/kennethfeng-che/posts
CNBlog: https://www.cnblogs.com/KennethYuen

