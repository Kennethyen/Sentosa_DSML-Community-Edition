
@[toc]
# 一、XGBoost Algorithm
&emsp;&emsp;This article aims to predict the transaction price of used cars, and constructs XGBoost regression prediction models through Python code and Sentosa_SSML community version, respectively. The models are evaluated, including the selection and analysis of evaluation indicators. Finally, the experimental conclusion is drawn to ensure the effectiveness and accuracy of the model in the regression prediction of second-hand car prices.

**Dataset Introduction**
&emsp;&emsp;The task is to predict the transaction price of second-hand cars. The data comes from the second-hand car transaction records of a certain trading platform, with a total data volume of over 40000, including 31 columns of variable information, of which 15 columns are anonymous variables. Overview of the dataset:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b3c971c2bf284eaea3ff098d32ca2766.png)

# 二、Comparison of algorithm implementation between Python code and Sentosa_DSML community edition
## (1) Data reading and statistical analysis
1、Implementation in Python code

```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import rcParams
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```
&emsp;&emsp;Data reading

```python
file_path = r'.\二手汽车价格.csv'
output_dir = r'.\xgb'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件未找到: {file_path}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
df = pd.read_csv(file_path)

print(df.isnull().sum())
print(df.head())
>>   SaleID    name   regDate  model  ...      v_11      v_12      v_13      v_14
0       0     736  20040402   30.0  ...  2.804097 -2.420821  0.795292  0.914763
1       1    2262  20030301   40.0  ...  2.096338 -1.030483 -1.722674  0.245522
2       2   14874  20040403  115.0  ...  1.803559  1.565330 -0.832687 -0.229963
3       3   71865  19960908  109.0  ...  1.285940 -0.501868 -2.438353 -0.478699
4       4  111080  20120103  110.0  ...  0.910783  0.931110  2.834518  1.923482
```
&emsp;&emsp;statistical analysis

```python
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']

stats_df = pd.DataFrame(columns=[
    '列名', '数据类型', '最大值', '最小值', '平均值', '非空值数量', '空值数量',
    '众数', 'True数量', 'False数量', '标准差', '方差', '中位数', '峰度', '偏度',
    '极值数量', '异常值数量'
])

def detect_extremes_and_outliers(column, extreme_factor=3, outlier_factor=5):
    if not np.issubdtype(column.dtype, np.number):
        return None, None
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
    if np.issubdtype(dtype, np.number):
        max_value = col_data.max()
        min_value = col_data.min()
        mean_value = col_data.mean()
        std_value = col_data.std()
        var_value = col_data.var()
        median_value = col_data.median()
        kurtosis_value = col_data.kurt()
        skew_value = col_data.skew()
        extreme_count, outlier_count = detect_extremes_and_outliers(col_data)
    else:
        max_value = min_value = mean_value = std_value = var_value = median_value = kurtosis_value = skew_value = None
        extreme_count = outlier_count = None

    non_null_count = col_data.count()
    null_count = col_data.isna().sum()
    mode_value = col_data.mode().iloc[0] if not col_data.mode().empty else None
    true_count = col_data[col_data == True].count() if dtype == 'bool' else None
    false_count = col_data[col_data == False].count() if dtype == 'bool' else None

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

def save_plot(filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f"{filename}_{timestamp}.png")
    plt.savefig(file_path)
    plt.close()

for col in df.columns:
    plt.figure(figsize=(10, 6))
    df[col].dropna().hist(bins=30)
    plt.title(f"{col} - 数据分布图")
    plt.ylabel("频率")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"{col}_数据分布图_{timestamp}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()

if 'car_brand' in df.columns and 'price' in df.columns:
    grouped_data = df.groupby('car_brand')['price'].count()
    plt.figure(figsize=(8, 8))
    plt.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title("品牌和价格分布饼状图", fontsize=16)
    plt.axis('equal')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"car_brand_price_distribution_{timestamp}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4d6bde2810d84979aee7c232ca50fd6f.jpeg#pic_center.png =500x300)
2、Implementation of Sentosa_DSML Community Edition 
&emsp;&emsp;Firstly, perform data reading by using text operators to directly read the data, and configure the reading attributes on the right side
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b56080930e014261aa20b8d84cca3621.png#pic_center)
&emsp;&emsp;Next, the descriptive operator can be used to perform statistical analysis on the data, obtaining the data distribution map, extreme values, outliers, and other results for each column of data. Connect the description operator and set the extremum multiplier to 3 and the outlier multiplier to 5 on the right side.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a058fdd311c540639090122aff8163e3.png#pic_center)
&emsp;&emsp;Right click to execute and obtain the data statistical analysis results. You can calculate and display the data distribution map, storage type, maximum value, minimum value, average value, non null number, null number, mode, median, extreme value, and outlier number for each column of the data. The results are as follows:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bdddf7f8687f4abead3c2a8a33207573.jpeg#pic_center)
&emsp;&emsp;Describing the results of operator execution helps us understand the data and conduct subsequent analysis.
## (二) 数据处理

1、Implementation in Python code
&emsp;&emsp;Perform data processing operations
```python
def handle_power(power, threshold=600, fill_value=600):
    return fill_value if power > threshold else power

def handle_not_repaired_damage(damage, missing_value='-', fill_value=0.0):
    return fill_value if damage == missing_value else damage

def extract_date_parts(date, part):
    if part == 'year':
        return str(date)[:4]
    elif part == 'month':
        return str(date)[4:6]
    elif part == 'day':
        return str(date)[6:8]

def fix_invalid_month(month, invalid_value='00', default='01'):
    return default if month == invalid_value else month

columns_to_fill_with_mode = ['model', 'bodyType', 'fuelType', 'gearbox']

for col in columns_to_fill_with_mode:
    mode_value = df[col].mode().iloc[0]
    df[col].fillna(mode_value, inplace=True)

df = (
    df.fillna({
        'model': df['model'].mode()[0],
        'bodyType': df['bodyType'].mode()[0],
        'fuelType': df['fuelType'].mode()[0],
        'gearbox': df['gearbox'].mode()[0]
    })
    .assign(power=lambda x: x['power'].apply(handle_power).fillna(600))
    .assign(notRepairedDamage=lambda x: x['notRepairedDamage'].apply(handle_not_repaired_damage).astype(float))
    .assign(
        regDate_year=lambda x: x['regDate'].apply(lambda y: str(extract_date_parts(y, 'year'))),
        regDate_month=lambda x: x['regDate'].apply(lambda y: str(extract_date_parts(y, 'month'))).apply(
            fix_invalid_month),
        regDate_day=lambda x: x['regDate'].apply(lambda y: str(extract_date_parts(y, 'day')))
    )

    .assign(
        regDate=lambda x: pd.to_datetime(x['regDate_year'] + x['regDate_month'] + x['regDate_day'],
                                         format='%Y%m%d', errors='coerce'),
        creatDate=lambda x: pd.to_datetime(x['creatDate'].astype(str), format='%Y%m%d', errors='coerce')
    )
    .assign(
        car_day=lambda x: (x['creatDate'] - x['regDate']).dt.days,
        car_year=lambda x: (x['car_day'] / 365).round(2) 
    )
    .assign(log1p_price=lambda x: np.log1p(x['price']))
)
print(df.head())
>>   SaleID    name    regDate  model  ...  regDate_day  car_day  car_year  log1p_price
0       0     736 2004-04-02   30.0  ...           02     4385     12.01     7.523481
1       1    2262 2003-03-01   40.0  ...           01     4757     13.03     8.188967
2       2   14874 2004-04-03  115.0  ...           03     4382     12.01     8.736007
3       3   71865 1996-09-08  109.0  ...           08     7125     19.52     7.783641
4       4  111080 2012-01-03  110.0  ...           03     1531      4.19     8.556606

print(df.dtypes)
>>SaleID                        int64
name                          int64
regDate              datetime64[ns]
model                       float64
brand                         int64
bodyType                    float64
fuelType                    float64
gearbox                     float64
power                         int64
kilometer                   float64
notRepairedDamage           float64
regionCode                    int64
seller                        int64
offerType                     int64
creatDate            datetime64[ns]
price                         int64
v_0                         float64
v_1                         float64
...
```

2、Implementation of Sentosa_DSML Community Edition
&emsp;&emsp;By describing the execution results of the operator, it can be observed that columns "model", "bodyType", "fuelType", "gearbox", "power", and "notRepairedDamage" need to be processed for missing and outlier values. Firstly, connect the outlier missing value filling operator, click on the configuration column selection, and select the column that needs to be processed for outlier missing values.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f617987e30324a0989788d56b19006e1.png#pic_center)
&emsp;&emsp;Then, select the filling method for missing values in the configuration column for outliers. Select 'model', 'bodyType', 'fuelType', 'gearbox' columns to retain outliers and fill in missing values with mode,
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9cc89bc5996e4eb89af0fd1bc7f5237b.jpeg#pic_center)
&emsp;&emsp;Select the input rule for handling outliers in the 'power' column, specify the detection rule for outliers as' power>600 ', choose to fill in missing values using the missing value method, and use a fixed value of 600 to fill in missing values.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be8a829af8764c748136408194106495.jpeg#pic_center)
&emsp;&emsp;Select the input rule for handling outliers in the "notRepairedDamage" column, specify the detection rule for outliers as' notRepairedDamage '==' ', choose to fill in missing values using the missing value method, and use a fixed value of 0.0 to fill in missing values.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3a28f5b32ef64161b4f58a05075615c7.jpeg#pic_center)
&emsp;&emsp;Then, use the generator column operator to extract year, month, and day information separately, and generate corresponding columns. The expressions for generating year, month, and day columns are: substr (` regDate `, 1,4), substr (` regDate `, 5,2), and substr (` regDate `, 7,2), respectively.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/50f402b9459d4464adbe432948a56f26.png#pic_center)
&emsp;&emsp;The result of generating column operators after processing is as follows:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1be060b0a9f74e86ac92cf274f25b59a.jpeg#pic_center)
&emsp;&emsp;To handle invalid or missing month information, use the padding operator to process the month column 'regDate_ month', with the padding condition 'regDate_ month==00', and fill the 'regDate' column with the padding value '01'.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/06b43df282564f7f8b48295a62a49837.png#pic_center)
&emsp;&emsp;For valid data in the regDate column, use concat (regDate_ year, regDate_ month, regDate_ day) to fill the regDate column. By filling in a new regDate from the valid year, month, and day columns, it is possible to fix incomplete or abnormal situations in some parts of the original data.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/055400ad68304a6488970da0237a7e0f.png#pic_center)
&emsp;&emsp;Modify the columns' regDate 'and' creatData 'to be of type String (Intege cannot be directly modified to Date type) and the column' notRepairedDamage 'to be of type Double using formatting operators.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/870bbc22d34b44e9aa85c1aa3baa5f1f.png#pic_center)
&emsp;&emsp;Change the 'regDate' and 'creatData' columns to Date type (format: yyyyMMdd).
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2e9679eeba6844659014a670035a1e15.png#pic_center)
&emsp;& emsp; After the format modification is completed, use the generate column operator.
&emsp;& emsp; 1. Generate a 'car day' column with the expression DATEDIFF (` creatDate `, ` regDate `), and calculate the date difference between the car registration time (regDate) and the online time (creatDate).
&emsp;& emsp; 2. Generate a 'car year' list showing the number of years the car has been in use. The expression is DATEDIFF (` creatDate `, ` regDate `)/365, which calculates the number of years used.
&emsp;& emsp; 3. Generate the 'log1p_price' column with the expression log1p (` price `), and avoid errors when the price is 0 by calculating the natural logarithm of the price.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/99361abc49d74651980f2bb30b537831.png#pic_center)
&emsp;& emsp; The generated column execution result is shown below:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/49e45ca32c0f43c0823b7c07528a13e1.jpeg#pic_center)
&emsp;&emsp; These steps lay a solid data foundation for subsequent modeling and analysis, reducing the impact of data anomalies and enhancing the model's ability to understand data.
## (三) Feature selection and correlation analysis

1、Implementation in Python code
&emsp;&emsp;Histogram and Pearson correlation coefficient calculation
```python
def plot_log1p_price_distribution(df, column='log1p_price', bins=20,output_dir=None):
    """
    绘制指定列的分布直方图及正态分布曲线
    参数:
    df: pd.DataFrame - 输入数据框
    column: str - 要绘制的列名
    bins: int - 直方图的组数
    output_dir: str - 保存图片的路径
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=True, stat='density',
                   color='orange', edgecolor='black', alpha=0.6)

    mean = df[column].mean()
    std_dev = df[column].std()

    x = np.linspace(df[column].min(), df[column].max(), 100)
    p = np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))

    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Curve')
    plt.title('Distribution of log1p_price with Normal Distribution Curve', fontsize=16)
    plt.xlabel('log1p_price', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    plt.tight_layout()
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, 'log1p_price_distribution.png')
        plt.savefig(save_path, dpi=300)
    plt.show()

plot_log1p_price_distribution(df,output_dir=output_dir)

numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
log1p_price_corr = correlation_matrix['log1p_price'].drop('log1p_price')

print(log1p_price_corr)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/08b08f4fcfbe436ea795595cfe8a4562.png#pic_center.png =500x300)
&emsp;&emsp;删除'SaleID', 'name', 'regDate', 'model', 'brand', 'regionCode', 'seller', 'offerType', 'creatDate', 'price', 'v_4', 'v_7', 'v_13', 'regDate_year', 'regDate_month', 'regDate_day'列并进行流式归一化：

```python
columns_to_drop = ['SaleID', 'name', 'regDate', 'model', 'brand', 'regionCode', 'seller', 'offerType',
                   'creatDate', 'price', 'v_4', 'v_7', 'v_13', 'regDate_year', 'regDate_month', 'regDate_day']

df = df.drop(columns=columns_to_drop)
print(df.head())
>>   bodyType  fuelType  gearbox  power  ...      v_14  car_day  car_year  log1p_price
0       1.0       0.0      0.0     60  ...  0.914763     4385     12.01     7.523481
1       2.0       0.0      0.0      0  ...  0.245522     4757     13.03     8.188967
2       1.0       0.0      0.0    163  ... -0.229963     4382     12.01     8.736007
3       0.0       0.0      1.0    193  ... -0.478699     7125     19.52     7.783641
4       1.0       0.0      0.0     68  ...  1.923482     1531      4.19     8.556606

columns_to_normalize = df.columns.drop('log1p_price')
scaler = MaxAbsScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
df = df.round(3)
print(df.head())
>>   bodyType  fuelType  gearbox  power  ...   v_14  car_day  car_year  log1p_price
0     0.143       0.0      0.0  0.100  ...  0.106    0.475     0.475        7.523
1     0.286       0.0      0.0  0.000  ...  0.028    0.516     0.516        8.189
2     0.143       0.0      0.0  0.272  ... -0.027    0.475     0.475        8.736
3     0.000       0.0      1.0  0.322  ... -0.055    0.772     0.772        7.784
4     0.143       0.0      0.0  0.113  ...  0.222    0.166     0.166        8.557
```

2、Implementation of Sentosa_DSML Community Edition
&emsp;&emsp;Select the histogram chart analysis operator, set the log1p_price column as the statistical column, select a grouping method of 20 groups, and then enable the option to display normal distribution, which is used to show the distribution of the values in the log1p_price column in different intervals.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/14b82e014e004a1d87e5f00e6651697d.png#pic_center)
&emsp;&emsp;The histogram results obtained are as follows:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/623c10a4c0924654a82c8be5ba32b3c2.jpeg#pic_center)
&emsp;&emsp;Connect the Pearson correlation coefficient operator and calculate the correlation between each column. Set the column on the right that needs to calculate the Pearson correlation coefficient to analyze the relationship between features and provide a basis for data modeling and feature selection.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9768bef459c34a63a37e5f9af482c38c.png#pic_center)
&emsp;&emsp;The calculated results are as follows:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/372c654fd99d4a0283b0d337d56fc443.png)
&emsp;&emsp;Through further processing, redundant features are removed, and the deletion and renaming operators are connected. Columns such as' SalelD ',' name ',' regDate ',' model ',' brand ',' regionCode ',' sellar ',' offerType ',' creatDate ',' price ',' vv-4 ',' vv-7 ',' vv13 ',' regDate_ year ',' regDate_ month ', and' regDate_ day 'are deleted.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/16d6d5eaa44e432d95488a3b2c62abbf.png#pic_center)
&emsp;&emsp;Connect the flow normalization operator for feature processing, select the normalization column on the right, and choose absolute value normalization as the algorithm type.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/966573d43abd4d6f9f3f2eb47717bb4c.png#pic_center)
&emsp;&emsp;You can right-click on the operator to preview the result of feature processing.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/49ce09c93f1d4e7a80ffd8cfcc8abe7e.jpeg#pic_center)
## (4) Sample partitioning and model training

1、Implementation in Python code
```python
X = df.drop(columns=['log1p_price'])
y = df['log1p_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=100,             
    learning_rate=1,               
    max_depth=6,                   
    min_child_weight=1,            
    subsample=1,                   
    colsample_bytree=0.8,          
    objective='reg:squarederror',  
    eval_metric='rmse',            
    reg_alpha=0,                  
    reg_lambda=1,                  
    scale_pos_weight=1,                    
)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
```

2、Implementation of Sentosa_DSML Community Edition

&emsp;&emsp;Connect the sample partitioning operator to divide the data processed and feature engineered into training and testing sets for subsequent model training and validation processes. The ratio of training and testing samples is 8:2.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/148fc88452a64f9aad53520ecff3be7d.png#pic_center)
&emsp;&emsp;Then, connect the type operator to display the storage type, measurement type, and model type of the data, and set the model type of the 'log1p_price' column to Label.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7b69fc7410bf44299be8d906d21a298c.png#pic_center)
&emsp;&emsp;After the sample partitioning is completed, connect the XGBoost regression operator and configure the operator properties on the right side. The evaluation metric is the loss function of the algorithm, which includes root mean square error, root mean square logarithmic error, mean absolute error, and gamma regression deviation; Learning rate, maximum depth of the tree, minimum leaf node sample weight sum, subsampling rate, minimum splitting loss, proportion of randomly sampled columns per tree, L1 regularization term and L2 regularization term are all used to prevent algorithm overfitting. When the sum of sub node sample weights is not greater than the set minimum sum of leaf node sample weights, no further partitioning is performed on the node. The three parameters of adding node method, maximum number of boxes, and single precision are only effective when the tree construction method is set to 'hist'. The minimum splitting loss specifies the minimum decrease in the loss function required for node splitting.
&emsp;&emsp;In this case, the attribute configuration in the regression model is as follows: iteration count: 100, learning rate: 1, minimum splitting loss: 0, maximum depth of number: 6, minimum leaf node sample weight sum: 1, subsampling rate: 1, proportion of randomly sampled columns per tree: 0.8, tree construction algorithm: auto, weight adjustment for imbalanced positive and negative samples: 1, L2 regularization term: 1, L1 regularization term: 0, learning objective is reg: squareror, evaluation metric is root mean square error, initial prediction score is 0.5, and feature importance and residual histogram are calculated.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/868c678b1c154b39aa1109737ee300cd.png#pic_center)
&emsp;&emsp;Right click to execute the XGBoost regression model.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/732653551fda43c8a365630cb3e2f023.jpeg#pic_center)
## (5) Model evaluation and model visualization
1、Implementation in Python code

```python
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    mse = mean_squared_error(y_true, y_pred)

    return {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'SMAPE': smape,
        'MSE': mse
    }

train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

print("训练集评估结果:")
print(train_metrics)
>>训练集评估结果:
{'R2': 0.9762793552927467, 'MAE': 0.12232836257076264, 'RMSE': 0.18761878906931295, 'MAPE': 1.5998275558939563, 'SMAPE': 1.5950003598874698, 'MSE': 0.035200810011835344}

print("\n测试集评估结果:")
print(test_metrics)
>>测试集评估结果:
{'R2': 0.9465739577985525, 'MAE': 0.16364796002127327, 'RMSE': 0.2815951292200689, 'MAPE': 2.176241755969303, 'SMAPE': 2.1652435034262068, 'MSE': 0.0792958168004673}
```
&emsp;&emsp;Draw important feature rows and residual histograms
```python
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='weight', max_num_features=10, color='orange')  
plt.title('特征重要性图', fontsize=16)
plt.xlabel('重要性', fontsize=14)
plt.ylabel('特征', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(train_residuals, bins=30, kde=True, color='blue') 
plt.title('残差分布', fontsize=16)
plt.xlabel('残差', fontsize=14)
plt.ylabel('技术', fontsize=14)
plt.axvline(x=0, color='red', linestyle='--')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/231c3b2651ab4bcb81618cefe923a3e8.jpeg#pic_center.png =400x300)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4dd45f343a9d47ccb15dfd0e87e7755b.jpeg#pic_center.png =500x300)

```python
test_data = pd.DataFrame(X_test)
test_data['log1p_price'] = y_test
test_data['predicted_log1p_price'] = y_test_pred

test_data_subset = test_data.head(200)

original_values = y_test[:200]
predicted_values = y_test_pred[:200]

x_axis = range(1, 201)

plt.figure(figsize=(12, 6))
plt.plot(x_axis, original_values.values, label='Original Values', color='orange')
plt.plot(x_axis, predicted_values, label='Predicted Values', color='green')

plt.title('Comparison of Original and Predicted Values')
plt.ylabel('log1p_price')
plt.legend()
plt.grid()

plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/464bc17cf9384e3db7426cb0701591f4.jpeg#pic_center.png =600x300)

2、Implementation of Sentosa_DSML Community Edition

&emsp;&emsp;The connection evaluation operator evaluates the model.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/41574353c9cb44ee8cf8c3cb7f6f64e1.png#pic_center)
&emsp;&emsp;The evaluation results of the training and testing sets are as follows:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c2e1332325a48b380b7ecdb7cebafad.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/74aa506e85594691823ea7f229067bf4.jpeg#pic_center)&emsp;&emsp;Connect the filtering operator to filter the test set data, with the expression 'Partition_Column'=='Testing ',
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/97c2987ceca0433db7df5bc5d59c9407.png#pic_center)
&emsp;&emsp;Then connect the line chart analysis operator, select the Lable column predicted value column and the original value column, and use the line chart for comparative analysis.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/689dd3bf93304c7388a47eaa6fb2d68a.png#pic_center)
&emsp;&emsp;Right click to execute to obtain a comparison chart between the predicted results of the test set and the original values.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d875193d336245bfb38c4393ca8a5113.jpeg#pic_center)
&emsp;&emsp;Right click on the model to view model information such as feature importance maps and residual histograms. The results are as follows:![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9962e5c7d24b4cc5b6f3dc5f85678570.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/22ad16821b2c4a3f84fa28c9e6af44e2.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cd17146053b64921b90a225d7940f39c.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5e164ab0d1da4d12ad43be5047cd0c17.jpeg#pic_center)&emsp;&emsp;By connecting various operators and configuring relevant parameters, a second-hand car price prediction model based on XGBoost regression algorithm was completed. From data import and cleaning, to feature engineering, model training, to final prediction, performance evaluation, and visualization analysis, the entire process of modeling is completed. By calculating multiple evaluation indicators of the model, such as mean square error (MSE) and R ² value, the performance of the model is comprehensively measured. Combining visual analysis, the comparison between actual values and predicted values demonstrates the excellent performance of the model in the task of predicting second-hand car prices. Through this series of operations, the powerful performance of XGBoost in complex datasets is fully demonstrated, providing an accurate and reliable solution for predicting second-hand car prices.
# 3、summarize

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

