# 1、XGBoost Algorithm
&emsp;&emsp;This article will utilize the diabetes dataset to construct an XGBoost classification prediction model through Python code and the Sentosa_DSML community edition, respectively. Subsequently, the model will be evaluated, including the selection and analysis of evaluation metrics. Finally, the experimental results and conclusions will be presented, demonstrating the effectiveness and accuracy of the model in predicting diabetes classification, providing technical means and decision support for early diagnosis and intervention of diabetes.
# 2、Comparison of algorithm implementation between Python code and Sentosa_DSML community edition
## (1) Data reading and statistical analysis
1、Implementation in Python code
```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from matplotlib import rcParams
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

file_path = r'.\xgboost分类案例-糖尿病结果预测.csv'
output_dir = r'.\xgb分类'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"文件未找到: {file_path}")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_csv(file_path)

print("缺失值统计:")
print(df.isnull().sum())

print("原始数据前5行:")
print(df.head())
```
&emsp;&emsp;After reading in, perform statistical analysis on the data information
```python
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']
stats_df = pd.DataFrame(columns=[
    '列名', '数据类型', '最大值', '最小值', '平均值', '非空值数量', '空值数量',
    '众数', 'True数量', 'False数量', '标准差', '方差', '中位数', '峰度', '偏度',
    '极值数量', '异常值数量'
])

def detect_extremes_and_outliers(column, extreme_factor=3, outlier_factor=6):
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
>> 列名     数据类型     最大值    最小值  ...         峰度        偏度  极值数量 异常值数量
0               gender   object     NaN    NaN  ...        NaN       NaN  None  None
1                  age  float64   80.00   0.08  ...  -1.003835 -0.051979     0     0
2         hypertension    int64    1.00   0.00  ...   8.441441  3.231296  7485  7485
3        heart_disease    int64    1.00   0.00  ...  20.409952  4.733872  3942  3942
4      smoking_history   object     NaN    NaN  ...        NaN       NaN  None  None
5                  bmi  float64   95.69  10.01  ...   3.520772  1.043836  1258    46
6          HbA1c_level  float64    9.00   3.50  ...   0.215392 -0.066854     0     0
7  blood_glucose_level    int64  300.00  80.00  ...   1.737624  0.821655     0     0
8             diabetes    int64    1.00   0.00  ...   6.858005  2.976217  8500  8500

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

grouped_data = df.groupby('smoking_history')['diabetes'].count()
plt.figure(figsize=(8, 8))
plt.pie(grouped_data, labels=grouped_data.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title("饼状图\n维饼状图", fontsize=16)
plt.axis('equal')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f"smoking_history_diabetes_distribution_{timestamp}.png"
file_path = os.path.join(output_dir, file_name)
plt.savefig(file_path)
plt.close() 
```

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/73475c3bb3414712bde457505bb72b4f.jpeg#pic_center.png =500x400)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8654ab0638e047dfb50f360a2311bd30.png#pic_center.png =400x400)
2、Implementation of Sentosa_DSML Community Edition 

&emsp;&emsp;First, perform data input by directly reading the data using a text operator and selecting the data path, 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fd2e37063b454de8b34ba04d2ca80b8f.png#pic_center)
&emsp;&emsp;Next, the description operator can be utilized to perform statistical analysis on the data, obtaining results such as the data distribution diagram, extreme values, and outliers for each column of data. Connect the description operator, and set the extreme value multiplier to 3 and the outlier multiplier to 6 on the right side. 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b29fe4a2a20d4f74a55dac66e9f3fe0c.png#pic_center)
&emsp;&emsp;After clicking execute, the results of data statistical analysis can be obtained.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2649e1c2926245ddb5bca231c34ce136.jpeg#pic_center)
&emsp;&emsp;You can also connect graph operators, such as pie charts, to make statistics on the relationship between different smoking histories and diabetes,
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b59ab641150b42de9ad30cb1f26026b4.png#pic_center)
&emsp;&emsp;The results obtained are as follows:![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8513c24bbdb24ed79ad3661dc3a224ad.jpeg#pic_center)
## (2)Data preprocessing
1、Implementation in Python code
```python
df_filtered = df[df['gender'] != 'Other']
if df_filtered.empty:
    raise ValueError(" `gender`='Other'")
else:
    print(df_filtered.head())

if 'Partition_Column' in df.columns:
    df['Partition_Column'] = df['Partition_Column'].astype('category')

df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

X = df.drop(columns=['diabetes'])
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
```
2、Implementation of Sentosa_DSML Community Edition
&emsp;&emsp;Connect the filtering operator after the text operator, with the filtering condition of 'gender'='Other', without retaining the filtering term, that is, filtering out data with a value of 'Other' in the 'gender' column.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/776b305c47b14774b2eeaaf73b44bc8f.png#pic_center)
&emsp;&emsp;Connect the sample partitioning operator, divide the training set and test set ratio,
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b4f8b26b9eee4084b495773f922c66d6.png#pic_center)
Then, connect the type operator to display the storage type, measurement type, and model type of the data, and set the model type of the diabetes column to Label.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/15d12f5f3b8f4525a7d2a52b10395570.png#pic_center)
## (3)Model Training and Evaluation
1、Implementation in Python code
```python
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)

params = {
    'n_estimators': 300,
    'learning_rate': 0.3,
    'min_split_loss': 0,
    'max_depth': 30,
    'min_child_weight': 1,
    'subsample': 1,
    'colsample_bytree': 0.8,
    'lambda': 1,
    'alpha': 0,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'missing': np.nan
}

xgb_model = xgb.XGBClassifier(**params, use_label_encoder=False)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

def evaluate_model(y_true, y_pred, dataset_name=''):
    accuracy = accuracy_score(y_true, y_pred)
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"评估结果 - {dataset_name}")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"加权精确率 (Weighted Precision): {weighted_precision:.4f}")
    print(f"加权召回率 (Weighted Recall): {weighted_recall:.4f}")
    print(f"加权 F1 分数 (Weighted F1 Score): {weighted_f1:.4f}\n")

    return {
        'accuracy': accuracy,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
train_eval_results = evaluate_model(y_train, y_train_pred, dataset_name='训练集 (Training Set)')
>评估结果 - 训练集 (Training Set)
准确率 (Accuracy): 0.9991
加权精确率 (Weighted Precision): 0.9991
加权召回率 (Weighted Recall): 0.9991
加权 F1 分数 (Weighted F1 Score): 0.9991

test_eval_results = evaluate_model(y_test, y_test_pred, dataset_name='测试集 (Test Set)')

>评估结果 - 测试集 (Test Set)
准确率 (Accuracy): 0.9657
加权精确率 (Weighted Precision): 0.9641
加权召回率 (Weighted Recall): 0.9657
加权 F1 分数 (Weighted F1 Score): 0.9643
```
Evaluate the performance of classification models on the test set by plotting ROC curves.
```python
def save_plot(filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f"{filename}_{timestamp}.png")
    plt.savefig(file_path)
    plt.close()
    
def plot_roc_curve(model, X_test, y_test):
    """绘制ROC曲线"""
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC 曲线 (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) 曲线')
    plt.legend(loc='lower right')
    save_plot("ROC曲线")
    
plot_roc_curve(xgb_model, X_test, y_test)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be6c56f1309341d3877971b3c6dfa446.png#pic_center.png =400x300)
2、Implementation of Sentosa_DSML Community Edition
&emsp;&emsp;After preprocessing is completed, connect the XGBoost classification operator and configure the operator properties on the right side. In the operator properties, the evaluation metric is the loss function of the algorithm, which includes logarithmic loss and classification error rate; Learning rate, maximum depth of the tree, minimum leaf node sample weight sum, subsampling rate, minimum splitting loss, proportion of randomly sampled columns per tree, L1 regularization term and L2 regularization term are all used to prevent algorithm overfitting. When the sum of the weights of the sub node samples is not greater than the set minimum sum of the weights of the leaf node samples, the node will not be further divided. The minimum splitting loss specifies the minimum decrease in the loss function required for node splitting. When the tree construction method is hist, three attributes need to be configured: node mode, maximum number of boxes, and single precision.
&emsp;& emsp; In this case, the attribute configuration in the classification model is as follows: iteration number: 300, learning rate: 0.3, minimum splitting loss: 0, maximum depth of number: 30, minimum leaf node sample weight sum: 1, subsampling rate: 1, tree construction algorithm: auto, proportion of randomly sampled columns per tree: 0.8, L2 regularization term: 1, L1 regularization term: 0, evaluation metric is logarithmic loss, initial prediction score is 0.5, and the confusion matrix between feature importance and training data is calculated.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/561420d843b54849b4906a8ec32730a5.png#pic_center)
&emsp;&emsp;Right click to execute to obtain the XGBoost classification model.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6be3a539d37a417f979f2a5f4c57dc81.jpeg#pic_center)
&emsp;&emsp;By connecting the evaluation operator and ROC-AUC evaluation operator after the classification model, the prediction results of the model's training and testing sets can be evaluated.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fa5d3c407cc340a1a1e37feb257dcdc3.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e91cd391d1204f0e9ebd3831f90a5338.png#pic_center)
&emsp;&emsp;Evaluate the performance of the model on the training and testing sets, mainly using accuracy, weighted precision, weighted recall, and weighted F1 score. The results are as follows:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ef3845b2cce74253a6c22efc382f49ec.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bfbbd64a4608449aab44eda7e7560fcc.jpeg#pic_center)
&emsp;&emsp;The ROC-AUC operator is used to evaluate the correctness of the classification model trained on the current data, display the ROC curve and AUC value of the classification results, and evaluate the classification performance of the model. The execution result is as follows:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cdfbdb7f387b40b9a82c95af009b9f30.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5fa8a79aca844464a099ac188192681e.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/99bf355c68014a18bba442c226f198ef.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0a846a0c1d1345b0b42150feff15b72e.jpeg#pic_center)
&emsp;&emsp;The table operator in chart analysis can also be used to output model data in tabular form.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2f929eb5cf964a9a8ef925766960dfd4.png#pic_center)
&emsp;&emsp;The execution result of the table operator is as follows:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/adc8885c0a6a47ed95cf3fdba266ac84.jpeg#pic_center)
## (4)Model visualization

1、Implementation in Python code
```python
def save_plot(filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f"{filename}_{timestamp}.png")
    plt.savefig(file_path)
    plt.close()
    
def plot_confusion_matrix(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title("混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    save_plot("混淆矩阵")
    
def print_model_params(model):
    params = model.get_params()
    print("模型参数:")
    for key, value in params.items():
        print(f"{key}: {value}")
        
def plot_feature_importance(model):
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, importance_type='weight', max_num_features=10)
    plt.title('特征重要性图')
    plt.xlabel('特征重要性 (Weight)')
    plt.ylabel('特征')
    save_plot("特征重要性图")

print_model_params(xgb_model)
plot_feature_importance(xgb_model)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/10678f96adb14cd1bf718ca22afcb529.png#pic_center.png =400x300)
2、Implementation of Sentosa_DSML Community Edition
&emsp;&emsp;Right click to view model information to display model results such as feature importance maps, confusion matrices, decision trees, etc.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fed92dbfeb6740b9b283fb3596b2e7e0.png)
&emsp;&emsp;The model information is as follows:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d1a01511e7be4253a0a0745160ecccba.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0b25c9cd3fc74eaa87b91b4b98ef1412.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6044f6ed77624e309010ba2c7dc8b842.jpeg#pic_center)
&emsp;&emsp;Through connection operators and configuration parameters, the whole process of diabetes classification prediction based on XGBoost algorithm is completed, from data import, pre-processing, model training to prediction and performance evaluation. Through the model evaluation operator, we can know the accuracy, recall rate, F1 score and other key evaluation indicators of the model in detail, so as to judge the performance of the model in the diabetes classification task.
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

