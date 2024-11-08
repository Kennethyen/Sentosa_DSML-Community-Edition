@[toc]
# Chapter 2: Product Features of the Sentosa_DSML Community Edition
## 1.Public Functions
The entry to the operator flow construction page is shown in the figure, including the operator control area, operator flow control area, theme switching, operator set area, and canvas. The operator set area is where all operator classifications are grouped. The canvas is used to construct and set the properties of operators and operator flows by dragging operators from the operator set area onto the canvas.
![](https://i-blog.csdnimg.cn/direct/fb9fc36141a4439eb4f7b27b005c24ac.png#pic_center)

## 2.算子功能
The Sentosa_DSML Community Edition currently provides a total of 120 operators, categorized into 11 types: Data Input, Data Output, Row Processing, Column Processing, Data Integration, Statistical Analysis, Feature Engineering, Machine Learning, Linear Programming, Chart Analysis, and Extended Programming. Based on their roles in data analysis, they can also be grouped into Data Input, Data Processing (including Row Processing, Column Processing, Data Integration, and Statistical Analysis), Machine Learning & Linear Programming, Chart Analysis, and Data Output.

(1) Data Input operators primarily provide functionality to read data from local files, file systems, databases, and streaming data sources. Text and EXCEL operators also offer file upload capabilities, allowing users to upload files from the client to the server or HDFS for use in constructing operator flows. In the following section on configuring the properties of input operators, a configuration for deleting and renaming duplicate columns has been added. After configuring the input properties, users can modify data columns using the operator's built-in function to delete and rename columns.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7722cf9e71a2483e8b9d9bf9b6717600.png#pic_center)

Hive Database Input (HiveSourceNode): Reads data from a Hive database.
Database Connection (JDBCSourceNode): Supports reading data from five types of databases: MySQL, Oracle, DB2, SQL Server, and PostgreSQL.
Text (FileSourceNode): Supports reading data from local files and HDFS. It also supports uploading files to the server locally or to HDFS, and creating new folders within directories.
Excel Input (ExcelSourceNode): Reads Excel data. It also supports uploading files to the server locally or to HDFS, and creating new folders within directories.
Fit Data Generation (FitDataGenerateNode): Used to generate different types of data according to various distribution patterns.
Random Data Generation (RandomDataGetNode): Used to randomly generate different types of data.
Markov Data Source (MarkovSourceNode): Generates Markov chain data.
Kafka Input (Kafa2DatasetNode): Reads data from Kafka.
HBase Input (HbaseSourceNode): Reads data from an HBase database.
XML Input (XMLSourceNode): Supports reading XML data from local files and HDFS. It also supports uploading files to the server locally or to HDFS.
ES Input (ESSourceNode): Reads data from an ES database.
	
(2) Data Output operators primarily provide functionality to write processed data from the operator flow to files or databases. File output supports multiple formats, and data written to files can be downloaded locally through file browsing interfaces. The specific functions of Data Output operators are shown in the following table:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5fbc1d715e294ac7a311a6fa4a68394f.png#pic_center)
File Output (FileWriterNode): Supports writing result data to HDFS and local file systems.
Database Output (JDBCOutputNode): Supports writing result data to JDBC-compatible relational databases.
Hive Database Output Operator (HiveWriterNode): Supports writing data to a Hive database on the cluster when launched in YARN mode.
Kafka Output Operator (KafkaNode): Supports outputting datasets in Kafka format, enriching the data output methods of the operator platform.
HBase Output Operator (HbaseOutputNode): Supports writing data to an HBase database.
Excel Output Operator (ExcelOutputNode): Supports writing result data to Excel files on HDFS and local file systems.
XML Output Operator (XMLOutputNode): Supports writing result data to XML files on HDFS and local file systems.
ES Database Output Operator (ESOutputNode): Supports writing data to an ElasticSearch database.

(3) Data Processing Operators consist of six categories: Row Processing, Column Processing, Data Merging, Data Statistics, Feature Selection, and Extended Programming. These operators are primarily used to process data based on user requirements or machine learning modeling needs. The specific functions of the data processing operators are outlined in the following table:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a4e1183bd4a9454c8975bb6b9be60cb7.png#pic_center)
FillNode: Fills selected columns with corresponding fill values.
SortNode: Sorts data.
FilterNode: Filters data.
AggregateNode: Aggregates data.
SampleNode: Samples data.
TimeSeriesResampleNode: Resamples time series data based on set parameters.
TimeSeriesCleaningNode: Cleans and supplements time series data based on set parameters.
RePartitionNode: Re-partitions data storage based on selected columns.
DistinctNode: Removes user-defined duplicate rows from data.
OutlierAndMissingValProNode: Handles outliers and missing values in data with specific strategies.
PartitionNode: Randomly divides data into training, testing, and validation sets (validation set can be optional).
DiffNode: A row-processing operator that calculates the difference row-by-row for specified columns.
DatasetBalanceNode: Selects data sets based on user-specified conditions and adjusts the selected data sets based on user-specified coefficients.
ShuffleNode: Shuffles the data set by rows.
SampleStratifiedNode: Primarily used for data sampling to ensure a more balanced distribution across different categories.
GenerateRowNode: Generates multiple rows of data.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7962a86b0d2042019647320c4c926451.png#pic_center)
GenerateColumnNode: Generates a new column of data based on parameter settings.
Array2DatasetNode: Reduces the dimensionality of matrix data for use with cplex operators.
SelectNode: Selects columns from data based on expression settings.
TypeNode: Views data types and sets test types.
FormatNode: Views and modifies data storage types.
PolynomialExpressionNode: Expands features into a multivariate space, used for polynomial transformations of feature values.
DeleteRenameNode: Deletes data columns and renames column headers.
TransposeNode: Transposes the rows and columns of a data table, converting rows to columns and columns to rows.
ColumnAdjustNode: Adjusts the order of columns.
ExcelFunctionCalcuateNode: Calculates results row-by-row based on Excel formulas.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7164133472d547fd887364dc223da62d.png#pic_center)
MergeNode: Merges data.
UnionNode: Appends data.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a9138dc48c9441b19ce6f5a4eb315584.png#pic_center)

SpearmanCorrelationNode: Implements the Spearman correlation coefficient algorithm.
PearsonCorrelationNode: Implements the Pearson correlation coefficient algorithm.
DescribeNode: Summarizes the incoming dataset by column, calculating the number of outliers, mode, and extreme values based on parameters.
ChiSquareNode: Measures the degree of deviation between the actual observed values and the theoretically inferred values in a statistical sample.
LBTestNode: Determines whether a time series is a pure random sequence.
ADFNode: Tests for the presence of a unit root in a sequence: if the sequence is stationary, there is no unit root; otherwise, there is a unit root.
ACFNode (Autocorrelation Function Node): Measures the correlation between observations in a time series separated by k time units (yt and yt-k).
PACFNode (Partial Autocorrelation Function Node): Describes the direct relationship between an observation and its lags.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e7cc5ea0a7ed4432a30f78966835045e.png#pic_center)
QuantileDiscretizerNode (Streaming Quantile Discretizer): Implements quantile discretization, converting a continuous column of data into categorical data.
StandardScalerNode (Streaming Standardization): An operator for standardizing data.
BuildPCANode (PCA): Implements dimensionality reduction of data. Automatically reduces high-dimensional data to an appropriate dimension through analysis of data samples.
RescaleNode (Streaming Normalization): An operator for implementing normalization algorithms.
ChiSquareSelectorNode (Chi-Square Feature Selection): For labels of discrete types, it selects effective features from discrete-type features through chi-square testing.
BuildRescaleNode (Normalization): The streaming normalization operator is transformed from a dataset in/out mode to a modeling-like operator. It generates a model that retains the statistical values of the data during modeling, allowing for the same processing of prediction data.
BuildStandardScaler (Standardization): The streaming standardization operator is transformed from a dataset in/out mode to a modeling-like operator. It generates a model that retains the statistical values of the data during modeling, allowing for the same processing of prediction data.
BuildQdNode (Quantile): The streaming quantile operator is transformed from a dataset in/out mode to a modeling-like operator. It generates a model that retains the statistical values of the data during modeling, allowing for the same processing of prediction data.
BinarizerNode (Binarization): An improvement on K-means to prevent clustering from falling into local optimal solutions.
BucketizerNode (Bucketing): An operator based on dichotomization for classification, which can discretize data.
FeatureImportanceNode (Feature Importance): Describes the importance of other features in the data relative to the label column.
TSNENode (t-SNE): Used for dimensionality reduction of high-dimensional data.
BulidRobustScalerNode (Robust Scaler): Scales data with outliers using robust statistics, removing the median and scaling the data based on a quantile range (default is IQR).
Information Value (IV): Calculates the information value for feature columns used in classification models.
SMOTE: A method of oversampling for imbalanced data, which analyzes minority class samples and artificially synthesizes new samples from them to add to the original dataset.
SVD: Decomposes a matrix into singular values and singular vectors.
BuildICANode (ICA): Implements a dimensionality reduction algorithm that recovers independent components from observed data.
IGFeatureSelectNode (IG): The IG operator calculates the mutual information between features and labels as an evaluation index for feature importance. Users can make feature selections based on this value.
FisherScoreNode (Fisher Score): The FisherScore operator calculates a score between features and labels as an evaluation index for feature importance. Users can make feature selections based on this value.
BuildRecursiveFeatureEliminationNode (RFE): The RFE operator excludes unimportant features through multiple iterations to achieve the purpose of feature selection.
CategoricalFeatureEncodingNode (Streaming Categorical Feature Encoding): An operator for converting discrete variables into continuous variables.
BuildCategoricalFeatureEncodingNode (Categorical Feature Encoding): An operator for converting discrete variables into continuous variables.
TargetEncodingNode (Streaming Target Encoding): Used to encode discrete feature variables by replacing categorical values with the mean of the target variable.
BuildTargetEncodingNode (Target Encoding): Used to encode discrete feature variables by replacing categorical values with the mean of the target variable.
OneHotEncodingNode (Streaming One-Hot Encoding): Uses an N-bit state register to encode N states, with each state having its own independent register bit and only one bit being active at any time.
BuildOneHotEncodingNode (One-Hot Encoding): Uses an N-bit state register to encode N states, with each state having its own independent register bit and only one bit being active at any time.
StreamingCountEncodingNode (Streaming Count Encoding): Used for count encoding of discrete types, replacing categorical values with their occurrence frequencies.
BuildCountEncodingNode (Count Encoding): Used for count encoding of discrete types, replacing categorical values with their occurrence frequencies.
StreamingBaseNEncodingNode (Streaming Base-N Encoding): Used to encode discrete feature variables by encoding the order as a base-N number.
BuildBaseNEncodingNode (Base-N Encoding): Used to encode discrete feature variables by encoding the order as a base-N number.
StreamingHashEncodingNode (Streaming Hash Encoding): Used to encode discrete feature variables by encoding discrete values using a hash algorithm.
BuildHashEncodingNode (Hash Encoding): Used to encode discrete feature variables by encoding discrete values using a hash algorithm.
StreamingWOEEncodingNode (Streaming Weight of Evidence Encoding): Used to encode discrete feature variables.
BuildWOEEncodingNode (Weight of Evidence Encoding): Used to encode discrete feature variables.

(4) Machine Learning Operators provide mainstream machine learning algorithms. Users can mine deeper patterns and knowledge from data by modeling it. These models are then utilized for predictions to improve business decisions. Machine Learning Operators encompass algorithms such as classification, regression, clustering, and more. Additionally, model evaluation operators are provided to facilitate users in assessing whether the generated models meet their requirements. The specific functions of Machine Learning Operators are detailed in the following table.：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b85f5be6ba774b07ab39c85565029bce.png#pic_center)
Logical Regression Classification (BuildLGRNode): A classification algorithm that specifies an activation function on the basis of linear regression to classify data.

Decision Tree Classification (BuildDTCNode): A simple and easy-to-use non-parametric classifier model that continuously splits on user-selected feature columns, increasing the purity of the target variable at each branch, achieving data classification.

Gradient Boosting Decision Tree Classification (BuildGBTCNode): Gradient Boosting Trees are a Boosting ensemble model that combines multiple decision trees for prediction. The multiple decision trees are sequentially combined, with each decision tree model correcting the prediction errors of all previous models to achieve classification.

XGBoost Classification (BuildXGBoostClassifier): An optimized distributed gradient boosting classification algorithm that provides parallel tree boosting.

Random Forest Classification (BuildRFCNode): Internally integrates a large number of decision tree models. Each model selects a subset of features and a subset of training samples. Ultimately, multiple decision tree models jointly determine the predicted value to classify data.

Naive Bayes Classification (BulidNBNode): Predicts by applying Bayes' theorem to calculate the conditional probability distribution of each label for a given observation.

Support Vector Machine Classification (BuildSVM): Classifies data by constructing hyperplanes or sets of hyperplanes in high-dimensional space.

Multilayer Perceptron Classification (BuildMLPNode): Implements artificial neural networks for data classification.

LightGBM Classification (BuildLightGBMClassifierNode): Belongs to the Boosting ensemble model, and like XGBoost, it is an efficient implementation of GBDT. LightGBM performs better than XGBoost in many aspects. It has the following advantages: faster training efficiency, low memory usage, higher accuracy, support for parallelized learning, and the ability to handle large-scale data.

Factorization Machines Classification (BuildFMClassifierNode): A machine learning algorithm based on matrix factorization that addresses feature combinations and high-dimensional sparse matrix problems. Firstly, it combines features by introducing pairwise feature interactions to improve model scores. Secondly, it addresses the curse of dimensionality by introducing latent vectors (through matrix factorization of the parameter matrix) to estimate feature parameters. Currently, the FM algorithm is one of the verified recommendation schemes with good performance in the recommendation field.

AdaBoost Classification (BuildAdaboostClassifierNode): A Boosting ensemble method whose main idea is to boost weak learners into strong learners. It adjusts the sample weights in the training set based on the prediction performance of the learner obtained from the previous iteration, and then trains a new base learner accordingly. The final ensemble result is a combination of multiple base learners.

KNN Classification (BuildKNNClassifierNode): To predict an instance, the distance to all instances needs to be calculated. That is, which category's samples are closer to the target sample.

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b76175a6161c422dbe8791b0b13c6c1f.png#pic_center)

Linear Regression (BuildLRNode): Assumes a linear relationship between all feature variables and the target variable. It trains to find the weights and intercept of each feature.

Decision Tree Regression (BuildDTRNode): A simple and easy-to-use non-parametric classifier model that continuously splits on user-selected feature columns, increasing the purity of the target variable at each branch, achieving data regression.

Gradient Boosting Decision Tree Regression (BuildGBTRNode): Gradient Boosting Trees are a Boosting ensemble model that combines multiple decision trees for prediction. The multiple decision trees are sequentially combined, with each decision tree model correcting the prediction errors of all previous models to achieve regression.

Isotonic Regression (BuildRNode): A regression model that performs non-parametric estimation of given data within a monotone function space.

XGBoost Regression (BuildXGBoostRegression): An optimized distributed gradient boosting regression algorithm that provides parallel tree boosting.

Random Forest Regression (BuildRFRNode): Internally integrates a large number of decision tree models. Each model selects a subset of features and a subset of training samples. Ultimately, multiple decision tree models jointly determine the predicted value to achieve data regression.

Generalized Linear Regression (BuildGLRNode): Establishes a relationship between the mathematical expectation of the response variable and the predictor variables of the linear combination through a link function. Its characteristics include not forcibly changing the natural measure of the data, allowing for non-linear and non-constant variance structures.

LightGBM Regression (BuildLightGBMRegressionNode): Belongs to the Boosting ensemble model and, like XGBoost, is an efficient implementation of GBDT. LightGBM performs better than XGBoost in many aspects. It has the following advantages: faster training efficiency, low memory usage, higher accuracy, support for parallelized learning, and the ability to handle large-scale data.

Factorization Machines Regression (BuildFMRegressionNode): A machine learning algorithm based on matrix factorization that addresses feature combinations and high-dimensional sparse matrix problems. Firstly, it combines features by introducing pairwise feature interactions to improve model scores. Secondly, it addresses the curse of dimensionality by introducing latent vectors (through matrix factorization of the parameter matrix) to estimate feature parameters. Currently, the FM algorithm is one of the verified recommendation schemes with good performance in the recommendation field.

AdaBoost Regression (BuildAdaboostRegressionNode): A Boosting ensemble method whose main idea is to boost weak learners into strong learners. It adjusts the sample weights in the training set based on the prediction performance of the learner obtained from the previous iteration and then trains a new base learner accordingly. The final ensemble result is a combination of multiple base learners.

KNN Regression (BuildKNNRegressionNode): The KNN operator generally uses the average method when making regression predictions.

Gaussian Process Regression (BuildGPRegressionNode): Gaussian Process Regression is a non-parametric model that uses a Gaussian process prior for regression analysis of data.

Multilayer Perceptron Regression (BuildMLPRegressionNode): Multilayer Perceptron is a feedforward artificial neural network model that maps multiple input datasets to a single output dataset. The layers of the multilayer perceptron are fully connected, with the bottom layer being the input layer, the middle layers being hidden layers, and the final layer being the output layer.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/debe02d3305541f1be10a8bfcbe6478e.png#pic_center)

Frequent Pattern Growth (FP-Growth)：Frequent Pattern Growth constructs a frequent pattern tree to generate frequent itemsets or frequent item pairs with fewer passes through the dataset. It also supports user-defined minimum confidence and minimum support levels.
PrefixSpan：PrefixSpan is a method used to mine frequent sequences that satisfy a minimum support threshold.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/61d1f7ffa2be4da98259ce1630f69c9f.png#pic_center)

ARIMAX (ARIMAXNode)：The ARIMA model with exogenous regressors, also known as the extended ARIMA model. The introduction of exogenous regressors enhances the predictive power of the model, and these regressors are typically variables that are highly correlated with the dependent variable.

ARIMA (ARIMANode)：One of the commonly used time series models. If you only need to predict future data based on the historical data of a single target variable, you can use the ARIMA algorithm. If there are other input variables besides the target variable, you can choose the ARIMAX model.

HoltWinters (HoltWintersNode)：One of the commonly used time series models. If you need to predict future data based on the historical data of a single target variable with an obvious periodic pattern, you can use the HoltWinters algorithm.

Single Exponential Smoothing (SESNode)：Single Exponential Smoothing, also known as Simple Exponential Smoothing, is used when there is no obvious trend in the time series data.

Double Exponential Smoothing (HoltLinearNode)：Double Exponential Smoothing, which is a re-smoothing of Single Exponential Smoothing, is applicable to time series data with a linear trend.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ea8e997f8d1d46979e144ccadfcd21cb.png#pic_center)

Accelerated Failure Time Regression (BuildAFTSRNode)：This is a parametric survival regression model that examines data. It describes a model for the logarithm of survival time.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b28fae1e704e4d7e9b555dd64d1ae597.png#pic_center)

KMeans Clustering (BuildKMeansNode)：This is an iterative clustering analysis algorithm. The steps involve randomly selecting K objects as the initial cluster centers, then calculating the distance between each object and each seed cluster center, and assigning each object to the nearest cluster center.

Bisecting K-Means Clustering (BuildBKMeansNode)：This is an improvement on K-means clustering to prevent it from falling into local optimal solutions. The main idea is to first treat all points as one cluster, and then bisect this cluster. Afterward, the cluster that can most significantly reduce the clustering cost function is divided into two clusters. This process continues until the number of clusters equals the user-specified number k.

Gaussian Mixture Model Clustering (BuildGMNode)：The Gaussian Mixture Model uses Gaussian probability density functions to accurately quantify things. It is a model that decomposes things into several components based on Gaussian probability density functions.

Fuzzy C-Means Clustering (BuildFCMeansNode)：This is a partition-based clustering algorithm that aims to maximize the similarity between objects assigned to the same cluster while minimizing the similarity between different clusters.

Canopy Clustering (BuildCanopyClusterNode)：This is a fast, coarse clustering algorithm that does not require the user to specify the number of clusters beforehand. The user needs to specify two distance thresholds, T1 and T2, where T1 > T2. T2 can be considered the core clustering range, and T1 the peripheral clustering range.

Canopy K-Means Clustering (BuildCanopyKMeansNode)：This combines the advantages of both Canopy and K-means clustering algorithms. First, Canopy clustering is used to quickly perform a "coarse" clustering of the data. After obtaining the k value, K-means clustering is then used for further "fine" clustering.

Latent Dirichlet Allocation (LDA) Clustering (BuildLDANode)：Also known as a three-layer Bayesian probabilistic model, it includes three layers of structure: words, topics, and documents. The generative model assumes that each word in an article is obtained through a process of "selecting a topic with a certain probability and then selecting a word from this topic with a certain probability." The distribution from documents to topics follows a multinomial distribution, and the distribution from topics to words also follows a multinomial distribution.

DBSCAN Clustering (Density-Based Spatial Clustering of Applications with Noise)：This is a representative density-based clustering algorithm. It defines a cluster as the largest set of density-connected points and can divide regions with sufficiently high density into clusters. It can discover clusters of arbitrary shape in noisy spatial databases.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aecf1f227b454d6cb3306f6e9526d899.png#pic_center)

Anomaly Detection (BuildIFnode) ：Detecting abnormal data in datasets.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/144ac6f0974f44329c975cb5aa85f76f.png#pic_center)
Collaborative Filtering (BuildALSNode) - A commonly used method in recommendation systems. This algorithm aims to fill in the missing entries in the user-product association matrix. In this algorithm, both users and products are described by a small set of latent factors, which can be used to predict the missing entries in the user-product association matrix.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6955054188b4460abf969502c29a017c.png#pic_center)
Evaluation (EvaluationNode) ： Used to assess the accuracy of a model trained on the current data, displaying specific values for various evaluation metrics of the model.

Confusion Matrix (ConfusionMatrixNode)： Used to display the confusion matrix of the classification results produced by a classification operator, facilitating the evaluation of classification results by users.

ROC-AUC Evaluation (ROCAUCNode)：The ROC-AUC operator (ROCAUCNode) is used after a classification model to evaluate the accuracy of the classification model trained on the current data. It displays the ROC curve and AUC value of the classification results, allowing users to assess the classification performance of the model.

Time Series Model Evaluation (TimeSeriesModeEvaluateNode) ： This operator evaluates the performance of a dataset that has undergone time series prediction using various metrics through the time series model evaluation process.

(5) Chart Analysis Operators primarily visualize data through bar charts, pie charts, tables, and other graphical representations. The specific functions of the chart analysis operators are shown in the following table:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/251fdb1842514f5c820d3a2c3e867756.png#pic_center)
Chart Operators:
ScattergramNode: Displays data in the form of a scatter plot.
BarNode: Displays data in the form of a bar chart.
LineNode: Displays data in the form of a line chart.
PieNode: Displays data in the form of a pie chart.
HistogramNode: Displays data in the form of a histogram, showing the distribution of data and optionally overlaying a normal distribution curve.
BubbleNode: Displays data in the form of a 2D bubble chart.
ParallelgramNode: Displays data in the form of a parallel relationship diagram.
RadarNode: Displays data in the form of a 2D radar chart.
StackBarNode: Displays data in the form of a 2D stacked bar chart.
BoxplotNode: Displays data in the form of a 2D box plot.
Scattergram3DNode: Displays data in the form of a 3D scatter plot.
Bubble3DNode: Displays data in the form of a 3D bubble chart.
SurfaceNode: Displays data in the form of a 3D surface plot.
MapColorBlockNode: Displays different regions on a map with different colors, indicating the distribution of data across different geographical areas.
SequenceDiagramNode: Displays time series data or time series results predicted by "time series prediction operators" on the operator platform in the form of points or lines, facilitating the observation and analysis of time data.
MapBubbleNode: A data visualization operator based on Baidu Maps, displaying data in the form of a scatter plot on the map.
MapBarNode: A data visualization operator based on Baidu Maps, displaying data in the form of a bar chart on the map.
MapPieNode: A data visualization operator based on Baidu Maps, displaying data in the form of a pie chart on the map.
MapHeatMapNode: A data visualization operator based on Baidu Maps, displaying data in the form of a heatmap, showing the geographical distribution of data.
TableOutputNode: Displays output data.
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6494f8d6f3304105abf11762021af138.png#pic_center)


Extended Programming:
SQL (SQLNode): An operator that supports SQL query statements, currently limited to SELECT operations. This allows users to query data using SQL syntax within the operator platform.
PySpark (PySparkNode): Provides the ability to write PySpark statements for data processing, enabling customers to quickly complete a series of data operations through programming. With PySparkNode, users can leverage the powerful data processing capabilities of Apache Spark within a Python environment, facilitating complex data transformations, aggregations, and analyses.


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

