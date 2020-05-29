# Machine learning: Regression project
***

## 1- Missing values visualization
Any data science project starts with data visualization and cleaning. I would like to share here visualization functions that I created and are very usefull to recycle in any data science project. 

Data cleaning is the first thing to do before digging into the data and extracting insight. The best way to start treating missing values is by visualizing the percentage of missing values per column, some columns have >90% and other less than 5% missing values, those columns should not be treated equally. Some columns need to be dropped completely and others need to be filld for example with the values in the previous or following rows, the mean or median. The function ```msv1()```shows the percentage of missing values per column. It requires 6 main input parameters:

* data: your dataset
* thresh (default thresh=20): The threshold in percentage of the missing values 
* color (default color='black'): The bars color
* edgecolor (default edgecolor='black'): The bars edgecolor
* width (default width=15): the figure's width
* height (default height=3): the figure's height

```Python
msv1(data, thresh=20, color='black', edgecolor='black', width=15, height=3)
```
![msv1 image](https://i.ibb.co/KG8QbgG/Screen-Shot-2020-05-29-at-13-23-25.png)

Ideally, the columns with more than 80% missing values should be dropped, because those columns have low variance and low variance features are irrelevant and introduce noise to machine learning models. So, after dropping for the example the columns with more than 80% missing values:
```Python
data.dropna(thresh=len(data)*0.8, axis=1)
```

We can visualize again the columns with missing values that need preprocessing. The function `msv2()`shows the missing values percentage per column with annotations. It takes the same input parameters as the `msv1()`function (excluding the `thresh` argument)
```Python
msv2(data, width=12, height=8, color='silver', edgecolor='black')
```
![msv2 image](https://i.ibb.co/nRdC8dr/Screen-Shot-2020-05-29-at-13-58-25.png)

We should figure out how to treat those columns based on the exploratory data analysis, columns elements and the domain knowledge. A couple missing values can be fixed easily by filling the columns with the values in the previous rows 
```python
data.fillna(method=`ffill')
```
or with the median/mean
```python
data.fillna(data.column.median())
data.fillna(data.column.mean())

```
It's important to keep in mind that the modification we do to the column should not affect or skew our data. For more info: [My kaggle kernel](https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking)
***
## 2- Regularization function (Version 1.0):

In regularization we work with 3 main algorithms: Ridge (L2), Lasso (L1) an ElasticNet that is a combination of both L2 and L1 regressors. I would like to introduce a function *r_reg()* that does all the regularization work **just with one line of code.** The function does all the regression pipeline:

1. Split the data to train/test
2. Scale the data
3. Gridsearch for the best hyperparameters
4. Predict the target
5. Evaluate the prediction

The function takes as input parameters:

* x: the features
* y: the target
* modelo: Ridge(default), Lasso, ElasticNetCV
* scaler: RobustScaler(default), MinMaxSclaer, StandardScaler

Example:
```python
r_reg(x=features, y=target, modulo=Ridge, scaler=RobustScaler)
```

*Future versions will include more input parameters in this function to make it more flexible such as: The personalization of the hyperparameters search. *[WORK IN PROGRESS]**

Tutorial: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
