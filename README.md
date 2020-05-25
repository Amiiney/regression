# Regularization function (Version 1.0):

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

**Future versions will include more input parameters in this function to make it more flexible such as: The personalization of the hyperparameters search. *[WORK IN PROGRESS]***

Tutorial: https://www.kaggle.com/amiiiney/price-prediction-regularization-stacking
