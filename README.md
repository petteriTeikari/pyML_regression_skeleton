# pyML_regression_skeleton

Python init for Python-only machine learning methods if you want to benchmark the `caret` package against some new funky Python models.

Place for example `df_ZCA_cor.csv` or the original non-hot encoded data `TestPredMod.csv` to `test_private_data`, and run the main demo file `demo_py_regression.py`.

## R code for ZCAcor transform

From: Kessy, Lewin, and Strimmer (2018) _"Optimal whitening and decorrelation"_, [doi:10.1080/00031305.2016.1277159](https://doi.org/10.1080/00031305.2016.1277159), the authors who created the package: https://cran.r-project.org/web/packages/whitening/index.html

## Results

Standard 3-layer MLP network with concrete dropout, taken from:
https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout-keras.ipynb

_i.e. 3 dropout probabilities + 1 for `mean` + 1 for `log_var`_

```python
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_5 (InputLayer)            (None, 58)           0                                            
__________________________________________________________________________________________________
concrete_dropout_1 (ConcreteDr  (None, 58)           3423        input_5[0][0]                    
__________________________________________________________________________________________________
concrete_dropout_2 (ConcreteDr  (None, 58)           3423        concrete_dropout_21[0][0]        
__________________________________________________________________________________________________
concrete_dropout_3 (ConcreteDr  (None, 58)           3423        concrete_dropout_22[0][0]        
__________________________________________________________________________________________________
concrete_dropout_4 (ConcreteDr  (None, 1)            60          concrete_dropout_23[0][0]        
__________________________________________________________________________________________________
concrete_dropout_5 (ConcreteDr  (None, 1)            60          concrete_dropout_23[0][0]        
__________________________________________________________________________________________________
output (Concatenate)            (None, 2)            0           concrete_dropout_24[0][0]        
                                                                 concrete_dropout_25[0][0]        
==================================================================================================
Total params: 10,389
Trainable params: 10,389
Non-trainable params: 0

```

### z-standardized

3 repeats (mean & stdev below of the repeats), length scale = 1e-4, number of features per each layer = 58 (the amount of features)

```python
RMSE Test mean:  20.897048178203246
RMSE Test stdev:  0.0492192304907618
Dropout probabilities (per dense layer):  [1.5786922e-05 2.5202235e-05 8.8937231e-06 1.1876257e-05 2.9436257e-01]
Aleatoric uncertainty (exp from logvar):  0.6873182576718747
Epistemic uncertainty (var of means): 0.153345257861877 
R^2 =  0.08881353567778605
```

`nb_reps` = 16, `K_test` = 100, best `ll` = 1e-4 from `lscale= [1, 1e-1, 1e-2, 1e-3, 1e-4]`. In other words, the less you regularize, the "better" the fit becomes for our test set in this range.


```python
RMSE Test mean:  21.066091801101543
RMSE Test stdev:  0.05654301392338118
Dropout probabilities (per dense layer):  [1.5242346e-05 2.1328844e-05 6.4392938e-05 8.9678297e-06 2.4842893e-01]
Aleatoric uncertainty (exp from logvar):  0.7082485026630876
Epistemic uncertainty (var of means): 0.45814347850488174 
R^2 =  0.09959331771393598
```

#### TODO! the same with cross-validation

### ZCAcor

_"Optimal whitening and decorrelation"_, [doi:10.1080/00031305.2016.1277159](https://doi.org/10.1080/00031305.2016.1277159)

3 repeats (mean & stdev below of the repeats), length scale = 1e-4, number of features per each layer = 58 (the amount of features)

```python
RMSE Test mean:  21.128462781242433
RMSE Test stdev:  0.09694193545003267
Dropout probabilities (per dense layer):  [1.6511696e-05 2.4299350e-05 3.6801528e-05 9.7002685e-06 2.1191585e-01]
Aleatoric uncertainty:  0.8934794409915705
Epistemic uncertainty: 0.23795429843345484 
R^2 =  0.15268120241824223
```

`nb_reps` = 16, `K_test` = 100, best `ll` = 1e-2 from `lscale= [1, 1e-1, 1e-2, 1e-3, 1e-4]`. Stronger regularization gave the better test mean now than with _z_-standardization. But pretty close the results anyway, most likely statistically not significant.

```python
RMSE Test mean:  21.063691003190005
RMSE Test stdev:  0.0677725253588539
Dropout probabilities (per dense layer):  [1.5095859e-05 2.3792471e-05 3.8986793e-05 9.4188708e-06 2.4920020e-01]
Aleatoric uncertainty (exp from logvar):  0.7277302535590989
Epistemic uncertainty (var of means): 0.22482351792628677 
```
With ZCAcor, the aleatoric and epistemic uncertainties increased with more repeats, whereas with z-standardization, both uncertainties when up?

### Attention for the dense units

`nb_reps` = 3, `K_test` = 100, best `ll` = 1e-2, `no of dense units` = 3, `attncfg` = 'oninput'

```python
RMSE Test mean:  19.16054
RMSE Test stdev:  0.151641
Aleatoric uncertainty (exp from logvar):  16.679714
Epistemic uncertainty (var of means): 1.124963
R^2: 0.176519
```

## TODO!

* cross-validation from `skilearn`
* Make a publically available example file
* add pip-requirements.txt
* try the MCBN for regression, Teye et al. (2018): "Bayesian Uncertainty Estimation for Batch Normalized Deep Networks" https://arxiv.org/abs/1802.06455
* add proper error/loss metrics, e.g. (predictive) log likelihood ((P)LL) [https://github.com/yaringal/DropoutUncertaintyExps], Continuous Ranked Probability Score (CRPS) [https://arxiv.org/abs/1802.06455], entropy and mutual information [https://arxiv.org/abs/1803.08533]
