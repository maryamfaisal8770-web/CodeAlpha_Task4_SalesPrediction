# CodeAlpha Task 4 — Sales Prediction using Python

## Objective
Build a regression model that predicts sales from advertising expenditure and examine how the available advertising channels relate to sales.

## Dataset
`Advertising.csv` contains 200 observations with:
- TV advertising spend
- Radio advertising spend
- Newspaper advertising spend
- Sales (target)

The dataset also contained an `Unnamed: 0` index column, which was removed during cleaning because it is only an identifier and should not be used as a predictive feature.

## Method
1. Loaded the supplied dataset with Pandas.
2. Removed the unnecessary index column.
3. Converted the modeling columns to numeric values and removed rows with missing modeling values.
4. Selected TV, Radio and Newspaper as input features and Sales as the target.
5. Split the data into 80% training and 20% testing data using `random_state=42`.
6. Trained a multiple Linear Regression model using scikit-learn.
7. Evaluated the model with MAE, RMSE and R².
8. Examined correlations and model coefficients to understand feature impact.
9. Created visualizations for each advertising channel and model predictions.

## Results
- Test MAE: **1.461**
- Test RMSE: **1.782**
- Test R²: **0.899**

The fitted model coefficients were:
- TV: **0.045**
- Radio: **0.189**
- Newspaper: **0.003**

Within this linear model, all three coefficients are positive. The coefficient magnitudes indicate that TV and Radio have larger estimated marginal associations with sales than Newspaper after accounting for the other channels. These coefficients should be interpreted as associations in this observational dataset, not proof that increasing a channel will automatically cause the same increase in sales.

## Important dataset limitation
The supplied dataset contains advertising channels but does **not** contain target-segment, platform, customer, or date/time columns. Therefore, the analysis does not invent those variables. It focuses on the factors that are actually present in the dataset.

Also, this is a regression-based sales prediction task rather than a time-series forecast because no date column is provided.

## Marketing insight
The model can be used as a simple baseline for estimating expected sales from a proposed advertising mix. The feature-impact analysis can help compare channels, while actual budget decisions should also consider campaign cost, reach, conversion, and business constraints that are not included in this dataset.

## Files
- `Advertising.csv` — supplied dataset
- `sales_prediction.py` — complete Python source code
- `sales_prediction.ipynb` — executed Jupyter Notebook
- `feature_impact_summary.csv` — feature correlations and model coefficients
- `test_predictions.csv` — actual and predicted test-set sales
- `tv_vs_sales.png`, `radio_vs_sales.png`, `newspaper_vs_sales.png` — channel plots
- `actual_vs_predicted.png` — model evaluation plot
- `feature_coefficients.png` — coefficient comparison
- `VIDEO_EXPLANATION.md` — short guide for the required video explanation
- `requirements.txt` — required Python packages
