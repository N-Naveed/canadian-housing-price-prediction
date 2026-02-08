# Canadian Housing Price Prediction Using Linear Regression

This repository contains a research project that examines how well a simple linear regression model can predict housing prices across Canada. The full research paper is included below, and the Jupyter notebook with all code and analysis can be found in the `notebook/` folder. The project demonstrates applied data analysis, statistical reasoning, and research maturity.

---

## How to Read This Repository

- **README.md** contains a research-paper-style write-up explaining the goal, methodology, results, and limitations of the project.
- **`notebook/Canadian_Housing_Prediction.ipynb`** contains the full analysis, including all the code, preprocessing steps, visualizations, and explanatory markdown.
- **`data/Canadian House Prices.csv`** contains the dataset used in the analysis.
- The project prioritizes **interpretability and statistical reasoning** over maximum predictive accuracy.

---

## Abstract

Housing prices vary across Canada due to many overlapping factors, making them challenging to predict. In this project, I explore how well a simple linear regression model can predict housing prices using a publicly available Kaggle dataset of listings from many Canadian cities. The goal of this project is not to maximize accuracy, but rather to remain easily understandable while still being statistically sound.  

The model is trained using various features such as structural characteristics, demographic information, and regional indicators, with steps taken to prevent data leakage and ensure reproducibility. Furthermore, housing prices are log-transformed to account for their strong skew; however, performance is evaluated using both log-based metrics and real dollar measures to ensure the results are statistically meaningful while also interpretable in real-world applications.  

The results show that the model is stable across all five cross-validation folds, with an average R² score of approximately **0.638**. While the model produces larger errors for higher-priced homes, this is expected and reflects the scale of housing prices rather than a weakness of the model. Overall, this project demonstrates that even simple models can provide useful insight into complex datasets when applied carefully.

---

## Introduction

Housing prices are crucial when it comes to economic planning, investment decisions, and household decisions. For this reason, predicting these prices has become a popular application of data analysis and machine learning. However, this has proven to be quite challenging, as housing markets are complex, due to them being affected by a variety of features. These include, but are not limited to, property characteristics, location, and regional conditions.

Many approaches taken to predict housing prices use complex models that aim to maximize accuracy. However, in order to do this, these models often end up sacrificing interpretability, thus making it harder for less experienced users to understand their models and results. On the other hand, simpler models, as this project uses, allow for more transparency but may not be able to capture all the complexity found in real-world housing data.

This project intentionally aims to be interpretable rather than highly accurate. Through the use of linear regression, the goal is to better understand the influence that housing features and regional information have on prices. Through this approach, the project highlights not only the usefulness of simple models when applied to a large and complex housing dataset, but also their limitations.

---

## Dataset Description

This project uses a publicly available Kaggle dataset, which contains the listings of residential homes from major cities all across Canada. Each entry represents an individual property listing and includes various key information such as the price, number of bedrooms and bathrooms, city, province, and local population.

In addition, the dataset covers a wide range of Canadian housing markets, resulting in substantial variation in both prices and property characteristics. The data also include certain entries that may not be suitable for the model, and thus the following steps were taken: duplicate listings were removed, observations with missing key values were excluded, and extreme price outliers within the top 1% of listings were filtered out. All these steps were taken to improve the stability of the model and ensure that the dataset was suitable for linear regression.

---

## Methodology

Linear regression relies on several key assumptions, including a linear relationship between predictors and the target variable, independence of observations, and constant variance of residuals. To ensure that these requirements were satisfied, housing prices were modeled in log space. This allowed for the skewness in the data to be reduced, and helped stabilize residual variance. Residual plots were examined to assess how model errors behaved across the price distribution. While some heteroskedasticity remains, particularly at higher price levels, this behavior is expected in real-world housing data and does not invalidate the model’s use as an interpretable baseline.

Before the model was trained, various preprocessing and feature engineering steps were taken in an effort to improve its interpretability and stability. Key features provided in the dataset were retained, while others were added to further improve the model’s performance. These added features included one that represented the total rooms in the property, to capture its overall size, and the other was the average housing price in each province. Importantly, the province-average feature was calculated using only the training set and then mapped to both the training and test sets, ensuring that no information from the test set leaked into the model during training.

Furthermore, categorical variables, such as city and province, were one-hot encoded after the train-test split to allow them to be evaluated while avoiding potential data leakage. Finally, both the population and the target variable (price) were log-transformed after the train–test split to account for their skewed distributions. All in all, these transformations helped improve the performance of the linear regression model while ensuring that it remained easy to understand.

---

## Model and Training

After completing the preprocessing and feature engineering steps, a linear regression model was trained in log space to predict the housing prices using the features. Before training, the dataset was first split 80/20 into a training set and a testing set, with a fixed random seed to ensure reproducibility. Once this was done, categorical variables were one-hot encoded so that they too could be evaluated. Finally, the matrices were aligned so that both sets shared the same feature space.

Once all the data was prepared, the model was trained on the training set and then tested on the testing data. However, rather than just doing this on one split and basing the assessment of the model solely upon those results, a five-fold cross-validation was performed on the training data, with the R² score being the metric for evaluation. This approach allowed not only for the model's accuracy to be assessed, but also for its consistency across different subsets, ensuring that it was stable.

---

## Evaluation and Results

The model's performance was evaluated using various metrics, both in log space and in real-world dollars. The standard regression metrics, such as R² and RMSE, were computed in log space to assess the overall fit of the model, as it was trained on log-transformed prices. This approach ensures that the evaluation metrics are consistent with the model’s objective function and underlying assumptions. However, to ensure that the model was interpretable, further error metrics were calculated in dollar terms. These included median absolute error, mean absolute error, and mean absolute percentage error. Using both logarithmic and dollar-based metrics provides a more complete understanding of model performance, balancing statistical validity with real-world interpretability.

The model achieved an R² score of approximately **0.6227** in log space, with RMSE of **0.4283**. Furthermore, the R² score was computed across a five-fold cross-validation, with a mean R² of approximately **0.6379** and a low standard deviation of **0.0069**, indicating that the results remained consistent throughout. Evaluation in dollar terms showed larger errors for higher-priced homes, which is expected as the price range widens significantly in the upper end of the market. Median absolute error was **$145,971.58**, mean absolute error was **$483,029.32**, and mean absolute percentage error was **36.29%**. Overall, the results suggest that while the model is not highly accurate, it is still able to provide a stable and understandable baseline for housing price patterns across Canada.

---

## Error Analysis

To better understand how the model performs across the housing market, an error analysis was conducted. Absolute prediction errors, in real dollar terms, were measured for five different price ranges, from very low to very high. The results indicate that lower- and mid-priced homes were predicted with relatively small errors, whereas higher-priced homes had larger errors. Residual plots support this observation, showing an increasing spread in prediction errors as actual prices increase. Log-residual plots revealed a fan-shaped pattern, indicating heteroskedasticity, which is partially reduced by the log transformation.

The analysis highlights that the model performs more reliably when evaluated in relative terms rather than absolute dollar values. These findings suggest that more flexible models or segmented approaches could further improve performance, particularly for high-priced properties.

---

## Limitations and Future Work

Although the linear regression model provides an interpretable baseline, it has limitations. Housing prices are influenced by complex, non-linear relationships that a simple linear model cannot fully capture. High-priced homes remain challenging to predict, and aggregated regional features may overlook local variations. Province-level average prices improve interpretability but may slightly inflate predictive strength.

Future work could explore more flexible models, such as regularized regression or tree-based methods, and incorporate additional features like property size or neighborhood ratings. These approaches may improve performance while maintaining interpretability.

---

## Conclusion

This project explored the use of a simple linear regression model in predicting housing prices across Canada using a publicly available dataset. By prioritizing interpretability over maximum accuracy, the model provides clear insight into how basic property features and regional factors influence housing prices. Although higher-priced homes are more challenging to predict, this reflects the complexity and scale of the housing market rather than poor model design. Overall, the results show that even simple, well-applied models can offer meaningful understanding of complex real-world data.

---
   
## Reproducibility

To reproduce this analysis:

1. Download the dataset *"Canadian House Prices for Top Cities"* from Kaggle.
2. Place the CSV file in the `data/` folder
3. Install the required Python packages:
  ```bash
   pip install -r requirements.txt
  ```
4. Run the notebook from top to bottom using Python 3.x.
5. A fixed random seed (`SEED = 42`) ensures reproducible train-test splits and cross-validation results.

---

## Acknowledgements

The dataset used in this project, "Canadian house prices for top cities," was sourced from [Kaggle](https://www.kaggle.com/datasets/jeremylarcher/canadian-house-prices-for-top-cities/code) and credited to its original contributors. This project was completed independently.
