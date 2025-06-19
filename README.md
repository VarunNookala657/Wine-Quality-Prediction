# Wine Quality Prediction using Tree Ensemble Methods

## Overview
This project focuses on predicting wine quality based on its physicochemical properties using various tree ensemble machine learning methods. It explores the effectiveness and efficiency of different models, including Gradient Boosting, XGBoost, Random Forest, AdaBoost, and Extremely Randomized Trees, on a small and noisy dataset of red and white wines.

## Project Contents
* **Analysis Code & Data Loading:** `Data_Mining_Final_Project_Submission_Code.ipynb` - The Google Colab notebook containing all the Python code for data loading (directly from public URLs), preprocessing, model training, evaluation, and hyperparameter tuning.
    * **Click here to open and run the notebook directly in Google Colab:** https://github.com/VarunNookala657/Wine-Quality-Prediction/blob/main/Data_Mining_Final_Project_Submission_Code.ipynb

---

## Introduction

Assessing the quality of wine is crucial for both consumers and manufacturers. Certification of product quality is being used by industries worldwide to boost sales and enhance their market value. Traditional methods involved testing the quality of wine at the end of production, which was time-consuming and required significant resources. This included the need for various human experts to assess product quality, making the process very expensive. However, relying solely on human experts for wine quality assessment can be challenging as each individual has their own subjective opinion.

Wine quality is also usually calculated using a traditional method called sensory science, where in a blindfolded sommelier tastes a wine and rates it. This method is clearly subjective, so we are interested investigating if there is a more scientific approach based on objective qualities and components of the wine. In addition, we are particularly interested in investigating the difference between tree ensemble methods as prediction tools. These machine learning tools have been on the forefront of data science and we want to enhance our knowledge of them by assessing them against each other in this environment. Hence, we combine these two problems and compare 5 different tree ensemble methods based on performance and time, and also explore hyper parameter tuning for these models.

## Analysis Approach
The project employs a machine learning pipeline involving:
1.  **Data Loading:** The necessary Red and White Wine Quality datasets are loaded directly from their UCI Machine Learning Repository URLs within the notebook.
2.  **Data Preprocessing:** Combining red and white wine datasets, adding a 'type' column, and preparing features for modeling.
3.  **Model Selection:** Comparing five tree ensemble methods:
    * AdaBoost
    * Random Forest
    * Gradient Boosting
    * XGBoost
    * Extremely Randomized Trees (Extra Trees)
4.  **Hyperparameter Tuning:** Exploring optimal parameters for selected models to enhance performance.
5.  **Model Evaluation:** Assessing models based on performance metrics (e.g., RMSE) and training time.

## Results: Comparing the Models

Based on the analysis, here's a summary of the model comparisons:

* **Extra Random Trees (ERT):** Performed the best with the lowest RMSE (Root Mean Squared Error) and shortest time to train, making it the most efficient model in these circumstances.
* **Gradient Boosting and XGBoost:** Also exhibited good performance with relatively low RMSEs, but required longer training times compared to ERT.
* **Random Forest:** Performed similarly to Gradient Boosting and XGBoost in terms of RMSE, but demonstrated faster training times.
* **AdaBoost:** Had the highest RMSE and the longest training time, indicating it was the least effective model for this dataset.

**Overall, it seems that bagging models (Random Forest and ERT) were more effective than boosting models (AdaBoost, Gradient Boosting, and XGBoost) on this particular dataset, which was small and noisy. Bagging models are generally more robust to noise, while boosting models may overfit in such situations.**

## Conclusion

In terms of quality of wine, we were able to create several relatively powerful models using tree ensemble methods. We especially uncovered alcohol content of wine as the sole strong predictor for its quality as determined by wine experts. From our results we see that with a noisy, dense, and relatively small dataset, XGBoost does not enjoy the benefits of its many extra features, and instead Gradient Boost outperforms it. Even still, Extremely Randomized Trees is shown to be the best, and most efficient, model in these circumstances.

---

## How to Reproduce the Analysis
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YourUsername]/Wine-Quality-Prediction.git
    cd Wine-Quality-Prediction
    ```
2.  **Open in Google Colab:**
    * Navigate to the `notebooks/` folder (or the root if you didn't create a folder) and click on `Data_Mining_Final_Project_Submission_Code.ipynb`.
    * Click the "Open in Colab" button at the top of the GitHub preview page to open it directly in Google Colab. The notebook will automatically download the necessary datasets from their online URLs.

## Technologies Used
* Python
* Pandas (for data manipulation)
* NumPy (for numerical operations)
* Scikit-learn (for machine learning models and utilities)
* XGBoost (for eXtreme Gradient Boosting)
* LightGBM (if used in your notebook, otherwise remove)
* Matplotlib / Seaborn (for visualization, if used)

## Author
Varun Tej Nookala
