# Asteroid Classification for Impact Risk Assessment

This machine learning project aims to classify asteroids into either dangerous or not dangerous categories based on various features. The dataset used in this project contains information about asteroids such as size, velocity, distance from Earth, and other relevant parameters.

## Dataset
The dataset used in this project is sourced from NASA's Near Earth Object Program and contains information about known asteroids. The dataset can be accessed [here](https://www.kaggle.com/datasets/shrutimehta/nasa-asteroids-classification).

## Requirements
The following packages are required to run the code:
- pandas
- numpy
- scikit-learn
- lightgbm
- xgboost
- skopt
- matplotlib

## Code Structure
The code for this project is organized into the following parts:

- `input`: This folder contains the input data used for the project.

- `src`: This folder contains the source code for the project. It includes the data preprocessing, feature selection using mutual information classification, PCA for dimensionality reduction, and model training and evaluation.

- `model`: This folder contains the output model generated.

- `notebook`: This folder contains Jupyter Notebook files used for exploratory data analysis, feature selection, and model training.

## Methodology
1. Data Preprocessing: The dataset is loaded and cleaned by handling missing values and scaling the values as necessary.

2. Feature Selection: Mutual information classification is used to select the most informative features from the dataset. Three features are chosen based on their mutual information scores.

3. Dimensionality Reduction: Principal Component Analysis (PCA) is applied to compress the three selected features into two principal components that capture the maximum variance in the data.

4. Model Training: The stacked ensemble algorithm is employed, consisting of LightGBM, K-Nearest Neighbors (KNN), and Decision Tree classifiers. The models are trained on the compressed two-dimensional feature set.

5. Model Evaluation: The trained models are evaluated using a test dataset. Accuracy and F1 score are used as performance metrics to assess the models' classification performance.

6. Output Visualization: Visualize the decision boundary of the model.

## Results
The stacked ensemble model achieved outstanding performance on the test set, with an accuracy of 99% and an F1 score of 98% on test set.

## Conclusion
In conclusion, this project demonstrates how machine learning techniques can be utilized for classifying asteroids based on their impact risk. The stacked ensemble model, consisting of LightGBM, KNN, and Decision Tree classifiers, showcased remarkable accuracy and F1 score on the test set. The developed solution can aid in identifying potentially dangerous asteroids and support impact risk assessment efforts.
