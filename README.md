# Mini-Project
SC1015 - Mini Project

Suicide-Rate-Prediction
Problem Formulation
How different factors, (age group, GDP, income level, country and region etc.)  affecting suicide rates between Year 1985 to 2016?


DATA SET (master_2.csv)
In this project, we are using the 2 relating datasets found in Kaggle.
The first dataset is listing the SUICIDE_NO with respect to variables such as COUNTRY, GENDER, AGE GROUP, GDP and Generation.
In view of the limited variables in the first dataset, we looked for the second dataset which add on the data with COUNTRY REGION and INCOME LEVEL.


IMOPORT THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
sb.set()
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_treefrom sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

CHECKING FOR OUTLIERS
Outliers can mislead the training process of machine learning algorithms takes longer training times, less accurate and poor results.

INVESTIGATING CORRELATION AND VISUALIZING WITH HEATMAP
Correlation measures the strength or of relationship between two variables such as POPULATION/ HDI_FOR_YEAR/ GDP_FOR_YEAR_($)/ GDP_PER_CAPITA_($) and SUICIDES_NO

DATA PREPROCESSING
This preprocessing includes:- Data Cleaning (Remove NULL Values for the numerical data)

SPLITTING THE DATASET
As we work with datasets, a machine learning algorithm works in two stages. We have split the data around 30%-70% between testing and training stages.
Model use for Machine Learning - Linear Regression, Decision Tree, Random Forest, Grid Search

VARIABLES USED TO PREDICT THE SUICIDE RATE
Converted SUICIDE_NO → SUICIDE_RATE of ‘High’ or ‘Low’ based on Mean
Categorical Predictor: Sex, Age, Admin Region
Numerical Predictor: GDP, Population

LINEAR REGRESSION
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x).

PERFORMANCE EVALUATION
Using Linear Regression getting Accuracy/Score:-
Accuracy for Train Data Set: 0.3912

DECISION TREE 
Using OneHotEncoding to perform task
PERFORMANCE EVALUATION
Using Decision Tree getting Accuracy/Score:-
Accuracy for Train Data Set: 0.9287
Accuracy for Test Data Set: 0.9278

RANDOM FOREST
PERFORMANCE EVALUATION
Using Random Forest getting Accuracy/Score:-
Accuracy for Train Data Set: 0.9204
Accuracy for Test Data Set: 0.9183

GRID SEARCH
PERFORMANCE EVALUATION
Using Grid Search getting Accuracy/Score:-
Accuracy for Train Data Set: 0.9697
Accuracy for Test Data Set: 0.9600
