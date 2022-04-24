# Mini-Project
<div>SC1015 - Mini Project</div)>

<h3>Suicide-Rate-Prediction</h3>

<h2>Problem Formulation</h2>
<div>How different factors, (age group, GDP, income level, country and region etc.)  affecting suicide rates between Year 1985 to 2016?</div>


<h2>DATA SET (master_2.csv)</h2>
<div>In this project, we are using the 2 relating datasets found in Kaggle.
The first dataset is listing the SUICIDE_NO with respect to variables such as COUNTRY, GENDER, AGE GROUP, GDP and Generation.
In view of the limited variables in the first dataset, we looked for the second dataset which add on the data with COUNTRY REGION and INCOME LEVEL.</div>


<h2>IMOPORT THE NECESSARY LIBRARIES</h2>
<div>import numpy as np</div>
<div>import pandas as pd</div>
<div>import seaborn as sb</div>
<div>import matplotlib.pyplot as plt # we only need pyplot</div>
<div>sb.set()</div>
<div><div>from sklearn.linear_model import LinearRegression</div>
<div>from sklearn.tree import DecisionTreeClassifier</div>
<div>from sklearn.model_selection import train_test_split</div>
<div>from sklearn.metrics import confusion_matrix</div>
<div>from sklearn.tree import plot_treefrom sklearn.model_selection import train_test_split</div>
<div>from sklearn.utils import resample</div>
<div>from sklearn.ensemble import RandomForestClassifier</div>
<div>from sklearn.model_selection import GridSearchCV</div>

<h2>CHECKING FOR OUTLIERS</h2>
<div>Outliers can mislead the training process of machine learning algorithms takes longer training times, less accurate and poor results.</div>

<h2>INVESTIGATING CORRELATION AND VISUALIZING WITH HEATMAP</h2>
<div>Correlation measures the strength or of relationship between two variables as below:
  <li>POPULATION</li>
  <li>HDI_FOR_YEAR</li>
  <li>GDP_FOR_YEAR_($)</li>
  <li>GDP_PER_CAPITA_($)</li>
  <li>SUICIDES_NO</li>
 </div>

<h2>DATA PREPROCESSING</h2>
<div>This preprocessing includes:- Data Cleaning (Remove NULL Values for the numerical data)</div>

<h2>SPLITTING THE DATASET</h2>
<div>As we work with datasets, a machine learning algorithm works in two stages. We have split the data around 30%-70% between testing and training stages.
Model use for Machine Learning:
  <li>Linear Regression</li>
  <li>Decision Tree</li>
  <li>Random Forest</li>
  <li>Grid Search</li>
</div>

<h2>VARIABLES USED TO PREDICT THE SUICIDE RATE</h2>
<div>
Converted SUICIDE_NO → SUICIDE_RATE of ‘High’ or ‘Low’ based on Mean
Categorical Predictor: Sex, Age, Admin Region
Numerical Predictor: GDP, Population
</div>

<h2>LINEAR REGRESSION</h2>
<div>Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x).
PERFORMANCE EVALUATION
Using Linear Regression getting Accuracy/Score:-
Accuracy for Train Data Set: 0.3912
</div>

<h2>DECISION TREE</h2> 
Using OneHotEncoding to perform task
PERFORMANCE EVALUATION
Using Decision Tree getting Accuracy/Score:-
Accuracy for Train Data Set: 0.9287
Accuracy for Test Data Set: 0.9278

<h2>RANDOM FOREST</h2>
PERFORMANCE EVALUATION
Using Random Forest getting Accuracy/Score:-
Accuracy for Train Data Set: 0.9204
Accuracy for Test Data Set: 0.9183

<h2>GRID SEARCH</h2>
<PERFORMANCE EVALUATION
Using Grid Search getting Accuracy/Score:-
Accuracy for Train Data Set: 0.9697
Accuracy for Test Data Set: 0.9600

<h2>CONTRIBUTOR</h2>
  <li>Jin Fuyi</li>
  <li>Teng Fanru</li>
  <li>Ku Qian Yu</li>
