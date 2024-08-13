# polynomial Regression

Configure the project. Indeed you create a dataset in csv format.


```python
! rm -rf *.csv
! unzip ./dataset/archive.zip
! mv dataset.csv data.csv

```

    Archive:  ./dataset/archive.zip
      inflating: dataset.csv             


Import needed libraries


```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

%matplotlib inline
```

Read data from data.csv using pandas and store in data frame structure. Also shuffle data to have uniform distribution. 


```python
df = pd.read_csv("data.csv")
df.head()
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>NationalITy</th>
      <th>PlaceofBirth</th>
      <th>StageID</th>
      <th>GradeID</th>
      <th>SectionID</th>
      <th>Topic</th>
      <th>Semester</th>
      <th>Relation</th>
      <th>raisedhands</th>
      <th>VisITedResources</th>
      <th>AnnouncementsView</th>
      <th>Discussion</th>
      <th>ParentAnsweringSurvey</th>
      <th>ParentschoolSatisfaction</th>
      <th>StudentAbsenceDays</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>F</td>
      <td>KW</td>
      <td>KuwaIT</td>
      <td>MiddleSchool</td>
      <td>G-07</td>
      <td>A</td>
      <td>English</td>
      <td>F</td>
      <td>Father</td>
      <td>19</td>
      <td>30</td>
      <td>26</td>
      <td>19</td>
      <td>Yes</td>
      <td>Bad</td>
      <td>Above-7</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>F</td>
      <td>Lybia</td>
      <td>Lybia</td>
      <td>MiddleSchool</td>
      <td>G-07</td>
      <td>B</td>
      <td>Biology</td>
      <td>F</td>
      <td>Mum</td>
      <td>10</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>No</td>
      <td>Good</td>
      <td>Above-7</td>
      <td>L</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>Jordan</td>
      <td>Palestine</td>
      <td>MiddleSchool</td>
      <td>G-07</td>
      <td>A</td>
      <td>Biology</td>
      <td>F</td>
      <td>Mum</td>
      <td>78</td>
      <td>91</td>
      <td>50</td>
      <td>40</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Under-7</td>
      <td>H</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>Palestine</td>
      <td>Jordan</td>
      <td>MiddleSchool</td>
      <td>G-06</td>
      <td>A</td>
      <td>English</td>
      <td>S</td>
      <td>Mum</td>
      <td>92</td>
      <td>31</td>
      <td>42</td>
      <td>27</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Under-7</td>
      <td>H</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>Palestine</td>
      <td>Palestine</td>
      <td>MiddleSchool</td>
      <td>G-07</td>
      <td>A</td>
      <td>Biology</td>
      <td>S</td>
      <td>Father</td>
      <td>89</td>
      <td>92</td>
      <td>89</td>
      <td>83</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Under-7</td>
      <td>H</td>
    </tr>
  </tbody>
</table>
</div>




```python
categorical_attr = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class']

le = LabelEncoder()
df[categorical_attr] = df[categorical_attr].apply(le.fit_transform, axis=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>NationalITy</th>
      <th>PlaceofBirth</th>
      <th>StageID</th>
      <th>GradeID</th>
      <th>SectionID</th>
      <th>Topic</th>
      <th>Semester</th>
      <th>Relation</th>
      <th>raisedhands</th>
      <th>VisITedResources</th>
      <th>AnnouncementsView</th>
      <th>Discussion</th>
      <th>ParentAnsweringSurvey</th>
      <th>ParentschoolSatisfaction</th>
      <th>StudentAbsenceDays</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>30</td>
      <td>26</td>
      <td>19</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>9</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>78</td>
      <td>91</td>
      <td>50</td>
      <td>40</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>92</td>
      <td>31</td>
      <td>42</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>7</td>
      <td>7</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>89</td>
      <td>92</td>
      <td>89</td>
      <td>83</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# summarize data
df.describe() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gender</th>
      <th>NationalITy</th>
      <th>PlaceofBirth</th>
      <th>StageID</th>
      <th>GradeID</th>
      <th>SectionID</th>
      <th>Topic</th>
      <th>Semester</th>
      <th>Relation</th>
      <th>raisedhands</th>
      <th>VisITedResources</th>
      <th>AnnouncementsView</th>
      <th>Discussion</th>
      <th>ParentAnsweringSurvey</th>
      <th>ParentschoolSatisfaction</th>
      <th>StudentAbsenceDays</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.635417</td>
      <td>4.345833</td>
      <td>4.397917</td>
      <td>1.345833</td>
      <td>2.906250</td>
      <td>0.472917</td>
      <td>5.256250</td>
      <td>0.489583</td>
      <td>0.410417</td>
      <td>46.775000</td>
      <td>54.797917</td>
      <td>37.918750</td>
      <td>43.283333</td>
      <td>0.562500</td>
      <td>0.608333</td>
      <td>0.602083</td>
      <td>1.143750</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.481815</td>
      <td>2.469265</td>
      <td>2.628334</td>
      <td>0.603732</td>
      <td>2.464267</td>
      <td>0.612411</td>
      <td>3.388388</td>
      <td>0.500413</td>
      <td>0.492423</td>
      <td>30.779223</td>
      <td>33.080007</td>
      <td>26.611244</td>
      <td>27.637735</td>
      <td>0.496596</td>
      <td>0.488632</td>
      <td>0.489979</td>
      <td>0.846312</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.750000</td>
      <td>20.000000</td>
      <td>14.000000</td>
      <td>20.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>50.000000</td>
      <td>65.000000</td>
      <td>33.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>75.000000</td>
      <td>84.000000</td>
      <td>58.000000</td>
      <td>70.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>2.000000</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>11.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>100.000000</td>
      <td>99.000000</td>
      <td>98.000000</td>
      <td>99.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>



Print the histogram chart of data


```python
print(df.columns)
```

    Index(['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
           'SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands',
           'VisITedResources', 'AnnouncementsView', 'Discussion',
           'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
           'StudentAbsenceDays', 'Class'],
          dtype='object')



```python
viz = df[['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class']]
fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(20, 8))

axes = axes.flatten()

for i, column in enumerate(viz.columns):
    viz[column].hist(ax=axes[i])
    axes[i].set_title(column)

fig.delaxes(axes[17])
plt.tight_layout()
plt.show()
```


    
![png](polynomial_regression_files/polynomial_regression_11_0.png)
    


Print scatter chart of data to recognize the patterns of data. I draw them based on VisITedResources parameter.


```python
fig, axs = plt.subplots(3, 6, figsize=(15, 7))
fig.delaxes(axs[2, 4])
fig.delaxes(axs[2, 5])  # To remove extra empty plots since 3x6 = 18 but we only need 16 plots

columns = df.columns.drop('VisITedResources')
row, col = 0, 0

for column in columns:
    if col == 6:
        col = 0
        row += 1
    axs[row, col].scatter(df[column], df['VisITedResources'], color="blue")
    axs[row, col].set_ylabel(column)
    axs[row, col].set_xlabel('VisITedResources')
    col += 1

fig.suptitle("Each parameter Vs VisITedResources", fontsize=22)

plt.tight_layout()
plt.show()
```


    
![png](polynomial_regression_files/polynomial_regression_13_0.png)
    


Make the dataset minimal to work with it more easily


```python
df = df[['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']]
```


```python
# print(df)
train, test = train_test_split(df, test_size=0.20, random_state=42)
# test, evaluate = train_test_split(test, test_size=0.5, random_state=42)
```


```python

fig, ax1 = plt.subplots(1, figsize=(5, 5))

# Plot the original data
ax1.scatter(train["AnnouncementsView"], train["VisITedResources"], color="blue", label="Train")
ax1.scatter(test["AnnouncementsView"], test["VisITedResources"], color="red", label="Test")


# Set labels and title
ax1.set_ylabel("Chance of Admit")
ax1.set_xlabel("AnnouncementsView")
ax1.set_title("AnnouncementsView vs VisITedResources")
plt.tight_layout()
plt.show()
```


    
![png](polynomial_regression_files/polynomial_regression_17_0.png)
    


Find the best fitted line based on distribution of data. 

For better clarity and to make more sense, I will choose just one of the parameters based on VisITedResources for training. This is a mistake because the rest of the data also contains information and helps the learning process. However, for practice, instead of a multiple model, I want to use a single model which is clearer.


```python
train_x = np.asanyarray(train[['AnnouncementsView']])
train_y = np.asanyarray(train[['VisITedResources']])
poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly
```




    array([[1.000e+00, 1.300e+01, 1.690e+02],
           [1.000e+00, 0.000e+00, 0.000e+00],
           [1.000e+00, 7.200e+01, 5.184e+03],
           ...,
           [1.000e+00, 8.200e+01, 6.724e+03],
           [1.000e+00, 5.100e+01, 2.601e+03],
           [1.000e+00, 5.000e+00, 2.500e+01]])




```python
reg = linear_model.LinearRegression()
train_y_ = reg.fit(train_x_poly, train_y)


print("Coefficients:\t", reg.coef_)
print("Intercept:\t", reg.intercept_)
```

    Coefficients:	 [[ 0.          1.45712653 -0.00834579]]
    Intercept:	 [18.3228313]


Draw the data scatters plot and fitted polynomial line.


```python
plt.scatter(train[['AnnouncementsView']], train["VisITedResources"],  color='blue')
XX = np.arange(0.0, 100.0, 0.1)
yy = reg.intercept_[0]+ reg.coef_[0][1]*XX+ reg.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("AnnouncementsView")
plt.ylabel("VisITedResources")
```




    Text(0, 0.5, 'VisITedResources')




    
![png](polynomial_regression_files/polynomial_regression_23_1.png)
    


Testing model based on Test data. Measure the R2 and MSE.


```python
test_x = np.asanyarray(test[['AnnouncementsView']])
test_y = np.asanyarray(test[['VisITedResources']])

test_x_poly = poly.fit_transform(test_x)
test_y_ = reg.predict(test_x_poly)

print("Mean absolute error: %.2f" % mean_absolute_error(test_y_, test_y))
print("Residual sum of squares (MSE): %.2f" % mean_squared_error(test_y_, test_y))
print("R2-score: %.2f" % r2_score(test_y_, test_y))
```

    Mean absolute error: 22.35
    Residual sum of squares (MSE): 712.86
    R2-score: -0.98


R2-score is -0.98. So it means our model is so poor. Lets do it with simple regression also.


```python
train_y_ = reg.fit(train_x, train_y)

print("Coefficients:\t", reg.coef_)
print("Intercept:\t", reg.intercept_)
```

    Coefficients:	 [[0.72608861]]
    Intercept:	 [27.90304923]



```python
plt.scatter(train[['AnnouncementsView']], train["VisITedResources"],  color='blue')
plt.plot(train_x, reg.coef_[0][0]*train_x + reg.intercept_[0], "-r")
#               y = theta1 x + theta0
plt.ylabel("VisITedResources")
plt.xlabel("AnnouncementsView")
plt.show()
```


    
![png](polynomial_regression_files/polynomial_regression_28_0.png)
    



```python
test_x = np.asanyarray(test[['AnnouncementsView']])
test_y = np.asanyarray(test[['VisITedResources']])

test_y_ = reg.predict(test_x)

print("Mean absolute error: %.2f" % mean_absolute_error(test_y_, test_y))
print("Residual sum of squares (MSE): %.2f" % mean_squared_error(test_y_, test_y))
print("R2-score: %.2f" % r2_score(test_y_, test_y))
```

    Mean absolute error: 23.02
    Residual sum of squares (MSE): 715.26
    R2-score: -1.56


## As you see the polynomial has a better R2-score compare with linear but this data's need another solution. Regression method is not good for this type of problem! 

<font size="6" color="green">
Better types of problems that we can solve with polynomial regression are: Predicting GDP, Salary Based on Business Levels, House Prices, etc.
</font>

## Lets test other degrees for prediction:


```python
degrees = range(3, 10)
results = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    train_x_poly = poly.fit_transform(train_x)
    reg = linear_model.LinearRegression()
    reg.fit(train_x_poly, train_y)
    
    # The coefficients
    print(f'Degree: {degree}')
    print('Coefficients:', reg.coef_)
    print('Intercept:', reg.intercept_)
    
    # Plotting
    plt.scatter(train[['AnnouncementsView']], train["VisITedResources"], color='blue')
    XX = np.arange(0.0, 100.0, 0.1)
    yy = reg.intercept_[0]
    for power in range(1, degree + 1):
        yy += reg.coef_[0][power] * np.power(XX, power)
    plt.plot(XX, yy, label=f'Degree {degree}')
    
    # Evaluation
    test_x_poly = poly.fit_transform(test_x)
    test_y_pred = reg.predict(test_x_poly)
    
    mae = mean_absolute_error(test_y, test_y_pred)
    mse = mean_squared_error(test_y, test_y_pred)
    r2 = r2_score(test_y, test_y_pred)
    
    results.append((degree, mae, mse, r2))
    
    print("Mean absolute error: %.2f" % mae)
    print("Residual sum of squares (MSE): %.2f" % mse)
    print("R2-score: %.2f\n" % r2)
    
plt.xlabel("Announcements View")
plt.ylabel("Visited Resources")
plt.legend()
plt.show()

# Display summary results
print("\nSummary of Performance Metrics:")
for degree, mae, mse, r2 in results:
    print(f'Degree: {degree} | MAE: {mae:.2f} | MSE: {mse:.2f} | R2: {r2:.2f}')
```

    Degree: 3
    Coefficients: [[ 0.00000000e+00  1.87446936e+00 -2.03663236e-02  8.94008242e-05]]
    Intercept: [15.54568529]
    Mean absolute error: 22.58
    Residual sum of squares (MSE): 719.31
    R2-score: 0.30
    
    Degree: 4
    Coefficients: [[ 0.00000000e+00  2.01321173e+00 -2.75060802e-02  2.12359728e-04
      -6.70353033e-07]]
    Intercept: [14.98883662]
    Mean absolute error: 22.59
    Residual sum of squares (MSE): 718.26
    R2-score: 0.30
    
    Degree: 5
    Coefficients: [[ 0.00000000e+00  2.02327255e+00 -2.83155035e-02  2.36027012e-04
      -9.55781095e-07  1.20933364e-09]]
    Intercept: [14.96333848]
    Mean absolute error: 22.58
    Residual sum of squares (MSE): 718.13
    R2-score: 0.30
    
    Degree: 6
    Coefficients: [[ 0.00000000e+00  2.40704382e+00 -7.21893955e-02  2.15842312e-03
      -3.96908061e-05  3.63313236e-07 -1.27207598e-09]]
    Intercept: [14.29201797]
    Mean absolute error: 22.64
    Residual sum of squares (MSE): 720.17
    R2-score: 0.30
    
    Degree: 7
    Coefficients: [[ 0.00000000e+00 -2.92346324e-01  3.41454024e-01 -2.29254721e-02
       6.99873143e-04 -1.09337509e-05  8.46648541e-08 -2.57281734e-10]]
    Intercept: [17.78082524]
    Mean absolute error: 22.07
    Residual sum of squares (MSE): 698.40
    R2-score: 0.32
    
    Degree: 8
    Coefficients: [[ 0.00000000e+00 -2.44035337e+00  7.77661590e-01 -5.80382297e-02
       2.11557708e-03 -4.22533393e-05  4.70573086e-07 -2.73983399e-09
       6.49519189e-12]]
    Intercept: [19.72588503]
    Mean absolute error: 22.10
    Residual sum of squares (MSE): 698.68
    R2-score: 0.32
    
    Degree: 9
    Coefficients: [[ 0.00000000e+00  1.51759504e-04  3.93712019e-03  2.69722614e-02
      -2.45961801e-03  9.51963725e-05 -1.95043302e-06  2.21000675e-08
      -1.30929741e-10  3.16791901e-13]]
    Intercept: [19.69125765]
    Mean absolute error: 21.86
    Residual sum of squares (MSE): 689.39
    R2-score: 0.33
    



    
![png](polynomial_regression_files/polynomial_regression_33_1.png)
    


    
    Summary of Performance Metrics:
    Degree: 3 | MAE: 22.58 | MSE: 719.31 | R2: 0.30
    Degree: 4 | MAE: 22.59 | MSE: 718.26 | R2: 0.30
    Degree: 5 | MAE: 22.58 | MSE: 718.13 | R2: 0.30
    Degree: 6 | MAE: 22.64 | MSE: 720.17 | R2: 0.30
    Degree: 7 | MAE: 22.07 | MSE: 698.40 | R2: 0.32
    Degree: 8 | MAE: 22.10 | MSE: 698.68 | R2: 0.32
    Degree: 9 | MAE: 21.86 | MSE: 689.39 | R2: 0.33


<font size="6" color="cyan">
Based on the obtained results, the situation slightly improved, but none of them are good enough. Hence, we can confidently say that it is better not to train this dataset with this method and instead use another method, especially neural networks, to achieve better accuracy. This learning model is suitable for the dataset mentioned in the previous section. This notebook is solely for learning and to ensure the resulting outcome :)
</font>
