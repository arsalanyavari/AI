# K Nearest Neighbor Classification

Configure the project. Indeed you create a dataset in csv format.


```python
! rm -rf *.csv
! unzip ./dataset/archive.zip
! mv Graduate.csv data.csv

```

    Archive:  ./dataset/archive.zip
      inflating: Graduate.csv            
      inflating: play.csv                


Import needed libraries


```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import scikitplot as skplt

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
      <th>Serial No.</th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>362</td>
      <td>334</td>
      <td>116</td>
      <td>4</td>
      <td>4.0</td>
      <td>3.5</td>
      <td>9.54</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>74</td>
      <td>314</td>
      <td>108</td>
      <td>4</td>
      <td>4.5</td>
      <td>4.0</td>
      <td>9.04</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>375</td>
      <td>315</td>
      <td>105</td>
      <td>2</td>
      <td>2.0</td>
      <td>2.5</td>
      <td>7.65</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>156</td>
      <td>312</td>
      <td>109</td>
      <td>3</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>8.69</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>105</td>
      <td>326</td>
      <td>112</td>
      <td>3</td>
      <td>3.5</td>
      <td>3.0</td>
      <td>9.05</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.shape)
print("\n" + "#" * 80 + "\n")
print(df.columns)
print("\n" + "#" * 80 + "\n")
print(df.dtypes)
```

    (500, 9)
    
    ################################################################################
    
    Index(['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
           'LOR ', 'CGPA', 'Research', 'Admit'],
          dtype='object')
    
    ################################################################################
    
    Serial No.             int64
    GRE Score              int64
    TOEFL Score            int64
    University Rating      int64
    SOP                  float64
    LOR                  float64
    CGPA                 float64
    Research               int64
    Admit                  int64
    dtype: object



```python
df['Admit'].value_counts()
```




    Admit
    1    463
    0     37
    Name: count, dtype: int64




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
      <th>Serial No.</th>
      <th>GRE Score</th>
      <th>TOEFL Score</th>
      <th>University Rating</th>
      <th>SOP</th>
      <th>LOR</th>
      <th>CGPA</th>
      <th>Research</th>
      <th>Admit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.00000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>250.500000</td>
      <td>316.472000</td>
      <td>107.192000</td>
      <td>3.114000</td>
      <td>3.374000</td>
      <td>3.48400</td>
      <td>8.576440</td>
      <td>0.560000</td>
      <td>0.926000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>144.481833</td>
      <td>11.295148</td>
      <td>6.081868</td>
      <td>1.143512</td>
      <td>0.991004</td>
      <td>0.92545</td>
      <td>0.604813</td>
      <td>0.496884</td>
      <td>0.262033</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>290.000000</td>
      <td>92.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>6.800000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>125.750000</td>
      <td>308.000000</td>
      <td>103.000000</td>
      <td>2.000000</td>
      <td>2.500000</td>
      <td>3.00000</td>
      <td>8.127500</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>250.500000</td>
      <td>317.000000</td>
      <td>107.000000</td>
      <td>3.000000</td>
      <td>3.500000</td>
      <td>3.50000</td>
      <td>8.560000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>375.250000</td>
      <td>325.000000</td>
      <td>112.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.00000</td>
      <td>9.040000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>500.000000</td>
      <td>340.000000</td>
      <td>120.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.00000</td>
      <td>9.920000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Print the histogram chart of data


```python
print(df.columns)
```

    Index(['Serial No.', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
           'LOR ', 'CGPA', 'Research', 'Admit'],
          dtype='object')



```python
viz = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR ', 'CGPA', 'Research', 'Admit']]
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 7))

axes = axes.flatten()

for i, column in enumerate(viz.columns):
    viz[column].hist(ax=axes[i])
    axes[i].set_title(column)


plt.tight_layout()
plt.show()
```


    
![png](knn_files/knn_12_0.png)
    



```python
# print(df)
train, test = train_test_split(df, test_size=0.20, random_state=42)
# test, evaluate = train_test_split(test, test_size=0.5, random_state=42)
```

# Preprocessing 


```python
train_x = np.asanyarray(train[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']])
train_y = np.asanyarray(train['Admit'])
```


```python
test_x = np.asanyarray(test[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']])
test_y = np.asanyarray(test['Admit'])
```


```python
train_x = StandardScaler().fit(train_x).transform(train_x.astype(float))
test_x = StandardScaler().fit(test_x).transform(test_x.astype(float))
```

# Train Phase


```python
k = 5
model = KNeighborsClassifier(n_neighbors=k).fit(train_x, train_y)
test_y_ = model.predict(test_x)
```

## Evaluation


```python

print("Train Accuracy", accuracy_score(train_y, model.predict(train_x)))
print("Test Accuracy", accuracy_score(test_y, test_y_))

print("Test-set Precision", precision_score(test_y, test_y_))
print("Test-set Recall", recall_score(test_y, test_y_))
print("Test-set F1-Score", f1_score(test_y, test_y_))
```

    Train Accuracy 0.9525
    Test Accuracy 0.95
    Test-set Precision 0.9484536082474226
    Test-set Recall 1.0
    Test-set F1-Score 0.9735449735449735



```python
print("Logistic Regression Classification Report:")
print(classification_report(test_y, test_y_))

print("Logistic Regression Confusion Matrix:")
skplt.metrics.plot_confusion_matrix(test_y, test_y_)
```

    Logistic Regression Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      0.38      0.55         8
               1       0.95      1.00      0.97        92
    
        accuracy                           0.95       100
       macro avg       0.97      0.69      0.76       100
    weighted avg       0.95      0.95      0.94       100
    
    Logistic Regression Confusion Matrix:





    <Axes: title={'center': 'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>




    
![png](knn_files/knn_22_2.png)
    


For imbalanced data like the one you've provided, the F1-score is often a better metric for evaluation. This is because the F1-score considers both precision and recall, providing a balance between the two. In imbalanced datasets, precision and recall alone might not give a full picture of the model performance. 


```python
best_f1score = 0
best_model = None

for k in range(1, 20, 2):
    model = KNeighborsClassifier(n_neighbors=k).fit(train_x, train_y)
    test_y_ = model.predict(test_x)
    f1score = f1_score(test_y, test_y_)

    if f1score > best_f1score:
        best_f1score = f1score
        best_model = model

print(f"Best f1-score: {best_f1score}")
print(f"Best Model: {best_model}")
```

    Best f1-score: 0.9735449735449735
    Best Model: KNeighborsClassifier(n_neighbors=3)



```python
test_y_ = best_model.predict(test_x)

print("Logistic Regression Classification Report:")
print(classification_report(test_y, test_y_))

print("Logistic Regression Confusion Matrix:")
skplt.metrics.plot_confusion_matrix(test_y, test_y_)
```

    Logistic Regression Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      0.38      0.55         8
               1       0.95      1.00      0.97        92
    
        accuracy                           0.95       100
       macro avg       0.97      0.69      0.76       100
    weighted avg       0.95      0.95      0.94       100
    
    Logistic Regression Confusion Matrix:





    <Axes: title={'center': 'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>




    
![png](knn_files/knn_25_2.png)
    

