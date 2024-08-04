# Support Vector Machine Classification

Configure the project. Indeed you create a dataset in csv format.


```python
! rm -rf *.xlsx
! unzip ./dataset/archive.zip
! mv *.xlsx data.xlsx

```

    Archive:  ./dataset/archive.zip
      inflating: Student-Employability-Datasets.xlsx  


Import needed libraries


```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, jaccard_score
import scikitplot as skplt

%matplotlib inline
```

Read data from data.csv using pandas and store in data frame structure. Also shuffle data to have uniform distribution. 


```python
df = pd.read_excel("data.xlsx")
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
      <th>Name of Student</th>
      <th>GENERAL APPEARANCE</th>
      <th>MANNER OF SPEAKING</th>
      <th>PHYSICAL CONDITION</th>
      <th>MENTAL ALERTNESS</th>
      <th>SELF-CONFIDENCE</th>
      <th>ABILITY TO PRESENT IDEAS</th>
      <th>COMMUNICATION SKILLS</th>
      <th>Student Performance Rating</th>
      <th>CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Student 2428</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>Employable</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Student 1548</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>LessEmployable</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Student 882</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>LessEmployable</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Student 332</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>LessEmployable</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Student 2244</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>LessEmployable</td>
    </tr>
  </tbody>
</table>
</div>



## Preprocessing


```python
print(df.shape)
print(df.columns)
print(df.dtypes)
```

    (2982, 10)
    Index(['Name of Student', 'GENERAL APPEARANCE', 'MANNER OF SPEAKING',
           'PHYSICAL CONDITION', 'MENTAL ALERTNESS', 'SELF-CONFIDENCE',
           'ABILITY TO PRESENT IDEAS', 'COMMUNICATION SKILLS',
           'Student Performance Rating', 'CLASS'],
          dtype='object')
    Name of Student               object
    GENERAL APPEARANCE             int64
    MANNER OF SPEAKING             int64
    PHYSICAL CONDITION             int64
    MENTAL ALERTNESS               int64
    SELF-CONFIDENCE                int64
    ABILITY TO PRESENT IDEAS       int64
    COMMUNICATION SKILLS           int64
    Student Performance Rating     int64
    CLASS                         object
    dtype: object



```python
df['CLASS'].value_counts()
```




    CLASS
    Employable        1729
    LessEmployable    1253
    Name: count, dtype: int64




```python
df = df.drop('Name of Student', axis=1)
categorical_attr = ['CLASS']

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
      <th>GENERAL APPEARANCE</th>
      <th>MANNER OF SPEAKING</th>
      <th>PHYSICAL CONDITION</th>
      <th>MENTAL ALERTNESS</th>
      <th>SELF-CONFIDENCE</th>
      <th>ABILITY TO PRESENT IDEAS</th>
      <th>COMMUNICATION SKILLS</th>
      <th>Student Performance Rating</th>
      <th>CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
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
      <th>GENERAL APPEARANCE</th>
      <th>MANNER OF SPEAKING</th>
      <th>PHYSICAL CONDITION</th>
      <th>MENTAL ALERTNESS</th>
      <th>SELF-CONFIDENCE</th>
      <th>ABILITY TO PRESENT IDEAS</th>
      <th>COMMUNICATION SKILLS</th>
      <th>Student Performance Rating</th>
      <th>CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2982.000000</td>
      <td>2982.000000</td>
      <td>2982.000000</td>
      <td>2982.000000</td>
      <td>2982.000000</td>
      <td>2982.000000</td>
      <td>2982.000000</td>
      <td>2982.000000</td>
      <td>2982.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.246814</td>
      <td>3.884641</td>
      <td>3.972166</td>
      <td>3.962777</td>
      <td>3.910798</td>
      <td>3.813883</td>
      <td>3.525486</td>
      <td>4.610664</td>
      <td>0.420188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.678501</td>
      <td>0.757013</td>
      <td>0.744135</td>
      <td>0.781982</td>
      <td>0.807602</td>
      <td>0.739390</td>
      <td>0.743881</td>
      <td>0.692845</td>
      <td>0.493672</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Print the histogram chart of data


```python
print(df.columns)
```

    Index(['GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION',
           'MENTAL ALERTNESS', 'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS',
           'COMMUNICATION SKILLS', 'Student Performance Rating', 'CLASS'],
          dtype='object')



```python
viz = df[['GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION', 'MENTAL ALERTNESS', 'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS', 'COMMUNICATION SKILLS', 'Student Performance Rating', 'CLASS']]
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 7))

axes = axes.flatten()

for i, column in enumerate(viz.columns):
    viz[column].hist(ax=axes[i])
    axes[i].set_title(column)


fig.delaxes(axes[9])
plt.tight_layout()
plt.show()
```


    
![png](svm_files/svm_14_0.png)
    



```python
# print(df)
train, test = train_test_split(df, test_size=0.20, random_state=42)
# test, evaluate = train_test_split(test, test_size=0.5, random_state=42)
```

## Fit model based on data. 


```python
train_x = np.asanyarray(train[['GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION', 'MENTAL ALERTNESS', 'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS', 'COMMUNICATION SKILLS', 'Student Performance Rating']])
train_y = np.asanyarray(train[['CLASS']])
```


```python
test_x = np.asanyarray(test[['GENERAL APPEARANCE', 'MANNER OF SPEAKING', 'PHYSICAL CONDITION', 'MENTAL ALERTNESS', 'SELF-CONFIDENCE', 'ABILITY TO PRESENT IDEAS', 'COMMUNICATION SKILLS', 'Student Performance Rating']])
test_y = np.asanyarray(test[['CLASS']])
```


```python
model = svm.SVC(kernel="rbf")
model.fit(train_x, train_y)
test_y_ = model.predict(test_x)
```

    /home/andre/code/AI/venv/lib/python3.11/site-packages/sklearn/utils/validation.py:1339: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)


## Evaluation


```python
print("The jaccard score for 0's:")
print(jaccard_score(test_y, test_y_, pos_label=0))
print("\nThe jaccard score for 1's:")
print(jaccard_score(test_y, test_y_, pos_label=1))
```

    The jaccard score for 0's:
    0.7702020202020202
    
    The jaccard score for 1's:
    0.6883561643835616



```python
accuracy = accuracy_score(test_y, test_y_)
precision = precision_score(test_y, test_y_)
recall = recall_score(test_y, test_y_)



print("Logistic Regression Classification Report:")
print(classification_report(test_y, test_y_))

print("Logistic Regression Confusion Matrix:")
skplt.metrics.plot_confusion_matrix(test_y, test_y_)
```

    Logistic Regression Classification Report:
                  precision    recall  f1-score   support
    
               0       0.85      0.89      0.87       343
               1       0.84      0.79      0.82       254
    
        accuracy                           0.85       597
       macro avg       0.85      0.84      0.84       597
    weighted avg       0.85      0.85      0.85       597
    
    Logistic Regression Confusion Matrix:





    <Axes: title={'center': 'Confusion Matrix'}, xlabel='Predicted label', ylabel='True label'>




    
![png](svm_files/svm_22_2.png)
    

