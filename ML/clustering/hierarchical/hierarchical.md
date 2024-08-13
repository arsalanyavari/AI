# Hierarchical Clustering

Configure the project. Indeed you create a dataset in csv format.


```python
! rm -rf *.csv
! unzip ./dataset/archive.zip
! mv *.csv data.csv
```

    Archive:  ./dataset/archive.zip
      inflating: MathE dataset (4).csv   


Import needed libraries


```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import euclidean_distances

%matplotlib inline
```

# Make Sample to practice on K-Means


```python
clusters_head = [[-4, 4], [5, 4], [-3, -1], [6, -4], [2, -3], [1, 1], [-5, -5]]
X, _ = make_blobs(n_samples=100, centers=clusters_head, cluster_std=0.9, random_state=40)

Z = linkage(X)
'''
linkage methods in hierarchical clustering can be considered similar to 
agglomerative clustering. In agglomerative clustering, each data point 
starts as an individual cluster, and pairs of clusters are merged step 
by step based on a linkage criterion until a single cluster is formed.

- Single linkage: Minimum distance between clusters.
- Complete linkage: Maximum distance between clusters.
- Average linkage: Average distance between clusters.
'''

plt.figure(figsize=(10, 6))

num_clusters = 5
max_d = Z[-num_clusters, 2]
dendrogram(Z, color_threshold=max_d)

line_margin = 0.1

plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample Index')
plt.axhline(y=max_d+line_margin, color='black', linestyle='--')
plt.show()
```


    
![png](hierarchical_files/hierarchical_6_0.png)
    


# Real Data

Read data from data.csv using pandas and store in data frame structure. Also shuffle data to have uniform distribution. 


```python
df = pd.read_csv("data.csv", sep=";", encoding='ISO-8859-1')
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
      <th>Student ID</th>
      <th>Student Country</th>
      <th>Question ID</th>
      <th>Type of Answer</th>
      <th>Question Level</th>
      <th>Topic</th>
      <th>Subtopic</th>
      <th>Keywords</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>751</td>
      <td>Lithuania</td>
      <td>568</td>
      <td>0</td>
      <td>Basic</td>
      <td>Analytic Geometry</td>
      <td>Analytic Geometry</td>
      <td>Cartesian equations of a plane,Cartesian equat...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>Lithuania</td>
      <td>1144</td>
      <td>1</td>
      <td>Basic</td>
      <td>Set Theory</td>
      <td>Set Theory</td>
      <td>Subset</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>Portugal</td>
      <td>337</td>
      <td>1</td>
      <td>Basic</td>
      <td>Complex Numbers</td>
      <td>Complex Numbers</td>
      <td>Algebraic form,Principal argument,De Moivre fo...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>80</td>
      <td>Lithuania</td>
      <td>955</td>
      <td>0</td>
      <td>Basic</td>
      <td>Optimization</td>
      <td>Linear Optimization</td>
      <td>Linear programming</td>
    </tr>
    <tr>
      <th>4</th>
      <td>966</td>
      <td>Portugal</td>
      <td>444</td>
      <td>0</td>
      <td>Basic</td>
      <td>Linear Algebra</td>
      <td>Linear Transformations</td>
      <td>Range,Kernel</td>
    </tr>
  </tbody>
</table>
</div>



data cleaning


```python
df = df.drop(columns=['Student ID', 'Keywords'])
print(df.columns)
```

    Index(['Student Country', 'Question ID', 'Type of Answer', 'Question Level',
           'Topic', 'Subtopic'],
          dtype='object')


make data numeric


```python
categorical_attr = ['Student Country', 'Question Level', 'Topic', 'Subtopic']

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
      <th>Student Country</th>
      <th>Question ID</th>
      <th>Type of Answer</th>
      <th>Question Level</th>
      <th>Topic</th>
      <th>Subtopic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>568</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1144</td>
      <td>1</td>
      <td>1</td>
      <td>12</td>
      <td>21</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>337</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>955</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>444</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
num_entities = len(df)
num_unique_question_ids = df['Question ID'].nunique()

if num_entities == num_unique_question_ids:
    print("Each entity related to unique question")
```


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
      <th>Student Country</th>
      <th>Question ID</th>
      <th>Type of Answer</th>
      <th>Question Level</th>
      <th>Topic</th>
      <th>Subtopic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9546.000000</td>
      <td>9546.000000</td>
      <td>9546.000000</td>
      <td>9546.000000</td>
      <td>9546.000000</td>
      <td>9546.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.747748</td>
      <td>478.912319</td>
      <td>0.468259</td>
      <td>0.821705</td>
      <td>6.197779</td>
      <td>14.741043</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.322671</td>
      <td>249.244061</td>
      <td>0.499018</td>
      <td>0.382781</td>
      <td>2.717820</td>
      <td>7.725488</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>77.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>323.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>428.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>571.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>1549.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>13.000000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>



scale data from 0 to 1


```python
X = df.values
scaled_x = MinMaxScaler().fit_transform(X)
scaled_x[-10:-5]
```




    array([[0.14285714, 0.48029891, 1.        , 1.        , 0.53846154,
            0.65217391],
           [0.42857143, 0.24592391, 1.        , 0.        , 0.53846154,
            1.        ],
           [0.42857143, 0.23505435, 0.        , 1.        , 0.53846154,
            1.        ],
           [0.14285714, 0.03804348, 1.        , 0.        , 0.53846154,
            0.69565217],
           [0.42857143, 0.25611413, 0.        , 1.        , 0.53846154,
            1.        ]])



calculate euclidean distances matrix


```python
dist_matrix = euclidean_distances(scaled_x,scaled_x) 
print(dist_matrix)
```

    [[0.         1.66172614 1.02608114 ... 1.59356415 1.72632018 1.1847848 ]
     [1.66172614 0.         1.31124534 ... 0.84460709 0.71192234 0.787654  ]
     [1.02608114 1.31124534 0.         ... 1.11124386 1.34218407 0.62975975]
     ...
     [1.59356415 0.84460709 1.11124386 ... 0.         0.98835919 0.97426296]
     [1.72632018 0.71192234 1.34218407 ... 0.98835919 0.         0.82669249]
     [1.1847848  0.787654   0.62975975 ... 0.97426296 0.82669249 0.        ]]



```python
Z = linkage(dist_matrix, 'complete')

```

    /tmp/ipykernel_148084/1895303000.py:1: ClusterWarning: scipy.cluster: The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix
      Z = linkage(dist_matrix, 'complete')



```python
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
```


    
![png](hierarchical_files/hierarchical_21_0.png)
    


### Lets split data in 2 cluster


```python
from scipy.cluster.hierarchy import fcluster

max_clusters = 2
clusters = fcluster(Z, max_clusters, criterion='maxclust')

plt.figure(figsize=(10, 6))
dendrogram(Z, labels=clusters, color_threshold=Z[-(max_clusters-1), 2])
plt.show()

```


    
![png](hierarchical_files/hierarchical_23_0.png)
    


<font color="green" size=5>
As you see data split into 2 clusters.
</font>

<hr>

### Lets see how many cluster exist in specific depth


```python

max_d = 30
clusters = fcluster(Z, max_d, criterion='distance')
num_clusters = len(set(clusters))
line_margin = 0.5

plt.figure(figsize=(10, 6))
dendrogram(Z, color_threshold=Z[-(num_clusters-1), 2])
plt.axhline(y=max_d+line_margin, color='black', linestyle='--')
plt.show()
print(f"number of cluster that under {max_d} depth: {num_clusters}")
```


    
![png](hierarchical_files/hierarchical_26_0.png)
    


    number of cluster that under 30 depth: 17

