# Collaborative Recommender System

Configure the project. Indeed you create a dataset in csv format.


```python
! rm -rf *.csv
! unzip ./dataset/archive.zip

```

    Archive:  ./dataset/archive.zip
      inflating: anime.csv               
      inflating: rating.csv              


Import needed libraries


```python
import pandas as pd
from IPython.display import Image
from scipy.stats import pearsonr

%matplotlib inline
```

### User-Based Collaborative Filtering
- Approach: Finds users similar to the target user based on historical interactions.
- Process:
  1. Identify users with similar preferences.
  2. Recommend items liked by these similar users.
- Pros:
  - Simple to understand and implement.
  - Often effective with sufficient user data.
- Cons:
  - Performance degrades with large datasets.
  - Struggles with new users (cold start problem).

### Item-Based Collaborative Filtering
- Approach: Finds items similar to the ones the target user has interacted with.
- Process:
  1. Identify items similar to what the user likes.
  2. Recommend these similar items.
- Pros:
  - More scalable with large datasets.
  - Can leverage item characteristics and interactions.
- Cons:
  - Requires significant item interaction data.
  - Might not capture nuanced user preferences.

Both approaches aim to provide personalized recommendations but differ in their method and scalability.


```python
url = "https://www.scaler.com/topics/images/collaborative.webp"
Image(url=url)
```




<img src="https://www.scaler.com/topics/images/collaborative.webp"/>



### I decided to implement User Based Approach to avoid memory issues.

Read data from csv files using pandas and store in data frame structure. Also shuffle data to have uniform distribution. 


```python
anime_df = pd.read_csv("anime.csv")
anime_df = anime_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

rating_df = pd.read_csv("rating.csv")
rating_df = rating_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
```


```python
anime_df
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
      <th>anime_id</th>
      <th>name</th>
      <th>genre</th>
      <th>type</th>
      <th>episodes</th>
      <th>rating</th>
      <th>members</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17209</td>
      <td>Suzy&amp;#039;s Zoo: Daisuki! Witzy - Happy Birthday</td>
      <td>Kids</td>
      <td>Special</td>
      <td>1</td>
      <td>6.17</td>
      <td>158</td>
    </tr>
    <tr>
      <th>1</th>
      <td>173</td>
      <td>Tactics</td>
      <td>Comedy, Drama, Fantasy, Mystery, Shounen, Supe...</td>
      <td>TV</td>
      <td>25</td>
      <td>7.34</td>
      <td>27358</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3616</td>
      <td>Kamen no Maid Guy</td>
      <td>Action, Comedy, Ecchi, Super Power</td>
      <td>TV</td>
      <td>12</td>
      <td>7.14</td>
      <td>27761</td>
    </tr>
    <tr>
      <th>3</th>
      <td>18799</td>
      <td>Take Your Way</td>
      <td>Action, Music, Seinen, Supernatural</td>
      <td>Music</td>
      <td>1</td>
      <td>6.66</td>
      <td>1387</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18831</td>
      <td>Rinkaku</td>
      <td>Dementia, Horror, Music</td>
      <td>Music</td>
      <td>1</td>
      <td>5.60</td>
      <td>606</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12289</th>
      <td>4638</td>
      <td>Milkyway</td>
      <td>Hentai, Romance</td>
      <td>OVA</td>
      <td>2</td>
      <td>5.82</td>
      <td>695</td>
    </tr>
    <tr>
      <th>12290</th>
      <td>5272</td>
      <td>Tondemo Nezumi Daikatsuyaku</td>
      <td>Adventure</td>
      <td>Movie</td>
      <td>1</td>
      <td>6.53</td>
      <td>252</td>
    </tr>
    <tr>
      <th>12291</th>
      <td>1262</td>
      <td>Macross II: Lovers Again</td>
      <td>Adventure, Mecha, Military, Sci-Fi, Shounen, S...</td>
      <td>OVA</td>
      <td>6</td>
      <td>6.47</td>
      <td>6760</td>
    </tr>
    <tr>
      <th>12292</th>
      <td>22819</td>
      <td>Aikatsu! Movie</td>
      <td>Music, School, Shoujo, Slice of Life</td>
      <td>Movie</td>
      <td>1</td>
      <td>7.79</td>
      <td>2813</td>
    </tr>
    <tr>
      <th>12293</th>
      <td>2364</td>
      <td>Virus: Virus Buster Serge</td>
      <td>Action, Adventure, Mecha, Police, Sci-Fi</td>
      <td>TV</td>
      <td>12</td>
      <td>5.59</td>
      <td>2250</td>
    </tr>
  </tbody>
</table>
<p>12294 rows × 7 columns</p>
</div>




```python
rating_df
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
      <th>user_id</th>
      <th>anime_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59257</td>
      <td>650</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8203</td>
      <td>591</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15395</td>
      <td>209</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1280</td>
      <td>6702</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9259</td>
      <td>32998</td>
      <td>8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2925805</th>
      <td>24906</td>
      <td>9065</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2925806</th>
      <td>48795</td>
      <td>18411</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2925807</th>
      <td>43226</td>
      <td>28299</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2925808</th>
      <td>62082</td>
      <td>17397</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2925809</th>
      <td>42635</td>
      <td>302</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>2925810 rows × 3 columns</p>
</div>




```python
rating_df['rating'].value_counts()
```




    rating
    -1     625635
     7     570953
     8     567465
     9     362429
     6     303423
     10    253867
     5     144328
     4      53456
     3      22105
     2      12758
     1       9391
    Name: count, dtype: int64




```python
userInput = [
    {'Title': 'Boku dake ga Inai Machi', 'Rating': 10.0},
    {'Title': 'Violet Evergarden', 'Rating': 9.5},
    {'Title': 'Goblin Slayer', 'Rating': 6.0},
    {'Title': 'Berserk', 'Rating': 8.0},
    {'Title': 'Shingeki no Kyojin', 'Rating': 7.0},
    {'Title': 'Tokyo Ghoul', 'Rating': 6.5},
    {'Title': 'Orange', 'Rating': 6.0},
    {'Title': 'Death Parade', 'Rating': 8.0},
    {'Title': 'Death Note', 'Rating': 7.5},
    {'Title': 'Bungou Stray Dogs', 'Rating': 7.5},
    {'Title': 'Tenki no Ko', 'Rating': 8.0},
    {'Title': 'Kimi no Na wa.', 'Rating': 8.0},
    {'Title': 'Kimi no Suizou wo Tabetai', 'Rating': 8.5},
    {'Title': 'Mononoke Hime', 'Rating': 7.5},
    {'Title': 'Sen to Chihiro no Kamikakushi', 'Rating': 7.5},
    {'Title': 'Koe no Katachi', 'Rating': 8.5},
    {'Title': 'Ao Haru Ride', 'Rating': 5.5},
    {'Title': 'Toki wo Kakeru Shoujo', 'Rating': 7.0},
    {'Title': 'Another', 'Rating': 7.5},
    {'Title': 'Kimetsu no Yaiba', 'Rating': 7.0},
    {'Title': 'Shigatsu wa Kimi no Uso', 'Rating': 8.0},
    {'Title': 'Byousoku 5 Centimeter', 'Rating': 6.0},
    {'Title': 'Kokoro ga Sakebitagatterunda.', 'Rating': 7.5},
    {'Title': 'Schick x Evangelion', 'Rating': 5.0}
]

inputAnime = pd.DataFrame(userInput)
print(inputAnime)
```

                                Title  Rating
    0         Boku dake ga Inai Machi    10.0
    1               Violet Evergarden     9.5
    2                   Goblin Slayer     6.0
    3                         Berserk     8.0
    4              Shingeki no Kyojin     7.0
    5                     Tokyo Ghoul     6.5
    6                          Orange     6.0
    7                    Death Parade     8.0
    8                      Death Note     7.5
    9               Bungou Stray Dogs     7.5
    10                    Tenki no Ko     8.0
    11                 Kimi no Na wa.     8.0
    12      Kimi no Suizou wo Tabetai     8.5
    13                  Mononoke Hime     7.5
    14  Sen to Chihiro no Kamikakushi     7.5
    15                 Koe no Katachi     8.5
    16                   Ao Haru Ride     5.5
    17          Toki wo Kakeru Shoujo     7.0
    18                        Another     7.5
    19               Kimetsu no Yaiba     7.0
    20        Shigatsu wa Kimi no Uso     8.0
    21          Byousoku 5 Centimeter     6.0
    22  Kokoro ga Sakebitagatterunda.     7.5
    23            Schick x Evangelion     5.0



```python
anime_df.columns[:25]
```




    Index(['anime_id', 'name', 'genre', 'type', 'episodes', 'rating', 'members'], dtype='object')




```python
inputAnime = pd.merge(inputAnime, anime_df[['anime_id', 'name']], how='left', left_on='Title', right_on='name')
inputAnime = inputAnime.drop(columns='name')
inputAnime
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
      <th>Title</th>
      <th>Rating</th>
      <th>anime_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Boku dake ga Inai Machi</td>
      <td>10.0</td>
      <td>31043.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Violet Evergarden</td>
      <td>9.5</td>
      <td>33352.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Goblin Slayer</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Berserk</td>
      <td>8.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shingeki no Kyojin</td>
      <td>7.0</td>
      <td>16498.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tokyo Ghoul</td>
      <td>6.5</td>
      <td>22319.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Orange</td>
      <td>6.0</td>
      <td>32729.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Death Parade</td>
      <td>8.0</td>
      <td>28223.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Death Note</td>
      <td>7.5</td>
      <td>1535.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bungou Stray Dogs</td>
      <td>7.5</td>
      <td>31478.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Tenki no Ko</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Kimi no Na wa.</td>
      <td>8.0</td>
      <td>32281.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Kimi no Suizou wo Tabetai</td>
      <td>8.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Mononoke Hime</td>
      <td>7.5</td>
      <td>164.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sen to Chihiro no Kamikakushi</td>
      <td>7.5</td>
      <td>199.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Koe no Katachi</td>
      <td>8.5</td>
      <td>28851.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ao Haru Ride</td>
      <td>5.5</td>
      <td>21995.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Toki wo Kakeru Shoujo</td>
      <td>7.0</td>
      <td>2236.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Another</td>
      <td>7.5</td>
      <td>11111.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Kimetsu no Yaiba</td>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Shigatsu wa Kimi no Uso</td>
      <td>8.0</td>
      <td>23273.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Byousoku 5 Centimeter</td>
      <td>6.0</td>
      <td>1689.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Kokoro ga Sakebitagatterunda.</td>
      <td>7.5</td>
      <td>28725.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Schick x Evangelion</td>
      <td>5.0</td>
      <td>31115.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
inputAnime = inputAnime.dropna(subset=['anime_id'])
inputAnime = inputAnime.reset_index(drop=True)

inputAnime
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
      <th>Title</th>
      <th>Rating</th>
      <th>anime_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Boku dake ga Inai Machi</td>
      <td>10.0</td>
      <td>31043.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Violet Evergarden</td>
      <td>9.5</td>
      <td>33352.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Berserk</td>
      <td>8.0</td>
      <td>33.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Shingeki no Kyojin</td>
      <td>7.0</td>
      <td>16498.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tokyo Ghoul</td>
      <td>6.5</td>
      <td>22319.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Orange</td>
      <td>6.0</td>
      <td>32729.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Death Parade</td>
      <td>8.0</td>
      <td>28223.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Death Note</td>
      <td>7.5</td>
      <td>1535.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bungou Stray Dogs</td>
      <td>7.5</td>
      <td>31478.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Kimi no Na wa.</td>
      <td>8.0</td>
      <td>32281.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Mononoke Hime</td>
      <td>7.5</td>
      <td>164.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Sen to Chihiro no Kamikakushi</td>
      <td>7.5</td>
      <td>199.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Koe no Katachi</td>
      <td>8.5</td>
      <td>28851.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Ao Haru Ride</td>
      <td>5.5</td>
      <td>21995.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Toki wo Kakeru Shoujo</td>
      <td>7.0</td>
      <td>2236.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Another</td>
      <td>7.5</td>
      <td>11111.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Shigatsu wa Kimi no Uso</td>
      <td>8.0</td>
      <td>23273.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Byousoku 5 Centimeter</td>
      <td>6.0</td>
      <td>1689.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Kokoro ga Sakebitagatterunda.</td>
      <td>7.5</td>
      <td>28725.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Schick x Evangelion</td>
      <td>5.0</td>
      <td>31115.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
userSubset = rating_df[rating_df['anime_id'].isin(inputAnime['anime_id'].tolist())]
userSubset
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
      <th>user_id</th>
      <th>anime_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>220</th>
      <td>6853</td>
      <td>23273</td>
      <td>9</td>
    </tr>
    <tr>
      <th>402</th>
      <td>10089</td>
      <td>21995</td>
      <td>8</td>
    </tr>
    <tr>
      <th>601</th>
      <td>3539</td>
      <td>33</td>
      <td>10</td>
    </tr>
    <tr>
      <th>760</th>
      <td>28955</td>
      <td>32281</td>
      <td>10</td>
    </tr>
    <tr>
      <th>762</th>
      <td>2398</td>
      <td>22319</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2925331</th>
      <td>121</td>
      <td>1535</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2925458</th>
      <td>1256</td>
      <td>1535</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2925556</th>
      <td>1442</td>
      <td>164</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2925669</th>
      <td>9625</td>
      <td>21995</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2925685</th>
      <td>3652</td>
      <td>164</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>17078 rows × 3 columns</p>
</div>




```python
userSubsetGroup = userSubset.groupby('user_id').agg(
    counts=('anime_id', 'count'),                           # Count of anime
    anime_ids=('anime_id', lambda x: list(x))               # List of anime IDs
).reset_index()

userSubsetGroup = userSubsetGroup.sort_values(by='counts', ascending=False)
userSubsetGroup
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
      <th>user_id</th>
      <th>counts</th>
      <th>anime_ids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>552</th>
      <td>687</td>
      <td>15</td>
      <td>[21995, 2236, 1689, 22319, 1535, 164, 28725, 2...</td>
    </tr>
    <tr>
      <th>920</th>
      <td>1145</td>
      <td>14</td>
      <td>[16498, 31043, 21995, 28725, 2236, 31478, 2327...</td>
    </tr>
    <tr>
      <th>2479</th>
      <td>3338</td>
      <td>14</td>
      <td>[28851, 32281, 31478, 22319, 21995, 2236, 1689...</td>
    </tr>
    <tr>
      <th>625</th>
      <td>784</td>
      <td>14</td>
      <td>[28223, 22319, 1535, 32729, 21995, 31043, 1649...</td>
    </tr>
    <tr>
      <th>606</th>
      <td>760</td>
      <td>13</td>
      <td>[22319, 28223, 31043, 31478, 199, 32729, 1535,...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7082</th>
      <td>68091</td>
      <td>1</td>
      <td>[28725]</td>
    </tr>
    <tr>
      <th>7083</th>
      <td>68177</td>
      <td>1</td>
      <td>[28725]</td>
    </tr>
    <tr>
      <th>7084</th>
      <td>68320</td>
      <td>1</td>
      <td>[28725]</td>
    </tr>
    <tr>
      <th>7085</th>
      <td>68405</td>
      <td>1</td>
      <td>[28725]</td>
    </tr>
    <tr>
      <th>7086</th>
      <td>68559</td>
      <td>1</td>
      <td>[28725]</td>
    </tr>
  </tbody>
</table>
<p>7133 rows × 3 columns</p>
</div>




```python
userSubsetGroup = userSubsetGroup[userSubsetGroup['counts'] >= 10]
userSubsetGroup
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
      <th>user_id</th>
      <th>counts</th>
      <th>anime_ids</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>552</th>
      <td>687</td>
      <td>15</td>
      <td>[21995, 2236, 1689, 22319, 1535, 164, 28725, 2...</td>
    </tr>
    <tr>
      <th>920</th>
      <td>1145</td>
      <td>14</td>
      <td>[16498, 31043, 21995, 28725, 2236, 31478, 2327...</td>
    </tr>
    <tr>
      <th>2479</th>
      <td>3338</td>
      <td>14</td>
      <td>[28851, 32281, 31478, 22319, 21995, 2236, 1689...</td>
    </tr>
    <tr>
      <th>625</th>
      <td>784</td>
      <td>14</td>
      <td>[28223, 22319, 1535, 32729, 21995, 31043, 1649...</td>
    </tr>
    <tr>
      <th>606</th>
      <td>760</td>
      <td>13</td>
      <td>[22319, 28223, 31043, 31478, 199, 32729, 1535,...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>960</th>
      <td>1197</td>
      <td>10</td>
      <td>[1689, 16498, 21995, 28725, 28223, 23273, 3147...</td>
    </tr>
    <tr>
      <th>1519</th>
      <td>1870</td>
      <td>10</td>
      <td>[21995, 164, 31478, 31043, 28223, 22319, 11111...</td>
    </tr>
    <tr>
      <th>1530</th>
      <td>1889</td>
      <td>10</td>
      <td>[21995, 23273, 11111, 199, 33, 31043, 164, 168...</td>
    </tr>
    <tr>
      <th>1752</th>
      <td>2243</td>
      <td>10</td>
      <td>[21995, 32729, 1689, 28223, 22319, 11111, 3228...</td>
    </tr>
    <tr>
      <th>1495</th>
      <td>1837</td>
      <td>10</td>
      <td>[28223, 22319, 31043, 164, 2236, 199, 23273, 2...</td>
    </tr>
  </tbody>
</table>
<p>97 rows × 3 columns</p>
</div>




```python
animeRating_dict = inputAnime.set_index('anime_id')['Rating'].to_dict()

pearsonCorrelation_dict = {}

for index, row in userSubsetGroup.iterrows():
    user_id = row['user_id']
    user_anime_ids = row['anime_ids']
    
    # Get corresponding ratings for the user's anime ids
    user_ratings = [animeRating_dict[anime_id] for anime_id in user_anime_ids if anime_id in animeRating_dict]
    
    # Calculate Pearson correlation with inputAnime ratings (use it as a baseline)
    if user_ratings:
        input_ratings = [animeRating_dict[anime_id] for anime_id in inputAnime['anime_id'] if anime_id in user_anime_ids]
        
        # Ensure both lists have the same length
        if len(user_ratings) == len(input_ratings) and len(user_ratings) > 0:
            correlation, _ = pearsonr(user_ratings, input_ratings)
            pearsonCorrelation_dict[user_id] = correlation
        else:
            pearsonCorrelation_dict[user_id] = 0
    else:
        pearsonCorrelation_dict[user_id] = 0

# Display the results
for user_id, correlation in pearsonCorrelation_dict.items():
    print(f"User ID: {user_id}, Pearson Correlation: {correlation}")
```

    User ID: 687, Pearson Correlation: -0.23953974895397492
    User ID: 1145, Pearson Correlation: -0.018181818181818202
    User ID: 3338, Pearson Correlation: 0.18984771573604062
    User ID: 784, Pearson Correlation: 0.14410480349344976
    User ID: 760, Pearson Correlation: -0.33180428134556567
    User ID: 392, Pearson Correlation: 0.12532981530343007
    User ID: 446, Pearson Correlation: -0.6027397260273972
    User ID: 17, Pearson Correlation: -0.4708029197080291
    User ID: 342, Pearson Correlation: 0.10817941952506599
    User ID: 786, Pearson Correlation: -0.0633245382585752
    User ID: 1497, Pearson Correlation: 0.294811320754717
    User ID: 963, Pearson Correlation: 0.08401084010840111
    User ID: 2378, Pearson Correlation: 0.3827751196172249
    User ID: 813, Pearson Correlation: -0.1868131868131868
    User ID: 958, Pearson Correlation: -0.3411764705882353
    User ID: 938, Pearson Correlation: 0.6970873786407766
    User ID: 198, Pearson Correlation: 0.1
    User ID: 562, Pearson Correlation: 0.1047957371225577
    User ID: 1013, Pearson Correlation: 0.6026490066225164
    User ID: 1597, Pearson Correlation: -0.2529411764705883
    User ID: 3299, Pearson Correlation: -0.10982658959537578
    User ID: 1222, Pearson Correlation: -0.03311258278145697
    User ID: 123, Pearson Correlation: 0.15761589403973508
    User ID: 77, Pearson Correlation: 0.10773899848254934
    User ID: 1378, Pearson Correlation: 0.2
    User ID: 1344, Pearson Correlation: 0.039999999999999994
    User ID: 1327, Pearson Correlation: -0.3511111111111111
    User ID: 1813, Pearson Correlation: -0.041176470588235294
    User ID: 271, Pearson Correlation: 0.536082474226804
    User ID: 1290, Pearson Correlation: 0.05962521294718907
    User ID: 1019, Pearson Correlation: 0.31746031746031733
    User ID: 478, Pearson Correlation: 0.3470588235294118
    User ID: 2864, Pearson Correlation: -0.19111111111111112
    User ID: 1456, Pearson Correlation: 0.27472527472527475
    User ID: 1274, Pearson Correlation: -0.1502145922746781
    User ID: 651, Pearson Correlation: 0.10724637681159421
    User ID: 611, Pearson Correlation: -0.33333333333333337
    User ID: 2318, Pearson Correlation: 0.09130434782608696
    User ID: 1176, Pearson Correlation: 0.6577777777777776
    User ID: 2200, Pearson Correlation: -0.6019417475728154
    User ID: 585, Pearson Correlation: 1.7094345118926147e-18
    User ID: 2016, Pearson Correlation: 0.35069444444444436
    User ID: 1394, Pearson Correlation: -0.021428571428571443
    User ID: 1116, Pearson Correlation: -0.03125000000000002
    User ID: 1620, Pearson Correlation: -0.12780898876404492
    User ID: 1435, Pearson Correlation: 0.3058252427184466
    User ID: 1442, Pearson Correlation: -0.0298804780876494
    User ID: 395, Pearson Correlation: 0.06382978723404255
    User ID: 3284, Pearson Correlation: 0.28864353312302843
    User ID: 3360, Pearson Correlation: -0.04532163742690057
    User ID: 2951, Pearson Correlation: -0.8857142857142858
    User ID: 1418, Pearson Correlation: -0.35714285714285704
    User ID: 2555, Pearson Correlation: 0.017182130584192438
    User ID: 2867, Pearson Correlation: 0.032407407407407406
    User ID: 79, Pearson Correlation: -0.02325581395348838
    User ID: 139, Pearson Correlation: -0.028037383177570086
    User ID: 1815, Pearson Correlation: -0.20833333333333337
    User ID: 1814, Pearson Correlation: -0.3707165109034268
    User ID: 1918, Pearson Correlation: 0.05241935483870963
    User ID: 2256, Pearson Correlation: -0.4411247803163445
    User ID: 228, Pearson Correlation: -0.2588652482269503
    User ID: 565, Pearson Correlation: 0.024822695035461
    User ID: 744, Pearson Correlation: 0.45895522388059706
    User ID: 4102, Pearson Correlation: -0.6903409090909091
    User ID: 1501, Pearson Correlation: -0.16465863453815263
    User ID: 1504, Pearson Correlation: -0.5151515151515152
    User ID: 1343, Pearson Correlation: 0.019607843137254957
    User ID: 4565, Pearson Correlation: 0.13043478260869568
    User ID: 861, Pearson Correlation: -0.44
    User ID: 2202, Pearson Correlation: -0.023454157782515993
    User ID: 553, Pearson Correlation: 0.23809523809523808
    User ID: 4512, Pearson Correlation: 0.13043478260869568
    User ID: 1237, Pearson Correlation: -0.5329768270944742
    User ID: 1252, Pearson Correlation: -0.044176706827309245
    User ID: 1284, Pearson Correlation: -0.02083333333333333
    User ID: 926, Pearson Correlation: -0.11475409836065573
    User ID: 995, Pearson Correlation: -0.40909090909090906
    User ID: 1419, Pearson Correlation: 0.8333333333333333
    User ID: 1579, Pearson Correlation: -0.06339468302658491
    User ID: 1578, Pearson Correlation: -0.045871559633027505
    User ID: 244, Pearson Correlation: -0.20567375886524816
    User ID: 1711, Pearson Correlation: -0.17021276595744678
    User ID: 1400, Pearson Correlation: 0.483065953654189
    User ID: 894, Pearson Correlation: -0.018518518518518556
    User ID: 2025, Pearson Correlation: 0.3023255813953489
    User ID: 598, Pearson Correlation: -0.22340425531914884
    User ID: 1023, Pearson Correlation: 0.21985815602836875
    User ID: 2273, Pearson Correlation: 0.09274193548387095
    User ID: 1605, Pearson Correlation: -0.4262820512820513
    User ID: 1963, Pearson Correlation: 0.3055555555555556
    User ID: 1167, Pearson Correlation: -0.2121212121212121
    User ID: 444, Pearson Correlation: 0.45512820512820523
    User ID: 1197, Pearson Correlation: -0.5261044176706828
    User ID: 1870, Pearson Correlation: -0.3750000000000001
    User ID: 1889, Pearson Correlation: -0.2126537785588753
    User ID: 2243, Pearson Correlation: 0.0657439446366782
    User ID: 1837, Pearson Correlation: -0.22699386503067487



```python
pearsonDF = pd.DataFrame.from_dict(pearsonCorrelation_dict, orient='index')
pearsonDF.columns = ['Similarity Index']
pearsonDF['user_id'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF
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
      <th>Similarity Index</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.239540</td>
      <td>687</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.018182</td>
      <td>1145</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.189848</td>
      <td>3338</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.144105</td>
      <td>784</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.331804</td>
      <td>760</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>92</th>
      <td>-0.526104</td>
      <td>1197</td>
    </tr>
    <tr>
      <th>93</th>
      <td>-0.375000</td>
      <td>1870</td>
    </tr>
    <tr>
      <th>94</th>
      <td>-0.212654</td>
      <td>1889</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.065744</td>
      <td>2243</td>
    </tr>
    <tr>
      <th>96</th>
      <td>-0.226994</td>
      <td>1837</td>
    </tr>
  </tbody>
</table>
<p>97 rows × 2 columns</p>
</div>




```python
topUsers = pearsonDF.sort_values(by='Similarity Index', ascending=False)[0:40]
topUsers
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
      <th>Similarity Index</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>77</th>
      <td>0.833333</td>
      <td>1419</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.697087</td>
      <td>938</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.657778</td>
      <td>1176</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.602649</td>
      <td>1013</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.536082</td>
      <td>271</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.483066</td>
      <td>1400</td>
    </tr>
    <tr>
      <th>62</th>
      <td>0.458955</td>
      <td>744</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.455128</td>
      <td>444</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.382775</td>
      <td>2378</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.350694</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.347059</td>
      <td>478</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.317460</td>
      <td>1019</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.305825</td>
      <td>1435</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.305556</td>
      <td>1963</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.302326</td>
      <td>2025</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.294811</td>
      <td>1497</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.288644</td>
      <td>3284</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.274725</td>
      <td>1456</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.238095</td>
      <td>553</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.219858</td>
      <td>1023</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.200000</td>
      <td>1378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.189848</td>
      <td>3338</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.157616</td>
      <td>123</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.144105</td>
      <td>784</td>
    </tr>
    <tr>
      <th>67</th>
      <td>0.130435</td>
      <td>4565</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.130435</td>
      <td>4512</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.125330</td>
      <td>392</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.108179</td>
      <td>342</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.107739</td>
      <td>77</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.107246</td>
      <td>651</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.104796</td>
      <td>562</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.100000</td>
      <td>198</td>
    </tr>
    <tr>
      <th>87</th>
      <td>0.092742</td>
      <td>2273</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.091304</td>
      <td>2318</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.084011</td>
      <td>963</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.065744</td>
      <td>2243</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.063830</td>
      <td>395</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.059625</td>
      <td>1290</td>
    </tr>
    <tr>
      <th>58</th>
      <td>0.052419</td>
      <td>1918</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.040000</td>
      <td>1344</td>
    </tr>
  </tbody>
</table>
</div>




```python
topUsersRating=topUsers.merge(rating_df, left_on='user_id', right_on='user_id', how='inner')
topUsersRating
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
      <th>Similarity Index</th>
      <th>user_id</th>
      <th>anime_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.833333</td>
      <td>1419</td>
      <td>23319</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.833333</td>
      <td>1419</td>
      <td>31043</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.833333</td>
      <td>1419</td>
      <td>2904</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.833333</td>
      <td>1419</td>
      <td>13391</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.833333</td>
      <td>1419</td>
      <td>14189</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>18181</th>
      <td>0.040000</td>
      <td>1344</td>
      <td>239</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>18182</th>
      <td>0.040000</td>
      <td>1344</td>
      <td>30413</td>
      <td>9</td>
    </tr>
    <tr>
      <th>18183</th>
      <td>0.040000</td>
      <td>1344</td>
      <td>9863</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>18184</th>
      <td>0.040000</td>
      <td>1344</td>
      <td>9041</td>
      <td>7</td>
    </tr>
    <tr>
      <th>18185</th>
      <td>0.040000</td>
      <td>1344</td>
      <td>3654</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>18186 rows × 4 columns</p>
</div>




```python
topUsersRating['weighted Rating'] = topUsersRating['Similarity Index']*topUsersRating['rating']
topUsersRating.head
```




    <bound method NDFrame.head of        Similarity Index  user_id  anime_id  rating  weighted Rating
    0              0.833333     1419     23319      10         8.333333
    1              0.833333     1419     31043      10         8.333333
    2              0.833333     1419      2904      10         8.333333
    3              0.833333     1419     13391       9         7.500000
    4              0.833333     1419     14189      10         8.333333
    ...                 ...      ...       ...     ...              ...
    18181          0.040000     1344       239      -1        -0.040000
    18182          0.040000     1344     30413       9         0.360000
    18183          0.040000     1344      9863      -1        -0.040000
    18184          0.040000     1344      9041       7         0.280000
    18185          0.040000     1344      3654      -1        -0.040000
    
    [18186 rows x 5 columns]>




```python
tempTopUsersRating = topUsersRating.groupby('anime_id').sum()[['Similarity Index','weighted Rating']]
tempTopUsersRating.columns = ['sum of similarity Index','sum of weighted Rating']
tempTopUsersRating
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
      <th>sum of similarity Index</th>
      <th>sum of weighted Rating</th>
    </tr>
    <tr>
      <th>anime_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2.944419</td>
      <td>19.664045</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.845858</td>
      <td>5.965146</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.268708</td>
      <td>6.559716</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.003768</td>
      <td>8.823478</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.012594</td>
      <td>6.341339</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>34085</th>
      <td>0.092742</td>
      <td>0.649194</td>
    </tr>
    <tr>
      <th>34103</th>
      <td>0.468151</td>
      <td>2.845150</td>
    </tr>
    <tr>
      <th>34136</th>
      <td>0.040000</td>
      <td>0.320000</td>
    </tr>
    <tr>
      <th>34173</th>
      <td>0.274725</td>
      <td>-0.274725</td>
    </tr>
    <tr>
      <th>34240</th>
      <td>3.388897</td>
      <td>29.199424</td>
    </tr>
  </tbody>
</table>
<p>3503 rows × 2 columns</p>
</div>




```python
recommendation_df = pd.DataFrame()
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum of weighted Rating']/tempTopUsersRating['sum of similarity Index']
recommendation_df['anime_id'] = tempTopUsersRating.index
recommendation_df.head()
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
      <th>weighted average recommendation score</th>
      <th>anime_id</th>
    </tr>
    <tr>
      <th>anime_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6.678413</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7.052186</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.170389</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.790355</td>
      <td>7</td>
    </tr>
    <tr>
      <th>15</th>
      <td>6.262468</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)
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
      <th>weighted average recommendation score</th>
      <th>anime_id</th>
    </tr>
    <tr>
      <th>anime_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3247</th>
      <td>10.0</td>
      <td>3247</td>
    </tr>
    <tr>
      <th>10153</th>
      <td>10.0</td>
      <td>10153</td>
    </tr>
    <tr>
      <th>28479</th>
      <td>10.0</td>
      <td>28479</td>
    </tr>
    <tr>
      <th>28683</th>
      <td>10.0</td>
      <td>28683</td>
    </tr>
    <tr>
      <th>441</th>
      <td>10.0</td>
      <td>441</td>
    </tr>
    <tr>
      <th>446</th>
      <td>10.0</td>
      <td>446</td>
    </tr>
    <tr>
      <th>2825</th>
      <td>10.0</td>
      <td>2825</td>
    </tr>
    <tr>
      <th>449</th>
      <td>10.0</td>
      <td>449</td>
    </tr>
    <tr>
      <th>16143</th>
      <td>10.0</td>
      <td>16143</td>
    </tr>
    <tr>
      <th>1566</th>
      <td>10.0</td>
      <td>1566</td>
    </tr>
  </tbody>
</table>
</div>




```python
anime_df.loc[anime_df['anime_id'].isin(recommendation_df.head(10)['anime_id'].tolist())]

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
      <th>anime_id</th>
      <th>name</th>
      <th>genre</th>
      <th>type</th>
      <th>episodes</th>
      <th>rating</th>
      <th>members</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2839</th>
      <td>449</td>
      <td>InuYasha: Guren no Houraijima</td>
      <td>Adventure, Comedy, Demons, Drama, Historical, ...</td>
      <td>Movie</td>
      <td>1</td>
      <td>7.62</td>
      <td>50008</td>
    </tr>
    <tr>
      <th>3141</th>
      <td>16143</td>
      <td>One Piece: Kinkyuu Kikaku One Piece Kanzen Kou...</td>
      <td>Adventure, Comedy, Fantasy, Shounen</td>
      <td>Special</td>
      <td>1</td>
      <td>7.36</td>
      <td>5914</td>
    </tr>
    <tr>
      <th>6798</th>
      <td>10153</td>
      <td>Mahou Shoujo Lyrical Nanoha: The Movie 2nd A&amp;#...</td>
      <td>Action, Comedy, Drama, Magic, Super Power</td>
      <td>Movie</td>
      <td>1</td>
      <td>8.34</td>
      <td>13315</td>
    </tr>
    <tr>
      <th>6895</th>
      <td>28683</td>
      <td>One Piece: Episode of Alabasta - Prologue</td>
      <td>Action, Adventure, Fantasy, Shounen</td>
      <td>OVA</td>
      <td>1</td>
      <td>7.41</td>
      <td>4225</td>
    </tr>
    <tr>
      <th>7325</th>
      <td>441</td>
      <td>Shoujo Kakumei Utena: Adolescence Mokushiroku</td>
      <td>Dementia, Drama, Fantasy, Romance, Shoujo</td>
      <td>Movie</td>
      <td>1</td>
      <td>7.59</td>
      <td>22219</td>
    </tr>
    <tr>
      <th>7968</th>
      <td>446</td>
      <td>Weiß Kreuz Glühen</td>
      <td>Action, Drama, Shounen</td>
      <td>TV</td>
      <td>13</td>
      <td>6.72</td>
      <td>7043</td>
    </tr>
    <tr>
      <th>8528</th>
      <td>3247</td>
      <td>Love Hina Final Selection</td>
      <td>Comedy, Ecchi, Harem, Romance</td>
      <td>OVA</td>
      <td>1</td>
      <td>7.32</td>
      <td>21824</td>
    </tr>
    <tr>
      <th>8636</th>
      <td>1566</td>
      <td>Ghost in the Shell: Stand Alone Complex - Soli...</td>
      <td>Mecha, Military, Mystery, Police, Sci-Fi, Seinen</td>
      <td>Special</td>
      <td>1</td>
      <td>8.22</td>
      <td>55247</td>
    </tr>
    <tr>
      <th>11574</th>
      <td>28479</td>
      <td>Detective Conan Movie 19: The Hellfire Sunflowers</td>
      <td>Action, Mystery, Police, Shounen</td>
      <td>Movie</td>
      <td>1</td>
      <td>7.77</td>
      <td>8600</td>
    </tr>
    <tr>
      <th>12274</th>
      <td>2825</td>
      <td>Arabian Nights: Sindbad no Bouken (TV)</td>
      <td>Adventure, Fantasy, Magic, Romance</td>
      <td>TV</td>
      <td>52</td>
      <td>7.26</td>
      <td>2631</td>
    </tr>
  </tbody>
</table>
</div>


