# Content Based Recommender System

Configure the project. Indeed you create a dataset in csv format.


```python
! rm -rf *.csv
! unzip ./dataset/archive.zip
```

    Archive:  ./dataset/archive.zip
      inflating: anime.csv               
      inflating: rating_complete.csv     

    


Import needed libraries


```python
import pandas as pd

%matplotlib inline
```

Read data from csv files using pandas and store in data frame structure. Also shuffle data to have uniform distribution. 


```python
anime_df = pd.read_csv("anime.csv")
anime_df = anime_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
anime_df.head()
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
      <th>MAL_ID</th>
      <th>Name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Aired</th>
      <th>Premiered</th>
      <th>...</th>
      <th>Score-10</th>
      <th>Score-9</th>
      <th>Score-8</th>
      <th>Score-7</th>
      <th>Score-6</th>
      <th>Score-5</th>
      <th>Score-4</th>
      <th>Score-3</th>
      <th>Score-2</th>
      <th>Score-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40176</td>
      <td>Miru Tights: Cosplay Satsuei Tights</td>
      <td>6.53</td>
      <td>Ecchi, School</td>
      <td>Unknown</td>
      <td>みるタイツ コスプレ撮影 タイツ</td>
      <td>Special</td>
      <td>1</td>
      <td>Aug 23, 2019</td>
      <td>Unknown</td>
      <td>...</td>
      <td>875.0</td>
      <td>350.0</td>
      <td>762.0</td>
      <td>1526.0</td>
      <td>1542.0</td>
      <td>924.0</td>
      <td>384.0</td>
      <td>245.0</td>
      <td>162.0</td>
      <td>148.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13969</td>
      <td>Thermae Romae x Yoyogi Animation Gakuin Collab...</td>
      <td>6.29</td>
      <td>Comedy, Historical, Seinen</td>
      <td>Unknown</td>
      <td>テルマエ・ロマエｘ代々木アニメーション学院企業コラボレーション</td>
      <td>Special</td>
      <td>1</td>
      <td>Jul 9, 2012</td>
      <td>Unknown</td>
      <td>...</td>
      <td>35.0</td>
      <td>47.0</td>
      <td>114.0</td>
      <td>253.0</td>
      <td>240.0</td>
      <td>162.0</td>
      <td>63.0</td>
      <td>29.0</td>
      <td>10.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13459</td>
      <td>Ribbon-chan</td>
      <td>Unknown</td>
      <td>Comedy</td>
      <td>Unknown</td>
      <td>リボンちゃん</td>
      <td>TV</td>
      <td>24</td>
      <td>Apr 4, 2012 to Mar 27, 2013</td>
      <td>Spring 2012</td>
      <td>...</td>
      <td>7.0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>Unknown</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15617</td>
      <td>Jinrui wa Suitai Shimashita Specials</td>
      <td>7.23</td>
      <td>Comedy, Fantasy, Seinen</td>
      <td>Humanity Has Declined Specials</td>
      <td>人類は衰退しました</td>
      <td>Special</td>
      <td>6</td>
      <td>Sep 19, 2012 to Feb 20, 2013</td>
      <td>Unknown</td>
      <td>...</td>
      <td>451.0</td>
      <td>885.0</td>
      <td>2432.0</td>
      <td>3038.0</td>
      <td>1388.0</td>
      <td>588.0</td>
      <td>130.0</td>
      <td>38.0</td>
      <td>22.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19157</td>
      <td>Youkai Watch</td>
      <td>6.54</td>
      <td>Comedy, Demons, Kids, Supernatural</td>
      <td>Yo-kai Watch</td>
      <td>妖怪ウォッチ</td>
      <td>TV</td>
      <td>214</td>
      <td>Jan 8, 2014 to Mar 30, 2018</td>
      <td>Winter 2014</td>
      <td>...</td>
      <td>517.0</td>
      <td>532.0</td>
      <td>1141.0</td>
      <td>1912.0</td>
      <td>1636.0</td>
      <td>1196.0</td>
      <td>500.0</td>
      <td>228.0</td>
      <td>138.0</td>
      <td>125.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
rates_df = pd.read_csv("rating_complete.csv")
rates_df = rates_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
rates_df.head()
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
      <td>126602</td>
      <td>18199</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>162615</td>
      <td>39036</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25497</td>
      <td>34124</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>360</td>
      <td>39607</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1032</td>
      <td>2608</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(rates_df['user_id'].value_counts())
print("\n" + "#" * 80 + "\n")
print(rates_df['anime_id'].value_counts())
```

    user_id
    68042     4734
    10255     4509
    162615    4474
    189037    4260
    38143     3544
              ... 
    125529       1
    310701       1
    287547       1
    182161       1
    44941        1
    Name: count, Length: 27341, dtype: int64
    
    ################################################################################
    
    anime_id
    32219    30
    1914     30
    1720     30
    36672    30
    3162     30
             ..
    40959     1
    40674     1
    39685     1
    40594     1
    42144     1
    Name: count, Length: 16872, dtype: int64



```python
# summarize data
rates_df.describe() 
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
      <th>count</th>
      <td>447990.000000</td>
      <td>447990.000000</td>
      <td>447990.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>48705.908194</td>
      <td>19412.419686</td>
      <td>6.258184</td>
    </tr>
    <tr>
      <th>std</th>
      <td>78817.293788</td>
      <td>14646.009559</td>
      <td>2.143759</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1578.000000</td>
      <td>4654.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10457.000000</td>
      <td>17731.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61131.250000</td>
      <td>34122.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>353328.000000</td>
      <td>48456.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Data Cleaning


```python
anime_df.columns
```




    Index(['MAL_ID', 'Name', 'Score', 'Genres', 'English name', 'Japanese name',
           'Type', 'Episodes', 'Aired', 'Premiered', 'Producers', 'Licensors',
           'Studios', 'Source', 'Duration', 'Rating', 'Ranked', 'Popularity',
           'Members', 'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped',
           'Plan to Watch', 'Score-10', 'Score-9', 'Score-8', 'Score-7', 'Score-6',
           'Score-5', 'Score-4', 'Score-3', 'Score-2', 'Score-1'],
          dtype='object')




```python
anime_df['Genres'] = anime_df.Genres.str.split(',')
anime_df.head()
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
      <th>MAL_ID</th>
      <th>Name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Aired</th>
      <th>Premiered</th>
      <th>...</th>
      <th>Score-10</th>
      <th>Score-9</th>
      <th>Score-8</th>
      <th>Score-7</th>
      <th>Score-6</th>
      <th>Score-5</th>
      <th>Score-4</th>
      <th>Score-3</th>
      <th>Score-2</th>
      <th>Score-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40176</td>
      <td>Miru Tights: Cosplay Satsuei Tights</td>
      <td>6.53</td>
      <td>[Ecchi,  School]</td>
      <td>Unknown</td>
      <td>みるタイツ コスプレ撮影 タイツ</td>
      <td>Special</td>
      <td>1</td>
      <td>Aug 23, 2019</td>
      <td>Unknown</td>
      <td>...</td>
      <td>875.0</td>
      <td>350.0</td>
      <td>762.0</td>
      <td>1526.0</td>
      <td>1542.0</td>
      <td>924.0</td>
      <td>384.0</td>
      <td>245.0</td>
      <td>162.0</td>
      <td>148.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13969</td>
      <td>Thermae Romae x Yoyogi Animation Gakuin Collab...</td>
      <td>6.29</td>
      <td>[Comedy,  Historical,  Seinen]</td>
      <td>Unknown</td>
      <td>テルマエ・ロマエｘ代々木アニメーション学院企業コラボレーション</td>
      <td>Special</td>
      <td>1</td>
      <td>Jul 9, 2012</td>
      <td>Unknown</td>
      <td>...</td>
      <td>35.0</td>
      <td>47.0</td>
      <td>114.0</td>
      <td>253.0</td>
      <td>240.0</td>
      <td>162.0</td>
      <td>63.0</td>
      <td>29.0</td>
      <td>10.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13459</td>
      <td>Ribbon-chan</td>
      <td>Unknown</td>
      <td>[Comedy]</td>
      <td>Unknown</td>
      <td>リボンちゃん</td>
      <td>TV</td>
      <td>24</td>
      <td>Apr 4, 2012 to Mar 27, 2013</td>
      <td>Spring 2012</td>
      <td>...</td>
      <td>7.0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>Unknown</td>
      <td>2.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15617</td>
      <td>Jinrui wa Suitai Shimashita Specials</td>
      <td>7.23</td>
      <td>[Comedy,  Fantasy,  Seinen]</td>
      <td>Humanity Has Declined Specials</td>
      <td>人類は衰退しました</td>
      <td>Special</td>
      <td>6</td>
      <td>Sep 19, 2012 to Feb 20, 2013</td>
      <td>Unknown</td>
      <td>...</td>
      <td>451.0</td>
      <td>885.0</td>
      <td>2432.0</td>
      <td>3038.0</td>
      <td>1388.0</td>
      <td>588.0</td>
      <td>130.0</td>
      <td>38.0</td>
      <td>22.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19157</td>
      <td>Youkai Watch</td>
      <td>6.54</td>
      <td>[Comedy,  Demons,  Kids,  Supernatural]</td>
      <td>Yo-kai Watch</td>
      <td>妖怪ウォッチ</td>
      <td>TV</td>
      <td>214</td>
      <td>Jan 8, 2014 to Mar 30, 2018</td>
      <td>Winter 2014</td>
      <td>...</td>
      <td>517.0</td>
      <td>532.0</td>
      <td>1141.0</td>
      <td>1912.0</td>
      <td>1636.0</td>
      <td>1196.0</td>
      <td>500.0</td>
      <td>228.0</td>
      <td>138.0</td>
      <td>125.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>




```python
animeWithGenres_df = anime_df.drop(columns=['Score-10', 'Score-9', 'Score-8', 'Score-7', 'Score-6', 'Score-5', 'Score-4', 'Score-3', 'Score-2', 'Score-1'])

for index, row in anime_df.iterrows():
    for genre in row['Genres']:
        animeWithGenres_df.at[index, genre] = 1

#Filling in the NaN values with 0 
animeWithGenres_df = animeWithGenres_df.fillna(0)


animeWithGenres_df
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
      <th>MAL_ID</th>
      <th>Name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Aired</th>
      <th>Premiered</th>
      <th>...</th>
      <th>Super Power</th>
      <th>Psychological</th>
      <th>Yuri</th>
      <th>Samurai</th>
      <th>Martial Arts</th>
      <th>Josei</th>
      <th>Shoujo</th>
      <th>Seinen</th>
      <th>Yaoi</th>
      <th>Shounen Ai</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40176</td>
      <td>Miru Tights: Cosplay Satsuei Tights</td>
      <td>6.53</td>
      <td>[Ecchi,  School]</td>
      <td>Unknown</td>
      <td>みるタイツ コスプレ撮影 タイツ</td>
      <td>Special</td>
      <td>1</td>
      <td>Aug 23, 2019</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13969</td>
      <td>Thermae Romae x Yoyogi Animation Gakuin Collab...</td>
      <td>6.29</td>
      <td>[Comedy,  Historical,  Seinen]</td>
      <td>Unknown</td>
      <td>テルマエ・ロマエｘ代々木アニメーション学院企業コラボレーション</td>
      <td>Special</td>
      <td>1</td>
      <td>Jul 9, 2012</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13459</td>
      <td>Ribbon-chan</td>
      <td>Unknown</td>
      <td>[Comedy]</td>
      <td>Unknown</td>
      <td>リボンちゃん</td>
      <td>TV</td>
      <td>24</td>
      <td>Apr 4, 2012 to Mar 27, 2013</td>
      <td>Spring 2012</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15617</td>
      <td>Jinrui wa Suitai Shimashita Specials</td>
      <td>7.23</td>
      <td>[Comedy,  Fantasy,  Seinen]</td>
      <td>Humanity Has Declined Specials</td>
      <td>人類は衰退しました</td>
      <td>Special</td>
      <td>6</td>
      <td>Sep 19, 2012 to Feb 20, 2013</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19157</td>
      <td>Youkai Watch</td>
      <td>6.54</td>
      <td>[Comedy,  Demons,  Kids,  Supernatural]</td>
      <td>Yo-kai Watch</td>
      <td>妖怪ウォッチ</td>
      <td>TV</td>
      <td>214</td>
      <td>Jan 8, 2014 to Mar 30, 2018</td>
      <td>Winter 2014</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17557</th>
      <td>32238</td>
      <td>Watashi wa, Kairaku Izonshou</td>
      <td>6.2</td>
      <td>[Hentai]</td>
      <td>Unknown</td>
      <td>私は、快楽依存症</td>
      <td>OVA</td>
      <td>2</td>
      <td>Feb 26, 2016 to May 20, 2016</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17558</th>
      <td>33552</td>
      <td>Mameshiba Bangai-hen</td>
      <td>5.75</td>
      <td>[Music,  Comedy]</td>
      <td>Unknown</td>
      <td>豆しば番外編</td>
      <td>Special</td>
      <td>5</td>
      <td>2008 to Jun 20, 2019</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17559</th>
      <td>8476</td>
      <td>Otome Youkai Zakuro</td>
      <td>7.47</td>
      <td>[Demons,  Historical,  Military,  Romance,  Se...</td>
      <td>Zakuro</td>
      <td>おとめ妖怪 ざくろ</td>
      <td>TV</td>
      <td>13</td>
      <td>Oct 5, 2010 to Dec 28, 2010</td>
      <td>Fall 2010</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17560</th>
      <td>953</td>
      <td>Jyu Oh Sei</td>
      <td>7.26</td>
      <td>[Action,  Sci-Fi,  Adventure,  Mystery,  Drama...</td>
      <td>Jyu-Oh-Sei:Planet of the Beast King</td>
      <td>獣王星</td>
      <td>TV</td>
      <td>11</td>
      <td>Apr 14, 2006 to Jun 23, 2006</td>
      <td>Spring 2006</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17561</th>
      <td>39769</td>
      <td>Kimi ni Sekai</td>
      <td>6.7</td>
      <td>[Sci-Fi,  Music,  Fantasy]</td>
      <td>Unknown</td>
      <td>君に世界</td>
      <td>Music</td>
      <td>1</td>
      <td>Apr 20, 2019</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>17562 rows × 109 columns</p>
</div>



## Get the input from user


```python
userInput = [
            {'Title':'ERASED', 'Rating':10},
            {'Title':'Violet Evergarden', 'Rating':9.5},
            {'Title':'Goblin Slayer', 'Rating':6},
            {'Title':"Berserk", 'Rating':8},
            {'Title':'Attack on Titan', 'Rating':7},
            {'Title':"Tokyo Ghoul", 'Rating':6.5},
            {'Title':"Orange", 'Rating':6},
            {'Title':"Death Parade", 'Rating':8},
            {'Title':"Death Note", 'Rating':7.5},
            {'Title':"Bungou Stray Dogs", 'Rating':7.5},
            {'Title':"Weathering With You", 'Rating':8},
            {'Title':"Your Name", 'Rating':8},
            {'Title':"I want to eat your pancreas", 'Rating':8.5},
            {'Title':"Princess Mononoke", 'Rating':7.5},
            {'Title':"Spirited Away", 'Rating':7.5},
            {'Title':"A Silent Voice", 'Rating':8.5},
            {'Title':"Ao Haru Ride", 'Rating':5.5},
            {'Title':"The Girl Who Leapt Through Time", 'Rating':7},
            {'Title':"Another", 'Rating':7.5},
            {'Title':"Demon Slayer", 'Rating':7},
            {'Title':"Your Lie in April", 'Rating':8},
            {'Title':"5 Centimeters per Second", 'Rating':6},
            {'Title':"The Anthem of the Heart", 'Rating':7.5},
            {'Title':"Evangelion", 'Rating':5}
         ] 
inputAnime = pd.DataFrame(userInput)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ERASED</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Violet Evergarden</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Goblin Slayer</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Berserk</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Attack on Titan</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tokyo Ghoul</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Orange</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Death Parade</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Death Note</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bungou Stray Dogs</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Weathering With You</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Your Name</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>I want to eat your pancreas</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Princess Mononoke</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Spirited Away</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>A Silent Voice</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ao Haru Ride</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>The Girl Who Leapt Through Time</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Another</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Demon Slayer</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Your Lie in April</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5 Centimeters per Second</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>The Anthem of the Heart</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Evangelion</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
animeWithGenres_df.columns[:25]
```




    Index(['MAL_ID', 'Name', 'Score', 'Genres', 'English name', 'Japanese name',
           'Type', 'Episodes', 'Aired', 'Premiered', 'Producers', 'Licensors',
           'Studios', 'Source', 'Duration', 'Rating', 'Ranked', 'Popularity',
           'Members', 'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped',
           'Plan to Watch'],
          dtype='object')




```python
anime_df['Name_lower'] = anime_df['Name'].str.lower()
anime_df['English_lower'] = anime_df['English name'].str.lower()
inputAnime['Title_lower'] = inputAnime['Title'].str.lower()

def find_best_match(Title, anime_df):
    if Title in anime_df['Name_lower'].values:
        return anime_df[anime_df['Name_lower'] == Title]['Name'].values[0]
    elif Title in anime_df['English_lower'].values:
        return anime_df[anime_df['English_lower'] == Title]['Name'].values[0]
    else:
        for idx, row in anime_df.iterrows():
            if Title in row['Name_lower'] or Title in row['English_lower']:
                return row['Name']
    return None

inputAnime['best_match'] = inputAnime['Title_lower'].apply(find_best_match, anime_df=anime_df)
inputAnime = inputAnime.dropna(subset=['best_match'])
inputAnime['Title'] = inputAnime['best_match']
anime_df.drop(['Name_lower', 'English_lower'], axis=1, inplace=True)
inputAnime.drop(['Title_lower', 'best_match'], axis=1, inplace=True)

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Boku dake ga Inai Machi</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Violet Evergarden</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Goblin Slayer</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Berserk</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Shingeki no Kyojin</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Tokyo Ghoul</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Orange</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Death Parade</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Death Note</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Bungou Stray Dogs</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Tenki no Ko</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Kimi no Na wa.</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Kimi no Suizou wo Tabetai</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Mononoke Hime</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Sen to Chihiro no Kamikakushi</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Koe no Katachi</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ao Haru Ride</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Toki wo Kakeru Shoujo</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Another</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Kimetsu no Yaiba</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Shigatsu wa Kimi no Uso</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Byousoku 5 Centimeter</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Kokoro ga Sakebitagatterunda.</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Schick x Evangelion</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
inputAnime = inputAnime.merge(anime_df[['MAL_ID', 'Name']], left_on='Title', right_on='Name', how='left')
inputAnime = inputAnime[['MAL_ID', 'Title', 'Rating']]

inputAnime = inputAnime.sort_values(by='MAL_ID')
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
      <th>MAL_ID</th>
      <th>Title</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>164</td>
      <td>Mononoke Hime</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>199</td>
      <td>Sen to Chihiro no Kamikakushi</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1535</td>
      <td>Death Note</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1689</td>
      <td>Byousoku 5 Centimeter</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2236</td>
      <td>Toki wo Kakeru Shoujo</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11111</td>
      <td>Another</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>16498</td>
      <td>Shingeki no Kyojin</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>21995</td>
      <td>Ao Haru Ride</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>22319</td>
      <td>Tokyo Ghoul</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>23273</td>
      <td>Shigatsu wa Kimi no Uso</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>28223</td>
      <td>Death Parade</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>28725</td>
      <td>Kokoro ga Sakebitagatterunda.</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>28851</td>
      <td>Koe no Katachi</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>31043</td>
      <td>Boku dake ga Inai Machi</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>31115</td>
      <td>Schick x Evangelion</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>31478</td>
      <td>Bungou Stray Dogs</td>
      <td>7.5</td>
    </tr>
    <tr>
      <th>16</th>
      <td>32281</td>
      <td>Kimi no Na wa.</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>32379</td>
      <td>Berserk</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>32729</td>
      <td>Orange</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>33352</td>
      <td>Violet Evergarden</td>
      <td>9.5</td>
    </tr>
    <tr>
      <th>20</th>
      <td>36098</td>
      <td>Kimi no Suizou wo Tabetai</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>21</th>
      <td>37349</td>
      <td>Goblin Slayer</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>38000</td>
      <td>Kimetsu no Yaiba</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>38826</td>
      <td>Tenki no Ko</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
userAnimes = animeWithGenres_df[animeWithGenres_df['MAL_ID'].isin(inputAnime['MAL_ID'].tolist())]
userAnimes = userAnimes.sort_values(by='MAL_ID')
userAnimes
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
      <th>MAL_ID</th>
      <th>Name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Aired</th>
      <th>Premiered</th>
      <th>...</th>
      <th>Super Power</th>
      <th>Psychological</th>
      <th>Yuri</th>
      <th>Samurai</th>
      <th>Martial Arts</th>
      <th>Josei</th>
      <th>Shoujo</th>
      <th>Seinen</th>
      <th>Yaoi</th>
      <th>Shounen Ai</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12056</th>
      <td>164</td>
      <td>Mononoke Hime</td>
      <td>8.72</td>
      <td>[Action,  Adventure,  Fantasy]</td>
      <td>Princess Mononoke</td>
      <td>もののけ姫</td>
      <td>Movie</td>
      <td>1</td>
      <td>Jul 12, 1997</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2741</th>
      <td>199</td>
      <td>Sen to Chihiro no Kamikakushi</td>
      <td>8.83</td>
      <td>[Adventure,  Supernatural,  Drama]</td>
      <td>Spirited Away</td>
      <td>千と千尋の神隠し</td>
      <td>Movie</td>
      <td>1</td>
      <td>Jul 20, 2001</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2052</th>
      <td>1535</td>
      <td>Death Note</td>
      <td>8.63</td>
      <td>[Mystery,  Police,  Psychological,  Supernatur...</td>
      <td>Death Note</td>
      <td>デスノート</td>
      <td>TV</td>
      <td>37</td>
      <td>Oct 4, 2006 to Jun 27, 2007</td>
      <td>Fall 2006</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2140</th>
      <td>1689</td>
      <td>Byousoku 5 Centimeter</td>
      <td>7.73</td>
      <td>[Drama,  Romance,  Slice of Life]</td>
      <td>5 Centimeters Per Second</td>
      <td>秒速５センチメートル</td>
      <td>Movie</td>
      <td>3</td>
      <td>Mar 3, 2007</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17371</th>
      <td>2236</td>
      <td>Toki wo Kakeru Shoujo</td>
      <td>8.2</td>
      <td>[Adventure,  Drama,  Romance,  Sci-Fi]</td>
      <td>The Girl Who Leapt Through Time</td>
      <td>時をかける少女</td>
      <td>Movie</td>
      <td>1</td>
      <td>Jul 15, 2006</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5164</th>
      <td>11111</td>
      <td>Another</td>
      <td>7.55</td>
      <td>[Mystery,  Horror,  Supernatural,  Thriller,  ...</td>
      <td>Another</td>
      <td>アナザー</td>
      <td>TV</td>
      <td>12</td>
      <td>Jan 10, 2012 to Mar 27, 2012</td>
      <td>Winter 2012</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3936</th>
      <td>16498</td>
      <td>Shingeki no Kyojin</td>
      <td>8.48</td>
      <td>[Action,  Military,  Mystery,  Super Power,  D...</td>
      <td>Attack on Titan</td>
      <td>進撃の巨人</td>
      <td>TV</td>
      <td>25</td>
      <td>Apr 7, 2013 to Sep 29, 2013</td>
      <td>Spring 2013</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16704</th>
      <td>21995</td>
      <td>Ao Haru Ride</td>
      <td>7.67</td>
      <td>[Comedy,  Drama,  Romance,  School,  Shoujo,  ...</td>
      <td>Blue Spring Ride</td>
      <td>アオハライド</td>
      <td>TV</td>
      <td>12</td>
      <td>Jul 8, 2014 to Sep 23, 2014</td>
      <td>Summer 2014</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16584</th>
      <td>22319</td>
      <td>Tokyo Ghoul</td>
      <td>7.81</td>
      <td>[Action,  Mystery,  Horror,  Psychological,  S...</td>
      <td>Tokyo Ghoul</td>
      <td>東京喰種-トーキョーグール-</td>
      <td>TV</td>
      <td>12</td>
      <td>Jul 4, 2014 to Sep 19, 2014</td>
      <td>Summer 2014</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4752</th>
      <td>23273</td>
      <td>Shigatsu wa Kimi no Uso</td>
      <td>8.74</td>
      <td>[Drama,  Music,  Romance,  School,  Shounen]</td>
      <td>Your Lie in April</td>
      <td>四月は君の嘘</td>
      <td>TV</td>
      <td>22</td>
      <td>Oct 10, 2014 to Mar 20, 2015</td>
      <td>Fall 2014</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7045</th>
      <td>28223</td>
      <td>Death Parade</td>
      <td>8.2</td>
      <td>[Game,  Mystery,  Psychological,  Drama,  Thri...</td>
      <td>Death Parade</td>
      <td>デス・パレード</td>
      <td>TV</td>
      <td>12</td>
      <td>Jan 10, 2015 to Mar 28, 2015</td>
      <td>Winter 2015</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7640</th>
      <td>28725</td>
      <td>Kokoro ga Sakebitagatterunda.</td>
      <td>7.96</td>
      <td>[Drama,  Romance,  School]</td>
      <td>The Anthem of the Heart</td>
      <td>心が叫びたがってるんだ。</td>
      <td>Movie</td>
      <td>1</td>
      <td>Sep 19, 2015</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11419</th>
      <td>28851</td>
      <td>Koe no Katachi</td>
      <td>9.0</td>
      <td>[Drama,  School,  Shounen]</td>
      <td>A Silent Voice</td>
      <td>聲の形</td>
      <td>Movie</td>
      <td>1</td>
      <td>Sep 17, 2016</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4293</th>
      <td>31043</td>
      <td>Boku dake ga Inai Machi</td>
      <td>8.37</td>
      <td>[Mystery,  Psychological,  Supernatural,  Seinen]</td>
      <td>ERASED</td>
      <td>僕だけがいない街</td>
      <td>TV</td>
      <td>12</td>
      <td>Jan 8, 2016 to Mar 25, 2016</td>
      <td>Winter 2016</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>537</th>
      <td>31115</td>
      <td>Schick x Evangelion</td>
      <td>6.06</td>
      <td>[Comedy,  Parody]</td>
      <td>Unknown</td>
      <td>Schick × エヴァンゲリオン</td>
      <td>Special</td>
      <td>2</td>
      <td>May 11, 2015</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9102</th>
      <td>31478</td>
      <td>Bungou Stray Dogs</td>
      <td>7.79</td>
      <td>[Action,  Comedy,  Mystery,  Seinen,  Super Po...</td>
      <td>Bungo Stray Dogs</td>
      <td>文豪ストレイドッグス</td>
      <td>TV</td>
      <td>12</td>
      <td>Apr 7, 2016 to Jun 23, 2016</td>
      <td>Spring 2016</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13520</th>
      <td>32281</td>
      <td>Kimi no Na wa.</td>
      <td>8.96</td>
      <td>[Romance,  Supernatural,  School,  Drama]</td>
      <td>Your Name.</td>
      <td>君の名は。</td>
      <td>Movie</td>
      <td>1</td>
      <td>Aug 26, 2016</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1844</th>
      <td>32379</td>
      <td>Berserk</td>
      <td>6.39</td>
      <td>[Action,  Adventure,  Demons,  Drama,  Fantasy...</td>
      <td>Berserk</td>
      <td>ベルセルク</td>
      <td>TV</td>
      <td>12</td>
      <td>Jul 1, 2016 to Sep 16, 2016</td>
      <td>Summer 2016</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5178</th>
      <td>32729</td>
      <td>Orange</td>
      <td>7.62</td>
      <td>[Sci-Fi,  Drama,  Romance,  School,  Shoujo]</td>
      <td>Orange</td>
      <td>orange（オレンジ）</td>
      <td>TV</td>
      <td>13</td>
      <td>Jul 4, 2016 to Sep 26, 2016</td>
      <td>Summer 2016</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1933</th>
      <td>33352</td>
      <td>Violet Evergarden</td>
      <td>8.64</td>
      <td>[Slice of Life,  Drama,  Fantasy]</td>
      <td>Violet Evergarden</td>
      <td>ヴァイオレット・エヴァーガーデン</td>
      <td>TV</td>
      <td>13</td>
      <td>Jan 11, 2018 to Apr 5, 2018</td>
      <td>Winter 2018</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14997</th>
      <td>36098</td>
      <td>Kimi no Suizou wo Tabetai</td>
      <td>8.59</td>
      <td>[Drama]</td>
      <td>I want to eat your pancreas</td>
      <td>君の膵臓をたべたい</td>
      <td>Movie</td>
      <td>1</td>
      <td>Sep 1, 2018</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9444</th>
      <td>37349</td>
      <td>Goblin Slayer</td>
      <td>7.46</td>
      <td>[Action,  Adventure,  Fantasy]</td>
      <td>Goblin Slayer</td>
      <td>ゴブリンスレイヤー</td>
      <td>TV</td>
      <td>12</td>
      <td>Oct 7, 2018 to Dec 30, 2018</td>
      <td>Fall 2018</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12900</th>
      <td>38000</td>
      <td>Kimetsu no Yaiba</td>
      <td>8.62</td>
      <td>[Action,  Demons,  Historical,  Shounen,  Supe...</td>
      <td>Demon Slayer:Kimetsu no Yaiba</td>
      <td>鬼滅の刃</td>
      <td>TV</td>
      <td>26</td>
      <td>Apr 6, 2019 to Sep 28, 2019</td>
      <td>Spring 2019</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17023</th>
      <td>38826</td>
      <td>Tenki no Ko</td>
      <td>8.41</td>
      <td>[Slice of Life,  Drama,  Romance,  Fantasy]</td>
      <td>Weathering With You</td>
      <td>天気の子</td>
      <td>Movie</td>
      <td>1</td>
      <td>Jul 19, 2019</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>24 rows × 109 columns</p>
</div>



### Remove the anime's that the user has seen from the whole list.


```python
animeWithGenres_df = animeWithGenres_df[~animeWithGenres_df.isin(userAnimes).all(axis=1)]
animeWithGenres_df
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
      <th>MAL_ID</th>
      <th>Name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Aired</th>
      <th>Premiered</th>
      <th>...</th>
      <th>Super Power</th>
      <th>Psychological</th>
      <th>Yuri</th>
      <th>Samurai</th>
      <th>Martial Arts</th>
      <th>Josei</th>
      <th>Shoujo</th>
      <th>Seinen</th>
      <th>Yaoi</th>
      <th>Shounen Ai</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40176</td>
      <td>Miru Tights: Cosplay Satsuei Tights</td>
      <td>6.53</td>
      <td>[Ecchi,  School]</td>
      <td>Unknown</td>
      <td>みるタイツ コスプレ撮影 タイツ</td>
      <td>Special</td>
      <td>1</td>
      <td>Aug 23, 2019</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13969</td>
      <td>Thermae Romae x Yoyogi Animation Gakuin Collab...</td>
      <td>6.29</td>
      <td>[Comedy,  Historical,  Seinen]</td>
      <td>Unknown</td>
      <td>テルマエ・ロマエｘ代々木アニメーション学院企業コラボレーション</td>
      <td>Special</td>
      <td>1</td>
      <td>Jul 9, 2012</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13459</td>
      <td>Ribbon-chan</td>
      <td>Unknown</td>
      <td>[Comedy]</td>
      <td>Unknown</td>
      <td>リボンちゃん</td>
      <td>TV</td>
      <td>24</td>
      <td>Apr 4, 2012 to Mar 27, 2013</td>
      <td>Spring 2012</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15617</td>
      <td>Jinrui wa Suitai Shimashita Specials</td>
      <td>7.23</td>
      <td>[Comedy,  Fantasy,  Seinen]</td>
      <td>Humanity Has Declined Specials</td>
      <td>人類は衰退しました</td>
      <td>Special</td>
      <td>6</td>
      <td>Sep 19, 2012 to Feb 20, 2013</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19157</td>
      <td>Youkai Watch</td>
      <td>6.54</td>
      <td>[Comedy,  Demons,  Kids,  Supernatural]</td>
      <td>Yo-kai Watch</td>
      <td>妖怪ウォッチ</td>
      <td>TV</td>
      <td>214</td>
      <td>Jan 8, 2014 to Mar 30, 2018</td>
      <td>Winter 2014</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17557</th>
      <td>32238</td>
      <td>Watashi wa, Kairaku Izonshou</td>
      <td>6.2</td>
      <td>[Hentai]</td>
      <td>Unknown</td>
      <td>私は、快楽依存症</td>
      <td>OVA</td>
      <td>2</td>
      <td>Feb 26, 2016 to May 20, 2016</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17558</th>
      <td>33552</td>
      <td>Mameshiba Bangai-hen</td>
      <td>5.75</td>
      <td>[Music,  Comedy]</td>
      <td>Unknown</td>
      <td>豆しば番外編</td>
      <td>Special</td>
      <td>5</td>
      <td>2008 to Jun 20, 2019</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17559</th>
      <td>8476</td>
      <td>Otome Youkai Zakuro</td>
      <td>7.47</td>
      <td>[Demons,  Historical,  Military,  Romance,  Se...</td>
      <td>Zakuro</td>
      <td>おとめ妖怪 ざくろ</td>
      <td>TV</td>
      <td>13</td>
      <td>Oct 5, 2010 to Dec 28, 2010</td>
      <td>Fall 2010</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17560</th>
      <td>953</td>
      <td>Jyu Oh Sei</td>
      <td>7.26</td>
      <td>[Action,  Sci-Fi,  Adventure,  Mystery,  Drama...</td>
      <td>Jyu-Oh-Sei:Planet of the Beast King</td>
      <td>獣王星</td>
      <td>TV</td>
      <td>11</td>
      <td>Apr 14, 2006 to Jun 23, 2006</td>
      <td>Spring 2006</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17561</th>
      <td>39769</td>
      <td>Kimi ni Sekai</td>
      <td>6.7</td>
      <td>[Sci-Fi,  Music,  Fantasy]</td>
      <td>Unknown</td>
      <td>君に世界</td>
      <td>Music</td>
      <td>1</td>
      <td>Apr 20, 2019</td>
      <td>Unknown</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>17538 rows × 109 columns</p>
</div>




```python
userAnimes = userAnimes.reset_index(drop=True)

userGenreTable = userAnimes.iloc[:, 25:]
userGenreTable
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
      <th>Ecchi</th>
      <th>School</th>
      <th>Comedy</th>
      <th>Historical</th>
      <th>Seinen</th>
      <th>Fantasy</th>
      <th>Demons</th>
      <th>Kids</th>
      <th>Supernatural</th>
      <th>Slice of Life</th>
      <th>...</th>
      <th>Super Power</th>
      <th>Psychological</th>
      <th>Yuri</th>
      <th>Samurai</th>
      <th>Martial Arts</th>
      <th>Josei</th>
      <th>Shoujo</th>
      <th>Seinen</th>
      <th>Yaoi</th>
      <th>Shounen Ai</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>24 rows × 84 columns</p>
</div>




```python
inputAnime.Rating
```




    0      7.5
    1      7.5
    2      7.5
    3      6.0
    4      7.0
    5      7.5
    6      7.0
    7      5.5
    8      6.5
    9      8.0
    10     8.0
    11     7.5
    12     8.5
    13    10.0
    14     5.0
    15     7.5
    16     8.0
    17     8.0
    18     6.0
    19     9.5
    20     8.5
    21     6.0
    22     7.0
    23     8.0
    Name: Rating, dtype: float64




```python
userProfile = userGenreTable.transpose().dot(inputAnime['Rating'])
userProfile
```




    Ecchi           0.0
     School        51.0
    Comedy         10.5
     Historical     7.0
     Seinen        32.0
                   ... 
    Josei           0.0
    Shoujo          0.0
    Seinen          0.0
    Yaoi            0.0
    Shounen Ai      0.0
    Length: 84, dtype: float64




```python
genreTable = animeWithGenres_df.set_index(animeWithGenres_df['MAL_ID'])
genreTable = genreTable.iloc[:, 25:]
genreTable.head()
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
      <th>Ecchi</th>
      <th>School</th>
      <th>Comedy</th>
      <th>Historical</th>
      <th>Seinen</th>
      <th>Fantasy</th>
      <th>Demons</th>
      <th>Kids</th>
      <th>Supernatural</th>
      <th>Slice of Life</th>
      <th>...</th>
      <th>Super Power</th>
      <th>Psychological</th>
      <th>Yuri</th>
      <th>Samurai</th>
      <th>Martial Arts</th>
      <th>Josei</th>
      <th>Shoujo</th>
      <th>Seinen</th>
      <th>Yaoi</th>
      <th>Shounen Ai</th>
    </tr>
    <tr>
      <th>MAL_ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40176</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13969</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13459</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15617</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19157</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 84 columns</p>
</div>




```python
genreTable.shape
```




    (17538, 84)




```python
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()
```




    MAL_ID
    40176    0.064721
    13969    0.062817
    13459    0.013325
    15617    0.112310
    19157    0.120558
    dtype: float64




```python
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
recommendationTable_df.head()
```




    MAL_ID
    35009    0.517132
    33       0.517132
    449      0.496193
    450      0.496193
    451      0.496193
    dtype: float64




```python
top_mal_ids = recommendationTable_df.head(10).keys()
anime_df.set_index('MAL_ID').loc[top_mal_ids].reset_index()
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
      <th>MAL_ID</th>
      <th>Name</th>
      <th>Score</th>
      <th>Genres</th>
      <th>English name</th>
      <th>Japanese name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Aired</th>
      <th>Premiered</th>
      <th>...</th>
      <th>Score-10</th>
      <th>Score-9</th>
      <th>Score-8</th>
      <th>Score-7</th>
      <th>Score-6</th>
      <th>Score-5</th>
      <th>Score-4</th>
      <th>Score-3</th>
      <th>Score-2</th>
      <th>Score-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35009</td>
      <td>Berserk Recap</td>
      <td>6.02</td>
      <td>[Action,  Adventure,  Demons,  Drama,  Fantasy...</td>
      <td>Unknown</td>
      <td>ベルセルク 第1期ダイジェスト映像</td>
      <td>Special</td>
      <td>1</td>
      <td>Mar 3, 2017</td>
      <td>Unknown</td>
      <td>...</td>
      <td>373.0</td>
      <td>212.0</td>
      <td>433.0</td>
      <td>797.0</td>
      <td>1019.0</td>
      <td>663.0</td>
      <td>293.0</td>
      <td>183.0</td>
      <td>140.0</td>
      <td>231.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>Kenpuu Denki Berserk</td>
      <td>8.49</td>
      <td>[Action,  Adventure,  Demons,  Drama,  Fantasy...</td>
      <td>Berserk</td>
      <td>剣風伝奇ベルセルク</td>
      <td>TV</td>
      <td>25</td>
      <td>Oct 8, 1997 to Apr 1, 1998</td>
      <td>Fall 1997</td>
      <td>...</td>
      <td>58627.0</td>
      <td>65906.0</td>
      <td>60815.0</td>
      <td>29055.0</td>
      <td>9477.0</td>
      <td>3899.0</td>
      <td>1748.0</td>
      <td>671.0</td>
      <td>456.0</td>
      <td>842.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>449</td>
      <td>InuYasha Movie 4: Guren no Houraijima</td>
      <td>7.54</td>
      <td>[Action,  Adventure,  Comedy,  Historical,  De...</td>
      <td>InuYasha the Movie 4:Fire on the Mystic Island</td>
      <td>犬夜叉 紅蓮の蓬莱島</td>
      <td>Movie</td>
      <td>1</td>
      <td>Dec 23, 2004</td>
      <td>Unknown</td>
      <td>...</td>
      <td>5230.0</td>
      <td>6127.0</td>
      <td>9865.0</td>
      <td>11837.0</td>
      <td>5135.0</td>
      <td>2190.0</td>
      <td>671.0</td>
      <td>225.0</td>
      <td>92.0</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>450</td>
      <td>InuYasha Movie 2: Kagami no Naka no Mugenjo</td>
      <td>7.66</td>
      <td>[Action,  Adventure,  Comedy,  Historical,  De...</td>
      <td>InuYasha the Movie 2:The Castle Beyond the Loo...</td>
      <td>犬夜叉 鏡の中の夢幻城</td>
      <td>Movie</td>
      <td>1</td>
      <td>Dec 21, 2002</td>
      <td>Unknown</td>
      <td>...</td>
      <td>6722.0</td>
      <td>7566.0</td>
      <td>11990.0</td>
      <td>12862.0</td>
      <td>5409.0</td>
      <td>2184.0</td>
      <td>607.0</td>
      <td>206.0</td>
      <td>96.0</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>451</td>
      <td>InuYasha Movie 3: Tenka Hadou no Ken</td>
      <td>7.8</td>
      <td>[Action,  Adventure,  Comedy,  Historical,  De...</td>
      <td>InuYasha the Movie 3:Swords of an Honorable Ruler</td>
      <td>犬夜叉 天下覇道の剣</td>
      <td>Movie</td>
      <td>1</td>
      <td>Dec 20, 2003</td>
      <td>Unknown</td>
      <td>...</td>
      <td>6718.0</td>
      <td>7647.0</td>
      <td>11985.0</td>
      <td>11322.0</td>
      <td>4397.0</td>
      <td>1687.0</td>
      <td>395.0</td>
      <td>160.0</td>
      <td>60.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>452</td>
      <td>InuYasha Movie 1: Toki wo Koeru Omoi</td>
      <td>7.56</td>
      <td>[Action,  Adventure,  Comedy,  Historical,  De...</td>
      <td>InuYasha the Movie:Affections Touching Across ...</td>
      <td>犬夜叉 時代を越える想い</td>
      <td>Movie</td>
      <td>1</td>
      <td>Dec 22, 2001</td>
      <td>Unknown</td>
      <td>...</td>
      <td>6033.0</td>
      <td>6802.0</td>
      <td>11048.0</td>
      <td>13002.0</td>
      <td>5767.0</td>
      <td>2369.0</td>
      <td>620.0</td>
      <td>255.0</td>
      <td>104.0</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>969</td>
      <td>Tsubasa Chronicle 2nd Season</td>
      <td>7.6</td>
      <td>[Action,  Adventure,  Fantasy,  Romance,  Supe...</td>
      <td>Tsubasa RESERVoir CHRoNiCLE Season Two</td>
      <td>ツバサ・クロニクル 第2シリーズ</td>
      <td>TV</td>
      <td>26</td>
      <td>Apr 29, 2006 to Nov 4, 2006</td>
      <td>Spring 2006</td>
      <td>...</td>
      <td>6346.0</td>
      <td>8909.0</td>
      <td>14525.0</td>
      <td>13920.0</td>
      <td>6024.0</td>
      <td>2757.0</td>
      <td>1101.0</td>
      <td>401.0</td>
      <td>215.0</td>
      <td>139.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4938</td>
      <td>Tsubasa: Shunraiki</td>
      <td>8.13</td>
      <td>[Action,  Adventure,  Mystery,  Supernatural, ...</td>
      <td>Tsubasa RESERVoir CHRoNiCLE:Spring Thunder Chr...</td>
      <td>ツバサ 春雷記</td>
      <td>OVA</td>
      <td>2</td>
      <td>Mar 17, 2009 to May 15, 2009</td>
      <td>Unknown</td>
      <td>...</td>
      <td>4738.0</td>
      <td>6561.0</td>
      <td>8069.0</td>
      <td>5065.0</td>
      <td>1749.0</td>
      <td>691.0</td>
      <td>186.0</td>
      <td>74.0</td>
      <td>32.0</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>34055</td>
      <td>Berserk 2nd Season</td>
      <td>6.69</td>
      <td>[Action,  Adventure,  Demons,  Drama,  Fantasy...</td>
      <td>Berserk:Season II</td>
      <td>ベルセルク</td>
      <td>TV</td>
      <td>12</td>
      <td>Apr 7, 2017 to Jun 23, 2017</td>
      <td>Spring 2017</td>
      <td>...</td>
      <td>5577.0</td>
      <td>6876.0</td>
      <td>12904.0</td>
      <td>15677.0</td>
      <td>9697.0</td>
      <td>5290.0</td>
      <td>4305.0</td>
      <td>2697.0</td>
      <td>2122.0</td>
      <td>2798.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2983</td>
      <td>Digital Devil Story: Megami Tensei</td>
      <td>5.21</td>
      <td>[Adventure,  Mystery,  Horror,  Demons,  Psych...</td>
      <td>Unknown</td>
      <td>デジタル・デビル物語〈ストーリ〉 女神転生</td>
      <td>OVA</td>
      <td>1</td>
      <td>Mar 25, 1987</td>
      <td>Unknown</td>
      <td>...</td>
      <td>60.0</td>
      <td>62.0</td>
      <td>170.0</td>
      <td>378.0</td>
      <td>609.0</td>
      <td>634.0</td>
      <td>452.0</td>
      <td>271.0</td>
      <td>165.0</td>
      <td>83.0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 35 columns</p>
</div>


