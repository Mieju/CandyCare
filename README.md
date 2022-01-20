# CandyCare

__PML WS 2021/22__ <br>
_by Julian Mierisch and Anna Martynova_

### Goals
1. Make Model which predicts which win percentage a given new candy has
2. Predict which combination has highest win propability
3. Cluster data

### Import packages and data


```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
# conda install -c conda-forge kneed
```


```python
candyDataAll = pd.read_csv('candy-data.csv')
candyDataAll
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
      <th>competitorname</th>
      <th>chocolate</th>
      <th>fruity</th>
      <th>caramel</th>
      <th>peanutyalmondy</th>
      <th>nougat</th>
      <th>crispedricewafer</th>
      <th>hard</th>
      <th>bar</th>
      <th>pluribus</th>
      <th>sugarpercent</th>
      <th>pricepercent</th>
      <th>winpercent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100 Grand</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.732</td>
      <td>0.860</td>
      <td>66.971725</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3 Musketeers</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.604</td>
      <td>0.511</td>
      <td>67.602936</td>
    </tr>
    <tr>
      <th>2</th>
      <td>One dime</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.011</td>
      <td>0.116</td>
      <td>32.261086</td>
    </tr>
    <tr>
      <th>3</th>
      <td>One quarter</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.011</td>
      <td>0.511</td>
      <td>46.116505</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Air Heads</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.906</td>
      <td>0.511</td>
      <td>52.341465</td>
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
    </tr>
    <tr>
      <th>80</th>
      <td>Twizzlers</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.220</td>
      <td>0.116</td>
      <td>45.466282</td>
    </tr>
    <tr>
      <th>81</th>
      <td>Warheads</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.093</td>
      <td>0.116</td>
      <td>39.011898</td>
    </tr>
    <tr>
      <th>82</th>
      <td>WelchÕs Fruit Snacks</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.313</td>
      <td>0.313</td>
      <td>44.375519</td>
    </tr>
    <tr>
      <th>83</th>
      <td>WertherÕs Original Caramel</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.186</td>
      <td>0.267</td>
      <td>41.904308</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Whoppers</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.872</td>
      <td>0.848</td>
      <td>49.524113</td>
    </tr>
  </tbody>
</table>
<p>85 rows × 13 columns</p>
</div>



## Data exploration 
### Inspect distribution of candy attributes in dataset


```python
#Show distribution of candy attributes in dataset
candyAttributes = candyDataAll.drop(columns = ['competitorname', 'sugarpercent', 'pricepercent', 'winpercent'])
data = {'Attributes': candyAttributes.columns, 'Values': candyAttributes.sum()/len(candyAttributes)}  
candyAttrPercent = pd.DataFrame(data).reset_index().drop(columns=['index']).sort_values(by=['Values'])

fig, ax = plt.subplots()
def add_value_label(x_list,y_list):
    for i in range(1, len(x_list)+1):
        plt.text(i,y_list[i-1]/2,y_list[i-1], ha="center")
plt.barh(candyAttrPercent['Attributes'], candyAttrPercent['Values'])
for index, value in enumerate(candyAttrPercent['Values']):
    plt.text(value, index, str("{:.0%}".format(value)))
plt.barh(candyAttrPercent['Attributes'], 1-candyAttrPercent['Values'], left=candyAttrPercent['Values'], color="lightgrey")
plt.title('Rate of Attributes')
plt.ylabel('Attributes')
plt.xlabel('Percentage')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
plt.show()
```


![png](README_files/README_7_0.png)



```python
#Attribute correlation analysis
candyData = candyDataAll.drop(columns=['competitorname'])
corr = candyData.corr(method='pearson')
heatmap = sns.heatmap(corr, linewidth=0.5,cmap="YlGnBu").invert_yaxis()
```


![png](README_files/README_8_0.png)


### Inspect feature distribution


```python
# Feature distribution
sugar = candyDataAll['sugarpercent']*100
price = candyDataAll['pricepercent']*100
win = candyDataAll['winpercent']
colors = ['tab:red', 'tab:green', 'tab:blue']
data = [sugar, price, win]
titles = ['Sugar percentage distribution','Price percentage distribution','Win percentage distribution']
labels = ['Sugar percentage','Price percentage','Win percentage']

fig, ax = plt.subplots(1,3)
fig.set_figwidth(15)

for i in range(3):
    ax[i].hist(data[i], bins=10, color=colors[i])
    ax[i].title.set_text(titles[i])
    ax[i].set_xlabel(labels[i])
    ax[i].set_ylabel('Amount')
```


![png](README_files/README_10_0.png)


## Cluster Analysis
To analyze which features are most beneficial for high win percentage, we have to cluster the datapoints and analyze the features, that influence the average win percentage (further: awp) of each cluster

### Detect suitable cluster amount


```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
# conda install -c conda-forge kneed

kmeans = KMeans(n_clusters = 2)
candyData = candyDataAll.drop(columns=['winpercent','competitorname'])

kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

fig, ax = plt.subplots(1,2)
fig.set_figwidth(15)

# https://realpython.com/k-means-clustering-python/#:~:text=The%20k%2Dmeans%20clustering%20method,data%20objects%20in%20a%20dataset.&text=These%20traits%20make%20implementing%20k,novice%20programmers%20and%20data%20scientists.
# to choose the appropriate number of clusters:
# silhouette coefficient = measure of cluster cohesion and separation
# values range between -1 and 1. Larger numbers indicate that samples are closer to their clusters than they are to other clusters

silhouette_coefficients = []
for k in range(2, 13):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(candyData)
    score = silhouette_score(candyData, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
ax[0].set_title("Silhouette coefficient")
ax[0].plot(range(2, 13), silhouette_coefficients)
ax[0].set_xticks(range(2, 13))
ax[0].set_xlabel("Number of Clusters")
ax[0].set_ylabel("Silhouette Coefficient")
ax[0].vlines(silhouette_coefficients.index(max(silhouette_coefficients))+2, ymin=min(silhouette_coefficients), ymax=1.01*max(silhouette_coefficients), colors='green', linestyles='dashdot', linewidth=1.3)


# to choose the appropriate number of clusters:
# elbow = measure of cluster cohesion and separation

sse = []
for k in range(1, 13):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(candyData)
    sse.append(kmeans.inertia_)
ax[1].set_title("Elbow method")
ax[1].plot(range(1, 13), sse)
ax[1].set_xticks(range(1, 13))
ax[1].set_xlabel("Number of Clusters")
ax[1].set_ylabel("SSE")
kl = KneeLocator(
    range(1, 13), sse, curve="convex", direction="decreasing")
ax[1].vlines(kl.elbow, ymin=min(sse), ymax=max(sse)*1.01, colors='green', linestyles='dashdot', linewidth=1.3)

plt.show()
print("The optimal cluster amount based on silhouette coefficient method is ", silhouette_coefficients.index(max(silhouette_coefficients))+2)
print("The optimal cluster amount based on elbow method is ", kl.elbow)
print()
print("Following, we will analyze both KMeans with ", kl.elbow, " and with ", silhouette_coefficients.index(max(silhouette_coefficients))+2, 
     " clusters to decide which cluster has highes average winpercent with acceptable cluster size (ca. >10)")


```


![png](README_files/README_12_0.png)


    The optimal cluster amount based on silhouette coefficient method is  11
    The optimal cluster amount based on elbow method is  4
    
    Following, we will analyze both KMeans with  4  and with  11  clusters to decide which cluster has highes average winpercent with acceptable cluster size (ca. >10)


### Analyze clusters with KMeans

Since winpercentage is our target data, we exclude it from data which we cluster. Afterwards, we map the average winpercentage on resulted cluster and select the cluster with highest average winpercentage for further analysis of its features


```python
def autolabel(rects,axes, bar_plot):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        axes.text(rect.get_x() + rect.get_width()/2., 1,
                "awp = "+ str(winRates[idx]),
                ha='center', va='bottom', rotation=90)

pca = PCA(2)
cD = pca.fit_transform(candyData)

fig, ax = plt.subplots(1,2)
fig2,ax2 = plt.subplots(1,2)
fig.set_figwidth(15)
fig2.set_figwidth(15)


kmeansS = KMeans(n_clusters=silhouette_coefficients.index(max(silhouette_coefficients))+2)
label = kmeansS.fit_predict(cD)
u_labels = np.unique(label)
clusterSizes = []
winRates = []

# plot scatters for 11 clusters
for i in u_labels:
    ax[1].scatter(cD[label == i, 0], cD[label == i, 1], label = i)
    clusterSizes.append(cD[label == i,0].size)
    winPercentages = []
    for k in cD[label==i]:
        winPercentages.append(candyDataAll['winpercent'][np.where(cD==k)[0][0]])
    winRates.append(round(sum(winPercentages)/len(winPercentages),1))
ax[1].set_title("KMeans with " + str(silhouette_coefficients.index(max(silhouette_coefficients))+2) + " clusters"),

# plot barchart
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(clusterSizes)))
#plt.set_facecolor('white')
bar_plt2 = ax2[1].bar(range(0,len(clusterSizes)),clusterSizes, color=colors, tick_label=range(0,len(clusterSizes)))
ax2[1].set_xlabel('Clusters')
ax2[1].set_ylabel('Size')
ax2[1].set_title('Cluster sizes of ' + str(len(clusterSizes)) + ' clusters')
maxClusterSIL = clusterSizes.index(max(clusterSizes))
autolabel(bar_plt2,ax2[1],bar_plt2)



kmeansE = KMeans(n_clusters=kl.elbow)
label = kmeansE.fit_predict(cD)
u_labels = np.unique(label)
clusterSizes = []
winRates = []

# plot scatters for 4 clusters
for i in u_labels:
    ax[0].scatter(cD[label == i, 0], cD[label == i, 1], label = i)
    clusterSizes.append(cD[label == i,0].size)
    winPercentages = []
    for k in cD[label==i]:
        winPercentages.append(candyDataAll['winpercent'][np.where(cD==k)[0][0]])
    winRates.append(round(sum(winPercentages)/len(winPercentages),1))
ax[0].set_title("KMeans with " + str(kl.elbow) + " clusters")

# plot barchart
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(clusterSizes)))
bar_plt = ax2[0].bar(range(0,len(clusterSizes)),clusterSizes, color=colors, tick_label=range(0,len(clusterSizes)))
ax2[0].set_xlabel('Clusters')
ax2[0].set_ylabel('Size')
ax2[0].set_title('Cluster sizes of ' + str(len(clusterSizes)) + ' clusters')
maxClusterEL = clusterSizes.index(max(clusterSizes))
autolabel(bar_plt,ax2[0],bar_plt)

plt.show()

print('The largest of ', kl.elbow, ' clusters is: Cluster ', maxClusterEL)
print('The largest of ', silhouette_coefficients.index(max(silhouette_coefficients))+2, ' clusters is: Cluster ', maxClusterSIL)

```


![png](README_files/README_15_0.png)



![png](README_files/README_15_1.png)


    The largest of  4  clusters is: Cluster  1
    The largest of  11  clusters is: Cluster  1


As we can see, the highest awp of 4 clusters is 63.8 with 25 datapoints; the highest awp of 11 clusters is 70.5 with 4 datapoints. Though the cluster with awp of 70.5 seems to be more beneficial, due to it containing only 4 datapoints, which is to little for a substantial analysis, we choose to analyze the features of the cluster wit awp=63.8 and 25 datapoints.


```python
awpIdx = winRates.index(max(winRates))
clusterCandies = pd.DataFrame(columns=candyDataAll.columns)

for k in cD[label==awpIdx]:
    clusterCandies = clusterCandies.append(candyDataAll.loc[np.where(cD==k)[0][0]])

clusterCandies = clusterCandies.drop(columns=['winpercent','competitorname'])
vals = []
for factor in clusterCandies.columns:
    vals.append(sum(clusterCandies[factor])/len(clusterCandies))
```


```python
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(vals)))
bar_plt = plt.bar(range(0,len(vals)), vals, color=colors, tick_label=clusterCandies.columns)
plt.xticks(rotation = 'vertical')
plt.xlabel('Features')
plt.ylabel('Average occurency in cluster')
plt.title('Average of feature rate')

for idx,bar_plt in enumerate(bar_plt):
        height = bar_plt.get_height()
        plt.text(bar_plt.get_x() + bar_plt.get_width()/2., height,
                round(vals[idx],2),
                ha='center', va='bottom', rotation=0)
```


![png](README_files/README_18_0.png)


# Non-linear-regression
## Dataset preparation


```python
candyDataProcessed = candyDataAll
candyDataProcessed['winpercent'] = candyDataProcessed['winpercent']/100
training_set, test_set = train_test_split(candyDataProcessed, test_size=0.3, random_state = 100)
validation_set, test_set = train_test_split(test_set, test_size=0.5, random_state = 100)

#sugar
X_train_s = training_set.drop(columns = ['competitorname', 'sugarpercent', 'pricepercent', 'winpercent'])
y_train_s = training_set['sugarpercent']

X_test_s = training_set.drop(columns = ['competitorname', 'sugarpercent', 'pricepercent', 'winpercent'])
y_test_s = training_set['sugarpercent']

X_validation_s = training_set.drop(columns = ['competitorname', 'sugarpercent', 'pricepercent', 'winpercent'])
y_validation_s = training_set['sugarpercent']

#price
X_train_p = training_set.drop(columns = ['competitorname', 'sugarpercent', 'pricepercent', 'winpercent'])
y_train_p = training_set['pricepercent']

X_test_p = training_set.drop(columns = ['competitorname', 'sugarpercent', 'pricepercent', 'winpercent'])
y_test_p = training_set['pricepercent']

X_validation_p = training_set.drop(columns = ['competitorname', 'sugarpercent', 'pricepercent', 'winpercent'])
y_validation_p = training_set['pricepercent']

#win
X_train_w = training_set.drop(columns = ['competitorname', 'sugarpercent', 'pricepercent', 'winpercent'])
y_train_w = training_set['winpercent']

X_test_w = training_set.drop(columns = ['competitorname', 'sugarpercent', 'pricepercent', 'winpercent'])
y_test_w = training_set['winpercent']

X_validation_w = training_set.drop(columns = ['competitorname', 'sugarpercent', 'pricepercent', 'winpercent'])
y_validation_w = training_set['winpercent']
```

## Random forest classifier
### Create sugarpercent-predictor


```python
clf_s=RandomForestRegressor(n_estimators=100)
clf_s.fit(X_train_s, y_train_s)
y_pred_s = clf_s.predict(X_test_s)
print("MAE =", mean_absolute_error(y_test_s,y_pred_s))
print("Score (Acc) =", clf_s.score(X_test_s,y_test_s))
```

    MAE =  0.16962848787379214
    Score (Acc) =  0.3819626013913967


### Create pricepercent-predictor


```python
clf_p=RandomForestRegressor(n_estimators=100)
clf_p.fit(X_train_p, y_train_p)
y_pred_p = clf_p.predict(X_test_p)
print("MAE =", mean_absolute_error(y_test_p,y_pred_p))
print("Score (Acc) =", clf_p.score(X_test_p,y_test_p))
```

    MAE = 0.14923434423975768
    Score (Acc) = 0.4233939012749507


### Create winpercent-predictor


```python
clf_w=RandomForestRegressor(n_estimators=100)
clf_w.fit(X_train_w, y_train_w)
y_pred_w = clf_w.predict(X_test_w)
print("MAE =", mean_absolute_error(y_test_w,y_pred_w))
print("Score (Acc) =", clf_w.score(X_test_w,y_test_w))
```

    MAE = 0.0005135200033241511
    Score (Acc) = 0.7406549644650849


### Create Methods


```python
# dataline should have the format of a Dataframe!
def predSugar_RF(dataline):
    return clf_s.predict(dataline)

def predPrice_RF(dataline):
    return clf_p.predict(dataline)

def predWin_RF(dataline):
    return clf_w.predict(dataline)
```

### Prediction price


```python
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train_s, list(map(int,y_train_s*1000)))
y_pred_s = clf.predict(X_test_s)/1000   
```


```python

```
