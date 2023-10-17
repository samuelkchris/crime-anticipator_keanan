import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

# Load Data
dataset = pd.read_csv('data.csv')
data = pd.read_csv('data.csv')
dataset.head()

for col in data:
    print(type(data[col][1]))

data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%Y %H:%M:%S')
data['timestamp']

# DATE TIME STAMP FUNCTION
column_1 = data.iloc[:, 0]

db = pd.DataFrame({"year": column_1.dt.year,
                   "month": column_1.dt.month,
                   "day": column_1.dt.day,
                   "hour": column_1.dt.hour,
                   "dayofyear": column_1.dt.dayofyear,
                   "week": column_1.dt.week,
                   "weekofyear": column_1.dt.weekofyear,
                   "dayofweek": column_1.dt.dayofweek,
                   "weekday": column_1.dt.weekday,
                   "quarter": column_1.dt.quarter,
                   })

dataset1 = dataset.drop('timestamp', axis=1)
data1 = pd.concat([db, dataset1], axis=1)

# Data Analysis
data1.info()
data1.dropna(inplace=True)
data1.head()

# Data Visualization & Analysis
sns.pairplot(data1, hue='act363')
sns.boxplot(x='act379', y='hour', data=data1, palette='winter_r')
sns.boxplot(x='act13', y='hour', data=data1, palette='winter_r')
sns.boxplot(x='act323', y='hour', data=data1, palette='winter_r')
sns.boxplot(x='act363', y='hour', data=data1, palette='winter_r')
df = pd.DataFrame(data=data1, columns=['act13', 'hour', 'day'])
df.plot.hexbin(x='act13', y='hour', gridsize=25)
df.plot(legend=False)
df1 = pd.DataFrame(data=data1, columns=['act13', 'act323', 'act379'])
df1.plot.kde()

# X & Y array
X = data1.iloc[:, [1, 2, 3, 4, 6, 16, 17]].values
y = data1.iloc[:, [10, 11, 12, 13, 14, 15]].values

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)

# Creating & Training KNN Model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
knn.score(X_train, y_train)

# Elbow Method For optimum value of K
error_rate = []
for i in range(1, 140):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 140), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=5)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# Creating & Training Decision Tree Model
dtree = DecisionTreeClassifier(max_depth=500, random_state=300)
dtree.fit(X_train,y_train)
y_pred=dtree.predict(X_test)
dtree.score(X_test,y_test)
dtree.score(X_train,y_train)
y_pred
treefeatures=dtree.feature_importances_
indices = np.argsort(treefeatures)
treefeatures


# Creating & Training Random Forest Model
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
# Save the trained model
dump(rfc, 'crime_prediction_model.joblib')

y_pred=rfc.predict(X_test)
rfc.score(X_test,y_test)
rfc.score(X_train,y_train)
om=rfc.feature_importances_
indices = np.argsort(om)

om
features = data1.columns
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), om[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

# Predicting the crime trends with linear regression
from sklearn.linear_model import LinearRegression

X = data1.iloc[:, [1, 2, 3, 4, 6, 16, 17]].values

y = data1.iloc[:, [10, 11, 12, 13, 14, 15]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=50)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = model.score(X_test, y_test)
print(score)


# save the model to disk
dump(model, 'crime_prediction.joblib')
