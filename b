B1 uber 
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('/content/uber.csv')
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 26782 entries, 0 to 26781
Data columns (total 9 columns):
 # Column Non-Null Count Dtype
--- ------ -------------- -----
 0 Unnamed: 0 26782 non-null int64
 1 key 26782 non-null object
 2 fare_amount 26782 non-null float64
 3 pickup_datetime 26782 non-null object
 4 pickup_longitude 26782 non-null float64
 5 pickup_latitude 26782 non-null float64
 6 dropoff_longitude 26782 non-null float64
 7 dropoff_latitude 26782 non-null float64
 8 passenger_count 26781 non-null float64
dtypes: float64(6), int64(1), object(2)
memory usage: 1.8+ MB
df.shape
(26782, 9)
df.head()
 Unnamed: 0 key fare_amount \
0 24238194 2015-05-07 19:52:06.0000003 7.5
1 27835199 2009-07-17 20:04:56.0000002 7.7
2 44984355 2009-08-24 21:45:00.00000061 12.9
3 25894730 2009-06-26 08:22:21.0000001 5.3
4 17610152 2014-08-28 17:47:00.000000188 16.0
 pickup_datetime pickup_longitude pickup_latitude \
0 2015-05-07 19:52:06 UTC -73.999817 40.738354
1 2009-07-17 20:04:56 UTC -73.994355 40.728225
2 2009-08-24 21:45:00 UTC -74.005043 40.740770
3 2009-06-26 08:22:21 UTC -73.976124 40.790844
4 2014-08-28 17:47:00 UTC -73.925023 40.744085
 dropoff_longitude dropoff_latitude passenger_count
0 -73.999512 40.723217 1.0
1 -73.994710 40.750325 1.0
2 -73.962565 40.772647 1.0
3 -73.965316 40.803349 3.0
4 -73.973082 40.761247 5.0
#find any null value present
df.isnull().sum()
Unnamed: 0 0
key 0
fare_amount 0
pickup_datetime 0
pickup_longitude 0
pickup_latitude 0
dropoff_longitude 0
dropoff_latitude 0
passenger_count 1
dtype: int64
#drop null rows
df.dropna(axis=0,inplace=True)
df.isnull().sum()
Unnamed: 0 0
key 0
fare_amount 0
pickup_datetime 0
pickup_longitude 0
pickup_latitude 0
dropoff_longitude 0
dropoff_latitude 0
passenger_count 0
dtype: int64
#Calculatin the distance between the pickup and drop co-ordinates
#using the Haversine formual for accuracy.
def haversine (lon_1, lon_2, lat_1, lat_2):

 lon_1, lon_2, lat_1, lat_2 = map(np.radians, [lon_1, lon_2, lat_1,
lat_2]) #Degrees to Radians


 diff_lon = lon_2 - lon_1
 diff_lat = lat_2 - lat_1

 km = 2 * 6371 * np.arcsin(np.sqrt(np.sin(diff_lat/2.0)**2 +
 np.cos(lat_1) * np.cos(lat_2) *
np.sin(diff_lon/2.0)**2))

 return km
#find distance travelled per ride
df['Distance']=
haversine(df['pickup_longitude'],df['dropoff_longitude'],

df['pickup_latitude'],df['dropoff_latitude'])
#round it to 2 decimal points
df['Distance'] = df['Distance'].astype(float).round(2)
df.head()
plt.scatter(df['Distance'], df['fare_amount'])
plt.xlabel("Distance")
plt.ylabel("fare_amount")
Text(0, 0.5, 'fare_amount')
#Outliers
#We can get rid of the trips with very large distances that are
outliers
# as well as trips with 0 distance.
df.drop(df[df['Distance'] > 60].index, inplace = True)
df.drop(df[df['Distance'] == 0].index, inplace = True)
df.drop(df[df['fare_amount'] == 0].index, inplace = True)
df.drop(df[df['fare_amount'] < 0].index, inplace = True)
df.shape
(25942, 10)
# removing rows with non-plausible fare amounts and distance travelled
df.drop(df[(df['fare_amount']>100) & (df['Distance']<1)].index,
inplace = True )
df.drop(df[(df['fare_amount']<100) & (df['Distance']>100)].index,
inplace = True )
df.shape
plt.scatter(df['Distance'], df['fare_amount'])
plt.xlabel("Distance")
plt.ylabel("fare_amount")
df.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 25939 entries, 0 to 26780
Data columns (total 10 columns):
 # Column Non-Null Count Dtype
--- ------ -------------- -----
 0 Unnamed: 0 25939 non-null int64
 1 key 25939 non-null object
 2 fare_amount 25939 non-null float64
 3 pickup_datetime 25939 non-null object
 4 pickup_longitude 25939 non-null float64
 5 pickup_latitude 25939 non-null float64
 6 dropoff_longitude 25939 non-null float64
 7 dropoff_latitude 25939 non-null float64
 8 passenger_count 25939 non-null float64
 9 Distance 25939 non-null float64
dtypes: float64(7), int64(1), object(2)
memory usage: 3.2+ MB
# Create New DataFrame of Specific column
df2 = pd.DataFrame().assign(fare=df['fare_amount'],
Distance=df['Distance'])
df2.info()
df2.shape
<class 'pandas.core.frame.DataFrame'>
Int64Index: 25939 entries, 0 to 26780
Data columns (total 2 columns):
 # Column Non-Null Count Dtype
--- ------ -------------- -----
 0 fare 25939 non-null float64
 1 Distance 25939 non-null float64
dtypes: float64(2)
memory usage: 1.6 MB
(25939, 2)
# plot target fare distribution
plt.figure(figsize=[8,4])
sns.distplot(df2['fare'], color='g',hist_kws=dict(edgecolor="black",
linewidth=2), bins=30)
plt.title('Target Variable Distribution')
plt.show()
/usr/local/lib/python3.7/dist-packages/seaborn/distributions.py:2619:
FutureWarning: `distplot` is a deprecated function and will be removed
in a future version. Please adapt your code to use either `displot` (a
figure-level function with similar flexibility) or `histplot` (an
axes-level function for histograms).
 warnings.warn(msg, FutureWarning)
#plots
plt.scatter(df2['Distance'], df2['fare'])
plt.xlabel("Distance")
plt.ylabel("fare_amount")
x=df2['fare']
y=df2['Distance']
#independant variable
X = df2['Distance'].values.reshape(-1, 1)
#dependant variable
Y= df2['fare'].values.reshape(-1, 1)
# scale by standardscalar
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
y_std = std.fit_transform(Y)
x_std = std.fit_transform(X)
#split in test-train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_std, y_std,
test_size=0.2, random_state=0)
#simple linear regression
from sklearn.linear_model import LinearRegression
l_reg = LinearRegression()
l_reg.fit(X_train, y_train)
LinearRegression()
#predict test values
y_pred = l_reg.predict(X_test)
#find the error
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,
y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,
y_pred))
print('Root Mean Squared Error:',
np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
Mean Absolute Error: 0.2385901323763794
Mean Squared Error: 0.18371277215345383
Root Mean Squared Error: 0.4286172793454014
#final plot
plt.subplot(2, 2, 1)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, l_reg.predict(X_train), color ="blue")
plt.title("Fare vs Distance (Training Set)")
plt.ylabel("fare_amount")
plt.xlabel("Distance")
plt.subplot(2, 2, 2)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, l_reg.predict(X_train), color ="blue")
plt.ylabel("fare_amount")
plt.xlabel("Distance")
plt.title("Fare vs Distance (Test Set)")
Text(0.5, 1.0, 'Fare vs Distance (Test Set)')

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

B2 email 


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

B3 bank 

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
#from google.colab import files
#uploded = files.upload()
import io
df = pd.read_csv('/content/Churn_Modelling.csv')
df.shape
(10000, 14)
df.drop(['CustomerId','RowNumber','Surname'], axis = 'columns',
inplace =True)
df.isna().sum()
CreditScore 0
Geography 0
Gender 0
Age 0
Tenure 0
Balance 0
NumOfProducts 0
HasCrCard 0
IsActiveMember 0
EstimatedSalary 0
Exited 0
dtype: int64
df.dtypes
CreditScore int64
Geography object
Gender object
Age int64
Tenure int64
Balance float64
NumOfProducts int64
HasCrCard int64
IsActiveMember int64
EstimatedSalary float64
Exited int64
dtype: object
df['Geography'].unique()
array(['France', 'Spain', 'Germany'], dtype=object)
#one hot encoding
df = pd.get_dummies(data = df, columns=['Geography'])
df.dtypes
df['Gender'].unique()
df['Gender'].replace(['Male', 'Female'],[1, 0], inplace= True)
df['Exited'].value_counts()
0 7963
1 2037
Name: Exited, dtype: int64
#separate outcome or target col
X = df.drop(['Exited'], axis=1)
y = df['Exited']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test =
train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
# feature scaling
#
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
 keras.layers.Dense(12, input_shape=(12,),activation='relu'),
 keras.layers.Dense(15, activation='relu'),
 keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
 loss='binary_crossentropy',
 metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100)
model.evaluate(X_test, y_test)
yp = model.predict(X_test)
Epoch 1/100
250/250 [==============================] - 1s 2ms/step - loss: 0.5908
- accuracy: 0.6936
Epoch 2/100
250/250 [==============================] - 0s 2ms/step - loss: 0.4449
- accuracy: 0.8120
Epoch 3/100
250/250 [==============================] - 0s 2ms/step - loss: 0.4208
- accuracy: 0.8205
Epoch 4/100
250/250 [==============================] - 0s 2ms/step - loss: 0.4082
- accuracy: 0.8232
Epoch 5/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3960
- accuracy: 0.8286
Epoch 6/100
250/250 [==============================] - 1s 4ms/step - loss: 0.3794
- accuracy: 0.8431
Epoch 7/100
250/250 [==============================] - 1s 4ms/step - loss: 0.3641
- accuracy: 0.8530
Epoch 8/100
250/250 [==============================] - 1s 5ms/step - loss: 0.3554
- accuracy: 0.8574
Epoch 9/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3505
- accuracy: 0.8554
Epoch 10/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3478
- accuracy: 0.8593
Epoch 11/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3449
- accuracy: 0.8600
Epoch 12/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3433
- accuracy: 0.8620
Epoch 13/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3422
- accuracy: 0.8589
Epoch 14/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3408
- accuracy: 0.8624
Epoch 15/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3401
- accuracy: 0.8615
Epoch 16/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3393
- accuracy: 0.8636
Epoch 17/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3374
- accuracy: 0.8639
Epoch 18/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3369
- accuracy: 0.8620
Epoch 19/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3368
- accuracy: 0.8640
Epoch 20/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3354
- accuracy: 0.8644
Epoch 21/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3350
- accuracy: 0.8637
Epoch 22/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3341
- accuracy: 0.8643
Epoch 23/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3336
- accuracy: 0.8620
Epoch 24/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3337
- accuracy: 0.8645
Epoch 25/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3324
- accuracy: 0.8639
Epoch 26/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3325
- accuracy: 0.8637
Epoch 27/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3321
- accuracy: 0.8639
Epoch 28/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3311
- accuracy: 0.8655
Epoch 29/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3306
- accuracy: 0.8673
Epoch 30/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3304
- accuracy: 0.8651
Epoch 31/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3303
- accuracy: 0.8656
Epoch 32/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3301
- accuracy: 0.8675
Epoch 33/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3296
- accuracy: 0.8662
Epoch 34/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3294
- accuracy: 0.8660
Epoch 35/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3291
- accuracy: 0.8659
Epoch 36/100
250/250 [==============================] - 1s 2ms/step - loss: 0.3285
- accuracy: 0.8648
Epoch 37/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3285
- accuracy: 0.8684
Epoch 38/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3277
- accuracy: 0.8683
Epoch 39/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3269
- accuracy: 0.8668
Epoch 40/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3276
- accuracy: 0.8656
Epoch 41/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3261
- accuracy: 0.8684
Epoch 42/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3267
- accuracy: 0.8656
Epoch 43/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3261
- accuracy: 0.8660
Epoch 44/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3262
- accuracy: 0.8674
Epoch 45/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3255
- accuracy: 0.8669
Epoch 46/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3258
- accuracy: 0.8679
Epoch 47/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3258
- accuracy: 0.8665
Epoch 48/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3254
- accuracy: 0.8669
Epoch 49/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3253
- accuracy: 0.8669
Epoch 50/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3243
- accuracy: 0.8669
Epoch 51/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3249
- accuracy: 0.8669
Epoch 52/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3244
- accuracy: 0.8681
Epoch 53/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3237
- accuracy: 0.8677
Epoch 54/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3243
- accuracy: 0.8692
Epoch 55/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3234
- accuracy: 0.8687
Epoch 56/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3234
- accuracy: 0.8680
Epoch 57/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3230
- accuracy: 0.8684
Epoch 58/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3234
- accuracy: 0.8684
Epoch 59/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3234
- accuracy: 0.8683
Epoch 60/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3228
- accuracy: 0.8687
Epoch 61/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3222
- accuracy: 0.8676
Epoch 62/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3229
- accuracy: 0.8692
Epoch 63/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3226
- accuracy: 0.8692
Epoch 64/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3225
- accuracy: 0.8683
Epoch 65/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3217
- accuracy: 0.8701
Epoch 66/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3218
- accuracy: 0.8704
Epoch 67/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3215
- accuracy: 0.8690
Epoch 68/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3213
- accuracy: 0.8696
Epoch 69/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3212
- accuracy: 0.8701
Epoch 70/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3208
- accuracy: 0.8681
Epoch 71/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3210
- accuracy: 0.8700
Epoch 72/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3210
- accuracy: 0.8698
Epoch 73/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3210
- accuracy: 0.8694
Epoch 74/100
250/250 [==============================] - 1s 3ms/step - loss: 0.3199
- accuracy: 0.8700
Epoch 75/100
250/250 [==============================] - 1s 3ms/step - loss: 0.3206
- accuracy: 0.8689
Epoch 76/100
250/250 [==============================] - 1s 2ms/step - loss: 0.3207
- accuracy: 0.8687
Epoch 77/100
250/250 [==============================] - 1s 3ms/step - loss: 0.3203
- accuracy: 0.8691
Epoch 78/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3198
- accuracy: 0.8701
Epoch 79/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3201
- accuracy: 0.8673
Epoch 80/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3199
- accuracy: 0.8705
Epoch 81/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3203
- accuracy: 0.8691
Epoch 82/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3197
- accuracy: 0.8702
Epoch 83/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3198
- accuracy: 0.8684
Epoch 84/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3192
- accuracy: 0.8692
Epoch 85/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3198
- accuracy: 0.8702
Epoch 86/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3188
- accuracy: 0.8709
Epoch 87/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3187
- accuracy: 0.8680
Epoch 88/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3192
- accuracy: 0.8690
Epoch 89/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3188
- accuracy: 0.8689
Epoch 90/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3188
- accuracy: 0.8704
Epoch 91/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3192
- accuracy: 0.8699
Epoch 92/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3177
- accuracy: 0.8700
Epoch 93/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3185
- accuracy: 0.8686
Epoch 94/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3181
- accuracy: 0.8700
Epoch 95/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3182
- accuracy: 0.8695
Epoch 96/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3180
- accuracy: 0.8706
Epoch 97/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3179
- accuracy: 0.8709
Epoch 98/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3181
- accuracy: 0.8702
Epoch 99/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3174
- accuracy: 0.8694
Epoch 100/100
250/250 [==============================] - 0s 2ms/step - loss: 0.3174
- accuracy: 0.8690
63/63 [==============================] - 0s 1ms/step - loss: 0.3420 -
accuracy: 0.8580
63/63 [==============================] - 0s 1ms/step
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#print(classification_report(y_test,yp))
y_pred = []
for element in yp:
 if element > 0.5:
 y_pred.append(1)
 else:
 y_pred.append(0)
#print(classification_report(y_test,y_pred))
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
cm
tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1518, 77],
 [ 207, 198]], dtype=int32)>



>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

B4 diabetes 

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/content/diabetes.csv')
df.head()
 Pregnancies Glucose BloodPressure SkinThickness Insulin
BMI \
0 6 148 72 35 0 33.6
1 1 85 66 29 0 26.6
2 8 183 64 0 0 23.3
3 1 89 66 23 94 28.1
4 0 137 40 35 168 43.1
 Pedigree Age Outcome
0 0.627 50 1
1 0.351 31 0
2 0.672 32 1
3 0.167 21 0
4 2.288 33 1
df.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], axis=1,
inplace=True)
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 6 columns):
 # Column Non-Null Count Dtype
--- ------ -------------- -----
 0 Glucose 768 non-null int64
 1 Insulin 768 non-null int64
 2 BMI 768 non-null float64
 3 Pedigree 768 non-null float64
 4 Age 768 non-null int64
 5 Outcome 768 non-null int64
dtypes: float64(2), int64(4)
memory usage: 36.1 KB
df.describe().T
 count mean std min 25% 50% \
Glucose 768.0 120.894531 31.972618 0.000 99.00000 117.0000
Insulin 768.0 79.799479 115.244002 0.000 0.00000 30.5000
BMI 768.0 31.992578 7.884160 0.000 27.30000 32.0000
Pedigree 768.0 0.471876 0.331329 0.078 0.24375 0.3725
Age 768.0 33.240885 11.760232 21.000 24.00000 29.0000
Outcome 768.0 0.348958 0.476951 0.000 0.00000 0.0000
 75% max
Glucose 140.25000 199.00
Insulin 127.25000 846.00
BMI 36.60000 67.10
Pedigree 0.62625 2.42
Age 41.00000 81.00
Outcome 1.00000 1.00
#aiming to impute nan values for the columns in accordance
#with their distribution
df[['Glucose','Insulin','BMI']].replace(0,np.NaN)
columns = ['Glucose','Insulin','BMI']
for col in columns:
 val = df[col].mean()
 df[col].replace(0, val)
#plot graph
graph = ['Glucose','Insulin','BMI','Age','Outcome']
sns.set()
print(sns.pairplot(df[graph],hue='Outcome', diag_kind='kde'))
<seaborn.axisgrid.PairGrid object at 0x7fbd33435f10>
#separate outcome or target col
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test,y_train,y_test =
train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
classifier =
KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
# evaluating model
conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
print(f1_score(y_test,y_pred))
# accuracy
print(accuracy_score(y_test,y_pred))
[[93 14]
 [18 29]]
0.6444444444444444
0.7922077922077922
# roc curve
from sklearn.metrics import roc_curve
plt.figure(dpi=100)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
from sklearn.metrics import roc_auc_score
temp=roc_auc_score(y_test,y_pred)
plt.plot(fpr,tpr,label = "%.2f" %temp)
plt.legend(loc = 'lower right')
plt.grid(True)

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

B5 sales data

import numpy as np
import pandas as pd
from google.colab import files
uploded = files.upload()
import io
df = pd.read_csv(io.BytesIO(uploded['Mall_Customers.csv']))
df.shape
<IPython.core.display.HTML object>
Saving Mall_Customers.csv to Mall_Customers (1).csv
(200, 5)
df.head()
 CustomerID Genre Age Annual Income (k$) Spending Score (1-100)
0 1 Male 19 15 39
1 2 Male 21 15 81
2 3 Female 20 16 6
3 4 Female 23 16 77
4 5 Female 31 17 40
df["A"]= df[["Annual Income (k$)"]]
df["B"]=df[["Spending Score (1-100)"]]
X=df[["A","B"]]
X.head()
 A B
0 15 39
1 15 81
2 16 6
3 16 77
4 17 40
# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# %matplotlib inline
plt.scatter(X["A"], X["B"], s = 30, c = 'b')
plt.show()
Kmean = KMeans(n_clusters=5)
Kmean.fit(X)
centers=Kmean.cluster_centers_
print(Kmean.cluster_centers_)
[[26.30434783 20.91304348]
 [86.53846154 82.12820513]
 [55.2962963 49.51851852]
 [25.72727273 79.36363636]
 [88.2 17.11428571]]
clusters = Kmean.fit_predict(X)
df["label"] = clusters
df.head(100)
col=['green','blue','black','yellow','orange',]
for i in range(5):
 a=col[i]
 # print(a)
 plt.scatter(df.A[df.label==i], df.B[df.label == i], c=a,
label='cluster 1')
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=300,
 c='r', label='centroid')
X1 = X.loc[:,["A","B"]].values
wcss=[]
for k in range(1,11):
 kmeans = KMeans(n_clusters = k, init = "k-means++")
 kmeans.fit(X1)
 wcss.append(kmeans.inertia_)
plt.figure(figsize =( 12,6))
<Figure size 864x432 with 0 Axes>
<Figure size 864x432 with 0 Axes>
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2,color="red",marker="8")
plt.xlabel("K Value")
plt.ylabel("WCSS")
plt.show()

