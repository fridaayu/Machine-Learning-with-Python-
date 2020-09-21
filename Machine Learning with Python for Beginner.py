#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
amazon = pd.read_csv("https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/online_raw.csv")
print('Ukuran dataset\n',amazon.shape)
print('Lima teratas dataset\n',amazon.head())
print('Informasi dataset\n',amazon.info())
print('Ringakasan statistik dataset\n',amazon.describe())


# In[2]:


#Korelasi
amazon_corr = amazon.corr()
print('Korelasi dataset :\n',amazon_corr)
print('Distribusi Label(Revenue)\n',amazon['Revenue'].value_counts())
print('Korelasi BounceRates-ExitRates:',amazon_corr.loc['BounceRates','ExitRates'])
print('Korelasi Revenue-PageValues:',amazon_corr.loc['Revenue','PageValues'])
print('Korelasi TrafficType-Weekend:',amazon_corr.loc['TrafficType','Weekend'])


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
# checking the Distribution of customers on Revenue
plt.rcParams['figure.figsize']=(12,5)
plt.subplot(1,2,1)
sns.countplot(amazon['Revenue'],palette='pastel')
plt.title('Buy or Not',fontsize=20,color='blue')
plt.xlabel('Revenue or Not',fontsize=14)
plt.ylabel('Count',fontsize=14)
# checking the Distribution of customers on Weekend
plt.subplot(1,2,2)
sns.countplot(amazon['Weekend'],palette='inferno')
plt.title('Purcahse on Weekend',fontsize=20,color='blue')
plt.xlabel('Weekend or Not',fontsize=14)
plt.ylabel('Count',fontsize=14)
plt.show()


# In[4]:


# visualizing the distribution of customers around the Region
plt.hist(amazon['Region'], color = 'lightblue')
plt.title('Distribution of Customers', fontsize = 20)
plt.xlabel('Region Codes', fontsize = 14)
plt.ylabel('Count Users', fontsize = 14)
plt.show()


# In[5]:


#Cek missing Value
print('Cheking Missing Value untuk masing-masing kolom:')
print(amazon.isnull().sum())
print('Total Missing Value:')
print(amazon.isnull().sum().sum())


# In[6]:


#Mengatasi Missing Value
#1. Drop
amazon_clean1 = amazon.dropna()
print('Ukuran data Amazon yang bersih:',amazon_clean1.shape)
print('Missing Values:',amazon_clean1.isnull().sum().sum())
#2. Fill with mean/ median
amazon.fillna(amazon.mean(),inplace=True)
print('Ukuran data Amazon yang bersih:',amazon.shape)

print('Missing Values:',amazon.isnull().sum().sum())


# In[12]:


#Data Processing : 
#1. Scaling
#Scaling data ini sangat perlu agar data2 terletak pada rentang yang sama,
#Jika rentang tidak sama maka, akan variabel tersebut akan mendominasi dalam perhitungan model
#sehingga akan menghasilkan model yang bias

#Scaling hanya bisa dilakukan pada variabel numerik,
#sehingga perlu diseleksi variabel mana saja yg numerik

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#buat list varibel numerik dari dataset Amazon
scaling_var = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues']
#Apply fit_transform ke scale untuk variabel yang terpilih
amazon[scaling_var]=scaler.fit_transform(amazon[scaling_var])
#Mengecek hasil dari scaling
print(amazon[scaling_var].describe().T[['min','max']])

#2. Mengubah dari string ke numerik
import numpy as np
from sklearn.preprocessing import LabelEncoder
#mengubah variabel 'Month' 
amazon['Month']=LabelEncoder().fit_transform(amazon['Month'])
#print(LabelEncoder().classes_)
print(np.sort(amazon['Month'].unique()))
print('')
#mengubah variabel/feature VisitorType
amazon['VisitorType']=LabelEncoder().fit_transform(amazon['VisitorType'])
#print(LabelEncoder().classes_)
print(np.sort(amazon['VisitorType'].unique()))


# In[13]:


#Pemodelan dengan Scikit-Learn
#Scikit-learn adalah library untuk machine learning bagi para pengguna python yang memungkinkan kita melakukan berbagai pekerjaan dalam Data Science, seperti regresi (regression), klasifikasi (classification), pengelompokkan/penggugusan (clustering), data preprocessing, dimensionality reduction, dan model selection (pembandingan, validasi, dan pemilihan parameter maupun model).”

#Klasifikasi
#1. pisahkan data menjadi Features(x) dan Label(y)
x=amazon.drop(['Revenue'],axis=1)
y=amazon['Revenue']
print('Ukuran variabel x:',x.shape)
print('Ukuran variabel y:',y.shape)
#Spliting data training dan tetsing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print('Ukuran x_train:',x_train.shape)
print('Ukuran x_test:',x_test.shape)
print('Ukuran y_train:',y_train.shape)
print('Ukuran y_test:',y_test.shape)


# In[14]:


#3. Modeling data training dengan decisionTreeclassifier
from sklearn.tree import DecisionTreeClassifier
modeltree=DecisionTreeClassifier()
model=modeltree.fit(x_train,y_train)
print(model)


# In[15]:


#4. Training model predict
y_pred=model.predict(x_test)
print(y_pred) #ukuran y_pred=y_test


# In[16]:


#5. Evaluasi Model Performance
from sklearn.metrics import confusion_matrix,classification_report
#Evaluasi model
print('Training Accuracy:',model.score(x_train,y_train))
print('Testing Accuracy:',model.score(x_test,y_test))
#Confusion matriks
conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
#Classifiacation report
class_report = classification_report(y_test,y_pred)
print(class_report)


# In[19]:


#LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

#Model Regresi logistik
reglog = LogisticRegression()
model_reglog=reglog.fit(x_train,y_train)
#training model: predict
y_pred=reglog.predict(x_test)

#Evaluasi Model performance
print('Training Accuracy :',model_reglog.score(x_train,y_train))
print('Testing Accuracy :',model_reglog.score(x_test,y_test))

#Confusion matrix
print('\nConfusion Matriks:')
print(confusion_matrix(y_test,y_pred))

#Classification_report
print('\nClassification Report:')
print(classification_report(y_test,y_pred))


# In[20]:


#Linear Regression
#load dataset
import pandas as pd
housing = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/housing_boston.csv')
#Data rescaling
from sklearn import preprocessing
data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
housing[['RM','LSTAT','PTRATIO','MEDV']] = data_scaler.fit_transform(housing[['RM','LSTAT','PTRATIO','MEDV']])
# getting dependent and independent variables
X = housing.drop(['MEDV'], axis = 1)
y = housing['MEDV']
# checking the shapes
print('Shape of X:', X.shape)
print('Shape of y:', y.shape)

# splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
# checking the shapes
print('Shape of X_train :', X_train.shape)
print('Shape of y_train :', y_train.shape)
print('Shape of X_test :', X_test.shape)
print('Shape of y_test :', y_test.shape)

##import regressor from Scikit-Learn
from sklearn.linear_model import LinearRegression
# Call the regressor
reg = LinearRegression()
# Fit the regressor to the training data
reg = reg.fit(X_train,y_train)
# Apply the regressor/model to the test data
y_pred = reg.predict(X_test)


# In[21]:


#Evaluasi model Regresi linear
from sklearn.metrics import mean_squared_error, mean_absolute_error  
import numpy as np
import matplotlib.pyplot as plt 

#Calculating MSE, lower the value better it is. 0 means perfect prediction
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error of testing set:', mse)
#Calculating MAE
mae = mean_absolute_error(y_test, y_pred)
print('Mean absolute error of testing set:', mae)
#Calculating RMSE
rmse = np.sqrt(mse)
print('Root Mean Squared Error of testing set:', rmse)

#Plotting y_test dan y_pred
plt.scatter(y_test, y_pred, c = 'green')
plt.xlabel('Price Actual')
plt.ylabel('Predicted value')
plt.title('True value vs predicted value : Linear Regression')
plt.show()


# In[34]:


#Klasifikasi dengan Kmeans

#import library
from sklearn.cluster import KMeans

#load dataset
dataset = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/pythonTutorial/mall_customers.csv')

#selecting features  
X = dataset[['annual_income','spending_score']]  #asumsikan feature yang terilih adalah annual_income dan spending_score

#Define KMeans as cluster_model  
cluster_model = KMeans(n_clusters = 5, random_state = 24)  
labels = cluster_model.fit_predict(X)


# In[35]:


#Inspect and Visualisation Cluster

#convert dataframe to array
X = X.values
#Separate X to xs and ys --> use for chart axis
xs = X[:,0]
ys = X[:,1]
# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs,ys,c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = cluster_model.cluster_centers_
# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x,centroids_y,marker='D', s=50)
plt.title('K Means Clustering', fontsize = 20)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[36]:


#Measuring Cluster Criteria
#Clustering yang baik adalah cluster yang data point-nya saling rapat/sangat berdekatan satu sama lain dan cukup berjauhan dengan objek/data point di cluster yang lain.
#Bisa menggunkan inertia. Inertia sendiri mengukur seberapa besar penyebaran object/data point data dalam satu cluster, semakin kecil nilai inertia maka semakin baik.

#import library
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Elbow Method - Inertia plot
inertia = []
#looping the inertia calculation for each k
for k in range(1, 10):
    #Assign KMeans as cluster_model
    cluster_model = KMeans(n_clusters = k, random_state =24)
    #Fit cluster_model to X
    cluster_model.fit(X)
    #Get the inertia value
    inertia_value = cluster_model.inertia_
    #Append the inertia_value to inertia list
    inertia.append(inertia_value)
    
##Inertia plot
plt.plot(range(1, 10), inertia)
plt.title('The Elbow Method - Inertia plot', fontsize = 20)
plt.xlabel('No.of Clusters')
plt.ylabel('Inertia')
plt.show()

