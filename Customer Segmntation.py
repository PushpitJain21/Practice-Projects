#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data= pd.read_csv(r"C:\Users\PushpitJain\Downloads\analytics_dataset - analytics_dataset.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.describe()


# # Handling Null Values

# In[6]:


data.isnull().sum()


# In[7]:


data=data.dropna()
data


# # Categorical Feature Analysis(Count Plots)

# In[8]:


data['Education'].unique()


# In[9]:


plt.figure(figsize=(8,8))
sns.countplot(x='Education', data=data, palette='Reds_r')


# In[10]:


data.Education.value_counts()


# In[11]:


data['Marital_Status'].value_counts()


# In[12]:


## Dropping records with marital status as Alone, YOLO, Absurd
data=data[(data.Marital_Status!='Alone')]

data=data[(data.Marital_Status!='Absurd')]

data=data[(data.Marital_Status!='YOLO')]


# In[13]:


data


# In[14]:


plt.figure(figsize=(8,8))
sns.countplot(x='Marital_Status', data=data, palette='Blues_r')


# In[15]:


plt.figure(figsize=(8,8))
sns.countplot(x='Kidhome', data=data, palette='BuGn_r')


# In[16]:


plt.figure(figsize=(8,8))
sns.countplot(x='Teenhome', data=data, palette='BuGn_r')


# # Finding Outliers

# In[17]:


sns.boxplot(x='Income', data=data,color='cyan')


# In[18]:


#Removing the outlier
data=data[data.Income!=666666]


# In[19]:


sns.boxplot(x='Income', data=data,color='cyan')


# In[20]:


sns.boxplot(x='MntWines', data=data,color='cyan')


# In[21]:


sns.boxplot(x='NumWebVisitsMonth', data=data,color='cyan')


# # Analytics

# In[22]:


#correlation
plt.figure(figsize=(20,20))
sns.heatmap(data.corr(),annot = True,cmap="RdYlGn")


# In[23]:


data['Total_Expend']=data['MntWines']+data['MntFruits']+data['MntMeatProducts']+data['MntFishProducts']+data['MntSweetProducts']+data['MntGoldProds']


# In[24]:


data.head()


# In[25]:


data.describe()


# In[26]:


sns.boxplot(x='Total_Expend', data=data,color='cyan')


# In[27]:


plt.figure(figsize=(8,8))
sns.barplot(x=data.Education, y=data.Total_Expend, palette='rainbow')


# In[28]:


plt.figure(figsize=(8,8))
sns.barplot(x=data.Education, y=data.NumStorePurchases, palette='summer_r')


# In[29]:


plt.figure(figsize=(8,8))
sns.barplot(x=data.Education, y=data.NumWebPurchases, palette='Accent')


# ## Observation: We can incur from above three plots that higher the education,higher is the number of purchases as well as total expenditure.

# In[30]:


data['Age']=2022-data['Year_Birth']


# In[31]:


plt.figure(figsize=(20,8))
sns.barplot(x=data.Age, y=data.NumWebPurchases)


# In[32]:


plt.figure(figsize=(20,8))
sns.barplot(x=data.Age, y=data.NumStorePurchases)


# # Observation: Age Group between 60-80 have more web purchases and age group of 27-30 and 75-79 have highest no. of store purchases

# In[33]:


plt.figure(figsize=(8,8))
sns.barplot(x=data.Kidhome, y=data.MntGoldProds, palette='jet_r')


# #  Observation: Graph suggests that people with kids at home tend to spend less in Gold Products

# In[34]:


plt.figure(figsize=(8,8))
sns.barplot(x=data.Teenhome, y=data.MntGoldProds, palette='jet_r')


# #  Observation: While on the other hand, people with teens at home or no teens at home tend to spend almost equal on Gold Products

# In[35]:


plt.figure(figsize=(8,8))
sns.barplot(x=data.Marital_Status, y=data.MntWines, palette='twilight_r')


# # Observation: Widow Customers seems to spend more on wines while Single and married people spend least on wine.

# #    

# # Data Preprocessing for Model Building

# In[36]:


#handling categorical varibles


# In[37]:


data['Education']=data['Education'].map({'Graduation':'0', 'PhD':'1', 'Master':'2', 'Basic':'3', '2n Cycle':'4'})


# In[38]:


data['Education']=data['Education'].astype(int)


# In[39]:


data['Marital_Status']=data['Marital_Status'].map({'Married':'0', 'Together':'1', 'Single':'2', 'Divorced':'3', 'Widow':'4'})


# In[40]:


data['Marital_Status']=data['Marital_Status'].astype(int)


# In[41]:


data.info()


# In[42]:



data['Year_Customer']=data['Dt_Customer'].str.split('-').str[2]


# In[43]:


data['Year_Customer']=data['Year_Customer'].astype(int)


# In[44]:


#No. of years the customer has been associated with the company
data['Year_Customer']=2022-data['Year_Customer']


# In[45]:


data.head()


# In[46]:


#removing irrelevant features
data.columns


# # Selecting number of cluster using WCSS and then Kmeans clustering based on RFM approach

# In[47]:


data['frequency']=data['NumWebPurchases']+data['NumStorePurchases']+data['NumCatalogPurchases']


# In[48]:



df1=data[['ID','Recency','frequency','Total_Expend']]
df1.head()


# In[49]:


df1.describe()


# In[50]:


#recency
wcss=[]
recency_cluster=df1[['Recency']]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
   
    kmeans.fit(recency_cluster)
    recency_cluster["clusters"] = kmeans.labels_
    
    wcss.append(kmeans.inertia_)


# In[51]:


sns.set()
plt.plot(range(1,11),wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[52]:


#Above graph indicates 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(df1[['Recency']])

df1['RecencyCluster'] = kmeans.predict(df1[['Recency']])


# In[53]:


def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

df1 = order_cluster('RecencyCluster', 'Recency',df1,False)


# In[54]:


# frequency


# In[55]:


df2=data[['ID','frequency']]
df2.head()


# In[56]:


wcss=[]
frequency_cluster=df1[['frequency']]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
   
    kmeans.fit(frequency_cluster)
    frequency_cluster["clusters"] = kmeans.labels_
    
    wcss.append(kmeans.inertia_)


# In[57]:


plt.plot(range(1,11),wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# In[58]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(df1[['frequency']])

df1['FrequencyCluster'] = kmeans.predict(df1[['frequency']])


# In[61]:


def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

df1 = order_cluster('FrequencyCluster', 'frequency',df1,True)


# In[62]:


#Revenue/Monetaryvalue

df3=data[['ID','Total_Expend']]
df3.head()


# In[63]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(df1[['Total_Expend']])

df1['ExpendCluster'] = kmeans.predict(df1[['Total_Expend']])


# In[64]:


def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

df1 = order_cluster('ExpendCluster', 'Total_Expend',df1,True)


# In[65]:


#Overall Score
df1['OverallScore'] = df1['RecencyCluster'] + df1['FrequencyCluster'] + df1['ExpendCluster']
df1.groupby('OverallScore')['Recency','frequency','Total_Expend'].mean()


# # Customer Cluster with Lowest Recency, Highest Frequency and highest Total Expenditure are the best ones for the company!! Hence, Customers with total Score of 6 are the most loyal customers for the company!!

# In[ ]:




