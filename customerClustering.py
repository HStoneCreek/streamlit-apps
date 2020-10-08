#!/usr/bin/env python
# coding: utf-8

# In[89]:



#https://www.kaggle.com/hendriksteinbach/mall-customer-segmentation-using-k-means/edit

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[90]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[142]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# In[92]:


# Data display coustomization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)


# In[93]:


# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


# In[150]:


# import all libraries and dependencies for machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan


# In[151]:


class CustomerClassification:
    def __init__(self):
        self.mall = pd.read_csv(r"Mall_Customers.csv")
        self.mall_d = None
        self.mall_df1 = None
        self.newCustomer = None
        self.cluster = None
        self.scaler = None
        
    def prepareData(self):
        mall_c = self.mall.drop(['CustomerID','Gender'],axis=1,inplace=True)
        self.mall_d= self.mall.copy()
        self.mall_d.drop_duplicates(subset=None,inplace=True)
        self.scaler = StandardScaler()
        mall_scaled = self.scaler.fit_transform(self.mall)
        self.mall_df1 = pd.DataFrame(mall_scaled, columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    def predict(self):
        self.cluster = KMeans(n_clusters=4, max_iter=150, random_state= 0)
        self.cluster.fit(self.mall_df1)
        self.mall_d['Cluster_Id'] = self.cluster.labels_
    def createNewCustomer(self):
        st.sidebar.title('Wähle die Eigenschaften des neuen Kunden aus:')
        age = st.sidebar.slider('Alter', 18, 100, 30)
        annualIncome = st.sidebar.slider('Jährliches Einkommen (in T€)', 20, 150, 30)
        annualSpending = st.sidebar.slider('Ausgaben Score', 0,100, 50)


        self.newCustomer = {'Age':[age], 'Annual Income (k$)':[annualIncome], 'Spending Score (1-100)':[annualSpending]}
        self.newCustomer = pd.DataFrame(self.newCustomer)
        summary = self.mall.append(self.newCustomer,ignore_index=True)
        newCustomerScaled = self.scaler.fit_transform(summary)[-1].reshape(1, -1)
        prediction = self.cluster.predict(newCustomerScaled)
        self.newCustomer['Cluster_Id'] = prediction
        
    def createScatterPlot(self,newCustomer):
        plt.figure(figsize = (20,8))
        plt.subplot(1,3,1)
        sns.scatterplot(x = 'Age', y = 'Annual Income (k$)',hue='Cluster_Id',data = self.mall_d,legend='full',s = 120,palette="Set1")
        sns.scatterplot(x='Age', y='Annual Income (k$)', data = self.newCustomer,s = 200,color='orange')
        plt.subplot(1,3,2)
        sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)',hue='Cluster_Id', data = self.mall_d,legend='full',s=120,palette="Set1")
        sns.scatterplot(x='Age', y='Annual Income (k$)', data = self.newCustomer,s = 200,color='orange')
        plt.subplot(1,3,3)
        sns.scatterplot(x = 'Spending Score (1-100)', y = 'Age',hue='Cluster_Id',data= self.mall_d,legend='full',s=120,palette="Set1")
        sns.scatterplot(x='Age', y='Annual Income (k$)', data = self.newCustomer,s = 200,color='orange')
        st.pyplot(plt)
    def createViolinPlot(self, newCustomer):
        st.table(newCustomer)
        fig, axes = plt.subplots(1,3, figsize=(20,5))


        sns.violinplot(x = 'Cluster_Id', y = 'Age', data = self.mall_d,ax=axes[0])
        sns.scatterplot(x='Cluster_Id', y='Age', data = self.newCustomer,ax=axes[0], s= 100,color='red')
        sns.violinplot(x = 'Cluster_Id', y = 'Annual Income (k$)', data = self.mall_d,ax=axes[1])
        sns.scatterplot(x='Cluster_Id', y='Annual Income (k$)', data = self.newCustomer,ax=axes[1], s= 100,color='red')
        sns.violinplot(x = 'Cluster_Id', y = 'Spending Score (1-100)', data=self.mall_d,ax=axes[2])
        sns.scatterplot(x='Cluster_Id', y='Spending Score (1-100)', data = self.newCustomer,ax=axes[2], s= 100,color='red')

        st.pyplot(fig)
    def run(self):
        i = 0
        if i <1:
            self.prepareData()
            self.predict()
            i +=1    
        self.createNewCustomer()
        st.title('3. Fallstudie Kunden Klassifikation')
        st.header('Am Beispiel von Besuchern einer Einkaufsstraße')
        textIntro = """
        Stellen Sie sich vor Sie sind Besitzer einer gut besuchten Einkaufsstraße. Über Kundenkarten können Sie die Umsätze
        der Kunden zuordnen, zusätzlich erhalten Sie von den Kreditkartenunternehmen weitere Informationen. Wie können Sie
        Ihre Kunden nun untergliedern? Welche Angebote passen zu welcher Gruppe?
        Alogirithmen helfen uns dabei die Daten besser zu verstehen. Die Basis von diesem Beispiel stellt ein echter 
        Open Source Datensatz von der Data Science Platform Kaggle dar. Das zugrundeliegende Model wurde auf die Kundenbesuche trainiert.
        Sie können nun über die Auswahlmöglichkeiten links einen neuen Kunden simulieren und gucken wie dieser in bestehende
        Kluster eingeordnet würde.
        """
        st.write(textIntro)
        newCustomer = self.newCustomer.rename(columns={'Age':'Alter', 'Annual Income':'Jährliches Einkommen',
                                                      'Spending Score (1-100)': 'Ausgabenscore'})
        st.subheader('Ihr Kunde hat folgende Eigenschaften:')
        st.write(newCustomer)
        st.subheader('Der Kunde wird folgendem Cluster zugeordnet:')
        st.write(self.newCustomer['Cluster_Id'])
        st.subheader('Beschreibung des Cluster:')
        st.write('Hier den Text noch kopieren')
        st.subheader('Clustering der Daten am Beispiel der Einflussfaktoren')
        self.createScatterPlot(self.newCustomer)
        st.subheader('Verteilung der Einflussfaktoren pro Cluster')
        self.createViolinPlot(self.newCustomer)

        
    


# In[ ]:




