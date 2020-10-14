#!/usr/bin/env python
# coding: utf-8

# In[89]:



#https://www.kaggle.com/hendriksteinbach/mall-customer-segmentation-using-k-means/edit

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[90]:


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

plt.rcParams["axes.labelsize"] = 20


# In[1]:


class CustomerClassification:
    def __init__(self):
        self.mall = pd.read_csv(r"Mall_Customers.csv")
        self.mall_d = None
        self.mall_df1 = None
        self.newCustomer = None
        self.cluster = None
        self.scaler = None
        self.summary = None
        
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
        #plt.subplot(1,3,1)
        st.markdown("""<div><h4>Verhältnis von Alter und jährlichem Einkommen</h4></div>""",unsafe_allow_html=True)
        sns.scatterplot(x = 'Age', y = 'Annual Income (k$)',hue='Cluster_Id',data = self.mall_d,legend='full',s = 120,palette="Set1")
        sns.scatterplot(x='Age', y='Annual Income (k$)',marker='X', data = self.newCustomer,s = 600,color='#707172')
        #plt.subplot(1,3,2)
        st.pyplot(plt)
        
        plt.figure(figsize = (20,8))
        st.markdown("""<div><h4>Verhältnis von jährlichem Einkommen und Ausgabenscore</h4></div>""",unsafe_allow_html=True)
        sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)',hue='Cluster_Id', data = self.mall_d,legend='full',s=120,palette="Set1")
        sns.scatterplot(x='Age', y='Spending Score (1-100)',marker='X', data = self.newCustomer,s = 600,color='#707172')
        #plt.subplot(1,3,3)
        st.pyplot(plt)
        
        plt.figure(figsize = (20,8))
        st.markdown("""<div><h4>Verhältnis von Ausgabenscore und Alter</h4></div>""",unsafe_allow_html=True)
        sns.scatterplot(x = 'Spending Score (1-100)', y = 'Age',hue='Cluster_Id',data= self.mall_d,legend='full',s=120,palette="Set1")
        sns.scatterplot(x='Age', y='Age',marker='X', data = self.newCustomer,s = 600,color='#707172')
        st.pyplot(plt)
    def createViolinPlot(self, newCustomer):
        fig, axes = plt.subplots(1,3, figsize=(20,5))

        sns.violinplot(x = 'Cluster_Id', y = 'Age', inner=None, data = self.mall_d,ax=axes[0])
        sns.scatterplot(x='Cluster_Id', y='Age', data = self.newCustomer,ax=axes[0],marker='X', s= 300,color='#707172')
        sns.violinplot(x = 'Cluster_Id', y = 'Annual Income (k$)',inner=None, data = self.mall_d,ax=axes[1])
        sns.scatterplot(x='Cluster_Id', y='Annual Income (k$)',marker='X', data = self.newCustomer,ax=axes[1], s= 300,color='#707172')
        sns.violinplot(x = 'Cluster_Id', y = 'Spending Score (1-100)', inner=None,data=self.mall_d,ax=axes[2])
        sns.scatterplot(x='Cluster_Id', y='Spending Score (1-100)', marker='X', data = self.newCustomer,ax=axes[2], s= 300,color='#707172')

        st.pyplot(fig)
        
    def getSummary(self):
        self.summary = self.mall_d[['Age', 'Annual Income (k$)','Spending Score (1-100)','Cluster_Id']].groupby('Cluster_Id').mean()
        
    
    def getSummaryForCluster(self, cluster):
        summary = self.summary.loc[int(cluster)].to_dict()
        if cluster == 0:
            summary['Desc'] = 'Das Cluster ist geprägt von Kunden mittleren Alters, die über ein relativ hohes Einkommen verfügen. Die Ausgaben sind im Vergleich zu den anderen Gruppen am niedrigsten.'
        elif cluster == 1:
            summary['Desc'] = 'Jung und spendabel, so kann man dieses Cluster am Besten beschreiben. Ein Einkommen ist hingegen relativ niedrig.'
        elif cluster == 2:
            summary['Desc']  = 'In diesem Cluster sind vor allem Kunden fortgeschrittenen Alters zu finden mit durchschnittlichen Verdiensten und Ausgaben.'
        elif cluster == 3:
            summary['Desc'] = 'Gut verdienen, gut konsumieren. Mit Anfang 30 im Durchschnitt verdient diese Kundengruppe im Schnitt nicht nur viel, sondern hat auch die höchsten Konsumausgaben.'
        return summary
    def run(self):
        i = 0
        if i <1:
            self.prepareData()
            self.predict()
            i +=1    
        self.createNewCustomer()
        css = """
            <style>h3:hover{
            color: #0e3c8a;}
            h1{
            color: #f08200;
            }
            h2 {
            color: #707172
            }
            .customer {
            font-size: 15px;
            }
            .customer:hover{
            color: #f08200;
            }
            input[type="checkbox"] {
              position: absolute;
              top: 0;
              left: 0;
              width: 100%;
              height: 100%;
              opacity: 0;
            }
        
            label {
              cursor: pointer;
            }
            label {
              position: relative;
              display: block;
              padding-left: 30px;
            }
            label::before {
              content: "";
              position: absolute;
              width: 0;
              height: 0;
              top: 50%;
              left: 10px;
              border-left: 8px solid black;
              border-top: 8px solid transparent;
              border-bottom: 8px solid transparent;
              margin-top: -8px;
            }
            input[type="checkbox"]:checked ~ h2 label::before {
              border-left: 8px solid transparent;
              border-top: 8px solid black;
              border-right: 8px solid transparent;
              margin-left: -4px;
              margin-top: -4px;
            }

            #drop {
              max-height: 0;
              overflow: hidden;
              padding-left: 30px;
              transition: max-height 0.4s ease;
            }
            input[type="checkbox"]:checked ~ h2 ~ #drop {
              max-height: 200px;
            }
            </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        st.title('2. Fallstudie Kunden Clustering')
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
        
        
        
        
        newCustomer = self.newCustomer.rename(columns={'Age':'Alter', 'Annual Income (k$)':'Jährliches Einkommen (k€)',
                                                      'Spending Score (1-100)': 'Ausgabenscore'})
        st.subheader('Ihr Kunde hat folgende Eigenschaften:')
        st.markdown("""<div class='customer'>Alter: """+str(newCustomer['Alter'][0])+"""</div>""",unsafe_allow_html=True)
        st.markdown("""<div class='customer'>Jährliches Einkommen (k€): """+str(newCustomer['Jährliches Einkommen (k€)'][0])+"""</div>""",unsafe_allow_html=True)
        st.markdown("""<div class='customer'>Ausgabenscore: """+str(newCustomer['Ausgabenscore'][0])+"""</div>""",unsafe_allow_html=True)
        st.subheader('Der Kunde wird folgendem Cluster zugeordnet:')
        st.markdown("""<div class='customer'><b>"""+str(newCustomer['Cluster_Id'][0])+"""</b></div>""",unsafe_allow_html=True)
        st.subheader('Beschreibung des Cluster (im Durchschnitt):')
        self.getSummary()
        dictSummary = self.getSummaryForCluster(newCustomer['Cluster_Id'][0])
        st.markdown("""<div class='customer'>Alter: """+str(round(dictSummary['Age'],2))+"""</div>""",unsafe_allow_html=True)
        st.markdown("""<div class='customer'>Jährliches Einkommen (k€): """+str(round(dictSummary['Annual Income (k$)'],2))+"""</div>""",unsafe_allow_html=True)
        st.markdown("""<div class='customer'>Ausgabenscore: """+str(round(dictSummary['Spending Score (1-100)'],2))+"""</div>""",unsafe_allow_html=True)
        st.write('Beschreibung des Clusters:')
        
        st.markdown("""<div class='customer'><p style="color:#f08200"> """+str(dictSummary['Desc'])+"""</p></div>""",unsafe_allow_html=True)
        st.subheader('Verteilung der Einflussfaktoren pro Cluster')
        st.write('Im den folgenden Abbildungen sehen Sie Ihren erzeugten Kunden in Form des graumarkierten "x".')
        self.createViolinPlot(self.newCustomer)
        st.subheader('Darstellung der Cluster')
        self.createScatterPlot(self.newCustomer)
        
        html = """<div>
        <input type="checkbox" id="faq-1">
        <h2><label for="faq-1">Weitere Informationen </label></h2>
        <p id="drop">Hier werden sich Informationen zu der Datengrundlage und dem angewendeten Modell finden..</p>
        </div>"""
        st.markdown(html, unsafe_allow_html=True)
        
    


# In[ ]:




