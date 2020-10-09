#!/usr/bin/env python
# coding: utf-8

# [LendingClub](https://www.lendingclub.com/) is a US peer-to-peer lending company and the world's largest peer-to-peer lending platform. As explained by [Wikipedia](https://en.wikipedia.org/wiki/Lending_Club),
# 
# > Lending Club enables borrowers to create unsecured personal loans between \$1,000 and \$40,000. The standard loan period is three years. Investors can search and browse the loan listings on Lending Club website and select loans that they want to invest in based on the information supplied about the borrower, amount of loan, loan grade, and loan purpose. Investors make money from interest. Lending Club makes money by charging borrowers an origination fee and investors a service fee.
# 
# The goal of this project is to build a machine learning model to predict the probability that a loan will charge off. We will attempt to only use data available to investors via the LendingClub loan listing, including information about the borrower (income, employment length, FICO score, debt-to-income ratio, etc.) and the loan listing (the loan amount, loan purpose, loan grade, interest rate, installment, etc.). Such a predictive model could help LendingClub investors make better-informed investment decisions. We will only consider loans that LendingClub accepted under its credit underwriting policy.

# ## 1.1 References
# <a id="1.1"></a>

# * LendingClub information
#  * [LendingClub website](https://www.lendingclub.com/)
#  * [LendingClub Wikipedia page](https://en.wikipedia.org/wiki/Lending_Club)
# * Datasets
#  * [LendingClub statistics](https://www.lendingclub.com/info/download-data.action) - Original data source, aggregated on Kaggle
#  * [All Lending Club loan data](https://www.kaggle.com/wordsforthewise/lending-club) - The dataset used in this project, hosted on Kaggle
#  * [Lending Club Loan Data](https://www.kaggle.com/wendykan/lending-club-loan-data) - Another LendingClub dataset on Kaggle, not used in this project
# * This project
#  * [Kaggle notebook](https://www.kaggle.com/pileatedperch/predicting-charge-off-from-initial-listing-data)
#  * [GitHub repo](https://github.com/jgcorliss/lending-club)

# # 2. Import the Data
# <a id="2"></a>

# In[1]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import pickle

# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)

# Plotting options
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
sns.set(style='whitegrid')


# Read the data into a pandas dataframe:
tmpDict = { 'loan_amnt': 1000, # zuleihendeer Betrag
          'term': 10, #monate entweder 36, 60
          'int_rate': 10, #Zinssatz
          'installment': #zu Beginn fällige Rate
          'emp_length': 4, #Angestellt in dem Job seit x Jahren
          'issu_d':'Mar-2017', #entfällt
          'dti': 34,#berechne ich als Verhältnis monatlicher Zahlungen für Konsumkredite #A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, 
           #excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
          'earliest_cr_line':2000, #The year the borrower's earliest reported credit line was opened
          'open_acc':ccc,#The number of open credit lines in the borrower's credit file."
           'revol_util': xxx#Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
          'pub_rec': xxx,#Number of derogatory public records.
           'application_type_Join_App': #Indicates whether the loan is an individual application or a joint application with two co-borrowers
           'mort_acc': #Number of mortgage accounts.,
           'total_acc': #The total number of credit lines currently in the borrower's credit file.
          'pub_rec_bankruptcies': ,#Number of public record bankruptcies
          'initial_list_status_w': }#The initial listing status of the loan. Possible values are – W, F." I'm not sure what this means.
            
# In[36]:


columns = ['loan_amnt',
 'term',
 'int_rate',
 'installment',
 'emp_length',
 'dti',
 'earliest_cr_line',
 'open_acc',
 'pub_rec',
 'revol_util',
 'total_acc',
 'mort_acc',
 'pub_rec_bankruptcies',
 'log_annual_inc',
 'fico_score',
 'log_revol_bal',
 'sub_grade_A2',
 'sub_grade_A3',
 'sub_grade_A4',
 'sub_grade_A5',
 'sub_grade_B1',
 'sub_grade_B2',
 'sub_grade_B3',
 'sub_grade_B4',
 'sub_grade_B5',
 'sub_grade_C1',
 'sub_grade_C2',
 'sub_grade_C3',
 'sub_grade_C4',
 'sub_grade_C5',
 'sub_grade_D1',
 'sub_grade_D2',
 'sub_grade_D3',
 'sub_grade_D4',
 'sub_grade_D5',
 'sub_grade_E1',
 'sub_grade_E2',
 'sub_grade_E3',
 'sub_grade_E4',
 'sub_grade_E5',
 'sub_grade_F1',
 'sub_grade_F2',
 'sub_grade_F3',
 'sub_grade_F4',
 'sub_grade_F5',
 'sub_grade_G1',
 'sub_grade_G2',
 'sub_grade_G3',
 'sub_grade_G4',
 'sub_grade_G5',
 'home_ownership_OTHER',
 'home_ownership_OWN',
 'home_ownership_RENT',
 'verification_status_Source Verified',
 'verification_status_Verified',
 'purpose_credit_card',
 'purpose_debt_consolidation',
 'purpose_educational',
 'purpose_home_improvement',
 'purpose_house',
 'purpose_major_purchase',
 'purpose_medical',
 'purpose_moving',
 'purpose_other',
 'purpose_renewable_energy',
 'purpose_small_business',
 'purpose_vacation',
 'purpose_wedding',
 'addr_state_AL',
 'addr_state_AR',
 'addr_state_AZ',
 'addr_state_CA',
 'addr_state_CO',
 'addr_state_CT',
 'addr_state_DC',
 'addr_state_DE',
 'addr_state_FL',
 'addr_state_GA',
 'addr_state_HI',
 'addr_state_IA',
 'addr_state_ID',
 'addr_state_IL',
 'addr_state_IN',
 'addr_state_KS',
 'addr_state_KY',
 'addr_state_LA',
 'addr_state_MA',
 'addr_state_MD',
 'addr_state_ME',
 'addr_state_MI',
 'addr_state_MN',
 'addr_state_MO',
 'addr_state_MS',
 'addr_state_MT',
 'addr_state_NC',
 'addr_state_ND',
 'addr_state_NE',
 'addr_state_NH',
 'addr_state_NJ',
 'addr_state_NM',
 'addr_state_NV',
 'addr_state_NY',
 'addr_state_OH',
 'addr_state_OK',
 'addr_state_OR',
 'addr_state_PA',
 'addr_state_RI',
 'addr_state_SC',
 'addr_state_SD',
 'addr_state_TN',
 'addr_state_TX',
 'addr_state_UT',
 'addr_state_VA',
 'addr_state_VT',
 'addr_state_WA',
 'addr_state_WI',
 'addr_state_WV',
 'addr_state_WY',
 'initial_list_status_w',
 'application_type_Joint App']
newBorrower = pd.DataFrame(columns = columns)


# In[ ]:


liste = []
for i in 'ABCDEFG':
    for x in range(1,6):
        liste.append(i+str(x))
states = ['CA', 'NY', 'TX', 'FL', 'IL', 'NJ','OH']
        
loan_amnt = st.sidebar.slider('Wie hoch ist der Kreditbetrag?',0,40000,10000)
term = st.sidebar.radio('Laufzeit des Kredits?', (36,60))
int_rate = st.sidebar.slider('Wie hoch ist der Zinssatz?', 0, 15,5,0.25)
installment = (loan_amt*(1+(int_rate/100))^(term/12))/term
emp_length = st.sidebar.slider('Wie lange besteht das Beschäftigungsverhältnis?', 0, 15,5)
earliest_cr_line = st.sidebar.slider('Wann wurde die erste Kreditlinie beantragt?', 1995,2015,2010)
open_acc = st.sidebar.slider('Wie viele Kredite werden aktuell beansprucht?', 0, 30, 5)
revol_util = st.sidebar.slider('Wie viel Prozent der zur Verfügung stehenden Kreditlinie nutzt der Kunde aktuell?', 0, 1,0.1)

pub_rec = st.sidebar.slider('Wie viele negative Eintragungen bestehen?', 0,15,0)
application_type_Join_App = st.sidebar.radio('Wird der Antrag alleine gestellt?', ('Ja', 'Nein'))
if application_type_Join_App == 'Ja':
    application_type_Join_App = 0
else:
    application_type_Join_App = 1
annualIncome = st.sidebar.slider('Wie viel verdient der Kreditnehmer pro Jahr?', 0,200000, 50000) 
mort_acc = st.sidebar.slider('Wie viele Hauskredite bestehen?', 0, 10, 2)
total_acc = st.sidebar.slider('Wie viele Kredite befinden sich in der Kredithistorie?', 0,30,5)
pub_rec_bankruptcies = st.sidebar.slider('Wie viele Privatinsolvenzen sind veröffentlicht?',0,5,0)
initial_list_status_w = 1
grade = st.sidebar.select_slider('Welches Rating bekommt der Kunde?', options=liste)
home_ownership = st.sidebar.radio('Wie steht es um die Wohnverhältnisse?', ('OWN', 'RENT', 'OTHER'))
state = st.sidebar.select_slider('Aus welchem Staat kommt der Kreditnehmer?', options = states)

tmpDict = {
    'loan_amnt':loan_amt, 'term':term, 'int_rate':int_rate, 
    'installment':installment, 'emp_length':emp_length, 'dti':25,
    'earliest_cr_line':earliest_cr_line, 'open_acc':open_acc, 'pub_rec':pub_rec, 
    'revol_util':revol_util, 'total_acc':total_acc, 'mort_acc':mort_acc, 
    'pub_rec_bankruptcies':pub_rec_bankruptcies, 'log_annual_inc': np.log10(annualIncome+1),'fico_score':680, 
    'log_revol_bal','home_ownership_'+str(home_ownership):1, 
    'verification_status_Source Verified':0, 'verification_status_Verified':0,
    'initial_list_status_w':1, 'application_type_Joint App':application_type_Join_App, 'addr_state_'+str(state):1
}


# In[37]:


newBorrower.append(tmpDict, ignore_index=True)
newBorrower = newBorrower.fillna(0)

y_train = pd.read_pickle('y_train.p')
y_test = pd.read_pickle('y_test.p')
X_train = pd.read_pickle('X_train.p')
X_test = pd.read_pickle('X_test.p')
# The machine learning pipeline:

# In[5]:


with open("modelCreditDefault.p", 'rb') as input_file:
    modelTest = pickle.load(input_file)


# In[ ]:


prediction = modelTest.predict(newBorrower)


# In[ ]:


st.write(prediction)


# In[260]:


y_score = modelTest.predict_proba(X_test)[:,1]


# In[261]:


y_score

