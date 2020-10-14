#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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


mpl.style.use('ggplot')
sns.set(style='whitegrid')


# In[36]:


class CreditDefault:
    def __init__(self):
        self.columns = ['loan_amnt','term','int_rate','installment', 'emp_length', 'dti', 'earliest_cr_line',
 'open_acc','pub_rec', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 'log_annual_inc',
 'fico_score', 'log_revol_bal', 'sub_grade_A2', 'sub_grade_A3', 'sub_grade_A4', 'sub_grade_A5', 'sub_grade_B1', 'sub_grade_B2', 'sub_grade_B3',
 'sub_grade_B4', 'sub_grade_B5', 'sub_grade_C1', 'sub_grade_C2', 'sub_grade_C3', 'sub_grade_C4', 'sub_grade_C5',
 'sub_grade_D1', 'sub_grade_D2', 'sub_grade_D3', 'sub_grade_D4', 'sub_grade_D5', 'sub_grade_E1', 'sub_grade_E2',
 'sub_grade_E3', 'sub_grade_E4', 'sub_grade_E5', 'sub_grade_F1', 'sub_grade_F2', 'sub_grade_F3', 'sub_grade_F4', 'sub_grade_F5',
 'sub_grade_G1', 'sub_grade_G2', 'sub_grade_G3', 'sub_grade_G4', 'sub_grade_G5',
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
    def run(self):
        liste = []
        for i in 'ABCDEFG':
            for x in range(1,6):
                liste.append(i+str(x))
        states = ['CA', 'NY', 'TX', 'FL', 'IL', 'NJ','OH']
        st.sidebar.subheader('Krediteigenschaften:')
        loan_amnt = st.sidebar.slider('Wie hoch ist der Kreditbetrag?',0,40000,10000)
        term = st.sidebar.radio('Laufzeit des Kredits?', (36,60))
        int_rate = st.sidebar.slider('Wie hoch ist der Zinssatz?', 0.0, 15.0,5.0,0.25)
        installment = (loan_amnt*(1+(int_rate/100))**(term/12))/term
        application_type_Join_App = st.sidebar.radio('Wird der Antrag alleine gestellt?', ('Ja', 'Nein'))
        if application_type_Join_App == 'Ja':
            application_type_Join_App = 0
        else:
            application_type_Join_App = 1
        #todo einbauen
        tmpDict = {
            'Kreditkartenschulden':'purpose_credit_card',
            'Schuldenkonsolidierung': 'purpose_debt_consolidation',
            'Fortbildung' : 'purpose_educational',
            'Einrichtung': 'purpose_home_improvement',
            'Hauskauf': 'purpose_house',
            'Größere Anschaffung': 'purpose_major_purchase',
            'Gesundheit': 'purpose_medical',
            'Umzug': 'purpose_moving',
            'Weiteres': 'purpose_other',
            'Erneuerbare Energien': 'purpose_renewable_energy',
            'Kleines Unternehmen': 'purpose_small_business',
            'Reisen': 'purpose_vacation',
            'Hochzeit': 'purpose_wedding'
        }
        
        st.sidebar.subheader('Kundeneigenschaften:')
        st.sidebar.text('Allgemeine Aspekte')
        
        state = st.sidebar.select_slider('Aus welchem Staat kommt der Kreditnehmer?', options = states)
        home_ownership = st.sidebar.radio('Wie steht es um die Wohnverhältnisse?', ('OWN', 'RENT', 'OTHER'))
        emp_length = st.sidebar.slider('Wie lange besteht das Beschäftigungsverhältnis?', 0, 15,5)
        pub_rec = st.sidebar.slider('Wie viele negative Eintragungen bestehen?', 0,15,0)
        pub_rec_bankruptcies = st.sidebar.slider('Wie viele Privatinsolvenzen sind veröffentlicht?',0,5,0)
        initial_list_status_w = 1
        
        st.sidebar.text('Finanziell Aspekte')
        
        annualIncome = st.sidebar.slider('Wie viel verdient der Kreditnehmer pro Jahr?', 0,200000, 50000) 
        mort_acc = st.sidebar.slider('Wie viele Hauskredite bestehen?', 0, 10, 2)
        earliest_cr_line = st.sidebar.slider('Wann wurde die erste Kreditlinie beantragt?', 1995,2015,2010)
        total_acc = st.sidebar.slider('Wie viele Kredite befinden sich in der Kredithistorie?', 0,30,5)
        open_acc = st.sidebar.slider('Wie viele Kredite werden aktuell beansprucht?', 0, 30, 5)
        revol_util = st.sidebar.slider('Wie viel Prozent der zur Verfügung stehenden Kreditlinie nutzt der Kunde aktuell?', 0.0, 1.0,0.1)

        tmpDict = {
            'loan_amnt':loan_amnt, 'term':term, 'int_rate':int_rate, 
            'installment':installment, 'emp_length':emp_length, 'dti':100,
            'earliest_cr_line':earliest_cr_line, 'open_acc':open_acc, 'pub_rec':pub_rec, 
            'revol_util':revol_util, 'total_acc':total_acc, 'mort_acc':mort_acc, 
            'pub_rec_bankruptcies':pub_rec_bankruptcies, 'log_annual_inc': np.log10(annualIncome+1),'fico_score':680, 
            'log_revol_bal': 200000,'home_ownership_'+str(home_ownership):1, 'sub_grade_'+str(grade):1, 
            'verification_status_Source Verified':0, 'verification_status_Verified':0,
            'initial_list_status_w':1, 'application_type_Joint App':application_type_Join_App, 'addr_state_'+str(state):1
        }
        newBorrower = pd.DataFrame(data = tmpDict, columns = self.columns, index=[0])
        newBorrower = newBorrower.fillna(0)
        model = self.loadModel()
        
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
            </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        
        st.title('3. Fallstudie Kreditausfall')
        st.header('Am Beispiel einer amerikanischen Kreditplattform')
        textIntro = """
        Die Grundlage dieses Fallbeispiels bidlet ein Datensatz der Kreditvergabeplattform Lending Club. Auf der
        Plattform können Privatleute Kredite beantragen, die widerrum von privaten oder institutionellen Investoren
        finanziert werden. In dem Betrachtungszeitraum liegen knapp 2 Mio. Datensätze mit Krediten in unterschiedlichen
        Stadien. Für das Beispiel wurden nur die Kredite betrachtet, die entweder komplett abbezahlt wurden oder während der
        Laufzeit ausgefallen sind. Von diesen Daten wurden nur die zugänglichen Informationen genommen, die bei Kreditbeantragung
        zur Verfügung standen und ein Model trainiert, welches den Kreditausfall prognostizieren sollen. In der
        Praxis könnte ein derartiges Modell als Entscheidungshilfe dienen. Probieren Sie es aus! Erstellen Sie über die Slider 
        einen eigenen Kreditbeantrager.
        """
        st.write(textIntro)
        st.write('Auf Basis der eingestellten Parameter kommt das Modell zu folgender Entscheidung:')
        st.write(newBorrower)
        prediction = model.predict(newBorrower)
        if prediction == 0:
            st.write('Die Kreditausfallwahrscheinlichkeit wird als gering eingestuft.')
        elif prediction == 1:
            st.write('Die Kreditausfallwahrscheinlichkeit wird als hoch eingestuft.')
        st.write('''Die folgenden Grafiken sollen veranschaulichen, welche Einflussfaktoren für die Entscheidung
                 des Modells maßgeblich waren. Diese sogeannten SHAP-Values dienen als Interpretationshilfe.
                 ''')
        
    def loadModel(self):
        with open("modelCreditDefault.p", 'rb') as input_file:
            modelTest = pickle.load(input_file)
        return modelTest

