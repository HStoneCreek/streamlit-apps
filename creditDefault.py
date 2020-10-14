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
import xgboost
# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)


mpl.style.use('ggplot')
sns.set(style='whitegrid')


# In[36]:


class CreditDefault:
    def __init__(self):
        self.columns = ['loan_amnt', 'term', 'int_rate', 'installment', 'emp_length', 'dti', 'earliest_cr_line', 'open_acc', 'pub_rec', 'revol_util', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies', 'log_annual_inc', 'fico_score', 'log_revol_bal', 'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT', 'verification_status_Source Verified', 'verification_status_Verified', 'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_educational', 'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 'purpose_small_business', 'purpose_vacation', 'purpose_wedding', 'addr_state_CA', 'addr_state_FL', 'addr_state_IL', 'addr_state_NJ', 'addr_state_NY', 'addr_state_OH', 'addr_state_TX', 'initial_list_status_w', 'application_type_Joint App']
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
            'log_revol_bal': 200000,'home_ownership_'+str(home_ownership):1, 
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
        prediction = model.predict(newBorrower)
        if prediction == 0:
            st.markdown("""<div class='customer'><b>Die Kreditausfallwahrscheinlichkeit wird als gering eingestuft.</b></div>""",unsafe_allow_html=True)
        elif prediction == 1:
            
            st.markdown("""<div class='customer'><b>Die Kreditausfallwahrscheinlichkeit wird als hoch eingestuft.</b></div>""",unsafe_allow_html=True)
            
        st.subheader('Analyse der Einflussfaktoren auf die Prognose - SHAP-Values')
        st.write('''Die folgenden Grafiken sollen veranschaulichen, welche Einflussfaktoren für die Entscheidung
                 des Modells maßgeblich waren. Diese sogeannten SHAP-Values dienen als Interpretationshilfe.
                 ''')
        
        html = """<div>
        <input type="checkbox" id="faq-1">
        <h2><label for="faq-1">Weitere Informationen </label></h2>
        <p id="drop">Es folgen weitere Informationen</p>
        </div>"""
        st.markdown(html, unsafe_allow_html=True)
        
    def loadModel(self):
        model = pickle.load(open("xgb_reg.pkl", 'rb'))
        
        return model

