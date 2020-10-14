#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st

import textSentiment


from textSentiment import *
from customerClustering import *
from creditDefault import *


# In[ ]:


class Overview:
    def __init__(self, listOfCases):
        self.listOfCases = listOfCases
    def createContent(self):
        
        css = """
            <style>
            input[type="checkbox"] {
              position: absolute;
              top: 0;
              left: 0;
              width: 100%;
              height: 100%;
              opacity: 0;
            }
            h2 {
              font-size: 30px;
              margin: 20px 0 0;
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
        

        html = """<div>
        <input type="checkbox" id="faq-1">
        <h2><label for="faq-1">1. Textsentiment Analyse </label></h2>
        <p id="drop">Jede Sekunde werden Millionen Zeilen von Informationen veröffentlicht, die unterschiedliche Relevanz für den Bankalltag haben.
        Nachrichten, Gesetze und Veröffentlichungen, ob auf Twitter oder wissenschaftlicher Natur, stellen nur einen Teil der
        Informationsflut dar. Im Bereich der künstlichen Intelligenz haben sich zwei Teilbereiche herausgebildet, die den Menschen
        bei der Verarbeitung der Informationen unterstützen sollen. Entity Recognition identifiziert Personen, Orte, Daten und viele
        weitere Objekte in Texten und stellt diese heraus. Dadurch können Text direkt gefiltert und auch relevante Informationen hin untersucht werden.
        Sentiment Analyse versucht hingegen die "Stimmung" der Informationen zu identifizieren. In dem verwendeten Beispiel wird
        auf ein "vortrainiertes" Modell zurückgegriffen.</p>
        </div>"""
        html2="""
        <div>
        <input type="checkbox" id="faq-2">
        <h2><label for="faq-2">2. Kunden Clustering</label></h2>
        <p id="drop">Dem Kunden bedarfsgerechte Lösungen anzubieten, ist das Ziel von Vertriebs- und Marketingabteilungen. Doch wie sind die Kunden einzuordnen?
        Hier kann die Kunden Klassifikation unterstützen. Die Modelle erarbeiten "Cluster" von Kundengrruppen auf Basis von verschiedenen
        Merkmalen des einzelnen Kunden. Anschließend können die Kundengruppen gezielt angesprochen werden.</p>
        </div>
        """
        html3 ="""
        <div>
        <input type="checkbox" id="faq-3">
        <h2><label for="faq-3">3. Kreditausfall Prognose</label></h2>
        <p id="drop">Im Rahmen der zunehmenden Regulierung ist eine Risikomessung- und reduzierung notwendig, um Risikovorsorgen zu minimieren und 
        gleichzeitig den Gewinn zu stabilisieren. Vor diesem Hintergrund ist eine Prognose des Kreditausfalls bei Vergabe interessant.
        Die 3. Fallstudie beschäftigt sich mit eben diesem Problem.</p>
        </div>
        """
        html4 ="""
        <div>
        <input type="checkbox" id="faq-4">
        <h2><label for="faq-4">Allgemeine Informationen</label></h2>
        <p id="drop">Die für diese Fallstudien verwendeten Daten sind freizugängliche, anonymisierte Datensätze.
        Rückschlüsse auf private Personen sind nicht möglich. Die einzelne Quelle findet sich unten in der
        jeweiligen Fallstudie. </p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
        st.markdown(html2, unsafe_allow_html=True)
        st.markdown(html3, unsafe_allow_html=True)
        st.markdown(html4, unsafe_allow_html=True)


# In[ ]:


def main():
    
    
    case = st.sidebar.selectbox(label = 'Wählen Sie bitte die Fallstudie?', options = ['Überblick', 'Textanalyse', 'Kunden Clustering',
                                                                                       'Kreditausfall'])
    if case == 'Überblick':
        css = """
        <style>h1{
        color: #f08200; 'orange
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
        st.title('KI Erleben - Interaktive Fallstudien')
        intro = '''
        Künstliche Intelligenz? Nicht mehr als ein Buzzword? Oder doch?
        Die folgenden Anwendungsbeispiele sollen veranschaulichen, wie Alogrithmen auch in der Finanzwelt unterstützen können.
        '''
        st.markdown(intro)
        st.image('ki.jpg', use_column_width=True)
        ov = Overview([])
        ov.createContent()
        #st.header('3 Szenarien für den Bankalltag')
        #st.subheader('1. Textsentiment Analyse ')
        textSentiment = '''
        Jede Sekunde werden Millionen Zeilen von Informationen veröffentlicht, die unterschiedliche Relevanz für den Bankalltag haben.
        Nachrichten, Gesetze und Veröffentlichungen, ob auf Twitter oder wissenschaftlicher Natur, stellen nur einen Teil der
        Informationsflut dar. Im Bereich der künstlichen Intelligenz haben sich zwei Teilbereiche herausgebildet, die den Menschen
        bei der Verarbeitung der Informationen unterstützen sollen. Entity Recognition identifiziert Personen, Orte, Daten und viele
        weitere Objekte in Texten und stellt diese heraus. Dadurch können Text direkt gefiltert und auch relevante Informationen hin untersucht werden.
        Sentiment Analyse versucht hingegen die "Stimmung" der Informationen zu identifizieren. In dem verwendeten Beispiel wird
        auf ein "vortrainiertes" Modell zurückgegriffen.
        '''
        #st.write(textSentiment)

        #st.subheader('2. Kunden Klassifikation')
        textKundenKlassifikation = '''
        Dem Kunden bedarfsgerechte Lösungen anzubieten, ist das Ziel von Vertriebs- und Marketingabteilungen. Doch wie sind die Kunden einzuordnen?
        Hier kann die Kunden Klassifikation unterstützen. Die Modelle erarbeiten "Cluster" von Kundengrruppen auf Basis von verschiedenen
        Merkmalen des einzelnen Kunden. Anschließend können die Kundengruppen gezielt angesprochen werden.
        '''
        #st.write(textKundenKlassifikation)

        #st.subheader('3. Kreditausfall Prognose')
        textCreditDefault = '''
        Im Rahmen der zunehmenden Regulierung ist eine Risikomessung- und reduzierung notwendig, um Risikovorsorgen zu minimieren und 
        gleichzeitig den Gewinn zu stabilisieren. Vor diesem Hintergrund ist eine Prognose des Kreditausfalls bei Vergabe interessant.
        Die 3. Fallstudie beschäftigt sich mit eben diesem Problem.
        '''
        #st.write(textCreditDefault)
    if case == 'Textanalyse':
        textSenti = TextSentiment()
        textSenti.run()
    
    if case == 'Kunden Clustering':
        
        customerClass = CustomerClassification()
        customerClass.run()
    
    if case == 'Kreditausfall':
        creditDefault = CreditDefault()
        creditDefault.run()
    
    
if __name__ == '__main__':
    main()

