#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st

import textSentiment

from textSentiment import *
from customerClustering import *
#import creditDefault


# In[ ]:


def main():
    
    
    case = st.sidebar.selectbox(label = 'Wählen Sie bitte die Fallstudie?', options = ['Überblick', 'Textanalyse', 'Kunden Klassifikation',
                                                                                       'Kreditausfall'])
    if case == 'Überblick':
        st.title('KI Erleben - Interaktive Fallstudien')
        st.header('3 Szenarien für den Bankalltag')
        st.markdown("""<style>h3:hover{color: red;}</style><div><h3>Test</h3></div>""", unsafe_allow_html=True)
        st.subheader('1. Textsentiment Analyse ')
        textSentiment = '''
        Jede Sekunde werden Millionen Zeilen von Informationen veröffentlicht, die unterschiedliche Relevanz für den Bankalltag haben.
        Nachrichten, Gesetze und Veröffentlichungen, ob auf Twitter oder wissenschaftlicher Natur, stellen nur einen Teil der
        Informationsflut dar. Im Bereich der künstlichen Intelligenz haben sich zwei Teilbereiche herausgebildet, die den Menschen
        bei der Verarbeitung der Informationen unterstützen sollen. Entity Recognition identifiziert Personen, Orte, Daten und viele
        weitere Objekte in Texten und stellt diese heraus. Dadurch können Text direkt gefiltert und auch relevante Informationen hin untersucht werden.
        Sentiment Analyse versucht hingegen die "Stimmung" der Informationen zu identifizieren. In dem verwendeten Beispiel wird
        auf ein "vortrainiertes" Modell zurückgegriffen.
        '''
        st.write(textSentiment)

        st.subheader('2. Kunden Klassifikation')
        textKundenKlassifikation = '''
        Dem Kunden bedarfsgerechte Lösungen anzubieten, ist das Ziel von Vertriebs- und Marketingabteilungen. Doch wie sind die Kunden einzuordnen?
        Hier kann die Kunden Klassifikation unterstützen. Die Modelle erarbeiten "Cluster" von Kundengrruppen auf Basis von verschiedenen
        Merkmalen des einzelnen Kunden. Anschließend können die Kundengruppen gezielt angesprochen werden.
        '''
        st.write(textKundenKlassifikation)

        st.subheader('3. Kreditausfall Prognose')
        textCreditDefault = '''
        Im Rahmen der zunehmenden Regulierung ist eine Risikomessung- und reduzierung notwendig, um Risikovorsorgen zu minimieren und 
        gleichzeitig den Gewinn zu stabilisieren. Vor diesem Hintergrund ist eine Prognose des Kreditausfalls bei Vergabe interessant.
        Die 3. Fallstudie beschäftigt sich mit eben diesem Problem.
        '''
        st.write(textCreditDefault)
    if case == 'Textanalyse':
        TextSentiment.run()
    
    if case == 'Kunden Klassifikation':
        st.markdown("""<style>hover h3{color: red;}</style><div><h3>Test</h3></div>""", unsafe_allow_html=True)
        customerClass = CustomerClassification()
        customerClass.run()
    
    
if __name__ == '__main__':
    main()

