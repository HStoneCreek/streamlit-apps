#!/usr/bin/env python
# coding: utf-8

# In[2]:


import spacy_streamlit as sst
import streamlit as st
from transformers import pipeline

import os, sys
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge, Rectangle
st.set_option('deprecation.showPyplotGlobalUse', False)
import spacy
from spacy import load
#import en_core_web_md

#nlpEN = en_core_web_md.load()


# In[ ]:


class TextSentiment:
    def __init__(self):
        self.nlpEN = 'nlpEN'
        self.nlpDE = 'nlpDE'
        
    
        
    def run(self):
        
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
        
        @st.cache
        def loadPipeline():

            return pipeline('sentiment-analysis')
        
        
        DEFAULT_TEXT = """Instead of the anticipated million-mile battery, we got the $25,000 car, at Tesla’s eagerly awaited Battery Day on Tuesday.
    Elon Musk emphasised cutting the cost of batteries by more than half, rather than getting more miles out of them, meaning an electric car priced
    on a par with conventional vehicles becoming available in about three years.
    Tesla will get there partly by taking control of all stages of manufacturing of its batteries, including the basic cells,
    processing the raw materials and even buying lithium deposits still in the ground.
    """


        language = st.sidebar.selectbox(label='Wählen Sie bitte die Sprache', options=['Englisch', 'Deutsch'])
        if language == 'Deutsch':
            spacy_model = 'de_core_news_md'
            DEFAULT_TEXT = """
            Dies ist ein Beispiel."""
        elif language == 'Englisch':
            spacy_model = 'en_core_web_md'
            DEFAULT_TEXT = """Instead of the anticipated million-mile battery, we got the $25,000 car, at Tesla’s eagerly awaited Battery Day on Tuesday. Elon Musk emphasised cutting the cost of batteries by more than half, rather than getting more miles out of them, meaning an electric car priced on a par with conventional vehicles becoming available in about three years. Tesla will get there partly by taking control of all stages of manufacturing of its batteries, including the basic cells, processing the raw materials and even buying lithium deposits still in the ground.
        """

        st.title('1. Fallstudie Textanalyse')
        st.header('Am Beispiel von Entitätserkennung und Sentimentanalyse')
        st.write("""Mit Hilfe von neuronalen Netzen lassen sich aus Textbausteinen Kontexte erschließen und die Stimmung einer Nachricht bestimmen.""")
        text = st.text_area("Folgenden Text analysieren:", DEFAULT_TEXT, height=200)# max_chars=None
        doc = sst.process_text(spacy_model, text)

        sst.visualize_ner(
            doc,
            labels=["PERSON", "DATE", "GPE", "EVENT", "ORG",'CARDINAL', 'FAC', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'PERCENT', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART'],
            show_table=False,
            title="Analyse von Personen, Daten, Organisationen etc.",
            sidebar_title=None,
            colors = {'color': "#blue"}
        )

        #classifier = loadPipeline()

        #summary = classifier(text)

        #if summary[0]['label'] == 'NEGATIVE':
        #    score = round((summary[0]['score']*-1)+2)
        #elif summary[0]['label'] == 'POSITIVE':
            #score = round((summary[0]['score'])+2)
        score = 3
        st.write('Die Nachricht hat ein Sentiment von: {}'.format(score))
        plot = Tachometer().gauge(labels=['NEGATIVE', 'NEUTRAL','POSITIVE'],arrow=score, colors=['#DC143C','#778899','#32CD32'])
        st.pyplot(plot)
        
        html = """<div>
        <input type="checkbox" id="faq-1">
        <h2><label for="faq-1">Weitere Informationen </label></h2>
        <p id="drop">Grundlage für diese Fallstudie ist ein bereit trainiertes Neuronales Netz, welches auf die jeweilige
        Sprache vortrainiert wurde. Für weitere Informationen gucken Sie bitte hier: https://spacy.io/. Die Integration
        in Streamlit erfolgte dankenswerterweise durch: https://github.com/explosion/spacy-streamlit</p>
        </div>"""
        st.markdown(html, unsafe_allow_html=True)
        
class Tachometer:
    def __init__(self):
        self.name ='Text'
    
    def degree_range(self,n): 
        start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
        end = np.linspace(0,180,n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points
    def rot_text(self,ang): 
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation
    
    def gauge(self, labels=None, colors='jet_r', arrow=1, title='', fname=False):   
    
        N = len(labels)

        if arrow > N: 
            raise Exception("\n\nThe category ({}) is greated than             the length\nof the labels ({})".format(arrow, N))

        if isinstance(colors, str):
            cmap = cm.get_cmap(colors, N)
            cmap = cmap(np.arange(N))
            colors = cmap[::-1,:].tolist()
        if isinstance(colors, list): 
            if len(colors) == N:
                colors = colors[::-1]
            else: 
                raise Exception("\n\nnumber of colors {} not equal                 to number of categories{}\n".format(len(colors), N))


        #begins the plotting

        fig, ax = plt.subplots(figsize=(10,3))

        ang_range, mid_points = self.degree_range(N)

        labels = labels[::-1]

        #plots the sectors and the arcs

        patches = []
        for ang, c in zip(ang_range, colors): 
            # sectors
            patches.append(Wedge((0.,0.), .4, *ang, facecolor='w', lw=2))
            # arcs
            patches.append(Wedge((0.,0.), .4, *ang, width=0.10, facecolor=c, lw=2, alpha=0.5))


        #[ax.add_patch(p) for p in patches]

        for p in patches:
            ax.add_patch(p)
        #set the labels (e.g. 'LOW','MEDIUM',...)


        for mid, lab in zip(mid_points, labels): 

            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab,                 horizontalalignment='center', verticalalignment='center', fontsize=14,                 fontweight='bold', rotation = self.rot_text(mid))


        #set the bottom banner and the title

        r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
        ax.add_patch(r)

        ax.text(0, -0.05, title, horizontalalignment='center',              verticalalignment='center', fontsize=22, fontweight='bold')


        #plots the arrow now

        pos = mid_points[abs(arrow - N)]

        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)),                      width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))


        #removes frame and ticks, and makes axis equal and tight


        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()
        if fname:
            fig.savefig(fname, dpi=200)

