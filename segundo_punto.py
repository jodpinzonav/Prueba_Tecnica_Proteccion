"""
Análisis de sentimientos de los tweets:
Compararemos diferentes modelos: ML Tradicional vs Gemini vs Mistral Large
"""
import pandas as pd
import numpy as np
import re
import warnings
import time
import json
from typing import List, Disct, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


#API's de LLMs (Gemini y Mistral)
import google.generativeai as genai
from mistralai import Mistral

warnings.filterwarnings('ignore')

class LLMONFIG:
    """configurar las APIS para LLMs"""

    def __init__(self):
        # API KEYS -- en produccion usar variables de entorno
        self.GEMINI_API_KEY = "AIzaSyA8ePX40h2EUITvYvVY3qubZyjJvl9Mnmk"
        self.MISTRAL_API_KEY = "RBEK1rdTndtGGeOoNLGPuvStLZOc9bbA"

    def setup_gemini(self):
        """" Gemini 1.5 Flash """
        genai.configure(api_key=self.GEMINI_API_KEY)
        return genai.GenerativeModel('Gemini-1.5-flash')
    
    def setup_mistral(self):
        """ Mistral Large """
        return Mistral(api_key=self.MISTRAL_API_KEY)
    

class SentimentAnalysisPipeline:
    """
    Se tendran 3 enfoques:
    1. ML tradicional (SVM + TF-IDF)
    2. Gemini 1.5 Flash (Google)
    3. Mistral Large
    """
    def __init__ (self,random_state=42):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.llm_config = LLMONFIG()
    
    def load_and_explore_data(self,filepath: str) -> Tuple[pd.DataFrame,str,str]:
        """ cargar y explorar el dataset """
        print(" FASE 1: CARGA Y EXPLORACIÓN DE DATOS")


        df = pd.read_csv(filepath, encoding='utf-8')

        print(f"\n Dimensiones del dataset: {df.shape}")
        print(f" columnas del dataset: {df.columns.tolist()}")
        print(f"\n Vista preliminar de los datos:\n {df.head()}")

        expected_cols = ['user','text','date','emotion','sentiment']
        if all(col in df.columns for col in expected_cols):
            print(" Estructura de datos correcta.")
            text_col='text'
            label_col='sentiment'
            emotion_col='emotion'
        else:
            print("estructura diferente")
        print("vista previa de los primeros registros")
        print(df.head())

        if text_col and label_col:
            print(" columnas seleccionadas para el analisis")
            print(f" texto : '{text_col}'")
            print(f" sentimiento : '{label_col}'")
            print(f" Emoción : '{emotion_col}' (para analisis complementario)")

            print(f"\n ANALISIS DE SENTIMIENTOS:")
            print(f" valores unicos en '{label_col}':")
            sentiment_counts = df[label_col].value_counts()
            print(sentiment_counts)
            print(f"\n porcentajes:")
            print((sentiment_counts / len(df)* 100).round(2))

            if emotion_col and emotion_col in df.columns:
                print(f"\n ANALISIS DE EMOCIONES:")
                print(f" Valores unicos en '{emotion_col}'")
                emotion_counts = df[emotion_col].value_counts()
                print(emotion_counts)
                print(f"\n Total de emociones diferentes: {df[emotion_col].nunique()}")

            else:
                print(f"\n no se pudieron identificar las columnas necesarias")
                print(f" Columnas disponibles: {df.columns.tolist()}")
                print("por favor verificar la estructura del dataset")
        
        return df, text_col, label_col
    
    def preprocess_text(self, text: str) -> str:
        """ preprocesar tweets"""
    

