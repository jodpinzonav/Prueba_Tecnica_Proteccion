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
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
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
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text) # urls
        text = re.sub(r'@\w+', '', text)  # menciones
        text = re.sub(r'#(\w+)', r'\1', text)  # Hashtags (mantener texto)
        text = re.sub(r'[^a-záéíóúñü\s]', '', text)  # Caracteres especiales
        text = re.sub(r'\s+', ' ', text).strip()  # Espacios múltiples

        return text
    
    def prepare_sample(self,df: pd.DataFrame, text_column: str, label_column: str, sample_size: int=1000) -> pd.DataFrame:
        """ preparar una muestra del dataset """
        print("\n FASE 2: PREPARACIÓN DE DATOS Y MUESTRA BALANCEADA")

        # Preprocesar textos
        df_clean = df[[text_column, label_column]].copy().dropna()

        # IMPORTANTE: Normalizar sentimientos en inglés a español
        # overwhelmed, scared, etc. → NEGATIVO, POSITIVO, NEUTRO

        sentiment_mapping = {
            # Negativos
            'scared': 'NEGATIVO',
            'overwhelmed': 'NEGATIVO',
            'sad': 'NEGATIVO',
            'angry': 'NEGATIVO',
            'frustrated': 'NEGATIVO',
            'disappointed': 'NEGATIVO',
            'anxious': 'NEGATIVO',
            'worried': 'NEGATIVO',
            'upset': 'NEGATIVO',
            'mad': 'NEGATIVO',
            
            # Positivos
            'happy': 'POSITIVO',
            'excited': 'POSITIVO',
            'joyful': 'POSITIVO',
            'grateful': 'POSITIVO',
            'satisfied': 'POSITIVO',
            'delighted': 'POSITIVO',
            'proud': 'POSITIVO',
            'powerful': 'POSITIVO',
            
            # Neutros
            'neutral': 'NEUTRO',
            'calm': 'NEUTRO',
            'indifferent': 'NEUTRO',
            'okay': 'NEUTRO',
            'content': 'NEUTRO',
            'peaceful': 'NEUTRO',
        }


        df_clean['sentiment_normalized'] = df_clean[label_column].str.lower().map(sentiment_mapping)

        # Si algunos sentimientos no están en el mapeo, clasificarlos manualmente
        unmapped = df_clean[df_clean['sentiment_normalized'].isna()][label_column].unique()
        if len(unmapped) > 0:
            for sentiment in unmapped:
                print(f" Clasificando sentimiento no mapeado: '{sentiment}'")
            print("   Clasificándolos como NEUTRO por defecto...")
            df_clean['sentiment_normalized'].fillna('NEUTRO', inplace=True)


        # Usar la columna normalizada
        df_clean[label_column] = df_clean['sentiment_normalized']

        #elimino la columna virtual que creae
        df_clean = df_clean.drop('sentiment_normalized', axis=1)

        print(f"\n Registros válidos: {len(df_clean)}")
        print(f"\n Distribución de sentimientos después de normalización:")
        print(df_clean[label_column].value_counts())
        print(f"\n Porcentajes:")
        print((df_clean[label_column].value_counts(normalize=True) * 100).round(2))

     # el dataframe tiene que tener más registros que el tamaño de la muestra
        if len(df_clean) > sample_size:
            samples_per_class = sample_size // len(df_clean[label_column].unique())
            df_sample = df_clean.groupby(label_column, group_keys=False).apply(
                lambda x: x.sample(min(len(x), samples_per_class), 
                                 random_state=self.random_state)
            )
            # Ajustar al tamaño exacto si es necesario
            if len(df_sample) < sample_size:
                remaining = sample_size - len(df_sample)
                extra = df_clean.drop(df_sample.index).sample(remaining, random_state=self.random_state)
                df_sample = pd.concat([df_sample, extra])
            elif len(df_sample) > sample_size:
                df_sample = df_sample.sample(sample_size, random_state=self.random_state)
        else:
            df_sample = df_clean

        print(f"\n Tamaño de muestra final: {len(df_sample)} tweets")
        print(f"\n Distribución en la muestra:")
        print(df_sample[label_column].value_counts())

        # Preprocesamiento
        print("\n Aplicando preprocesamiento de texto...")
        df_sample['text_clean'] = df_sample[text_column].apply(self.preprocess_text)
        df_sample = df_sample[df_sample['text_clean'].str.len() > 5]  # Mínimo 5 caracteres

        print(f" Tweets después de limpieza: {len(df_sample)}")

        return df_sample
    
    def train_traditional_ml(self,X_train,X_test,y_train,y_test):
        """ ENFOQUE 1: ML TRADICIONAL (SVM + TF-IDF) """
        print("ENFOQUE 1: MACHINE LEARNING TRADICIONAL")
        print("Algoritmo: Support Vector Machine con TF-IDF")

        print("\n🔧 Configurando vectorizador TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2), #unigramas y bigramas
            min_df=2, # mínimo 2 documentos
            max_df=0.8, # máximo 80% de documentos
            sublinear_tf=True, #frecuencia sublineal
            strip_accents=None, #sin acentos
            )
        
        print("🔄 Transformando textos a vectores TF-IDF...")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        print(f"✅ Dimensionalidad de features: {X_train_tfidf.shape[1]}")
        # Optimización de hiperparámetros con Grid Search
        print("\n🔍 Optimizando hiperparámetros con Grid Search (CV=5)...")
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(
            SVC(random_state=self.random_state, class_weight='balanced'),
            param_grid,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_tfidf, y_train)
        
        print(f"\n✅ Optimización completada:")
        print(f"   Mejores parámetros: {grid_search.best_params_}")
        print(f"   Mejor F1-Score CV: {grid_search.best_score_:.4f}")

        # Predicciones en test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_tfidf)
        
        return {
            'model': best_model,
            'vectorizer': vectorizer,
            'predictions': y_pred,
            'name': 'ML Tradicional (SVM)',
            'training_time': 'rápido'
        }
    
    def classify_with_gemini(self, text: str, model) -> str:
        """Clasificación individual con Gemini 1.5 Flash"""
        prompt = f"""Clasifica el sentimiento del siguiente tweet en español.
                     Tweet: "{text}" Responde ÚNICAMENTE con una de estas tres palabras: POSITIVO, NEGATIVO, NEUTRO 
                     No agregues explicaciones ni puntuación adicional."""

        try:
            response = model.generate_content(prompt)
            sentiment = response.text.strip().upper()
            
            # Normalización robusta
            if 'POSITIVO' in sentiment or 'POSITIVE' in sentiment:
                return 'POSITIVO'
            elif 'NEGATIVO' in sentiment or 'NEGATIVE' in sentiment:
                return 'NEGATIVO'
            else:
                return 'NEUTRO'
                
        except Exception as e:
            print(f"⚠️  Error en Gemini: {e}")
            return 'NEUTRO'
    

    def train_llm_gemini(self, X_test, y_test) -> Dict:
        """
        ENFOQUE 2: Gemini 1.5 Flash
        API de Google, optimizada para velocidad
        """
        print("\n" + "="*80)
        print("ENFOQUE 2: GEMINI 1.5 FLASH (GOOGLE)")
        print("="*80)
        
        print("\n🤖 Inicializando Gemini 1.5 Flash...")
        model = self.llm_config.setup_gemini()
        
        predictions = []
        total = len(X_test)
        
        print(f"\n🔄 Clasificando {total} tweets con Gemini...")
        print("   Progreso:")
        
        start_time = time.time()
        
        for idx, text in enumerate(X_test, 1):
            if idx % 25 == 0 or idx == total:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (total - idx) / rate if rate > 0 else 0
                print(f"   [{idx}/{total}] ({idx/total*100:.1f}%) - "
                      f"{rate:.1f} tweets/seg - ETA: {eta:.0f}s")
            
            pred = self.classify_with_gemini(text, model)
            predictions.append(pred)
            
            time.sleep(0.5)  # Rate limiting suave
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Clasificación completada en {elapsed_time:.1f} segundos")
        print(f"   Velocidad promedio: {total/elapsed_time:.2f} tweets/segundo")
        
        return {
            'predictions': predictions,
            'name': 'Gemini 1.5 Flash',
            'time': elapsed_time
        }


    def evaluate_model(self, y_test, y_pred, model_name: str) -> Dict:
        """Evaluación exhaustiva con múltiples métricas"""
        print(f"\n{'='*80}")
        print(f"EVALUACIÓN: {model_name}")
        print(f"{'='*80}")
        
        # Convertir a listas para compatibilidad
        y_test_list = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
        y_pred_list = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
        
        # Normalizar etiquetas (todo a mayúsculas)
        y_test_norm = [str(y).upper() for y in y_test_list]
        y_pred_norm = [str(y).upper() for y in y_pred_list]
        
        # Métricas generales
        accuracy = accuracy_score(y_test_norm, y_pred_norm)
        f1_weighted = f1_score(y_test_norm, y_pred_norm, average='weighted')
        f1_macro = f1_score(y_test_norm, y_pred_norm, average='macro')
        
        print(f"\n📊 MÉTRICAS PRINCIPALES:")
        print(f"   {'Accuracy:':<25} {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   {'F1-Score (Weighted):':<25} {f1_weighted:.4f}")
        print(f"   {'F1-Score (Macro):':<25} {f1_macro:.4f}")
        
        # Reporte detallado por clase
        print(f"\n📋 REPORTE DETALLADO POR CLASE:")
        report = classification_report(y_test_norm, y_pred_norm, digits=4)
        print(report)
        
        # Matriz de confusión
        print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
        cm = confusion_matrix(y_test_norm, y_pred_norm)
        labels = sorted(set(y_test_norm))
        
        # Imprimir matriz formateada
        print(f"\n{'':>12}", end='')
        for label in labels:
            print(f"{label:>12}", end='')
        print()
        
        for i, label in enumerate(labels):
            print(f"{label:>12}", end='')
            for j in range(len(labels)):
                print(f"{cm[i][j]:>12}", end='')
            print()
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test_norm, y_pred_norm, 
                                                          output_dict=True)
        }
    
    def save_results_to_csv(self, df_sample: pd.DataFrame, X_test, y_test,
                           ml_predictions=None, gemini_predictions=None, 
                           mistral_predictions=None, mistral_details=None,
                           output_filename: str = "sentiment_analysis_results.csv"):
        """
        Guarda resultados completos en CSV con predicciones de todos los modelos
        
        Args:
            df_sample: DataFrame original con todos los datos
            X_test: Textos de prueba
            y_test: Etiquetas reales
            ml_predictions: Predicciones del modelo ML
            gemini_predictions: Predicciones de Gemini
            mistral_predictions: Predicciones de Mistral
            mistral_details: Detalles adicionales de Mistral (confianza, razón)
            output_filename: Nombre del archivo de salida
        """
        print("\n" + "="*80)
        print("EXPORTANDO RESULTADOS A CSV")
        print("="*80)
        
        # Crear DataFrame de resultados
        X_test_list = X_test.tolist() if hasattr(X_test, 'tolist') else list(X_test)
        y_test_list = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
        
        results_df = pd.DataFrame({
            'tweet_original': X_test_list,
            'sentimiento_real': y_test_list
        })
        
        # Agregar predicciones de cada modelo si existen
        if ml_predictions is not None:
            results_df['pred_ml_tradicional'] = ml_predictions
            results_df['correcto_ml'] = results_df['sentimiento_real'].str.upper() == \
                                        pd.Series(ml_predictions).str.upper()
        
        if gemini_predictions is not None:
            results_df['pred_gemini'] = gemini_predictions
            results_df['correcto_gemini'] = results_df['sentimiento_real'].str.upper() == \
                                            pd.Series(gemini_predictions).str.upper()
        
        if mistral_predictions is not None:
            results_df['pred_mistral'] = mistral_predictions
            results_df['correcto_mistral'] = results_df['sentimiento_real'].str.upper() == \
                                             pd.Series(mistral_predictions).str.upper()
            
            # Agregar detalles adicionales de Mistral si existen
            if mistral_details is not None:
                results_df['mistral_confianza'] = [d.get('confianza', 'N/A') for d in mistral_details]
                results_df['mistral_razon'] = [d.get('razón', 'N/A') for d in mistral_details]
        
        # Agregar columna de consenso (cuando todos los modelos coinciden)
        if all(p is not None for p in [ml_predictions, gemini_predictions, mistral_predictions]):
            results_df['consenso'] = (
                (results_df['pred_ml_tradicional'].str.upper() == 
                 results_df['pred_gemini'].str.upper()) &
                (results_df['pred_gemini'].str.upper() == 
                 results_df['pred_mistral'].str.upper())
            )
            
            # Predicción final (mayoría simple o consenso)
            def get_final_prediction(row):
                preds = [
                    row.get('pred_ml_tradicional', '').upper(),
                    row.get('pred_gemini', '').upper(),
                    row.get('pred_mistral', '').upper()
                ]
                preds = [p for p in preds if p]  # Remover vacíos
                if not preds:
                    return 'N/A'
                # Retornar el más común
                from collections import Counter
                return Counter(preds).most_common(1)[0][0]
            
            results_df['prediccion_final'] = results_df.apply(get_final_prediction, axis=1)
            results_df['correcto_final'] = results_df['sentimiento_real'].str.upper() == \
                                           results_df['prediccion_final']
        
        # Agregar índice como ID
        results_df.insert(0, 'id', range(1, len(results_df) + 1))
        
        # Guardar a CSV
        results_df.to_csv(output_filename, index=False, encoding='utf-8')
        
        print(f"\n✅ Resultados guardados en: {output_filename}")
        print(f"   Total de registros: {len(results_df)}")
        print(f"   Columnas: {results_df.columns.tolist()}")
        
        # Estadísticas de accuracy por modelo
        print(f"\n📊 ACCURACY POR MODELO (en el test set):")
        if ml_predictions is not None:
            ml_acc = results_df['correcto_ml'].mean()
            print(f"   ML Tradicional: {ml_acc:.2%} ({results_df['correcto_ml'].sum()}/{len(results_df)} correctos)")
        
        if gemini_predictions is not None:
            gemini_acc = results_df['correcto_gemini'].mean()
            print(f"   Gemini Flash:   {gemini_acc:.2%} ({results_df['correcto_gemini'].sum()}/{len(results_df)} correctos)")
        
        if mistral_predictions is not None:
            mistral_acc = results_df['correcto_mistral'].mean()
            print(f"   Mistral Large:  {mistral_acc:.2%} ({results_df['correcto_mistral'].sum()}/{len(results_df)} correctos)")
        
        if 'prediccion_final' in results_df.columns:
            final_acc = results_df['correcto_final'].mean()
            consenso_rate = results_df['consenso'].mean()
            print(f"\n   Predicción Final (Ensemble): {final_acc:.2%}")
            print(f"   Tasa de Consenso: {consenso_rate:.2%} (todos los modelos coinciden)")
        
        # Guardar también errores en un CSV separado
        if any(p is not None for p in [ml_predictions, gemini_predictions, mistral_predictions]):
            error_columns = ['correcto_ml', 'correcto_gemini', 'correcto_mistral']
            error_columns = [col for col in error_columns if col in results_df.columns]
            
            if error_columns:
                errors_df = results_df[~results_df[error_columns].all(axis=1)]
                if len(errors_df) > 0:
                    error_filename = output_filename.replace('.csv', '_ERRORS.csv')
                    errors_df.to_csv(error_filename, index=False, encoding='utf-8')
                    print(f"\n⚠️  Errores guardados en: {error_filename}")
                    print(f"   Tweets con al menos 1 error: {len(errors_df)}")
        
        # Vista previa
        print(f"\n📝 Vista previa de resultados:")
        print(results_df.head(3).to_string(index=False))
        
        return results_df


def main():
    pipeline = SentimentAnalysisPipeline(random_state=42)
    # Configuración
    FILEPATH = r"data\sentiment_analysis_dataset.csv"  
    SAMPLE_SIZE = 1000

    try:
        # 1. Cargar datos
        df, text_col, label_col = pipeline.load_and_explore_data(FILEPATH)
        
        if not text_col or not label_col:
            print("\n ERROR: No se detectaron columnas automáticamente")
            print("   Especifica manualmente en el código:")
            print("   text_col = 'nombre_columna_texto'")
            print("   label_col = 'nombre_columna_sentimiento'")
            return
        
        # 2. Preparar muestra
        df_sample = pipeline.prepare_sample(df, text_col, label_col, SAMPLE_SIZE)
        
        # 3. Split train/test estratificado
        X = df_sample['text_clean']
        y = df_sample[label_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"\n📊 DIVISIÓN DE DATOS:")
        print(f"   Entrenamiento: {len(X_train)} tweets")
        print(f"   Prueba: {len(X_test)} tweets")
        print(f"   Distribución test: {y_test.value_counts().to_dict()}")
        
        # 4. ENFOQUE 1: ML Tradicional
        print("\n" + "🚀"*40)
        ml_results = pipeline.train_traditional_ml(X_train, X_test, y_train, y_test)
        eval_ml = pipeline.evaluate_model(y_test, ml_results['predictions'], 
                                         ml_results['name'])

    # 5. ENFOQUE 2: Gemini
        print("\n" + "🚀"*40)
        print("\n⚠️  GEMINI: Configura tu API key en LLMConfig.GEMINI_API_KEY")
        use_gemini = input("¿Ejecutar clasificación con Gemini? (s/n): ").lower() == 's'
        
        if use_gemini:
            gemini_results = pipeline.train_llm_gemini(X_test, y_test)
            eval_gemini = pipeline.evaluate_model(y_test, gemini_results['predictions'],
                                                 gemini_results['name'])
        else:
            print("⏭️  Saltando Gemini")
            eval_gemini = {'accuracy': 0, 'f1_weighted': 0, 'f1_macro': 0}
        
        

        # 5. Guardar resultados
        ml_preds = ml_results['predictions']
        results_df = pipeline.save_results_to_csv(
                    df_sample=df_sample,
                    X_test=X_test,
                    y_test=y_test,
                    ml_predictions=ml_preds,
                    output_filename="sentiment_analysis_results.csv"
                )

        
    except Exception as e:
        print(f"\n ERROR inesperado: {e}")
        return

if __name__ == "__main__":
    main()