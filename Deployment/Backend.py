# Importar las librerías necesarias
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unicodedata

import pandas as pd

# Text processing and NLP
import re
import string
import unicodedata
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Visualization
# %matplotlib inline

# Machine Learning and Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Configurar FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar el modelo LSTM previamente entrenado
model = tf.keras.models.load_model('model_LSTM2_best.h5')

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
stop_words_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Configuración de Tokenizer
vocabulary_size = 5000
tweet_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocabulary_size)
max_length = 23

# Funciones de procesamiento de texto
def remove_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_punctuation(text):
    """Remove punctuation from the text."""
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)

def remove_emojis(text):
    """Remove emojis from the text."""
    emoji_pattern = re.compile(
        r'['
        u'\U0001F600-\U0001F64F'  # Emoticons
        u'\U0001F300-\U0001F5FF'  # Symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # Transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # Flags (iOS)
        u'\U00002702-\U000027B0'  # Miscellaneous symbols
        u'\U000024C2-\U0001F251'  # Other symbols
        ']+',
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def remove_html_tags(text):
    """Remove HTML tags from the text."""
    html_pattern = re.compile(r'<.*?>|&(?:[a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return html_pattern.sub('', text)

def preprocess_text(input_text):
    """Clean and preprocess the input text."""
    cleaned_text = str(input_text)
    cleaned_text = re.sub(r"\bI'm\b", "I am", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\byou're\b", "you are", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bthey're\b", "they are", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bcan't\b", "cannot", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bwon't\b", "will not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bdon't\b", "do not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bdoesn't\b", "does not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bain't\b", "am not", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bwe're\b", "we are", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bit's\b", "it is", cleaned_text, flags=re.IGNORECASE)

    cleaned_text = re.sub(r"&gt;", ">", cleaned_text)
    cleaned_text = re.sub(r"&lt;", "<", cleaned_text)
    cleaned_text = re.sub(r"&amp;", "&", cleaned_text)


    cleaned_text = re.sub(r"\bw/\b", "with", cleaned_text)  # "w/" → "with"
    cleaned_text = re.sub(r"\blmao\b", "laughing my ass off", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"<3", "love", cleaned_text)  # Corazón → "love"
    cleaned_text = re.sub(r"\bph0tos\b", "photos", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\bamirite\b", "am I right", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\btrfc\b", "traffic", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\b16yr\b", "16 year", cleaned_text)

    cleaned_text = str(cleaned_text).lower()  # Convert to lowercase

    # Remove unwanted patterns
    cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)  # Remove content inside brackets
    cleaned_text = re.sub(r'https?://\S+|www\.\S+', '', cleaned_text)  # Remove URLs
    cleaned_text = re.sub(r'<.*?>+', '', cleaned_text)  # Remove HTML tags
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)  # Replace newlines with spaces
    cleaned_text = re.sub(r'\w*\d\w*', '', cleaned_text)  # Remove words with numbers

    # Call additional cleaning functions
    cleaned_text = remove_urls(cleaned_text)  # Remove URLs
    cleaned_text = remove_emojis(cleaned_text)  # Remove emojis
    cleaned_text = remove_html_tags(cleaned_text)  # Remove HTML tags
    cleaned_text = remove_punctuation(cleaned_text)  # Remove punctuation

    return cleaned_text

def process_tweet_content(tweet):
    cleaned_tweet = preprocess_text(tweet)
    processed_tweet = ' '.join(lemmatizer.lemmatize(word) for word in cleaned_tweet.split() if word not in stop_words_list)

    return processed_tweet

def normalize_text(text):
    normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    return normalized_text

def remove_specific_words(tweet):
    tweet = normalize_text(tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation)).lower()
    cleaned_tweet = ' '.join(word for word in tweet.split() if word not in custom_words)

    return cleaned_tweet

def tokenize_corpus(corpus):
    return tweet_tokenizer.texts_to_sequences(corpus)
# Ruta principal
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Cargar dataset de prueba
    test_dataset = pd.read_csv("test.csv", encoding="latin-1")
    
    # Procesar los tweets y realizar predicciones
    test_dataset['text_clean'] = test_dataset['text'].apply(preprocess_text)
    test_tweets = test_dataset['text_clean'].values
    test_padded = pad_sequences(tweet_tokenizer.texts_to_sequences(test_tweets), maxlen=max_length, padding='post')
    
    # Realizar predicciones y preparar los resultados
    predictions = (model.predict(test_padded) >= 0.6).astype(int).flatten()
    results_df = pd.DataFrame({
        'Tweet': test_dataset['text'].values,
        'Prediction': predictions,
        'Label': ['REAL' if pred == 1 else 'FAKE' for pred in predictions]
    })
    
    # Filtrar los primeros 10 tweets reales y falsos
    real_tweets = results_df[results_df['Label'] == 'REAL'].head(10)
    fake_tweets = results_df[results_df['Label'] == 'FAKE'].head(10)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "real_tweets": real_tweets.to_dict(orient="records"),
        "fake_tweets": fake_tweets.to_dict(orient="records")
    })

# Ruta para predicción de un tweet ingresado
@app.post("/predict", response_class=HTMLResponse)
async def predict_tweet(request: Request, tweet: str = Form(...)):
    # Procesar y predecir el tweet ingresado
    tweet_clean = preprocess_text(tweet)
    tweet_sequence = pad_sequences(tweet_tokenizer.texts_to_sequences([tweet_clean]), maxlen=max_length, padding='post')
    prediction = (model.predict(tweet_sequence) >= 0.6).astype(int).flatten()[0]
    label = "REAL" if prediction == 1 else "FAKE"
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "tweet_input": tweet,
        "prediction": label
    })
