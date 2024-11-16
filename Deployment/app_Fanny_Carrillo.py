from fastapi import FastAPI, Form
from starlette.responses import HTMLResponse
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import string
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import torch
from transformers import BertTokenizer, BertForSequenceClassification



nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')


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
    cleaned_text = re.sub(r"\bu\b", "you", cleaned_text, flags=re.IGNORECASE)

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


def process_input_sentence(input_sentence):
    processed_sentence = process_tweet_content(input_sentence)
    processed_sentence = process_tweet_content(processed_sentence)
    #print(processed_sentence)
    processed_sentence = remove_specific_words(processed_sentence)
    #print(processed_sentence)
    tokenized_sentence = tokenize_corpus([processed_sentence])
    #print(tokenized_sentence)
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=max_length, padding='post')
    #print(padded_sentence)
    return padded_sentence

def predict_with_bert(sentence):
    """Realiza la predicción usando el modelo BERT."""
    sentence = process_tweet_content(sentence)
    sentence = remove_specific_words(sentence)
    inputs = tokenizer_bert(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model_bert(**inputs)
        logits = outputs.logits

    predicted_class = torch.argmax(logits, dim=1).item()
    probability = torch.softmax(logits, dim=1)[0][predicted_class].item()
    return "REAL" if predicted_class == 1 else "FAKE", probability


app = FastAPI()

# Cargar el modelo LSTM previamente entrenado
model_lstm = tf.keras.models.load_model('model_LSTM23_Final.h5')

# Cargar el modelo y el tokenizador BERT
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_bert.load_state_dict(torch.load('bert_finetuned_model.pth', map_location=torch.device('cpu')))
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')


# Cargar el dataset de pruebas
train_dataset = pd.read_csv("train.csv", encoding="latin-1")
test_dataset = pd.read_csv("test.csv", encoding="latin-1")

stop_words_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()  # <-- Usamos lemmatizer en lugar de stemmer



extra_stop_words = ['u', 'im', 'r']
extra_stop_words2 = [
    'u', 'im', 'r', 'ur', 'pls', 'thx',
    'b4', 'omw', 'ppl', 'msg', 'lvl',
    'sos', '911', 'help', 'asap',
    'wtf', 'omg', 'idk', 'nvm',
    'brb', 'btw', 'lmk', 'imo',
    'stay', 'safe', 'evacuate', 'fyi'
]

all_stop_words = stop_words_list + extra_stop_words + extra_stop_words2
word_stemmer = nltk.SnowballStemmer("english")


train_dataset['text_clean'] = train_dataset['text'].apply(process_tweet_content)
test_dataset['text_clean'] = test_dataset['text'].apply(process_tweet_content)

custom_words = {'im', 'u','â', 'ã', 'one', 'ã ã', 'ã ã', 'ã âª', 'ã â', 'â ã', 'â', 'âª','â', 'ã','aaa', 'rt','aa', 'ye'}

train_dataset['text_clean'] = train_dataset['text_clean'].apply(remove_specific_words)
test_dataset['text_clean'] = test_dataset['text_clean'].apply(remove_specific_words)


train_tweets = train_dataset['text_clean'].values
test_tweets = test_dataset['text_clean'].values

tweet_tokenizer = Tokenizer()
tweet_tokenizer.fit_on_texts(train_tweets)
vocabulary_size = len(tweet_tokenizer.word_index) + 1

max_length = 23
test_padded_sentences = pad_sequences(tokenize_corpus(test_tweets),maxlen=max_length,padding='post')


model = tf.keras.models.load_model('model_LSTM23_Final.h5')

pred_probs2 = model.predict(test_padded_sentences)
preds2 = (pred_probs2 >= 0.6).astype(int)  # Umbral en 0.5

# Contar los valores de 'REAL' y 'FAKE'
real_count = (preds2 == 1).sum()
fake_count = (preds2 == 0).sum()

# Imprimir los resultados
#print(f"REAL Tweets: {real_count}")
#print(f"FAKE Tweets: {fake_count}")


results_df = pd.DataFrame({ 'Tweet': test_dataset['text'].values,'Prediction': preds2.flatten(),  # Asegurar que la predicción sea unidimensional
'Label': ['REAL' if pred == 1 else 'FAKE' for pred in preds2.flatten()]
})

#print(results_df)#Esta no quiero que se muestre.


#Primera tabla con 10 resultados del test dataset para mostrar, real tweet
real_tweets = results_df[results_df['Label'] == 'REAL'].head(10)
print(real_tweets)


#Segunda tabla con 10 resultados del test dataset para mostrar, fake tweet
fake_tweets = results_df[results_df['Label'] == 'FAKE'].tail(10)
print(fake_tweets)



# Generar tablas de tweets reales y falsos
@app.get("/", response_class=HTMLResponse)
async def main_page():
    # Datos de tweets reales y falsos
    real_tweets = results_df[results_df['Label'] == 'REAL'].head(10).to_html(index=False)
    fake_tweets = results_df[results_df['Label'] == 'FAKE'].tail(10).to_html(index=False)
    
    return f'''
    <html>
        <head>
            <title>Disaster Tweets</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f0f2f5; }}
                h1, h2 {{ text-align: center; color: #333; }}
                .content {{ max-width: 800px; margin: auto; padding: 20px; background: white; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                table, th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #2BE060; color: white;  text-align: center; }}
                input[type="text"] {{ width: 100%; padding: 10px; margin: 8px 0; border: 1px solid #ccc; border-radius: 4px; }}
                button {{ background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }}
                button:hover {{ background-color: #0056b3; }}
                td:first-child {{ text-align: left; }}
                td:nth-child(2), #tweetTable td:nth-child(3) {{ text-align: center; }}
            </style>
        </head>
        <body>
            <div class="content">
                <h1>Disaster Tweets</h1>
                
                <h2>Real Disaster Tweets</h2>
                {real_tweets}
                
                <h2>Fake Disaster Tweets</h2>
                {fake_tweets}
                
                <h2>Enter a tweet to predict if it is REAL or FAKE</h2>
                <form action="/predict" method="post">
                    <input type="text" name="text" placeholder="Enter the text of the tweet" maxlength="280" required/>
                    <div style="text-align: center;">
                        <button type="submit" name="model_type" value="lstm">Predict LSTM</button>
                        <button type="submit" name="model_type" value="bert">Predict BERT</button>
                    </div>
                </form>
            </div>
        </body>
    </html>
    '''

@app.post('/predict', response_class=HTMLResponse)
async def predict(text: str = Form(...), model_type: str = Form(...)):
    if model_type == "lstm":
        processed_text = process_input_sentence(text)
        prediction = model_lstm.predict(processed_text)
        prediction_label = "REAL" if prediction >= 0.6 else "FAKE"
        probability = prediction[0][0] if prediction >= 0.6 else 1 - prediction[0][0]
    else:
        prediction_label, probability = predict_with_bert(text)
    
    return f"""
    <html>
        <head>
            <title>Prediction Result</title>
            <style>
                /* CSS de estilo general */
            </style>
        </head>
        <body>
            <div class="content">
                <h2>Prediction Result</h2>
                <p><strong>Tweet:</strong> {text}</p>
                <p><strong>Prediction:</strong> {prediction_label} (Probability: {probability:.2f})</p>
                <form action="/" method="get">
                    <button type="submit">Back</button>
                </form>
            </div>
        </body>
    </html>
    """

