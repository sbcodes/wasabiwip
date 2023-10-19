import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gensim
from gensim.corpora import Dictionary
import joblib
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.layers import Embedding, LSTM, Dense, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

SOME_FREE_SONGTEXT = "So one night I said to Kathy We gotta get away somehow Go somewhere south and somewhere warm But for God's sake let's go now. And Kathy she sort of looks at me And asks where I wanna go So I look back and I hear me say I don't care but we gotta go chorus and key change And all the other people Who slepwalk thru their days Just sort of faded out of sight When we two drove away And ev'ry day we travelled We were lookin' to get wise And we learned what was the truth And we learned what were the lies And in LA we bought a bus Sort of old and not too smart So for six hundred and fifty bucks We got out and made a start We hit the road down to the South And drove into Mexico That old bus was some old wreck But it just kept us on the road. chorus etc We drove up to Alabam And a farmer gave us some jobs We worked them crops all night and day And at night we slept like dogs We got paid and Kathy said to me It's time to make a move again And when I looked into her eyes I saw more than a friend. chorus etc And now we've stopped our travels And we sold the bus in Texas And we made our home in Austin And for sure it ain't no palace And Kathy and me we settled down And now our first kid's on the way Kathy and me and that old bus We did real good to get away."

def load_songs(filename, nrows=None):
        return pd.read_csv(filename, sep='\t', nrows=nrows)


def load_artists(filename, nrows=None):
        return pd.read_csv(filename, sep=',', nrows=nrows)

def load_albums(filename, nrows=None):
        return pd.read_csv(filename, sep='\t', nrows=nrows)

def random_key_from_dict(dictionary, seed=123):
        keys_list = list(dictionary.keys())
        np.random.seed(seed)
        random_index = np.random.choice(len(keys_list))
        return keys_list[random_index]


### Quick overview over some columns
df = load_songs('wasabi_songs.csv', nrows=100000)
for property in ['artist', 'title', 'publicationDateAlbum', 'album_genre',
                 'valence', 'arousal', 'valence_predicted', 'arousal_predicted', 'bpm']:
    print('Histogram for:', property)
    print(pd.value_counts(df[property]), '\n\n')
    

# remove rows that either have NaN in relevant feature or bpm cell(s)
clean_df = df[['artist', 'title', 'publicationDateAlbum', 'album_genre','bpm']].dropna()
clean_df.info()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Assuming df is your DataFrame
# Splitting the data into features (X) and the target (y)
X = clean_df.drop(columns=['bpm'])  # Features
y = clean_df['bpm']  # Target


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the text feature columns
text_feature_columns = ['artist', 'title', 'album_genre']

# Tokenize and pad the text data
tokenizer = Tokenizer()
max_sequence_length = 50  # Define a suitable maximum sequence length
vocab_size = 10000  # Define the vocabulary size
embedding_dim = 100  # Define the embedding dimension

for column in text_feature_columns:
    X_train[column] = X_train[column].astype(str)
    X_test[column] = X_test[column].astype(str)
    tokenizer.fit_on_texts(X_train[column])
    X_train[column] = pad_sequences(tokenizer.texts_to_sequences(X_train[column]), maxlen=max_sequence_length)
    X_test[column] = pad_sequences(tokenizer.texts_to_sequences(X_test[column]), maxlen=max_sequence_length)

# Create and train word embeddings (Word2Vec, GloVe, etc.) for your text features
# You would need to download or train these embeddings using an NLP library

# Define a Keras model that incorporates embedding layers for text features
model = Sequential()

for column in text_feature_columns:
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
    model.add(Flatten())  # Flatten the embedding output

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # Linear activation for regression

# Compile the model with an appropriate loss function for regression
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
model.fit([X_train[column] for column in text_feature_columns], y_train, epochs=100, batch_size=2, verbose=1)

# Evaluate the model on the test data
mse = model.evaluate([X_test[column] for column in text_feature_columns], y_test)
print(f'Mean Squared Error on Test Data: {mse}')

# You can use this model to make predictions on new data
