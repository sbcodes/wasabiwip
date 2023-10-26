import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Dense, Flatten, Input, LSTM, Dropout
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import joblib

def load_songs(filename, nrows=None):
    return pd.read_csv(filename, sep='\t', nrows=nrows)

# Load a subset of the data
df = load_songs('wasabi_songs.csv', nrows=50000)
print('songs loaded')

# Remove rows with missing values
clean_df = df[['artist', 'title', 'album_genre', 'bpm']].dropna()
clean_df.info()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('data cleaned')

# Split the data into features (X) and the target (y)
X = clean_df.drop(columns=['bpm'])  # Features
y = clean_df['bpm']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
joblib.dump(X_train, "train.pkl")
print('test data split')

# Define the text feature columns
text_feature_columns = ['artist', 'title', 'album_genre']

# Tokenize and pad the text data with the adjusted parameters
tokenizer = Tokenizer()
max_sequence_length = 40  # Adjusted sequence length
vocab_size = 15000  # Adjusted vocabulary size
embedding_dim = 50  # Adjusted embedding dimension

input_layers = []
embedding_layers = []

# Create a dictionary to hold the embedding layers for each text feature
embedding_dict = {}

for column in text_feature_columns:
    X_train[column] = X_train[column].astype(str)
    X_test[column] = X_test[column].astype(str)
    tokenizer.fit_on_texts(X_train[column])
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train[column]), maxlen=max_sequence_length)
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test[column]), maxlen=max_sequence_length)
    
    # Create input layers for each text feature
    input_layer = Input(shape=(max_sequence_length,), name=column)
    input_layers.append(input_layer)

    # Create embedding layers for each text feature
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(input_layer)
    embedding_layer = Flatten()(embedding_layer)
    
    # Store the embedding layer in the dictionary
    embedding_dict[column] = embedding_layer

# Concatenate the embedding layers
merged = keras.layers.concatenate(list(embedding_dict.values()))

# Define a Keras model with more complexity
x = Dense(128, activation='relu')(merged)
x = Dropout(0.5)(x)  # Use Dropout
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='linear')(x)

model = Model(inputs=input_layers, outputs=output)

# Compile and fit the model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit([X_train_seq for _ in text_feature_columns], y_train, epochs=80, batch_size=2, verbose=1)

# Evaluate the model on the test data
mse = model.evaluate([X_test_seq for _ in text_feature_columns], y_test)
print(f'Mean Squared Error on Test Data: {mse}')

model.save('initial_model_.keras')