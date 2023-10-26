import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Assuming you have a model file named 'my_model.h5' in your current directory
model = keras.models.load_model('my_model.h5')

# Define the text feature columns
text_feature_columns = ['title']

# Load your trained tokenizer or create a new one and fit it on your training data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train['title'])  # Assuming X_train contains your training data

# Replace 'your_song_title' with the song title you want to predict BPM for
your_song_title = "Your song title"
input_data = [your_song_title]

# Tokenize and pad the input data
max_sequence_length = 20  # Use the same max sequence length as during training
input_data = pad_sequences(tokenizer.texts_to_sequences(input_data), maxlen=max_sequence_length)

# Make predictions
predicted_bpm = model.predict([input_data])
print(f'Predicted BPM: {predicted_bpm[0][0]}')