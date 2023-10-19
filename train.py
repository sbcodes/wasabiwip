import pandas as pd
from sklearn.model_selection import train_test_split
from river.feature_extraction.text import TFIDF
from river.regression import LinearRegression
from river.compose import Pipeline

# Load your data
df = pd.read_csv('wasabi_songs.csv', sep='\t')

# Remove rows with missing values in relevant columns
df = df[['artist', 'title', 'album_genre', 'bpm']].dropna()

# Split the data into features (X) and the target (y)
X = df[['artist', 'title', 'album_genre']]
y = df['bpm']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing and regression model using a River pipeline
text_feature_columns = ['artist', 'title', 'album_genre']

tfidf_vectorizers = {}
for column in text_feature_columns:
    tfidf_vectorizer = TFIDF()
    tfidf_vectorizers[column] = tfidf_vectorizer

model = LinearRegression()

# Create a River pipeline
pipeline = Pipeline()

for column in text_feature_columns:
    pipeline |= (column, tfidf_vectorizers[column])

pipeline |= ('model', model)

# Train the model
for xi, yi in zip(X_train.iterrows(), y_train):
    x = xi[1].to_dict()
    pipeline = pipeline.learn_one(x, yi)

# Evaluate the model on the test data
y_pred = [pipeline.predict_one(xi[1].to_dict()) for xi in X_test.iterrows()]
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')

