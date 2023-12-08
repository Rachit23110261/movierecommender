from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Sample Movie Dataset
data = {
    'Title': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4'],
    'Genre': ['Action', 'Drama', 'Comedy', 'Action'],
    'Review': [
        'Exciting action sequences. Great movie!',
        'Touching drama with brilliant performances.',
        'Hilarious comedy, a must-watch!',
        'Action-packed with a thrilling plot.'
    ]
}

movies_df = pd.DataFrame(data)

# Text Processing and Feature Extraction
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['Review'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_input']

    # Transform user input using the same vectorizer
    user_tfidf = tfidf_vectorizer.transform([user_input])

    # Calculate Cosine Similarity
    cosine_similarities = linear_kernel(user_tfidf, tfidf_matrix).flatten()

    # Get movie recommendations based on similarity scores
    recommendations = movies_df.iloc[cosine_similarities.argsort()[::-1]]

    return render_template('recommendations.html', recommendations=recommendations[['Title', 'Genre', 'Review']])

if __name__ == '__main__':
    app.run(debug=True)
