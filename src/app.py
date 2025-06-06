from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

data = pd.read_csv(r"g:\gowthamy_pavanathn_saiakash_phase_2\gowthamy_pavanathn_saiakash_phase_2\preprocessed_data.csv")

df = pd.DataFrame(data)

# Preprocessing
df['genre'] = df['genre'].fillna('')
df['description'] = df['description'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genre'] + ' ' + df['description'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_movies(title, cosine_sim=cosine_sim):
    movie_indices = []
    if title in df['title'].values:
        idx = df[df['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
    return list(df['title'].iloc[movie_indices])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie_title']
    recommendations = recommend_movies(movie_title)
    return render_template('recommendations.html', movie_title=movie_title, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
