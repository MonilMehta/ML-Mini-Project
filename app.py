import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# ======= Setup & Load Data =======
datasets_path = os.path.join(os.getcwd(), 'datasets')
data_path = os.path.join(datasets_path, 'data.csv')
data = pd.read_csv(data_path, on_bad_lines='skip', engine='python')

genre_path = os.path.join(datasets_path, 'data_by_genres.csv')
year_path = os.path.join(datasets_path, 'data_by_year.csv')
genre_data = pd.read_csv(genre_path, on_bad_lines='skip', engine='python') if os.path.exists(genre_path) else None
year_data = pd.read_csv(year_path, on_bad_lines='skip', engine='python') if os.path.exists(year_path) else None

column_mapping = {
    'name': 'name', 'title': 'name', 'track_name': 'name',
    'artist': 'artists', 'artists': 'artists',
    'spotify_id': 'id', 'id': 'id', 'img': 'image_url',
    'release_date': 'year'
}

for std_col, mapping_col in column_mapping.items():
    if std_col in data.columns and mapping_col not in data.columns:
        data[mapping_col] = data[std_col]

if 'name' not in data.columns:
    data['name'] = [f"Unknown Song {i}" for i in range(len(data))]

if 'id' not in data.columns:
    data['id'] = [f"song_{i}" for i in range(len(data))]

if 'artists' not in data.columns:
    data['artists'] = 'Unknown Artist'

if 'popularity' not in data.columns:
    data['popularity'] = np.random.randint(30, 90, size=len(data))

if 'album_name' not in data.columns:
    data['album_name'] = 'Unknown'

all_possible_features = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence',
    'acousticness_artist', 'danceability_artist', 'energy_artist', 
    'instrumentalness_artist', 'liveness_artist', 'speechiness_artist', 'valence_artist'
]
features = [f for f in all_possible_features if f in data.columns]

# ======= Improved Feature Engineering =======
# Optionally add more features if available (e.g., genre, mood, lyrics embeddings)
# For demonstration, add genre as one-hot if present
if 'genre' in data.columns:
    genre_dummies = pd.get_dummies(data['genre'], prefix='genre')
    data = pd.concat([data, genre_dummies], axis=1)
    features += list(genre_dummies.columns)

scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# ======= Dimensionality Reduction (PCA) =======
pca_components = min(10, len(features))
pca = PCA(n_components=pca_components, random_state=42)
data_pca = pca.fit_transform(data[features])

# ======= Improved Clustering =======
# You can tune n_clusters or try other algorithms (e.g., AgglomerativeClustering)
cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=15, verbose=False, random_state=42))
])
cluster_pipeline.fit(data_pca)
data['cluster_label'] = cluster_pipeline.predict(data_pca)

# ======= Improved Similarity: Nearest Neighbors =======
neigh = NearestNeighbors(n_neighbors=21, metric='cosine')
neigh.fit(data_pca)

# ======= Backend Functions =======

def find_similar_songs(song_name, n=10):
    name_to_index = {name: i for i, name in enumerate(data['name'])}
    song_idx = name_to_index.get(song_name)
    if song_idx is None:
        matches = [i for i, name in enumerate(data['name']) if song_name.lower() in name.lower()]
        if matches:
            song_idx = matches[0]
        else:
            return get_popular_songs(n)
    song_vec = data_pca[song_idx].reshape(1, -1)
    dists, indices = neigh.kneighbors(song_vec, n_neighbors=n+1)
    indices = indices.flatten()
    indices = indices[indices != song_idx][:n]
    similar_songs = data.iloc[indices]
    return build_recommendation_list(similar_songs)

def get_popular_songs(n=10):
    if 'popularity' in data.columns:
        sample = data.sort_values('popularity', ascending=False).head(100).sample(n=min(n, 100))
    else:
        sample = data.sample(min(n, len(data)))
    return build_recommendation_list(sample)

def get_content_based_recommendations(seed_tracks, limit=10):
    weights = [0.5, 0.2, 0.15, 0.1, 0.05]
    all_recommendations = []
    for i, track in enumerate(seed_tracks[:5]):
        for rec in find_similar_songs(track, n=20):
            rec['weight'] = weights[i]
            all_recommendations.append(rec)
    if not all_recommendations:
        return get_popular_songs(limit)
    combined = {}
    for rec in all_recommendations:
        title = rec['title']
        if title in combined:
            combined[title]['weight'] += rec['weight']
        else:
            combined[title] = rec
    final = list(combined.values())
    final.sort(key=lambda x: x['weight'], reverse=True)
    for rec in final:
        rec.pop('weight', None)
    return final[:limit]

def build_recommendation_list(df):
    recs = []
    for _, row in df.iterrows():
        image_url = row.get('image_url', None)
        recs.append({
            'title': row['name'],
            'artist': row['artists'],
            'album': row.get('album_name', 'Unknown'),
            'spotify_id': row['id'],
            'image_url': image_url
        })
    return recs

# ======= Streamlit UI =======

st.title("🎧 Music Recommendation System")
st.markdown("Search for songs or select from existing ones:")

song_query = st.text_input("Enter song name:")
seed_tracks = st.multiselect("Or select seed tracks:", options=list(data['name'].unique()))
limit = st.slider("How many recommendations?", 5, 20, 10)

if st.button("Get Recommendations"):
    if seed_tracks:
        recs = get_content_based_recommendations(seed_tracks, limit)
    elif song_query:
        recs = find_similar_songs(song_query, limit)
    else:
        recs = get_popular_songs(limit)

    st.subheader("Recommended Songs")
    for rec in recs:
        with st.container():
            cols = st.columns([1, 4])
            if rec['image_url']:
                cols[0].image(rec['image_url'], width=80)
            else:
                cols[0].markdown("🎵")
            link = f"https://open.spotify.com/track/{rec['spotify_id']}"
            card = f"""
            <div style=\"background-color:#181818;padding:10px;border-radius:10px;\">
                <span style=\"font-size:18px;font-weight:bold;color:#1DB954;\">{rec['title']}</span><br>
                <span style=\"color:#fff;\">{rec['artist']}</span><br>
                <span style=\"color:#b3b3b3;\">Album: {rec['album']}</span><br>
                <a href=\"{link}\" target=\"_blank\" style=\"color:#1DB954;\">Open in Spotify</a>
            </div>
            """
            cols[1].markdown(card, unsafe_allow_html=True)
