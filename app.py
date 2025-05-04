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

# Add 'year' column if 'release_date' exists and can be converted
if 'release_date' in data.columns and 'year' not in data.columns:
    try:
        # Attempt to extract year, handling potential errors
        data['year'] = pd.to_datetime(data['release_date'], errors='coerce').dt.year
        # Drop rows where year extraction failed
        data.dropna(subset=['year'], inplace=True)
        data['year'] = data['year'].astype(int)
    except Exception as e:
        print(f"Could not process release_date into year: {e}")

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
    'instrumentalness_artist', 'liveness_artist', 'speechiness_artist', 'valence_artist',
    'year' # Add year here
]
features = [f for f in all_possible_features if f in data.columns and pd.api.types.is_numeric_dtype(data[f])]

# Ensure features list is not empty
if not features:
    st.error("Error: No valid numeric features found for recommendation.")
    st.stop()

# Drop rows with NaN in selected features before scaling
data.dropna(subset=features, inplace=True)

# Check if data is empty after dropping NaNs
if data.empty:
    st.error("Error: Data became empty after removing rows with missing feature values.")
    st.stop()

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
pca_components = min(10, len(features)) # Ensure components <= number of features
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

# ======= Load Music.csv for Images =======
music_csv_path = os.path.join(datasets_path, 'Music.csv')
if os.path.exists(music_csv_path):
    music_df = pd.read_csv(music_csv_path, on_bad_lines='skip', engine='python')
    # Standardize column names
    music_df.columns = [c.strip().lower().replace(' ', '_') for c in music_df.columns]
    # Use 'song' as the name column if present
    name_col = 'song' if 'song' in music_df.columns else 'name'
    # Build mapping from name and spotify_id to image_url
    name_to_img = {str(row[name_col]).strip().lower(): row['img'] for _, row in music_df.iterrows() if 'img' in row and pd.notnull(row['img'])}
    id_to_img = {str(row['spotify_id']).strip(): row['img'] for _, row in music_df.iterrows() if 'img' in row and pd.notnull(row['img']) and 'spotify_id' in row and pd.notnull(row['spotify_id'])}
else:
    name_to_img = {}
    id_to_img = {}

# ======= Backend Functions =======

def find_similar_songs(song_name, n=10):
    name_to_index = {name: i for i, name in enumerate(data['name'])}
    song_idx = name_to_index.get(song_name)
    if song_idx is None:
        matches = [i for i, name in enumerate(data['name']) if song_name.lower() in name.lower()]
        if matches:
            song_idx = matches[0]
        else:
            return get_popular_songs(n) # Fallback if song not found

    song_row = data.iloc[song_idx]
    song_artist = song_row['artists']
    song_is_hindi = is_hindi_song(song_row)
    song_vec = data_pca[song_idx].reshape(1, -1)

    # Fetch more neighbors initially to allow for re-ranking
    num_neighbors_to_fetch = max(n * 5, 50) # Fetch more candidates
    if num_neighbors_to_fetch >= len(data_pca):
        num_neighbors_to_fetch = len(data_pca) -1 # Avoid requesting more neighbors than available data points

    if num_neighbors_to_fetch <= 0:
         return get_popular_songs(n) # Fallback if not enough data

    dists, indices = neigh.kneighbors(song_vec, n_neighbors=num_neighbors_to_fetch)

    # Flatten and remove self
    indices = indices.flatten()
    dists = dists.flatten()
    mask = indices != song_idx
    indices = indices[mask]
    dists = dists[mask]

    # Get candidate songs DataFrame
    if len(indices) == 0:
        return get_popular_songs(n) # Fallback if no neighbors found

    similar_songs_df = data.iloc[indices].copy()
    similar_songs_df['distance'] = dists

    # --- Re-ranking based on Artist and Genre ---
    artist_weight_bonus = 1.0 # Higher bonus for same artist
    genre_match_bonus = 0.5 # Bonus for matching hindi status

    bonuses = []
    for index, row in similar_songs_df.iterrows():
        bonus = 0
        # Artist bonus
        if 'artists' in row and row['artists'] == song_artist:
            bonus += artist_weight_bonus
        # Genre bonus (matching 'Hindiness')
        if is_hindi_song(row) == song_is_hindi:
             bonus += genre_match_bonus
        # Potential future enhancement: Add more specific genre matching here
        # if 'genre' in row and 'genre' in song_row and row['genre'] == song_row['genre']: bonus += some_value
        bonuses.append(bonus)

    similar_songs_df['bonus'] = bonuses

    # Sort by bonus (descending) then distance (ascending)
    ranked_songs = similar_songs_df.sort_values(by=['bonus', 'distance'], ascending=[False, True])

    # --- Apply Hindi Filter if necessary ---
    if song_is_hindi:
        # Prioritize Hindi songs from the ranked list
        final_similar_songs = ranked_songs[ranked_songs.apply(is_hindi_song, axis=1)].head(n)
        # If not enough Hindi songs, supplement with top non-Hindi from ranked list
        if len(final_similar_songs) < n:
            needed = n - len(final_similar_songs)
            non_hindi_ranked = ranked_songs[~ranked_songs.index.isin(final_similar_songs.index)].head(needed)
            final_similar_songs = pd.concat([final_similar_songs, non_hindi_ranked])
    else:
        # If input song is not Hindi, just take top N from the bonus/distance ranked list
        final_similar_songs = ranked_songs.head(n)

    # Ensure we don't exceed n results
    final_similar_songs = final_similar_songs.head(n)

    return build_recommendation_list(final_similar_songs)

def get_popular_songs(n=10):
    if 'popularity' in data.columns:
        sample = data.sort_values('popularity', ascending=False).head(100)
    else:
        sample = data.sample(min(n, len(data)))
    # Filter for Hindi if most popular is Hindi
    if 'genre' in data.columns and len(sample) > 0:
        hindi_sample = sample[sample.apply(is_hindi_song, axis=1)]
        if len(hindi_sample) >= n:
            sample = hindi_sample
    sample = sample.sample(n=min(n, len(sample)))
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
        # Try to get image from Music.csv mapping
        image_url = row.get('image_url', None)
        if not image_url:
            # Try by id
            image_url = id_to_img.get(str(row.get('id', '')).strip(), None)
        if not image_url:
            # Try by name (case-insensitive)
            image_url = name_to_img.get(str(row.get('name', '')).strip().lower(), None)
        recs.append({
            'title': row['name'],
            'artist': row['artists'],
            'album': row.get('album_name', 'Unknown'),
            'spotify_id': row['id'],
            'image_url': image_url
        })
    return recs

# Helper: Detect if a song is Hindi/Indian
HINDI_GENRES = ['hindi', 'bollywood', 'indian']
def is_hindi_song(row):
    genre = str(row.get('genre', '')).lower()
    name = str(row.get('name', '')).lower()
    artist = str(row.get('artists', '')).lower()
    return any(g in genre or g in name or g in artist for g in HINDI_GENRES)

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
            img_url = rec['image_url']
            if img_url and isinstance(img_url, str) and img_url.startswith('http') and img_url.lower() != 'no':
                cols[0].image(img_url, width=80)
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
