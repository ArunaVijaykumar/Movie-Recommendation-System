#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[2]:


# Load data
data = pd.read_csv('movies_dataset2.csv')
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')
data['release_year'] = data['release_date'].dt.year.fillna(0).astype(int)


# In[3]:


# Feature extraction
data['genres'] = data['genre_ids'].apply(lambda x: ' '.join(map(str, eval(x))))
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['genres'])


# In[4]:


# Calculate similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Mapping titles to indices
indices = pd.Series(data.index, index=data['title'].str.lower()).to_dict()


# In[5]:


# Function to recommend movies
def recommend_movies(title, cosine_sim=cosine_sim):
    title = title.lower()
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 recommendations
    movie_indices = [i[0] for i in sim_scores]
    recommendations = data[['title', 'release_year', 'vote_average', 'poster_path']].iloc[movie_indices].drop_duplicates()
    recommendations['vote_average'] = recommendations['vote_average'].round(1)
    return recommendations


# In[6]:


import streamlit as st

# Set the layout to wide mode
st.set_page_config(layout="wide")

# Streamlit app
st.title('Movie Recommendation System')

# Add some CSS to style the input box and recommendations
st.markdown("""
    <style>
    .input-box {font-size: 18px;}
    .movie-box {
        border: 1px solid #ddd;
        padding: 20px; /* Increased padding */
        margin: 10px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        text-align: center;
        width: 200px; /* Fixed width for the movie box */
        height: 400px; /* Increased height for the movie box */
        overflow: hidden; /* Hide any overflow content */
    }
    .movie-box img {
        width: 100%; /* Ensure image fits the width of the box */
        height: auto; /* Maintain aspect ratio */
    }
    </style>
    """, unsafe_allow_html=True)

# Increase the size of the input box
movie_title = st.text_input('Enter a movie title', '', key='input_box', placeholder='Enter a movie title', help='Type a movie title to get recommendations')

if movie_title:
    recommendations = recommend_movies(movie_title)
    if not recommendations.empty:
        for i in range(0, len(recommendations), 5):
            cols = st.columns([1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1])  # Added spacing between columns
            for j in range(5):
                if i + j < len(recommendations):
                    col_index = j * 2
                    movie = recommendations.iloc[i + j]
                    with cols[col_index]:
                        st.markdown(f"""
                        <div class="movie-box">
                            <img src="https://image.tmdb.org/t/p/w500{movie['poster_path']}" />
                            <h4>{movie['title']} ({movie['release_year']})</h4>
                            <p>Rating: {movie['vote_average']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            st.write("")  # Empty row
    else:
        st.warning("Movie not found or no recommendations available.")

