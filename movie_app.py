import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
net_mov = pd.read_csv("cleaned_data.csv")


# Clean the data
def clean_data(x):
    return str.lower(x.replace(",", " "))


net_mov['description'] = net_mov['description'].apply(clean_data)
net_mov['director'] = net_mov['director'].apply(clean_data)
net_mov['cast'] = net_mov['cast'].apply(clean_data)
net_mov['Genre'] = net_mov['Genre'].apply(clean_data)


# Create a new feature 'soup' combining relevant features
def create_soup(x):
    return x['director'] + ' ' + x['cast'] + ' ' + x['Genre'] + ' ' + x['description']


net_mov['soup'] = net_mov.apply(create_soup, axis=1)

# Create CountVectorizer and fit_transform the 'soup' feature
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(net_mov['soup'])

# Compute the Cosine Similarity matrix based on the count_matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Construct reverse mapping
indices = pd.Series(net_mov.index, index=net_mov['title'])


# Recommendation function
def get_recommendations_new(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return net_mov['title'].iloc[movie_indices]


# Streamlit app
def main():
    st.title('Movie Recommendation System')

    st.sidebar.title('Enter Movie Title')
    user_input = st.sidebar.text_input('Input the movie title:', 'My Little Pony: A New Generation')

    if st.sidebar.button('Generate Recommendations'):
        recommendations = get_recommendations_new(user_input)
        st.subheader('Recommended Movies:')
        for movie in recommendations:
            st.write(movie)


if __name__ == '__main__':
    main()
