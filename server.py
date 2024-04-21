from flask import Flask, jsonify, request
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Đọc dữ liệu từ tệp CSV 'movies.csv'
movies_data = pd.read_csv('movies.csv')

# Xử lý dữ liệu trống
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Kết hợp các đặc trưng quan trọng thành một đặc trưng kết hợp
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Tạo vector đặc trưng từ dữ liệu kết hợp sử dụng TF-IDF
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Tính toán ma trận tương đồng cosine
similarity = cosine_similarity(feature_vectors)

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    # Nhận dữ liệu đầu vào từ yêu cầu POST
    data = request.get_json()

    # Nhận tên phim yêu thích từ dữ liệu đầu vào
    movie_name = data.get('movie_name', '')

    # Tìm kiếm phim tương tự
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0] if find_close_match else None

    if close_match:
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)[:10]

        recommended_movies = []
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            recommended_movies.append({'title': title_from_index})

        return jsonify({'status': 'success', 'recommended_movies': recommended_movies})
    else:
        return jsonify({'status': 'error', 'message': 'Movie not found'})

if __name__ == '__main__':
    app.run(debug=True)
