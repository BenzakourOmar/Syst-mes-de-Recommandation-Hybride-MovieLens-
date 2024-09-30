import pandas as pd
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import KNNBasic
from surprise import Dataset, Reader

# Chargement du fichier u.data (évaluations)
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df_ratings = pd.read_csv('u.data', sep='\t', names=column_names)

# Affichage des premières lignes du DataFrame df_ratings
print(df_ratings.head())

# Chargement du fichier u.item (informations sur les films)
movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 
                 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

movies_df = pd.read_csv('u.item', sep='|', names=movie_columns, encoding='latin-1')

# Sélection des colonnes pertinentes : movie_id, title, et les genres
movies_df = movies_df[['movie_id', 'title', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                       'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                       'Sci-Fi', 'Thriller', 'War', 'Western']]

# Affichage des premières lignes du DataFrame movies_df
print(movies_df.head())

# Exercie 2

# Chargement des données dans un DataFrame Pandas
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df_ratings = pd.read_csv('u.data', sep='\t', names=column_names)

# Définir un Reader pour Surprise avec les colonnes user_id, item_id, et rating
reader = Reader(rating_scale=(1, 5))

# Charger les données dans Surprise
data = Dataset.load_from_df(df_ratings[['user_id', 'item_id', 'rating']], reader)

# Diviser les données en train/test
trainset, testset = train_test_split(data, test_size=0.25)

# Définir un modèle basé sur la similarité cosinus entre utilisateurs
sim_options = {
    'name': 'cosine',  # Utiliser la similarité cosinus
    'user_based': True  # Comparer les utilisateurs plutôt que les films
}

# Utiliser KNNBasic avec les options de similarité définies
algo = KNNBasic(sim_options=sim_options)

# Entraîner le modèle
algo.fit(trainset)

# Faire des prédictions sur l'ensemble de test
predictions = algo.test(testset)

# Calculer les métriques RMSE et MAE
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# Exercice 3

# Créer une nouvelle colonne "genres" en combinant les genres en une seule chaîne de texte
movies_df['genres'] = movies_df.iloc[:, 2:].apply(lambda x: ' '.join(x.index[x == 1]), axis=1)

# Affichage des premières lignes pour vérifier la colonne "genres"
print(movies_df[['movie_id', 'title', 'genres']].head())

# Initialiser le TfidfVectorizer pour les genres
tfidf = TfidfVectorizer()

# Appliquer la transformation TF-IDF sur la colonne des genres
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])

# Affichage de la taille de la matrice TF-IDF
print(f"Taille de la matrice TF-IDF : {tfidf_matrix.shape}")

# Genres possibles dans la base de données
available_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
                    'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                    'Sci-Fi', 'Thriller', 'War', 'Western']

# Demander à l'utilisateur ses genres préférés
user_genres = input(f"Veuillez entrer vos genres préférés séparés par une virgule parmi {available_genres} : ")
user_genres = user_genres.split(',')

# Créer une chaîne de texte représentant les préférences de l'utilisateur
user_genre_str = ' '.join(user_genres)

# Créer un vecteur TF-IDF pour les préférences de l'utilisateur
user_tfidf = tfidf.transform([user_genre_str])

# Calculer les similarités cosinus entre les préférences de l'utilisateur et les films
cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

# Recommander les 10 films les plus similaires
similar_movies_indices = cosine_similarities.argsort()[-10:][::-1]

# Afficher les recommandations de films
recommended_movies = movies_df.iloc[similar_movies_indices][['movie_id', 'title', 'genres']]
print("Films recommandés :")
print(recommended_movies)

# Exercice 4

from surprise import KNNBasic
from surprise import Dataset, Reader

# Recharger les données et recréer le modèle de filtrage collaboratif
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_ratings[['user_id', 'item_id', 'rating']], reader)

trainset = data.build_full_trainset()

sim_options = {
    'name': 'cosine', 
    'user_based': True
}

# Modèle de filtrage collaboratif
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

# Prendre un utilisateur spécifique, par exemple user_id = 196
user_id = 196

# Obtenir tous les films que l'utilisateur n'a pas encore notés
user_rated_items = set([i for (i, _) in trainset.ur[trainset.to_inner_uid(user_id)]])
all_items = set(trainset.all_items())
unrated_items = all_items - user_rated_items

# Prédire les notes pour tous les films non notés par l'utilisateur
predictions = [algo.predict(user_id, trainset.to_raw_iid(i)) for i in unrated_items]

# Trier les films par la note prédite et prendre les 10 meilleurs
top_collab_recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]

# Obtenir les IDs des films recommandés
top_collab_movie_ids = [int(pred.iid) for pred in top_collab_recommendations]

# Afficher les titres et genres des films recommandés par le modèle collaboratif
recommended_movies_collab = movies_df[movies_df['movie_id'].isin(top_collab_movie_ids)]
print("Recommandations collaboratives :")
print(recommended_movies_collab[['movie_id', 'title', 'genres']])

# Utiliser les genres préférés de l'utilisateur (exemple de genres entrés par l'utilisateur)
user_genres = ['Action', 'Adventure', 'Sci-Fi']
user_genre_str = ' '.join(user_genres)

# Créer un vecteur TF-IDF pour les genres préférés de l'utilisateur
user_tfidf = tfidf.transform([user_genre_str])

# Filtrer les recommandations collaboratives basées sur la similarité de contenu
collab_movie_indices = movies_df[movies_df['movie_id'].isin(top_collab_movie_ids)].index
cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix[collab_movie_indices]).flatten()

# Trier les films collaboratifs par similarité de contenu
best_movies_indices = cosine_similarities.argsort()[-5:][::-1]

# Sélectionner les 5 meilleurs films en fonction des similarités
best_movies = recommended_movies_collab.iloc[best_movies_indices]
print("Meilleurs films recommandés avec filtrage hybride :")
print(best_movies[['movie_id', 'title', 'genres']])

# Exercice 5

# Genres disponibles dans la base de données MovieLens
available_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 
                    'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                    'Sci-Fi', 'Thriller', 'War', 'Western']

# Demander à l'utilisateur de spécifier ses genres préférés
print(f"Genres disponibles : {available_genres}")
user_input = input("Veuillez entrer vos genres préférés, séparés par des virgules : ")

# Traiter l'entrée utilisateur
user_genres = [genre.strip() for genre in user_input.split(',') if genre.strip() in available_genres]

if not user_genres:
    print("Aucun genre valide sélectionné, veuillez réessayer.")
else:
    print(f"Genres sélectionnés : {user_genres}")

from sklearn.metrics.pairwise import cosine_similarity

# Transformer les genres préférés de l'utilisateur en une chaîne de caractères
user_genre_str = ' '.join(user_genres)

# Créer un vecteur TF-IDF pour les préférences de l'utilisateur
user_tfidf = tfidf.transform([user_genre_str])

# Recalculer les similarités cosinus pour les films recommandés par le modèle collaboratif
collab_movie_indices = movies_df[movies_df['movie_id'].isin(top_collab_movie_ids)].index
cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix[collab_movie_indices]).flatten()

# Trier les films collaboratifs par similarité de contenu en fonction des genres préférés
best_movies_indices = cosine_similarities.argsort()[-5:][::-1]

# Sélectionner les 5 meilleurs films en fonction des similarités
best_movies = recommended_movies_collab.iloc[best_movies_indices]
print("Films recommandés avec personnalisation des genres :")
print(best_movies[['movie_id', 'title', 'genres']])
