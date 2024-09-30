# TP : Systèmes de Recommandation Hybride

## Objectif du TP
L'objectif de ce TP est de comprendre et d'implémenter un système de recommandation hybride qui combine deux approches courantes : le filtrage collaboratif et le filtrage basé sur le contenu. Vous utiliserez les données du célèbre dataset MovieLens pour recommander des films à des utilisateurs en fonction de leurs préférences et de leur historique.

## Compétences acquises
- Manipulation de données avec Pandas.
- Utilisation des algorithmes de filtrage collaboratif et basé sur le contenu.
- Construction d’un profil utilisateur à partir de ses préférences.
- Recommandation d’articles (films) basés sur des similitudes.

## Pré-requis
- Connaissance de base de Python et Pandas.
- Familiarité avec les bibliothèques scikit-learn et Surprise.

## Étapes du TP

### 1. Chargement des données MovieLens
Dans cette étape, vous allez charger deux fichiers du dataset MovieLens 100k :
- `u.data` : contient les évaluations (ratings) des utilisateurs pour les films.
- `u.item` : contient les informations sur les films (titre, genres, etc.).

#### Exercice :
- Chargez les fichiers et créez un DataFrame `df_ratings` avec les colonnes `user_id`, `item_id`, `rating`, et `timestamp`.
- Créez un autre DataFrame `movies_df` qui contient les informations sur les films (`movie_id`, `title`, et `features`).

### 2. Filtrage collaboratif (Collaborative Filtering)
L’objectif est de recommander des films à un utilisateur basé sur les évaluations similaires des autres utilisateurs.

#### Exercice :
- Implémentez un modèle de filtrage collaboratif basé sur la similarité des utilisateurs (cosine similarity) à l'aide de la bibliothèque **Surprise**.
- Divisez les données en un ensemble d'entraînement et de test, puis entraînez le modèle.
- Calculez la performance du modèle en utilisant la **RMSE** et la **MAE**.

### 3. Filtrage basé sur le contenu (Content-Based Filtering)
Vous allez maintenant implémenter une méthode qui recommande des films basés sur les genres préférés d’un utilisateur.

#### Exercice :
- Transformez les genres des films en une matrice de **TF-IDF** (utilisez `TfidfVectorizer`).
- Demandez à l’utilisateur de sélectionner ses genres préférés.
- Utilisez ces préférences pour calculer les similarités entre les genres sélectionnés et les films disponibles.

### 4. Système hybride
Le système de recommandation hybride combine les deux approches :
- Recommandations collaboratives basées sur les notes des utilisateurs similaires.
- Recommandations basées sur le contenu qui correspondent aux préférences de l’utilisateur.

#### Exercice :
- Récupérez les films recommandés via le filtrage collaboratif pour un utilisateur donné.
- Filtrez ces films en fonction des préférences de genres de l’utilisateur (utilisez la similarité calculée à l'étape précédente).
- Affichez les 5 meilleurs films recommandés avec leurs titres et genres.

### 5. Personnalisation
Ajoutez une fonctionnalité qui permet à l’utilisateur de sélectionner dynamiquement ses genres préférés via une entrée.

#### Exercice :
- Permettez à l'utilisateur de spécifier ses genres préférés (par exemple : Action, Comedy, etc.).
- Adaptez les recommandations pour qu'elles correspondent à ces genres.
