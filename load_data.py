import os
import pandas as pd


def load_data(directory, algorithm, given_user_id):     #load data
    data = {}
    movie_data = {}

    if algorithm == 'user_user_algorithm':      #data for user_user_algorithm
        file_path = os.path.join(directory, 'ratings.csv')
        ratings = pd.read_csv(file_path)

        for index, row in ratings.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            rating = row['rating']

            if user_id not in data:
                data[user_id] = {}

            data[user_id][item_id] = rating

    elif algorithm == 'item_item_algorithm':        #data for item_item_algorithm
        given_user_movies = []

        file_path = os.path.join(directory, 'ratings.csv')
        ratings = pd.read_csv(file_path)

        for index, row in ratings.iterrows():
            user_id = row['userId']
            item_id = row['movieId']
            rating = row['rating']

            if item_id not in data:
                data[item_id] = {}

            data[item_id][user_id] = rating

            if user_id == given_user_id:
                given_user_movies.append(item_id)

        file_path = os.path.join(directory, 'movies.csv') 
        movies = pd.read_csv(file_path)

        for index, row in movies.iterrows():
            movie_id = row['movieId']
            title = row['title']
            movie_data[movie_id] = title

        return data, movie_data, given_user_movies

    elif algorithm == 'tag_based_algorithm':        #data for item_item_algorithm
        file_path = os.path.join(directory, 'tags.csv')
        tags_data = pd.read_csv(file_path)

        for index, row in tags_data.iterrows():
            movie_id = row['movieId']
            tag = row['tag']

            if movie_id not in data:
                data[movie_id] = []

            data[movie_id].append(tag)


    file_path = os.path.join(directory, 'movies.csv')     #load movie csv for every algorithm including content_based_algorithm
    movies = pd.read_csv(file_path)

    for index, row in movies.iterrows():
        movie_id = row['movieId']
        title = row['title']
        movie_data[movie_id] = title

    return data, movie_data