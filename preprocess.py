import os
import pandas as pd
import numpy as np
import sys
import csv


from recommender_functions import *
from load_data import *

def write_recommendations_csv(csv_file_path, recommendations):
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['user_id', 'movies'])
        
        for user_id, movies in recommendations.items():
            csv_writer.writerow([user_id, ', '.join(movies)])


def main():
    #load data
    item_data, movie_data, given_user_movies = load_data('ml-latest-small', 'item_item_algorithm', 1)
    user_data, movie_data = load_data('ml-latest-small', 'user_user_algorithm', 0)
    tags, movie_data = load_data('ml-latest-small', 'tag_based_algorithm', 0)

    user_user_recommendations_jaccard = {}
    user_user_recommendations_dice = {}
    user_user_recommendations_cosine = {}
    user_user_recommendations_pearson = {}
    
    # for user_id, movie_id in user_data.items():
    #     #user-user algorithm
    #     recommendations_jaccard, score = user_user_algorithm(user_data, movie_data, user_id, 100, 'jaccard')
    #     user_user_recommendations_jaccard[user_id] = recommendations_jaccard

    #     recommendations_dice, score = user_user_algorithm(user_data, movie_data, user_id, 100, 'dice')
    #     user_user_recommendations_dice[user_id] = recommendations_dice

    #     recommendations_cosine, score = user_user_algorithm(user_data, movie_data, user_id, 100, 'cosine')
    #     user_user_recommendations_cosine[user_id] = recommendations_cosine

    #     recommendations_pearson, score = user_user_algorithm(user_data, movie_data, user_id, 100, 'pearson')
    #     user_user_recommendations_pearson[user_id] = recommendations_pearson
        
    # write_recommendations_csv('user_user_recommendations_jaccard.csv', user_user_recommendations_jaccard)
    # write_recommendations_csv('user_user_recommendations_dice.csv', user_user_recommendations_dice)
    # write_recommendations_csv('user_user_recommendations_cosine.csv', user_user_recommendations_cosine)
    # write_recommendations_csv('user_user_recommendations_pearson.csv', user_user_recommendations_pearson)
    


    # item_item_recommendations_jaccard = {}
    # item_item_recommendations_dice = {}
    # item_item_recommendations_cosine = {}
    # item_item_recommendations_pearson = {}

    # for user_id, movie in user_data.items():
    #     #item-item algorithm
    #     given_user_movies = list(user_data.get(user_id, {}).keys())
    #     recommendations_jaccard, score = item_item_algorithm(item_data, movie_data, given_user_movies, user_id, 100, 'jaccard')
    #     item_item_recommendations_jaccard[user_id] = recommendations_jaccard

        # recommendations_dice, score = item_item_algorithm(item_data, movie_data, given_user_movies, user_id, 100, 'dice')
        # item_item_recommendations_dice[user_id] = recommendations_dice

        # recommendations_cosine, score = item_item_algorithm(item_data, movie_data, given_user_movies, user_id, 100, 'cosine')
        # item_item_recommendations_cosine[user_id] = recommendations_cosine

        # recommendations_pearson, score = item_item_algorithm(item_data, movie_data, given_user_movies, user_id, 100, 'pearson')
        # item_item_recommendations_pearson[user_id] = recommendations_pearson
        


    content_based_recommendations_jaccard = {}
    content_based_recommendations_dice = {}
    content_based_recommendations_cosine = {}
    content_based_recommendations_pearson = {}

    for movie_id, title in movie_data.items():
        #content-based algorithm
        recommendations_jaccard, score = content_based_algorithm(movie_data, movie_id, 100, 'jaccard')
        content_based_recommendations_jaccard[movie_id] = recommendations_jaccard

        recommendations_dice, score = content_based_algorithm(movie_data, movie_id, 100, 'dice')
        content_based_recommendations_dice[movie_id] = recommendations_dice

        recommendations_cosine, score = content_based_algorithm(movie_data, movie_id, 100, 'cosine')
        content_based_recommendations_cosine[movie_id] = recommendations_cosine

        recommendations_pearson, score = content_based_algorithm(movie_data, movie_id, 100, 'pearson')
        content_based_recommendations_pearson[movie_id] = recommendations_pearson

    write_recommendations_csv('content_based_recommendations_jaccard.csv', content_based_recommendations_jaccard)
    write_recommendations_csv('content_based_recommendations_dice.csv', content_based_recommendations_dice)
    write_recommendations_csv('content_based_recommendations_cosine.csv', content_based_recommendations_cosine)
    write_recommendations_csv('content_based_recommendations_pearson.csv', content_based_recommendations_pearson)
    




    # tag_based_recommendations_jaccard = {}
    # tag_recommendations_dice = {}

    # for movie_id, tag in tags.items():
    #     #tag-based algorithm
    #     recommendations_jaccard, score = tag_based_algorithm(tags, movie_data, movie_id, 100, 'jaccard')
    #     tag_based_recommendations_jaccard[movie_id] = recommendations_jaccard

    #     recommendations_dice, score = tag_based_algorithm(tags, movie_data, movie_id, 100, 'dice')
    #     tag_recommendations_dice[movie_id] = recommendations_dice

    
if __name__ == "__main__":
    main()
