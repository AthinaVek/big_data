import os
import pandas as pd
import numpy as np
import sys

from recommender_functions import *
from load_data import *


def main():
    if len(sys.argv) != 11:
        print("Input must be in the form: recommender.py -d directory_of_data -n number_of_recommendations -s similarity_metric -a algorithm -i input_user_id")
        sys.exit(1)
    
    #get arguments
    directory_of_data = sys.argv[sys.argv.index("-d") + 1]
    n = int(sys.argv[sys.argv.index("-n") + 1])
    similarity_metric = sys.argv[sys.argv.index("-s") + 1]
    algorithm = sys.argv[sys.argv.index("-a") + 1]
    input_var = int(sys.argv[sys.argv.index("-i") + 1])

    if directory_of_data == 'ml-latest-small' and algorithm == 'user_user_algorithm':
        #real time
        if similarity_metric == 'jaccard':
            user_user_recommendations_jaccard = pd.read_csv('user_user_recommendations_jaccard.csv')
            user_recommendations = user_user_recommendations_jaccard[user_user_recommendations_jaccard['user_id'] == input_var]
            movies = user_recommendations['movies'].values[0].split(', ')
            print(f"Top {n} movie recommendations for user {input_var}: {', '.join(movies[:n])}")
        elif similarity_metric == 'dice':
            user_user_recommendations_dice = pd.read_csv('user_user_recommendations_dice.csv')
            user_recommendations = user_user_recommendations_dice[user_user_recommendations_dice['user_id'] == input_var]
            movies = user_recommendations['movies'].values[0].split(', ')
            print(f"Top {n} movie recommendations for user {input_var}: {', '.join(movies[:n])}")
        elif similarity_metric == 'cosine':
            user_user_recommendations_cosine = pd.read_csv('user_user_recommendations_cosine.csv')
            user_recommendations = user_user_recommendations_cosine[user_user_recommendations_cosine['user_id'] == input_var]
            movies = user_recommendations['movies'].values[0].split(', ')
            print(f"Top {n} movie recommendations for user {input_var}: {', '.join(movies[:n])}")
        elif similarity_metric == 'pearson':
            user_user_recommendations_pearson = pd.read_csv('user_user_recommendations_pearson.csv')
            user_recommendations = user_user_recommendations_pearson[user_user_recommendations_pearson['user_id'] == input_var]
            movies = user_recommendations['movies'].values[0].split(', ')
            print(f"Top {n} movie recommendations for user {input_var}: {', '.join(movies[:n])}")
    else:
        #load data
        if algorithm == 'item_item_algorithm':
            data, movie_data, given_user_movies = load_data(directory_of_data, algorithm, input_var)
        elif algorithm == 'content_based_algorithm' or algorithm == 'user_user_algorithm' or algorithm == 'tag_based_algorithm':
            data, movie_data = load_data(directory_of_data, algorithm, input_var)
        

        if algorithm == 'user_user_algorithm':
            recommendations, score = user_user_algorithm(data, movie_data, input_var, n, similarity_metric)
            print(f"Top {n} movie recommendations for user {input_var}: {recommendations}")
        elif algorithm == 'item_item_algorithm':
            recommendations, score = item_item_algorithm(data, movie_data, given_user_movies, input_var, n, similarity_metric)
            print(f"Top {n} movie recommendations for user {input_var}: {recommendations}")
        elif algorithm == 'tag_based_algorithm':
            recommendations, score = tag_based_algorithm(data, movie_data, input_var, n, similarity_metric)
            print(f"Top {n} movie recommendations for user {input_var}: {recommendations}")
        elif algorithm == 'content_based_algorithm':
            recommendations, score = content_based_algorithm(movie_data, input_var, n, similarity_metric)
            print(f"Top {n} movie recommendations for user {input_var}: {recommendations}")
        elif algorithm == 'hybrid_algorithm':
            recommendations = hybrid_algorithm(directory_of_data, input_var, n, similarity_metric)
            print(f"Top {n} movie recommendations for user {input_var}: {recommendations}")
            
if __name__ == "__main__":
    main()
