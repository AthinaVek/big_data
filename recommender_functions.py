import numpy as np
import pandas as pd
import math
import re

from recommender_similarity_metrics import *
from load_data import *


#user-user algorithm
def user_user_algorithm(user_ratings, movie_data, u_id, n, similarity_metric):
    similarity_vector = []
    rx = []
    other_rx = []

    for other_user_id, other_ratings in user_ratings.items():
        if other_user_id != u_id:
            for movie in other_ratings.keys():
                if movie in user_ratings[u_id].keys():
                    rx.append(user_ratings[u_id][movie])
                    other_rx.append(other_ratings[movie])

            if similarity_metric == 'jaccard':
                similarity = jaccard_similarity(rx, other_rx)
            elif similarity_metric == 'dice':
                similarity = dice_similarity(rx, other_rx)
            elif similarity_metric == 'cosine':
                similarity = cosine_similarity(rx, other_rx)
            elif similarity_metric == 'pearson':
                similarity = pearson_similarity(rx, other_rx)
            else:
                print ('Similarity metric does not exist.')
                return 0
            similarity_vector.append((other_user_id, similarity))

    similarity_vector.sort(key=lambda x: x[1], reverse=True)     #sort similarity value (highest first)
    top_k_users = similarity_vector[:128]     #(k = 128)

    recommendation_scores = []
    all_item_ids = {item_id for ratings in user_ratings.values() for item_id in ratings}

    for item_id in all_item_ids:
        sum1 = 0
        sum2 = 0

        for other_user_id, similarity in top_k_users:
            if item_id in user_ratings[other_user_id]:
                sum1 += similarity * user_ratings[other_user_id][item_id]
                sum2 += similarity

        r = sum1 / sum2 if sum2 != 0 else 0
        recommendation_scores.append((item_id, r))

    recommendation_scores.sort(key=lambda x: x[1], reverse=True)
    top_n_recommendations = [(movie_data[item_id]) for item_id, score in recommendation_scores[:n]]  # Get top n recommendations

    return top_n_recommendations, recommendation_scores[:n]



#item-item algorithm
def item_item_algorithm(movie_ratings, movie_data, given_user_movies, user_id, n, similarity_metric):
    recommendation_vector = []

    for movie_id, movie in movie_data.items():
        if movie_id not in given_user_movies:
            if movie_id not in movie_ratings:
                recommendation_vector.append((movie_id, 0))
            else:
                similarities = []
                for user_movie_id in given_user_movies:
                    rated_intersection = set(movie_ratings[movie_id]).intersection(movie_ratings[user_movie_id])
                    if rated_intersection:
                        s1 = []
                        s2 = []
                        for u in rated_intersection:
                            s1.append(movie_ratings[movie_id][u])
                            s2.append(movie_ratings[user_movie_id][u])

                        if similarity_metric == 'jaccard':
                            similarity = jaccard_similarity(s1, s2)
                        elif similarity_metric == 'dice':
                            similarity = dice_similarity(s1, s2)
                        elif similarity_metric == 'cosine':
                            similarity = cosine_similarity(s1, s2)
                        elif similarity_metric == 'pearson':
                            # print (movie)
                            similarity = pearson_similarity(s1, s2)
                        else:
                            print ('Similarity metric does not exist.')
                            return 0
                        similarities.append((user_movie_id, similarity))

                similarities.sort(key=lambda x: x[1], reverse=True)
                top_k_items = similarities[:128]     #(k = 128)

                numerator = 0
                denominator = 0
                for other_item_id, sim in top_k_items:
                    numerator += sim * movie_ratings[other_item_id][user_id]
                    denominator += sim

                recommendation_score = numerator / denominator if denominator != 0 else 0
                recommendation_vector.append((movie_id, recommendation_score))

    recommendation_vector.sort(key=lambda x: x[1], reverse=True)
    top_n_recommendations = [(movie_data[item_id]) for item_id, score in recommendation_vector[:n]]     #top n recommendations

    return top_n_recommendations, recommendation_vector[:n]



 #tag based algorithm
def tag_based_algorithm(tags, movie_data, movie_id, n, similarity_metric):
    movie_tags = set(tags.get(movie_id, {}))
    similarity_vector = []

    for other_movie_id, other_tags in tags.items():
        if movie_id != other_movie_id:
            other_tags_set = set(other_tags)
            if similarity_metric == 'jaccard':
                similarity = jaccard_similarity(movie_tags, other_tags_set)
            elif similarity_metric == 'dice':
                similarity = dice_similarity(movie_tags, other_tags_set)
            elif similarity_metric == 'cosine':
                similarity = cosine_similarity(movie_tags, other_tags_set)
            elif similarity_metric == 'pearson':
                similarity = pearson_similarity(movie_tags, other_tags_set)
            else:
                print ('Similarity metric does not exist.')
                return 0
            similarity_vector.append((other_movie_id, similarity))

    similarity_vector.sort(key=lambda x: x[1], reverse=True)
    top_n_recommendations = [movie_data[movie_id] for movie_id, score in similarity_vector[:n]]

    return top_n_recommendations, similarity_vector[:n]


#content based algorithm
def calculate_tfidf(titles):    #calculate tf-idf
    term_frequency = {}
    all_terms = set()

    for title in titles:
        term_frequency[title] = {}
        for term in set(title.split()):
            term_frequency[title][term] = title.count(term) / len(title.split())
            all_terms.add(term)

    document_frequency = {term: sum(1 for title in titles if term in title.split()) for term in all_terms}

    total_documents = len(titles)
    inverse_document_frequency = {term: math.log(total_documents / (document_frequency[term] + 1)) for term in all_terms}

    tfidf_matrix = {}
    for title in titles:
        tfidf_matrix[title] = [term_frequency[title].get(term, 0) * inverse_document_frequency[term] for term in all_terms]

    return tfidf_matrix



def content_based_algorithm(movie_data, movie_id, n, similarity_metric):
    all_movie_titles = []
    for m_id, title in movie_data.items():
        title = re.sub(r'\(\d{4}\)', '', title).strip()
        all_movie_titles.append(title)

    tfidf_matrix = calculate_tfidf(all_movie_titles)

    given_movie =  re.sub(r'\(\d{4}\)', '', movie_data[movie_id]).strip()

    input_vector = tfidf_matrix[given_movie]

    similarity_vector = []     #similarity scores
    for title, vector in tfidf_matrix.items():
        if title != given_movie:
            if similarity_metric == 'jaccard':
                similarity = jaccard_similarity(input_vector, vector)
            elif similarity_metric == 'dice':
                similarity = dice_similarity(input_vector, vector)
            elif similarity_metric == 'cosine':
                similarity = cosine_similarity(input_vector, vector)
            elif similarity_metric == 'pearson':
                similarity = pearson_similarity(input_vector, vector)
            else:
                print ('Similarity metric does not exist.')
                return 0
            similarity_vector.append((title, similarity))

    similarity_vector.sort(key=lambda x: x[1], reverse=True)
    top_n_recommendations = [movie_id for movie_id, score in similarity_vector[:n]]

    return top_n_recommendations, similarity_vector[:n]



#hybrid algorithm using user_user_algorithm, item_item_algorithm, content_based_algorithm
def hybrid_algorithm(dir_of_data, user_id, n, similarity_metric):
    user_ratings, movies = load_data(dir_of_data, 'user_user_algorithm', user_id)
    user_ratings_item, movies, given_user_movies = load_data(dir_of_data, 'item_item_algorithm', user_id)

    user_user_recommendations, user_user_scores = user_user_algorithm(user_ratings, movies, user_id, n, similarity_metric)
    item_item_recommendations, item_item_scores = item_item_algorithm(user_ratings_item, movies, given_user_movies,user_id, n, similarity_metric)
    
    combined_recommended_movies = set(user_user_recommendations).union(item_item_recommendations)      #combine results from user-user and item-item
    combined_recommended_dict = {}     #dictionary with these movies
    for title1 in combined_recommended_movies:
        for movie_id, title2 in movies.items():
            if title1 == title2:
                combined_recommended_dict[movie_id] = title1

    content_based_recommendations = []
    content_base_scores = []
    for movie_id, movie in combined_recommended_dict.items():
        content_base_movies, content_scores = content_based_algorithm(movies, movie_id, n, similarity_metric)
        content_based_recommendations.extend(content_base_movies)
        content_base_scores.extend(content_scores)

    all_recommended_movies = set(user_user_recommendations + item_item_recommendations + content_based_recommendations)
    all_recommended_scores = []
    all_recommended_scores.extend(user_user_scores)
    all_recommended_scores.extend(item_item_scores)
    all_recommended_scores.extend(content_base_scores)

    recommended_list = list(zip(all_recommended_movies, all_recommended_scores))    #make movies and scores into one list of tuples
    recommended_list = sorted(recommended_list, key=lambda x: x[1][1], reverse=True)

    top_n_recommendations = [movie[0] for movie in recommended_list[:n]]
    
    return top_n_recommendations