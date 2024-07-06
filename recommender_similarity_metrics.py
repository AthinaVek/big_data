from scipy.sparse import csr_matrix
import numpy as np

def jaccard_similarity(x, y):
    if isinstance(x, csr_matrix) and isinstance(y, csr_matrix):
        x = set(x.indices)
        y = set(y.indices)
    intersection = len(set(x) & set(y))
    union = len(set(x) | set(y))

    if union != 0:
        jaccard = intersection / union
        return jaccard
    else:
        return 0
       

def dice_similarity(x, y):
    intersection = len(set(x) & set(y))
    total_elements = len(set(x)) + len(set(y))

    if total_elements != 0:
        dice = 2 * intersection / total_elements
        return dice
    else:   
        return 0


def cosine_similarity(x, y):
    dot_product = sum(i * j for i, j in zip(x, y))

    norm_x = np.sqrt(sum(i**2 for i in x))
    norm_y = np.sqrt(sum(j**2 for j in y))

    if norm_x != 0 and norm_y != 0:
        cosine = dot_product / (norm_x * norm_y)
        return cosine
    else:
        return 0


def pearson_similarity(x, y):
    x = list(x)
    y = list(y)

    if len(x) == 0 or len(y) == 0:
        return 0

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    cov_xy = np.sum((x - mean_x) * (y - mean_y)) / len(x)

    std_x = np.sqrt(np.sum((x - mean_x)**2) / len(x))
    std_y = np.sqrt(np.sum((y - mean_y)**2) / len(y))

    if std_x == 0 or std_y == 0:
        return 0

    similarity = cov_xy / (std_x * std_y)
    return similarity