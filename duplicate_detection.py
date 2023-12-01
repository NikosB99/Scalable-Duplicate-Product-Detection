import json
from tqdm import tqdm
import re

from evaluation import calculate_all_true_pairs, find_if_a_pair_is_duplicate, \
    parameters_to_minimize_distance_in_duplicates
from extract_binary_repr import get_common_keys, get_map_values, get_binary_repr, train_one_hot_encoder_in_corpus
from calculate_distance import mcs_distance
from lsh import use_min_hash, lsh


def calculate_duplicates(data, k, shuffles, bands, a, b, g, mi, threshold):
    '''
    Detect duplicate values using the LSH and min_hashing. In the result sets there is the duplicates detected, as well
    as the candidate pairs that were found after the LSH method
    :param data: the json with all the dataset
    :param k: define the top-k most frequent keys that will be used in the representation along with the title
    :param shuffles: how many shuffles are going to be used from the min_hash method
    :param bands: in how many bands will the signatures extracted from the min_hash will be splitted into
    :param a: distance parameter-> determines over which similarity in the title the products will be considered duplicate
    :param b: distance parameter-> determines over which similarity the title will be considered in the similarity measurement
    :param g: distance parameter-> determines over which similarity the keys will be considered identical
    :param mi: distance parameter-> is the standard weight reserved for the similarity of the titles
    :param threshold: two products are considered duplicate if their distance is below this threshold
    :return: a list of the detected duplicate pairs and the list of the candidate pairs from the LSH method
    '''
    data_flattened = [product for key in list(data.keys()) for product in data[key]]
    complete_keys = {}
    for product in data_flattened:
        for k in product['featuresMap']:
            if k in complete_keys:
                complete_keys[k].append(product['featuresMap'][k])
            else:
                complete_keys[k] = []
                complete_keys[k].append(product['featuresMap'][k])

    # get the top most frequent keys
    top = get_common_keys(complete_keys, k)
    # get mappers  of the top-k most frequent keys (map the value of a key to another value in order to find
    # common values that look alike) and encoders
    mappers_and_encoders = [[key, get_map_values(complete_keys[key])] for key in top]

    # extract all titles
    titles = [j["title"] for j in data_flattened]
    # extract words
    ohe_title = train_one_hot_encoder_in_corpus(titles)

    # use mappers, encoders and the title bag of words to create the binary representation
    # of each tv for the top-k most frequent keys
    tvs_representation = []
    for tv in tqdm(data_flattened, desc="Calculate binary representation for each product", total=len(data_flattened)):
        tvs_representation.append(get_binary_repr(mappers_and_encoders, ohe_title, tv))

    min_hashed = use_min_hash(tvs_representation, shuffles)
    pairs_to_check = lsh(min_hashed, bands)

    detected_duplicates = []
    for pair in tqdm(pairs_to_check, desc="Calculating distance between pairs to detect duplicates",
                     total=len(pairs_to_check)):
        indices = pair.split("-")
        dist = mcs_distance(data_flattened[int(indices[0])], data_flattened[int(indices[1])], a, b, g, mi)
        if dist < threshold:
            detected_duplicates.append([int(indices[0]), int(indices[1])])
    return detected_duplicates, pairs_to_check


if __name__ == "__main__":
    file = open('tvs.json')
    data = json.load(file)

    duplicates = calculate_duplicates(data, 10, 10, 5, 0.95, 0.4, 0.75, 0.65, 0.5)

    # print(len(detected_duplicates))
    # total = calculate_all_true_pairs(data)
    # print(total)
    # maximum = sum([1 for i in pairs_to_check if find_if_a_pair_is_duplicate(data_flattened, i)])
    # print(maximum)
