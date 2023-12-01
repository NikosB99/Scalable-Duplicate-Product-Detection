import json
from tqdm import tqdm
import re

from evaluation import calculate_all_true_pairs, find_if_a_pair_is_duplicate
from extract_binary_repr import get_common_keys, get_map_values, get_binary_repr
from lsh import use_min_hash, lsh

if __name__ == "__main__":
    file = open('tvs.json')
    data = json.load(file)
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
    top = get_common_keys(complete_keys, 10)
    # get mappers (map the value of a key to another value in order to find common values that look alike) and encoders
    mappers_and_encoders = [[key, get_map_values(complete_keys[key])] for key in top]
    # use mappers and encoders to create the binary representation of each tv
    tvs_representation = []
    for tv in tqdm(data_flattened,desc="Calculate binary representation for each product",total=len(data_flattened)):
        tvs_representation.append(get_binary_repr(mappers_and_encoders, tv))

    min_hashed = use_min_hash(tvs_representation, 10)
    pairs_to_check = lsh(min_hashed, 10)
    print(len(pairs_to_check))
    total = calculate_all_true_pairs(data)
    print(total)
    maximum = sum([1 for i in pairs_to_check if find_if_a_pair_is_duplicate(data_flattened,i)])
    print(maximum)