import json
from multiprocessing.pool import Pool

from tqdm import tqdm
from sympy.solvers import solve
from sympy.abc import x, r

from extract_binary_repr import get_common_keys, get_map_values, get_binary_repr, train_one_hot_encoder_in_corpus
from calculate_distance import jaccard_distance_packed, msm_distance_packed
from lsh import use_min_hash, lsh


def calculate_duplicates(data_flattened, k, t, threshold, shuffles=0.1, a=0.95, b=0.4, g=0.75, mi=0.65,
                         distance: str = "msm"):
    '''
    Detect duplicate values using the LSH and min_hashing. In the result sets there is the duplicates detected, as well
    as the candidate pairs that were found after the LSH method
    :param data_flattened: the json with all the dataset flattened (each item is a single entry)
    :param k: define the top-k most frequent keys that will be used in the representation along with the title
    :param t: define the threshold between false positives and false negatives for the LSH method
    :param threshold: define the maximum distance that two items will have in order to be considered duplicate
    :param shuffles: percentage of the original representation that will determine the size of the signatures
    :param a: distance parameter-> determines over which similarity in the title the products will be considered duplicate
    :param b: distance parameter-> determines over which similarity the title will be considered in the similarity measurement
    :param g: distance parameter-> determines over which similarity the keys will be considered identical
    :param mi: distance parameter-> is the standard weight reserved for the similarity of the titles
    :param distance: determines the different distance used to calculate the duplicate pairs
    :return: a list of the detected duplicate pairs and the list of the candidate pairs from the LSH method
    '''
    complete_keys = {}
    for product in data_flattened:
        for n in product['featuresMap']:
            if n in complete_keys:
                complete_keys[n].append(product['featuresMap'][n])
            else:
                complete_keys[n] = []
                complete_keys[n].append(product['featuresMap'][n])

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

    num_shuffles = int(shuffles * len(tvs_representation[0]))  # define signature size as a portion of the original
    signatures = use_min_hash(tvs_representation, num_shuffles)  # generate the shorted signatures

    equations = [x * r - num_shuffles, t - ((1 / x) ** (1 / r))]
    solutions = solve(equations, x, r, dict=True)
    bands = int(solutions[0][x])  # signature length and parameter t determine the bands for the LSH

    pairs_to_check = lsh(signatures, bands)

    args = []
    args_jac = []
    for pair in pairs_to_check:
        y = pair.split("-")
        args.append((data_flattened[int(y[0])], data_flattened[int(y[1])], a, b, g, mi, pair))
        args_jac.append((signatures[int(y[0])], signatures[int(y[1])], pair))

    detected_duplicates = []
    if distance == "msm":
        with Pool(12) as p:
            results = list(tqdm(p.imap(msm_distance_packed, args), total=len(args),
                                desc="Calculating duplicates using distance MSM"))
            for res in results:
                if res[0] > threshold:
                    detected_duplicates.append(res[1])
    else:
        with Pool(12) as p:
            results = list(tqdm(p.imap(jaccard_distance_packed, args_jac), total=len(args_jac),
                                desc="Calculating duplicates using distance Jaccard"))
            for res in results:
                if res[0] > threshold:
                    detected_duplicates.append(res[1])

    return detected_duplicates, pairs_to_check
