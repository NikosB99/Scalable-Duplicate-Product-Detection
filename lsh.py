from random import shuffle
from tqdm import tqdm


def use_min_hash(list_of_representations, number_of_shuffles):
    '''
    Utilizes the min_has to shorten the large binary representations
    :param list_of_representations: the binary representations
    :param number_of_shuffles: how many times will the min_hash take place
    :return: a shorten version of the binary representation
    '''
    n = list(range(1, len(list_of_representations[0]) + 1))
    final_repr = [[] for _ in range(len(list_of_representations))]
    for _ in tqdm(range(number_of_shuffles), desc="Shortening the binary signatures"):
        shuffle(n)  # shuffles the numbers => shuffling each representation
        for index, b_repr in enumerate(list_of_representations):  # find the position of the first 1
            for x in n:
                if b_repr[x - 1] == 1:
                    final_repr[index].append(x)  # add the position found the first one
                    break
    return final_repr


def lsh(tv_signatures, b):
    '''
    Implementation of the Local Sensitivity Hashing. The shorten representations are splitted into bands (equal to the
    number of b) each containing r numbers. Then the numbers of each band are concatenated and a hash function is used.
    The function that is utilized here is the built in python function for hashing. Then for each two pairs that ended
    up in the same bucket (had the same hash function in at least one of the bands) a pair is added to the pairs_to_check
    in order to calculate the expensive distance.
    :param tv_signatures: the shortened signatures
    :param b: the number of the bands utilized
    :return: the pairs that require to be evaluated, as there are the possible duplicate values
    '''
    pairs_to_check = set()  # use set to avoid adding the same pair of tvs multiple times
    r = int(len(tv_signatures[0]) / b)
    for i in tqdm(range(0, len(tv_signatures[0]), r), desc="Calculating common hashes for each band"):
        hash_map = {}
        for index, signature in enumerate(tv_signatures):
            band = "".join([str(t) for t in signature[i:i + r]])
            h = hash(band)
            if h not in hash_map:
                hash_map[h] = []
            hash_map[h].append(index)
        for h in hash_map:
            if len(hash_map[h]) > 1:  # if there is more than one product indexed here
                for i1, x in enumerate(hash_map[h]):  # create all the pairs that requires to be checked
                    for i2, y in enumerate(hash_map[h][i1 + 1:]):  # avoid creating the same pair multiple times
                        pairs_to_check.add(f"{min(x, y)}-{max(x, y)}")
    return pairs_to_check
