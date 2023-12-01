import math
from tqdm import tqdm

from calculate_distance import mcs_distance


def calculate_all_true_pairs(original_json):
    total = 0
    for key in list(original_json.keys()):
        if len(original_json[key]) > 1:
            total += sum([i for i in range(1, len(original_json[key]))])
    return total


def find_if_a_pair_is_duplicate(flattened, pair):
    x = pair.split("-")
    tv1 = flattened[int(x[0])]
    tv2 = flattened[int(x[1])]
    return tv1["modelID"] == tv2["modelID"]


def parameters_to_minimize_distance_in_duplicates(data_flattened):
    a_s = [0.7, 0.8, 0.9, 0.95, 1]
    b_s = [0.5, 0.6, 0.7, 0.8]
    g = 0.75
    m = 0.65
    min_dist = math.inf
    best_param = []
    for a in a_s:
        for b in b_s:
            d = 0
            print(f"Testing {a},{b}")
            with open("temp_files/true.txt", "r") as f:
                for line in tqdm(f):
                    ids = line.strip().split("-")
                    d += mcs_distance(data_flattened[int(ids[0])], data_flattened[int(ids[1])], a, b, g, m)
                if d < min_dist:
                    min_dist = d
                    best_param = [a, b, g, m]
    return best_param, min_dist
