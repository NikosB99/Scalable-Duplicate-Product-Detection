import json
import math
from tqdm import tqdm

import numpy as np
from random import choices

from duplicate_detection import calculate_duplicates


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


def get_bootstrap(data_flattened: list, perc: float = 0.63):
    return choices(data_flattened, k=int(len(data_flattened) * perc))


def performance_lsh(detected):
    true = set()
    with open("temp_files/true.txt", "r") as f:
        for line in f:
            true.add(line.strip())
    found = len(true.intersection(detected))
    pair_quality = found / len(detected)
    pair_completeness = found / len(true)
    return pair_quality, pair_completeness, found


def performance(detected):
    true = set()
    with open("temp_files/true.txt", "r") as f:
        for line in f:
            true.add(line.strip())
    detected_set = set([f"{i[0]}-{i[1]}" for i in detected])
    found = len(true.intersection(detected_set))
    if found == 0:
        return 0, 0, 0
    precision = found / len(detected)
    recall = found / len(true)
    return precision, recall, found


if __name__ == "__main__":
    file = open('tvs.json')
    data = json.load(file)
    data_flattened = [product for key in list(data.keys()) for product in data[key]]

    with open("temp_files/lsh_performance_5.txt", "a+") as f:
        for t in np.arange(0.05, 1.05, 0.05):
            print(f"Testing for t = {t}")
            for iter in range(5):
                print(f"Iteration: {iter}")
                boot = get_bootstrap(data_flattened)
                duplicates, lsh_detected = calculate_duplicates(boot, 5, t, 1, distance="jaccard")
                pq, pc, found = performance_lsh(lsh_detected)
                f.write(f"{t},{pq},{found},{len(lsh_detected)}\n")

    with open("temp_files/all_performance.txt", "a+") as f:
        for distance in ["jaccard"]:
            for threshold in np.arange(0.5, 1, 0.05):
                print(f"Testing for threshold = {threshold} and distance = {distance}")
                for iter in range(5):
                    print(f"Iteration: {iter}")
                    boot = get_bootstrap(data_flattened)
                    duplicates, lsh_detected = calculate_duplicates(boot, 5, 0.1, 1, distance=distance)
                    precision, recall, found = performance(duplicates)
                    f.write(f"{distance},{1},{precision},{recall},{len(lsh_detected)}\n")

