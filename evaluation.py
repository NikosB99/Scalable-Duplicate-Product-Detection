def calculate_all_true_pairs(original_json):
    total = 0
    for key in list(original_json.keys()):
        if len(original_json[key]) > 1:
            total += sum([i for i in range(1, len(original_json[key]))])
    return total

def find_if_a_pair_is_duplicate(flattened,pair):
    x=pair.split("-")
    tv1=flattened[int(x[0])]
    tv2=flattened[int(x[1])]
    return tv1["modelID"]==tv2["modelID"]