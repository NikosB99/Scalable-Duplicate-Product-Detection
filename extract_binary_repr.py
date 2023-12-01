import re

import nltk
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


# list(encoded.A[0])
def get_binary_repr(mappers_and_encoders, ohe_title, json_obj):
    signature = []
    features = json_obj['featuresMap']
    for me in mappers_and_encoders:
        mapper = me[1][0]
        encoder = me[1][1]
        key = me[0]
        if key in features:
            mapped_value = mapper[features[key]]
            encoded = encoder.transform(np.array(mapped_value).reshape(-1, 1))
            combined = encoded.A[0]
            if len(encoded.A) > 1:
                for y in encoded.A[1:]:
                    combined = combined + y
            for b in combined:
                signature.append(b)
        else:
            for _ in encoder.transform(np.array("N/A").reshape(-1, 1)).A[0]:  # append zeros equal to the size of this
                signature.append(0)
    title = json_obj["title"]
    for b in get_binary_repr_for_title(title, ohe_title):
        signature.append(b)
    return signature


def get_common_keys(complete_keys: dict, k: int):
    '''
    :param complete_keys: a dictionary with all the keys in the dataset along with the respected values
    :param k: top-k keys
    :return: the top-k most common used keys in the dataset
    '''
    lengths = []
    for key in complete_keys:
        lengths.append([key, len(complete_keys[key])])
    lengths.sort(key=lambda x: x[1], reverse=True)
    return [i[0] for i in lengths[:k]]


def get_map_values(values: list):
    '''
    :param values: the different values for a single key across all dataset
    :return: a mapper (maps each original value to a transformed one) and an encoder that given a transformed
    value returns a binary representation using OneHotEncoder
    '''
    c = sum([1 if "x" in y else 0 for y in values])
    if c > 0.8 * len(values):  # exists in more than 80% of the names => resolution type
        return _handle_resolution(values)
    c = sum([1 if ":" in y else 0 for y in values])
    if c > 0.8 * len(values):  # exists in more than 80% of the names => ratio
        return _handle_ration(values)
    c = sum([1 if "\"" in y else 0 for y in values])
    if c > 0.8 * len(values):  # exists in more than 80% of the names => inches
        return _handle_inches(values)
    c = sum([1 if str(y).lower() in ['yes', 'no', '0', '1', 'unknown', 'not specified', 'true', 'false'] else 0 for y in
             values])
    if c > 0.8 * len(values):  # exists in more than 80% of the names => boolean
        return _handle_boolean(values)
    return _handle_general(values)  # return for normal strings


def _handle_resolution(values):
    map = {}
    for x in set(values):
        # keep only numbers and the x to create a unified version
        k = "".join([letter for letter in x if letter.isdigit() or letter == "x"])
        map[x] = [k]
    return map, _get_one_hot_encoder(map)


def _handle_ration(values):
    map = {}
    for x in set(values):
        # extract ratios
        ratios = re.findall(r'[0-9, ]*:[0-9, ]*', x)
        if len(ratios) == 0:
            map[x] = ["unknown"]
        else:
            map[x] = ratios
    return map, _get_one_hot_encoder(map)


def _handle_boolean(values):
    map = {}
    for x in set(values):
        if x.lower() in ["yes", "1", "true"]:
            map[x] = ["true"]
        elif x.lower() in ["no", "0", "false"]:
            map[x] = ["false"]
        elif x.lower() in ["unknown", 'not specified']:
            map[x] = ["unknown"]
        else:
            map[x] = ["true"]  # something exists there
    return map, _get_one_hot_encoder(map)


def _handle_inches(values):
    map = {}
    for x in set(values):
        temp = "".join(
            [i for i in x.replace("\"", "") if (i.isnumeric() or i in ["/", ",", ".", "-", " "])])  # remove inches
        s = temp.split("-")
        if "/" in s[0]:
            t = s[0].split("/")
            t = int(t[0]) / int(t[1])
            value = "{:.1f}".format(t)
        else:
            t = 0
            if len(s) > 1:  # contains fraction
                try:
                    t = s[1].split("/")
                    t = int(t[0]) / int(t[1])
                except:
                    t = 0
            value = "{:.1f}".format(float(s[0]) + t)
        map[x] = [value]
    return map, _get_one_hot_encoder(map)


def _handle_general(values):
    map = {key: [''.join(char for char in key.lower() if key.isalnum())] for key in values}
    return map, _get_one_hot_encoder(map)


def _get_one_hot_encoder(map):
    '''
    Creates and fit the OneHotEncoder to the transformed values
    :param map: the map between the original and the transformed values
    :return: an OneHotEncoder object
    '''
    diff_values = list(set([y for i in map for y in map[i]]))
    diff_values.insert(0, "N/A")
    ohe = OneHotEncoder(handle_unknown='ignore', categories='auto')
    ohe.fit(np.array(diff_values).reshape(-1, 1))
    return ohe


def train_one_hot_encoder_in_corpus(list_of_titles: list):
    nltk.download('stopwords')
    nltk.download('punkt')
    all = " ".join(list_of_titles).lower()

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(all)

    tokens = [i for i in tokens if i not in string.punctuation and i not in ["''", "\""]]  # remove punctuations
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    ohe = OneHotEncoder(categories='auto', handle_unknown='ignore')
    ohe.fit(np.array(list(set(tokens))).reshape(-1, 1))  # keep only uniques
    return ohe


def get_binary_repr_for_title(title: str, ohe: OneHotEncoder):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(title.lower())
    tokens = [i for i in tokens if i not in string.punctuation and i not in ["''", "\""]]  # remove punctuations
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    x = ohe.transform(np.array(tokens).reshape(-1, 1))
    sum = x.A[0]
    for s in x.A[1:]:
        sum += s
    return sum
