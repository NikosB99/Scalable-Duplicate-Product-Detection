import string

from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def mcs_distance(tva, tvb, a, b, g, mi):
    sim = 0
    avgSim = 0
    m = 0
    w = 0

    nmki: list = list(tva['featuresMap'].keys())
    nmkj: list = list(tvb['featuresMap'].keys())

    i = 0
    while i < len(nmki):
        q = nmki[i]
        j = 0
        while j < len(nmkj):
            r = nmkj[j]
            keySim = calcSim(q, r)
            if keySim >= g:
                valueSim = calcSim(tva['featuresMap'][q], tvb['featuresMap'][r])
                weight = keySim
                sim += sim + weight * valueSim
                m += 1
                w += weight
                nmki.remove(q)
                nmkj.remove(r)
                i -= 1  # Adjust index after removing element from nmki
                break  # Move to the next q in the outer loop
            j += 1
        i += 1

    if w > 0:
        avgSim = sim / w
    mwPerc = mw(exMw(tva, set(nmki)), exMw(tvb, set(nmkj)))
    titleSim = TMWMSim(tva, tvb, a, b)
    if titleSim == -1:
        theta_1 = m / minFeatures(tva, tvb)
        theta_2 = 1 - theta_1
        hSim = theta_1 * avgSim + theta_2 * mwPerc
    else:
        theta_1 = (1 - mi) * m / minFeatures(tva, tvb)
        theta_2 = 1 - theta_1
        hSim = theta_1 * avgSim + theta_2 * mwPerc + mi * titleSim
    return 1 - hSim


def calcSim(a: str, b: str) -> float:
    ag = list(ngrams(a, 3))
    bg = list(ngrams(b, 3))
    common = sum([1 if x1 in bg else 0 for x1 in ag])
    return 1-(len(ag) + len(bg) - common) / (len(ag) + len(bg))


def exMw(tv, list_of_keys: set) -> set:
    values = [tv["featuresMap"][key] for key in list_of_keys]
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(" ".join(values))
    tokens = [i for i in tokens if i not in string.punctuation and i not in ["''", "\""]]  # remove punctuations
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    return set(tokens)


def mw(b1: set, b2: set) -> float:
    '''
    Calculate jaccard similarity between the set of keys
    :param b1: values of the first tv
    :param b2: values of the second tv
    :return: the jaccard similarity
    '''
    return len(b1.intersection(b2)) / len(b1.union(b2))


def TMWMSim(tva, tvb, a, b):
    title_a = tva["title"]
    title_b = tvb["title"]
    ag = set(ngrams(title_a, 3))
    bg = set(ngrams(title_b, 3))
    similarity = mw(ag, bg)
    if similarity >= a:
        return 1
    elif similarity >= b:
        return similarity
    else:
        return -1


def minFeatures(tva, tvb) -> int:
    return min(len(list(tva["featuresMap"])), len(list((tvb["featuresMap"]))))
