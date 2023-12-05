# Scalable-Duplicate-Product-Detection
This repository explores a novel representation based on the top-k
most frequent keys and the item title. The proposed representation is
utilized by Min Hash and LSH to effectively reduce the required number of comparisons. The distance between each extracted candidate pair
is calculated, and if the value is below a user-provided threshold, the
pair is reported as a duplicate. Two distinct distance metrics, Jaccard
and MSM, were employed.

### Code 
The code was implemented in python. The structure is as follows:
* **extract_binary_repr.py**: Contains the necessary functions to extract the binary representation of a single item
* **lsh.py**: Contains the implementation of LSH and Min Hashing
* **calculate_distance.py**: Contains the implementation of both Jaccard and MSM distance. Additionally, it contains
two more additional functions which are used by the multi-threading process to calculate the two distances
respectively.
* **duplicate_detection.py**: Contains the whole process of extracting the duplicates based on the 
given parameters. Utilizes functions from _extract_binary_repr_, _lsh_ and _calculate_distance_
* **evaluation.py**: Contains the evaluation process, i.e., the bootstrap methodology and the functions
to calculate the TP, TN, FN, FP. It writes the results in files inside the _tmp_files_ folder
* **create_graphs.py**: Creates the graphs based on the data extracted by the evaluation script

### Execute code
Since the code is implemented in python, it is straightforward to be executed.
First the requirements (contained in the _requirements.txt_ file) needs to be installed:
```bash
pip install -r requirements.txt
 ```
Next the evaluation script can be executed using the following command:
```bash
python3 evaluation.py
 ```
