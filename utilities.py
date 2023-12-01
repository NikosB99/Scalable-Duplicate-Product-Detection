def extract_true_duplicates(data_flattened):
    true = set()
    for index, a in enumerate(data_flattened):
        for indexb, b in enumerate(data_flattened):
            if index != indexb and a["modelID"] == b["modelID"]:
                true.add(f"{min(index, indexb)}-{max(index, indexb)}\n")

    with open("temp_files/true.txt", "w") as f:
        for pair in true:
            f.write(pair)
