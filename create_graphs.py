import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("temp_files/lsh_performance.txt")
result_df = df.groupby('t')[['found', 'lsh_detected']].mean().reset_index()
result_df["Fraction of Comparisons"] = result_df['lsh_detected'] / (1023 * 1023)
result_df["Pair Quality"] = result_df['found'] / result_df['lsh_detected']
result_df["Pair Completeness"] = result_df["found"] / 399
result_df["F1*"] = (2 * result_df['Pair Quality'] * result_df['Pair Completeness']) / (
        result_df['Pair Quality'] + result_df['Pair Completeness'])

plt.figure()
result_df = result_df.sort_values(by='Fraction of Comparisons', ascending=False)
plt.plot(result_df['Fraction of Comparisons'], result_df['Pair Quality'], marker='o')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Quality')
plt.tight_layout()
plt.savefig('graphs/pair_quality.eps', format="eps")

plt.figure()
plt.plot(result_df['Fraction of Comparisons'], result_df['Pair Completeness'], marker='o')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Completeness')
plt.tight_layout()
plt.savefig('graphs/pair_completeness.eps', format="eps")

plt.figure()
plt.plot(result_df['Fraction of Comparisons'], result_df['F1*'], marker='o')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1*')
plt.tight_layout()
plt.savefig('graphs/pair_f1.eps', format="eps")

plt.figure()
df = pd.read_csv("temp_files/all_performance.txt")
df = df.sort_values(by='threshold', ascending=True)
df_msm = df[df["distance"] == "msm"]
plt.plot(df_msm['threshold'], df_msm['precision'], marker='o', label="MSM")
df_jacc = df[df["distance"] == "jacc"]
plt.plot(df_jacc['threshold'], df_jacc['precision'], marker='x', label="Jaccard")
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.tight_layout()
plt.legend()
plt.savefig('graphs/precision.eps', format="eps")

plt.figure()
df_msm = df[df["distance"] == "msm"]
plt.plot(df_msm['threshold'], df_msm['recall'], marker='o', label="MSM")
df_jacc = df[df["distance"] == "jacc"]
plt.plot(df_jacc['threshold'], df_jacc['recall'], marker='x', label="Jaccard")
plt.xlabel('Threshold')
plt.ylabel("Recall")
plt.tight_layout()
plt.legend()
plt.savefig('graphs/recall.eps', format="eps")

plt.figure()
df_msm = df[df["distance"] == "msm"]
df_jacc = df[df["distance"] == "jacc"]
df_msm["F1"] = (2 * df_msm['precision'] * df_msm['recall']) / (
        df_msm['precision'] + df_msm['recall'])
df_jacc["F1"] = (2 * df_jacc['precision'] * df_jacc['recall']) / (
        df_jacc['precision'] + df_jacc['recall'])
plt.plot(df_msm['threshold'], df_msm['F1'], marker='o', label="MSM")
plt.plot(df_jacc['threshold'], df_jacc['F1'], marker='x', label="Jaccard")
plt.xlabel('Threshold')
plt.ylabel('F1')
plt.tight_layout()
plt.legend()
plt.savefig('graphs/f1.eps', format="eps")
