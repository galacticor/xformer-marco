import pandas as pd

top100val = pd.read_csv("top100val.csv")


# condition function
condition_512 = lambda x: x <= 512
condition_1024 = lambda x: x >= 1024
condition_2048 = lambda x: x >= 2048
condition_512_1024 = lambda x: x >= 512 and x <= 1024
condition_1024_2048 = lambda x: x >= 1024 and x <= 2048
condition_512_2048 = lambda x: x >= 512 and x <= 2048

conditions = [condition_512, condition_1024, condition_2048, condition_512_1024, condition_1024_2048, condition_512_2048]
names = ["512", "1024", "2048", "512_1024", "1024_2048", "512_2048"]

for i in range(len(conditions)):
    condition = conditions[i]
    name = names[i]
    top100 = top100val.copy()

    # exclude query that doesnt meet the condition
    top100["is_condition"] = top100["seq_len"].apply(condition)
    qid_exclude = list(top100[ (top100.label == 1) & (top100.is_condition == False) ].qid.unique())

    # get top 100 query that have higher number of documents that meets condition
    tmpdf = pd.DataFrame({'total' : top100val.groupby(["qid", "is_condition"]).size()}).reset_index()
    tmpdf = tmpdf[(tmpdf.is_condition == True) & (~tmpdf.qid.isin(qid_exclude))].sort_values(["total"], ascending=False)[:100]
    qids_long = tmpdf.qid.tolist()

    # get the complete data, save it
    top100_fix = top100val[top100val.qid.isin(qids_long)]
    top100_fix.to_csv(f"test_{name}.csv", index=False)
