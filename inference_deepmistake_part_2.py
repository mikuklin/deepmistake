import os
import json
import argparse
import pandas as pd
import numpy as np
import scipy
import krippendorff
from deepmistake import DeepMistakeWiC
from scipy.stats import spearmanr

def prepare_dataset(input_dir, is_test=False):
    for language in os.listdir(f"{input_dir}/ref"):
        instances = pd.read_csv(f"{input_dir}/ref/{language}/instances.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
        uses = pd.read_csv(f"{input_dir}/ref/{language}/uses.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
        instances = instances.merge(uses.drop(columns=["lemma"]), left_on="identifier1", right_on="identifier").drop(columns=["identifier"])
        instances = instances.merge(uses.drop(columns=["lemma"]), left_on="identifier2", right_on="identifier", suffixes=("_1", "_2")).drop(columns=["identifier"])
        if is_test:
            labels = instances
            labels["mean_disagreement_cleaned"] = 1
            labels["judgments"] = 1
        else:
            labels = pd.read_csv(f"{input_dir}/ref/{language}/labels.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
            labels = labels.merge(uses.drop(columns=["lemma"]), left_on="identifier1", right_on="identifier").drop(columns=["identifier"])
            labels = labels.merge(uses.drop(columns=["lemma"]), left_on="identifier2", right_on="identifier", suffixes=("_1", "_2")).drop(columns=["identifier"])


        labels["score"] =  "1," + labels["mean_disagreement_cleaned"].apply(str)
        labels["sentence1"] = labels["context_1"]
        labels["sentence2"] = labels["context_2"]
        labels["start1"] = labels["indices_target_token_1"].apply(lambda x: int(x.split(":")[0]))
        labels["end1"] = labels["indices_target_token_1"].apply(lambda x: int(x.split(":")[1]))
        labels["start2"] = labels["indices_target_token_2"].apply(lambda x: int(x.split(":")[0]))
        labels["end2"] = labels["indices_target_token_2"].apply(lambda x: int(x.split(":")[1]))
        labels["grp"] = "COMPARE"
        labels["pos"] = "NOUN"
        labels["row"] = labels.index
        labels["tag"] = np.where(labels["mean_disagreement_cleaned"] > 0, "T", "F")
        labels["id"] = input_dir.split("/")[-1] + ".comedi_" + language + "." + labels["row"].astype(str)
        labels = labels.drop(columns=["identifier1", "identifier2", "judgments", "context_1", "context_2", "indices_target_token_1", "indices_target_token_2"])

        data = labels.drop(columns=["tag", "row", "score"])
        gold = labels[["tag", "row", "score", "id"]]
        json_data = data.to_json(orient='records', indent=4)
        json_gold = gold.to_json(orient='records', indent=4)
        if not os.path.exists(f'{input_dir}/temp1'):
            os.makedirs(f'{input_dir}/temp1')
        with open(f'{input_dir}/temp1/{language}.data', 'w+') as file:
             file.write(json_data)
        with open(f'{input_dir}/temp1/{language}.gold', 'w+') as file:
             file.write(json_gold)

def calc_std(a):
    a = np.array(a) / sum(a)
    m = 0
    for i, v in enumerate(a):
        m += (i + 1) * v
    d = 0
    for i, v in enumerate(a):
        d += ((i + 1 - m) ** 2) * v
    return np.sqrt(d)


def calc_pows(probs, disagr, n=4):
    pows = [1 for i in range(n)]
    
    def min_loss(pows, probs, y):
        new_probs = []
        for i in range(len(probs)):
            new_probs.append([])
            for j in range(n):
                new_probs[-1].append(probs[i][j] ** pows[j])
        stds = [calc_std(new_probs[i]) for i in range(len(new_probs))]
        y = [float(i) for i in y]
        sp, _ = spearmanr(stds, y)
        print(sp)
        if sp < 0:
            print(pows)
        return 1 - sp
    
    # optimizing bin edges
    result = scipy.optimize.minimize(min_loss, pows, args=(probs, disagr), method='nelder-mead')
    optimized_pows = result.x.tolist()
    
    return optimized_pows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepmistake_dir", default=None, type=str, required=True,
                        help="The checkpoint directory")
    parser.add_argument("--input_dir", default=None, type=str, required=True,
                        help="The input directory")
    parser.add_argument("--treshold_dir", default=None, type=str, required=False,
                        help="The directory for treshold")
    parser.add_argument("--mode", type=str, default='2class',
                        choices=['2class', 'mse', '4class_pows', '4class'])
    
    
    parsed_args = parser.parse_args()

    prepare_dataset(parsed_args.input_dir, True)
    dm_model = DeepMistakeWiC(ckpt_dir = parsed_args.deepmistake_dir, device="cuda:4")
    print(f"{parsed_args.input_dir}")
    dm_model.predict_dataset(f"{parsed_args.input_dir}/temp1", ".", f"{parsed_args.input_dir}/temp2")
    if parsed_args.mode == '4class_pows':
        prepare_dataset(parsed_args.treshold_dir)
        dm_model.predict_dataset(f"{parsed_args.treshold_dir}/temp1", ".", f"{parsed_args.treshold_dir}/temp2")

    if parsed_args.mode == '4class':
        for f in os.listdir(f"{parsed_args.input_dir}/temp2"):
            if f.split(".")[-1] == "scores":
                with open(f"{parsed_args.input_dir}/temp2/{f}", "r") as json_file:
                    j = json.load(json_file)
                df = pd.DataFrame(j)
                language = df.id[0].split(".")[-2].split("_")[1]
                instances = pd.read_csv(f"{parsed_args.input_dir}/ref/{language}/instances.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
                def clear(x):
                    return list(map(float, filter(lambda x: len(x) > 0, x.replace("[", "").replace("]", "").split(" "))))
                def calc_std(a):
                    a = np.array(a)
                    m = 0
                    for i, v in enumerate(a):
                        m += (i + 1) * v
                    d = 0
                    for i, v in enumerate(a):
                        d += ((i + 1 - m) ** 2) * v
                    return np.sqrt(d)
                scores = df.score.apply(lambda x: calc_std((np.array(clear(x[0])) + np.array(clear(x[1])))/2))
                instances["prediction"] = scores
                instances.to_csv(f"{parsed_args.input_dir}/res/{language}.tsv", sep="\t", index=False)

    elif parsed_args.mode == '4class_part_pows':
        d = {}
        for f in os.listdir(f"{parsed_args.treshold_dir}/temp2"):
            if f.split(".")[-1] == "scores":
                with open(f"{parsed_args.treshold_dir}/temp2/{f}", "r") as json_file:
                    j = json.load(json_file)
                df = pd.DataFrame(j)
                language = df.id[0].split(".")[-2].split("_")[1]
                labels = pd.read_csv(f"{parsed_args.treshold_dir}/ref/{language}/labels.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
                def clear(x):
                    return list(map(float, filter(lambda x: len(x) > 0, x.replace("[", "").replace("]", "").split(" "))))
                scores = df.score.apply(lambda x: (np.array(clear(x[0])) + np.array(clear(x[1])))/2)
                labels["prediction"] = scores
                d[language] = labels

        pows = {}
        for language in d.keys():
            pows[language] = calc_pows(d[language].prediction, d[language].mean_disagreement_cleaned)
        for f in os.listdir(f"{parsed_args.input_dir}/temp2"):
            if f.split(".")[-1] == "scores":
                with open(f"{parsed_args.input_dir}/temp2/{f}", "r") as json_file:
                    j = json.load(json_file)
                df = pd.DataFrame(j)
                language = df.id[0].split(".")[-2].split("_")[1]
                instances = pd.read_csv(f"{parsed_args.input_dir}/ref/{language}/instances.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
                def clear(x):
                    return list(map(float, filter(lambda x: len(x) > 0, x.replace("[", "").replace("]", "").split(" "))))
                def calc_std_with_pows(a, language):
                    a = np.array(a)
                    for i, v in enumerate(a):
                        a[i] = a[i] ** pows[language][i]
                    a = a / a.sum()
                    m = 0
                    for i, v in enumerate(a):
                        m += (i + 1) * v
                    d = 0
                    for i, v in enumerate(a):
                        d += ((i + 1 - m) ** 2) * v
                    return np.sqrt(d)
                scores = df.score.apply(lambda x: calc_std_with_pows((np.array(clear(x[0])) + np.array(clear(x[1])))/2, language))
                instances["prediction"] = scores
                instances.to_csv(f"{parsed_args.input_dir}/res/{language}.tsv", sep="\t", index=False)

    elif parsed_args.mode == '2class' or parsed_args.mode == 'mse':
        for f in os.listdir(f"{parsed_args.input_dir}/temp2"):
            if f.split(".")[-1] == "scores":
                with open(f"{parsed_args.input_dir}/temp2/{f}", "r") as json_file:
                    j = json.load(json_file)
                df = pd.DataFrame(j)
                language = df.id[0].split(".")[-2].split("_")[1]
                instances = pd.read_csv(f"{parsed_args.input_dir}/ref/{language}/instances.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
                scores = df.score.apply(lambda x: (float(x[0]) + float(x[1])) / 2 )
                instances["prediction"] = scores
                instances.to_csv(f"{parsed_args.input_dir}/res/{language}.tsv", sep="\t", index=False)
