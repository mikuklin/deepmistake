import os
import json
import argparse
import pandas as pd
import numpy as np
import scipy
import krippendorff
from deepmistake import DeepMistakeWiC

def prepare_dataset(input_dir, chkp_dir):
    for language in os.listdir(f"{input_dir}/ref"):
        labels = pd.read_csv(f"{input_dir}/ref/{language}/labels.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
        uses = pd.read_csv(f"{input_dir}/ref/{language}/uses.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')

        labels = labels.merge(uses.drop(columns=["lemma"]), left_on="identifier1", right_on="identifier").drop(columns=["identifier"])
        labels = labels.merge(uses.drop(columns=["lemma"]), left_on="identifier2", right_on="identifier", suffixes=("_1", "_2")).drop(columns=["identifier"])

        labels["score"] = labels["median_cleaned"]
        labels["sentence1"] = labels["context_1"]
        labels["sentence2"] = labels["context_2"]
        labels["start1"] = labels["indices_target_token_1"].apply(lambda x: int(x.split(":")[0]))
        labels["end1"] = labels["indices_target_token_1"].apply(lambda x: int(x.split(":")[1]))
        labels["start2"] = labels["indices_target_token_2"].apply(lambda x: int(x.split(":")[0]))
        labels["end2"] = labels["indices_target_token_2"].apply(lambda x: int(x.split(":")[1]))
        labels["grp"] = "COMPARE"
        labels["pos"] = "NOUN"
        labels["row"] = labels.index
        labels["tag"] = np.where(labels["score"] > 2, "T", "F")
        labels["id"] = input_dir + ".comedi_" + language + "." + labels["row"].astype(str)
        labels = labels.drop(columns=["identifier1", "identifier2", "judgments", "median_cleaned", "context_1", "context_2", "indices_target_token_1", "indices_target_token_2"])

        data = labels.drop(columns=["tag", "row", "score"])
        gold = labels[["tag", "row", "score", "id"]]
        json_data = data.to_json(orient='records', indent=4)
        json_gold = gold.to_json(orient='records', indent=4)
        if not os.path.exists(f'temp1/{input_dir}/{chkp_dir}'):
            os.makedirs(f'temp1/{input_dir}/{chkp_dir}')
        with open(f'temp1/{input_dir}/{chkp_dir}/{language}.data', 'w+') as file:
             file.write(json_data)
        with open(f'temp1/{input_dir}/{chkp_dir}/{language}.gold', 'w+') as file:
             file.write(json_gold)

def calc_threshold(cosine_sim_train, median_cleaned_train, n=3):
    min_sim = float(min(cosine_sim_train))
    max_sim = float(max(cosine_sim_train))
    delta = (max_sim - min_sim) / (n + 1)
    # initial bins
    bins = [min_sim + delta*(i+1) for i in range(n)]
    
    # loss function
    def min_loss(bins, cos_sim, y):
        bins = sorted([-np.inf] + list(bins) + [np.inf])
        binned_similarities = pd.cut(cos_sim, bins=bins, labels=[1.0, 2.0, 3.0, 4.0])
        y_pred = binned_similarities.tolist()
        y = [float(i) for i in y]
        data = [y, y_pred]
        alpha = krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal")
        return 1 - alpha
    
    # optimizing bin edges
    result = scipy.optimize.minimize(min_loss, bins, args=(cosine_sim_train, median_cleaned_train), method='nelder-mead')
    optimized_bins = sorted([-np.inf] + result.x.tolist() + [np.inf])
    
    return optimized_bins


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepmistake_dir", default=None, type=str, required=True,
                        help="The checkpoint directory")
    parser.add_argument("--input_dir", default=None, type=str, required=True,
                        help="The input directory")
    parser.add_argument("--treshold_dir", default=None, type=str, required=False,
                        help="The directory for treshold")
    parser.add_argument("--mode", type=str, default='2class',
                        choices=['2class', '4class', '2class_treshold'])
    
    
    parsed_args = parser.parse_args()

    prepare_dataset(parsed_args.input_dir, parsed_args.deepmistake_dir)
    dm_model = DeepMistakeWiC(ckpt_dir = parsed_args.deepmistake_dir, device="cuda:0")
    dm_model.predict_dataset(f"temp1/{parsed_args.input_dir}/{parsed_args.deepmistake_dir}", ".", f"temp2/{parsed_args.input_dir}/{parsed_args.deepmistake_dir}")
    if parsed_args.mode == '2class_treshold':
        prepare_dataset(parsed_args.treshold_dir, parsed_args.deepmistake_dir)
        dm_model.predict_dataset(f"temp1/{parsed_args.treshold_dir}/{parsed_args.deepmistake_dir}", ".", f"temp2/{parsed_args.treshold_dir}/{parsed_args.deepmistake_dir}")
    
    if parsed_args.mode == '2class':
        for f in os.listdir(f"temp2/{parsed_args.input_dir}/{parsed_args.deepmistake_dir}"):
            if f.split(".")[-1] == "scores":
                with open(f"temp2/{parsed_args.input_dir}/{parsed_args.deepmistake_dir}/{f}", "r") as json_file:
                    j = json.load(json_file)
                df = pd.DataFrame(j)
                language = df.id[0].split(".")[-2].split("_")[1]
                labels = pd.read_csv(f"{parsed_args.input_dir}/ref/{language}/labels.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
                scores = df.score.apply(lambda x: 1 if (float(x[0]) + float(x[1])) / 2 < 0.5 else 4)
                labels["prediction"] = scores
                labels.to_csv(f"{parsed_args.input_dir}/res/{language}.tsv", sep="\t", index=False)

    elif parsed_args.mode == '2class_treshold':
        d = {}
        for f in os.listdir(f"temp2/{parsed_args.treshold_dir}/{parsed_args.deepmistake_dir}"):
            if f.split(".")[-1] == "scores":
                with open(f"temp2/{parsed_args.treshold_dir}/{parsed_args.deepmistake_dir}/{f}", "r") as json_file:
                    j = json.load(json_file)
                df = pd.DataFrame(j)
                language = df.id[0].split(".")[-2].split("_")[1]
                labels = pd.read_csv(f"{parsed_args.treshold_dir}/ref/{language}/labels.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
                scores = df.score.apply(lambda x: (float(x[0]) + float(x[1])) / 2)
                labels["prediction"] = scores
                d[language] = labels

        bins = {}
        for language in d.keys():
            bins[language] = calc_threshold(d[language].prediction, d[language].median_cleaned)

        for f in os.listdir(f"temp2/{parsed_args.input_dir}/{parsed_args.deepmistake_dir}"):
            if f.split(".")[-1] == "scores":
                with open(f"temp2/{parsed_args.input_dir}/{parsed_args.deepmistake_dir}/{f}", "r") as json_file:
                    j = json.load(json_file)
                df = pd.DataFrame(j)
                language = df.id[0].split(".")[-2].split("_")[1]
                labels = pd.read_csv(f"{parsed_args.input_dir}/ref/{language}/labels.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
                scores = df.score.apply(lambda x: (float(x[0]) + float(x[1])) / 2)
                labels["prediction"] = pd.cut(scores, bins=bins[language], labels=[1.0, 2.0, 3.0, 4.0])
                labels.to_csv(f"{parsed_args.input_dir}/res/{language}.tsv", sep="\t", index=False)

    elif parsed_args.mode == '4class':
        for f in os.listdir(f"temp2/{parsed_args.input_dir}/{parsed_args.deepmistake_dir}"):
            if f.split(".")[-1] == "scores":
                with open(f"temp2/{parsed_args.input_dir}/{parsed_args.deepmistake_dir}/{f}", "r") as json_file:
                    j = json.load(json_file)
                df = pd.DataFrame(j)
                language = df.id[0].split(".")[-2].split("_")[1]
                labels = pd.read_csv(f"{parsed_args.input_dir}/ref/{language}/labels.tsv", sep='\t', encoding='utf-8', quoting=0, quotechar='"')
                def clear(x):
                    return list(filter(lambda x: len(x) > 0, x.replace("[", "").replace("]", "").split(" ")))
                scores = df.score.apply(lambda x: np.argmax(list(map(float, clear(x[0])))) + 1)
                a = list(map(float, clear(df.score[0][0])))
                print(a, np.argmax(a))
                print(scores.value_counts())
                labels["prediction"] = scores
                labels.to_csv(f"{parsed_args.input_dir}/res/{language}.tsv", sep="\t", index=False)    
