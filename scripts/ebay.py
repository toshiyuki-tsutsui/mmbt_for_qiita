import json, re
from os.path import join
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm


def format_ebay_dataset(dataset_root_path, df_train, df_test):
    print("Parsing data...")
    data = parse_data(dataset_root_path, df_train, df_test)
    data["train"], data["dev"] = train_test_split(
        (data["train"]), test_size=5000, stratify=[x["label"] for x in data["train"]]
    )
    print("Saving everything into format...")
    save_in_format(data, dataset_root_path)


def format_txt_file(content):
    for c in "<>/\\+=-_[]{}'\";:.,()*&^%$#@!~`":
        content = content.replace(c, " ")
    else:
        content = re.sub("\\s\\s+", " ", content)
        return content.lower().replace("\n", " ")


def parse_data(dataset_root_path, df_train, df_test):
    splits = ["train", "test"]
    data = {split: [] for split in splits}
    for split in splits:
        if split == "train":
            for index, row in df_train.iterrows():
                dobj = {}
                dobj["id"] = row.id
                dobj["label"] = row.label
                dobj["text"] = row.text
                dobj["img"] = row.img
                data[split].append(dobj)

        else:
            if split == "test":
                for index, row in df_test.iterrows():
                    dobj = {}
                    dobj["id"] = row.id
                    dobj["label"] = row.label
                    dobj["text"] = row.text
                    dobj["img"] = row.img
                    data[split].append(dobj)

            return data


def save_in_format(data, target_path):
    """
    Stores the data to @target_dir. It does not store metadata.
    """
    for split_name in data:
        jsonl_loc = join(target_path, split_name + ".jsonl")
        with open(jsonl_loc, "w") as (jsonl):
            for sample in tqdm(data[split_name]):
                jsonl.write("%s\n" % json.dumps(sample))
