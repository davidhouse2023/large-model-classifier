import json
import random
from math import ceil

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelBinarizer


class InstructionDataset(Dataset):
    def __init__(self, path, tokenizer):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, item in enumerate(f):
                temp = json.loads(item)
                data.append(temp)

        df_data = pd.DataFrame(data)

        # 定义所有可能的标签
        all_labels = ["code","very easy", "easy", "medium", "hard", "very hard"]

        # 初始化 MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=all_labels)

        labels = df_data["labels"]

        # 对标签进行多标签二进制编码
        labels_binarized = mlb.fit_transform(labels)

        # df_data["labels"]=labels_binarized

        self.data=df_data
        self.labels=labels_binarized

        # print(self.data.head())

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction = self.data.iloc[idx]["instruction"]
        label = self.labels[idx]
        inputs = self.tokenizer(instruction, truncation=True, padding="max_length", max_length=512)
        inputs = {
            "input_ids": torch.LongTensor(inputs["input_ids"]),
            "attention_mask": torch.LongTensor(inputs["attention_mask"]),
            "labels": torch.FloatTensor(label)
        }

        return inputs


class myDataset(Dataset):
    def __init__(self, path, tokenizer):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for i, item in enumerate(f):
                temp = json.loads(item)
                data.append(temp)
        self.data=data
        df_data = pd.DataFrame(data)
        # 创建 OneHotEncoder 实例
        lb = LabelBinarizer()

        # 进行 one-hot 编码
        one_hot_labels = lb.fit_transform(df_data["labels"])
        self.labels = one_hot_labels

        self.tokenizer = tokenizer


    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
        instruction = self.data[idx]["instruction"]
        label = self.labels[idx]
        # ['easy' 'hard' 'medium' 'very easy' 'very hard']

        inputs = self.tokenizer(instruction, truncation=True, padding="max_length", max_length=512)
        inputs = {
            "input_ids": torch.LongTensor(inputs["input_ids"]),
            "attention_mask": torch.LongTensor(inputs["attention_mask"]),
            "labels": torch.FloatTensor(label)
        }
        return inputs