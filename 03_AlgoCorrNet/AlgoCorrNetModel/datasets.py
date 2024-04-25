


# standard library imports
# None

# related third-party
import torch

# local application/library specific imports
from AlgoCorrNetModel.utils import encode


class SiameseInputTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        triplet = self.df.iloc[idx]
        premise = encode(triplet['premise'])
        hypothesis = encode(triplet['hypothesis'])
        label = int(triplet['label'])

        return premise, hypothesis, label

    def __len__(self):
        return self.df.shape[0]


class TripleInputTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        triplet = self.df.iloc[idx]
        anchor = encode(triplet['anchor'])
        positive = encode(triplet['positive'])
        negative = encode(triplet['negative'])

        return anchor, positive, negative

    def __len__(self):
        return self.df.shape[0]


class CorrelationEvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        triplet = self.df.iloc[idx]
        statement = encode(triplet['statement'])
        editorial = encode(triplet['editorial'])
        label = int(triplet['label'])
        return statement, editorial, label

    def __len__(self):
        return self.df.shape[0]


class PKEvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        statement = encode(row.iloc[0])
        editorials = row.iloc[1:]

        editorials_list = []
        for editorial in editorials:
            editorials_list.append(encode(editorial))

        return statement, editorials_list

    def __len__(self):
        return self.df.shape[0]


class EditorialIndexDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        editorial_index = idx
        editorial = encode(row['editorial'])
        return editorial, editorial_index

    def __len__(self):
        return self.df.shape[0]
