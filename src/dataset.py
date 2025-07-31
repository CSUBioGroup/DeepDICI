import torch
from torch.utils.data import Dataset
import numpy as np


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

class CustomDataSet(Dataset):
    def __init__(self, pairs, protein_feature_dict, drug_feature_dict):
        self.pairs = pairs
        self.protein_feature_dict = protein_feature_dict
        self.drug_feature_dict = drug_feature_dict
        self.max_protein_len = 1000
        self.max_drug_len = 100

    def __getitem__(self, item):
        pair = self.pairs[item].strip().split()
        compoundstr, proteinstr, label = pair[-3], pair[-2], pair[-1]
        compound = self.drug_feature_dict.get(compoundstr)
        if compound.shape[0] > self.max_drug_len:
            compound = compound[: self.max_drug_len]
        protein = self.protein_feature_dict[proteinstr].clone().detach().float()
        if protein.shape[0] > self.max_protein_len:
            protein = protein[: self.max_protein_len]
        label = np.int32(float(label))
        return compound, protein, label

    def __len__(self):
        return len(self.pairs)

def collate_fn(batch_data):
    N = len(batch_data)
    compound_max = 100
    compound_feature_dim = 768
    protein_max_len = 1000
    protein_feature_dim = 3072
    compound_new = torch.zeros((N, compound_max, compound_feature_dim), dtype=torch.float)
    protein_new = torch.zeros((N, protein_max_len, protein_feature_dim), dtype=torch.float)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i, (compound, protein, label) in enumerate(batch_data):
        compound_new[i, : min(compound.shape[0], compound_max)] = compound[: min(compound.shape[0], compound_max)]
        protein_new[i, : min(protein.shape[0], protein_max_len)] = protein[: min(protein.shape[0], protein_max_len)]
        labels_new[i] = label
    return compound_new, protein_new, labels_new
