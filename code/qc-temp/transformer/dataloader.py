import torch
from torch.utils.data import Dataset
from torchvision import datasets
import h5py
import json
import os
import numpy as np
from typing import Optional
from itertools import product


class QCTempDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        size: Optional[tuple] = None,
        Rb_per_a: Optional[float] = None,
        delta_per_omega: Optional[float] = None,
        query_beta=None,
        lattice: Optional[str] = None,
    ):
        self.datasets = []
        self.beta = []
        self.energy = []
        self.energy_error = []
        self.ns = []
        self.order_param = []
        for _folder in sorted(os.listdir(data_folder)):
            data_path = os.path.join(data_folder, _folder, "data.jld")
            meta_data_path = os.path.join(data_folder, _folder, "meta_data.json")
            with open(meta_data_path, 'r') as f:
                _dict = json.load(f)
                if (
                    (size and (_dict['nx'] != size[0] or _dict['ny'] != size[1]))
                    or (Rb_per_a and not np.isclose(Rb_per_a, float(_dict["Rb_per_a"])))
                    or (delta_per_omega and not np.isclose(delta_per_omega, float(_dict["Δ_per_Ω"])))
                    or (query_beta and not query_beta(_dict["β"]))
                    or (lattice and lattice != _dict['lattice'])
                ):
                    continue
                self.beta.append(_dict["β"])
                self.energy.append(_dict["energy"])
                self.energy_error.append(_dict["energy_error"])
            file = h5py.File(data_path, "r")
            self.datasets.append(file['occs'])
            self.ns.append(file['ns'])
            self.order_param.append(file['order_param'])

        self.length = sum([d.shape[1] for d in self.datasets])

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        dataset_size = self.length // len(self.datasets)
        dataset_idx = idx // dataset_size
        row_idx = idx % dataset_size
        spins = self.datasets[dataset_idx][:, row_idx]
        beta = self.beta[dataset_idx]
        return spins, beta

    def get_dataset(self, dataset_idx: int):
        return self.datasets[dataset_idx]

    def get_beta(self, dataset_idx: int):
        return self.beta[dataset_idx]

    def get_order_param(self, dataset_idx: int):
        return self.order_param[dataset_idx]

    def get_ns(self, dataset_idx: int):
        return self.ns[dataset_idx]

    def get_nsamples(self, dataset_idx: int = 0):
        return self.datasets[dataset_idx].shape[1]

    def get_natoms(self, dataset_idx: int = 0):
        return self.datasets[dataset_idx].shape[0]


class DataLoader:
    def __init__(
        self,
        dataset: QCTempDataset,
        batchsize: int,
        epochs: int,
        nbatches: int,
        dataset_index: Optional[int] = None,
        seed: int = 1234,
    ):
        np.random.seed(seed)
        self.dataset = dataset
        self.batchsize = batchsize
        self.epochs = epochs
        self.dataset_index = dataset_index
        self.nbatches = nbatches

        self.current_epoch = 0
        self.current_batch = 0
        dset_size = dataset.datasets[0].shape[1]  # assume all datasets are the same size
        if self.dataset_index is None:
            self.dataset_order = np.arange(len(self.dataset.datasets))
            np.random.shuffle(self.dataset_order)
        else:
            self.dataset_order = np.array([self.dataset_index])

        self.shuffle_order = np.arange(dset_size)
        np.random.shuffle(self.shuffle_order)
        self.shuffle_order = np.sort(self.shuffle_order.reshape(-1, self.batchsize), axis=1)

    def __iter__(self):
        for epoch in range(self.epochs):
            # shuffle values in epoch

            self.current_epoch = epoch
            for i in range(self.nbatches * len(self.dataset_order)):
                self.current_batch = i % self.nbatches
                dataset_index = (i // 400) % len(self.dataset_order)
                data = torch.tensor(
                    self.dataset.datasets[dataset_index][:, self.shuffle_order[i, :]],
                    dtype=torch.int64,
                )
                beta = torch.full((1, self.batchsize), self.dataset.beta[dataset_index])
                yield (data, beta)


# with open("/home/jkambulo/projects/def-rgmelko/jkambulo/data/qc-temp/16x16,Rb=1.2,Δ=1.1,β=0.1_data/meta_data.json", 'r') as f:
#     meta_data = json.load(f)
# f = h5py.File("/home/jkambulo/projects/def-rgmelko/jkambulo/data/qc-temp/16x16,Rb=1.2,Δ=1.1,β=0.1_data/data.jld", "r")
