import numpy as np
import torch
import pandas as pd
import os

from transformer_model import EncoderOnlyTransformerModel, compute_energy, int_to_binary, compute_all_probabilities
from dataloader import QCTempDataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity


def train_model_on_beta(model_label: str, beta_condition, load:bool=False):
    if load:
        transformer = EncoderOnlyTransformerModel.load(
            f"/home/jkambulo/projects/def-rgmelko/jkambulo/data/transformer_beta/model_{model_label}.pt"
        )
    else:
        dataset = QCTempDataset(
            "/home/jkambulo/projects/def-rgmelko/jkambulo/data/qc-temp",
            size=(4, 4),
            Rb_per_a=1.2,
            delta_per_omega=1.1,
            lattice='SquareLattice',
            query_beta=beta_condition,
        )
        dl = DataLoader(dataset, batchsize=100, epochs=1, nbatches=10_000, dataset_index=None)
        transformer = EncoderOnlyTransformerModel(
            atom_grid_shape=(4, 4),
            spin_states=2,
            embedding_size=4,
            nhead=4,
            dim_feedforward=1024,
            nlayers=5,
            dropout=0,
            n_phys_params=1,
        )
        transformer.start_training(
            dl,
            checkpoint_path=f'/home/jkambulo/projects/def-rgmelko/jkambulo/data/transformer_beta/model_trained_{model_label}.pt',
            data_file=f'/home/jkambulo/projects/def-rgmelko/jkambulo/data/transformer_beta/model_logs_{model_label}.txt',
            log_interval=100,
        )
    return transformer

def all_transformer_energy(transformer: EncoderOnlyTransformerModel):
    delta = 1.1
    rb = 1.2

    dataset = QCTempDataset("/home/jkambulo/projects/def-rgmelko/jkambulo/data/qc-temp",
                        size=(4,4), Rb_per_a=rb, delta_per_omega=delta, lattice='SquareLattice')  
    

    # print(len(dataset.beta))
    natoms = 16
    f = lambda x: np.sum(x*(2**np.arange(natoms))[None], axis=1)

    i_values = len(dataset.datasets)
    energy_values = np.zeros((i_values, 3)) # 0 = transformer, 1 = data, 2 = exact
    beta = dataset.beta
        
    for i in range(i_values):
        data = dataset.datasets[i][:].T
        val, _ = np.histogram(f(data), bins=np.arange(2**natoms+1))
        logp = torch.from_numpy(np.log(val/np.sum(val)))
        total_configs = int_to_binary(torch.arange(2**natoms), natoms)
        energy_values[i,1] = compute_energy(total_configs, logp, {'omega':25.132741228718345, 'delta_per_omega':1.1, 'rb_per_a':1.2},atoms=natoms)
        energy_values[i,2] = dataset.energy[i]

        total_configs, prob = compute_all_probabilities(transformer,phys_params=torch.tensor([beta[i]]))
        energy_values[i,0] = compute_energy(total_configs, prob, {'omega':25.132741228718345, 'delta_per_omega':1.1, 'rb_per_a':1.2})

    return beta, energy_values

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', nargs=2, type=int)
    args = parser.parse_args()
    
    k, tot = args.split
    
    path = "/home/jkambulo/projects/def-rgmelko/jkambulo/data/transformer_beta"
    iteration = [(lambda x: x == 1,"beta=1"),
                 (lambda x: x==0.1,"beta=1e-1"),
                 (lambda x: x==0.00001,"beta=1e-5"),
                 (lambda x: 0.011<=x<=0.012,"beta=11e-3"),
                 (lambda x: (0.00010 <= x <= 0.00012),"beta=1e-4"),
                 (lambda x: x == 0.00001 or x == 1,"beta=1e-5,1"),
                 (lambda x: x == 0.00001 or x == 1 or (0.011 <= x <= 0.012), "beta=1e-5,11e-3,1"),
                 ]

    for i, (beta_condition, model_label) in enumerate(iteration):
        if i % tot != k:
            continue

        transformer = train_model_on_beta(model_label, beta_condition)
        beta, energy_values = all_transformer_energy(transformer)
        
        df = pd.DataFrame({'beta':beta, "transformer": energy_values[:,0],"qmc":energy_values[:,1], "exact":energy_values[:,2]})
        df.to_csv(os.path.join(path, f'model_data_{model_label}.csv'))
