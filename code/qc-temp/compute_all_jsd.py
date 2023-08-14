import numpy as np
import matplotlib.pyplot as plt
from transformer.dataloader import QCTempDataset
from scipy.spatial.distance import jensenshannon
import h5py
import os
import matplotlib 
import multiprocessing
import pandas as pd

def compute_histograms(data):
    natoms = data.shape[1]
    f = lambda x: np.sum(x*(2**np.arange(natoms))[None], axis=1)
    ncomb = 2**natoms
        
    bins = np.arange(ncomb)
    hist = np.histogram(f(data), bins=bins, density=True)[0]  
    yield hist
    nsamples = int(1e7)

    for i in range(10):
        print('-', i)
        sample = np.random.choice(bins[:-1], size=(nsamples,), p=hist)
        resampled_hist, _ = np.histogram(sample, bins=bins, density=True)
        yield resampled_hist
        

# def compute_histograms_dataset(dataset: QCTempDataset, atoms=None):

#     with multiprocessing.Pool(5) as p:
#         hist_list = p.map(compute_histograms, (print(i) or data[:].T for i, data in enumerate(dataset.datasets)))
#     return bins, hist_list

def compute_all_JSD(real_hist, dataset):
    beta_list = dataset.beta
    omega_list = dataset.omega
    delta_per_omega_list = dataset.delta_per_omega
    rb_per_a_list  = dataset.rb_per_a
    
    df = pd.DataFrame(columns=["real-QMC_JSD", "real-QMC_JSD_std","QMC-QMC_JSD", "QMC-QMC_JSD_std"])
    for i, data in enumerate(dataset.datasets): 
        jsd1 = []
        jsd2 = []
        gen = compute_histograms(data[:].T)
        qmc_hist = next(gen)
        jsd1.append(jensenshannon(real_hist, qmc_hist))
        for j, resampled_hist in enumerate(gen):
            print(i, j)
            jsd1.append(jensenshannon(real_hist, resampled_hist))
            jsd2.append(jensenshannon(qmc_hist, resampled_hist))
            del resampled_hist
        df.loc[i, "real-QMC_JSD"] = np.mean(jsd1)
        df.loc[i, "real-QMC_JSD_std"] = np.std(jsd1)
        df.loc[i, "QMC-QMC_JSD"] = np.mean(jsd2)
        df.loc[i, 'QMC-QMC_JSD_std'] = np.std(jsd2)
    
    df['omega'] = omega_list
    df['beta'] = beta_list    
    df['delta_per_omega'] = delta_per_omega_list
    df['beta*omega'] = df['beta']*df['omega']
    df['rb_per_a'] = rb_per_a_list
    
    df.to_csv("complete_phase_JSD.csv")

if __name__ == "__main__":
    # dataset = QCTempDataset("/home/jkambulo/projects/def-rgmelko/jkambulo/data/qc-temp", 
    # size=(4,4), Rb_per_a=1.15, delta_per_omega=1.2)
    dataset = QCTempDataset("/home/jkambulo/projects/def-rgmelko/jkambulo/data/qc-temp", 
                            size=(4,4),
                            # query_beta=lambda x: (x > 2) & (x < 4), 
                            # query_delta_per_omega=lambda x: (x > 1.3) & (x < 1.9)
                            )

 
    path = "/home/jkambulo/projects/def-rgmelko/jkambulo/data/quera_data"
    with h5py.File(os.path.join(path, 'split_acquila.h5'), 'r') as file:
        shots, natoms = file['postSequence/all'].shape
        real_hist = next(compute_histograms(1 - file['postSequence/all'][:]))

    print('hey')
    
    compute_all_JSD(real_hist, dataset)
