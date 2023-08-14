import torch
import numpy as np
import torch.nn.functional as F

from models import Model, RNN_1D, Transformer, RetNet
from hamiltonian import Hamiltonian
from dataloader import DataLoader
import time

def train_data(
    model: Model,
    hamiltonian: Hamiltonian,
    dataloader: DataLoader,
    epochs: int = 10,
    batchsize: int = 100,
    energy_samples: int = 100,
    energy_iterations: int = 50,
    lr: float = 0.001,
    max_iterations: int = np.inf,
) -> tuple:
    """_summary_

    Args:
        model (Model): _description_
        hamiltonian (Hamiltonian): _description_
        dataloader (DataLoader, optional): _description_. Defaults to None.
        epochs (int, optional): _description_. Defaults to 10.
        batchsize (int, optional): _description_. Defaults to 100.
        iterations (int, optional): _description_. Defaults to 1000.

    Returns:
        tuple: _description_
    """
    t = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    nbatches = min(len(dataloader) // batchsize, max_iterations)
    N = model.N
    energy = []
    standard_deviation = []
    loss_list = []
    i = 0
    print('test')
    for epoch in range(1, epochs + 1):
        print("epoch", epoch)
        dataset = dataloader.run(batchsize, max_iterations=max_iterations)
        for samples in dataset:
            samples = samples.type(torch.int64)
            optimizer.zero_grad()

            # Evaluate the loss function in AD mode
            logp = model(samples)
            if i % energy_iterations == energy_iterations - 1:
                with torch.no_grad():
                    generated_samples = model.sample(energy_samples)
                    eloc = hamiltonian.compute_energy(generated_samples, model.logp)
                avg_E = torch.mean(eloc).item() / float(N)
                std_E = torch.std(eloc).item() / np.sqrt(float(N))
            else:
                avg_E = np.nan
                std_E = np.nan

            loss = -torch.mean(logp * F.one_hot(samples, num_classes=2))

            # Update the parameters
            loss.backward()
            optimizer.step()
            with open('logs.txt', 'a+') as f:
                f.write(
                    f"epoch: {epoch} | iter: {i%nbatches + 1}/{nbatches} | energy: {avg_E:.4f} +/- {std_E:.4f} | loss: {loss:0.5f} | time: {time.time()-t:.2f}\n"
                )
                print(
                    fr"epoch: {epoch} | iter: {i%nbatches + 1}/{nbatches} | energy: {avg_E:.4f} +/- {std_E:.4f} | loss: {loss:0.5f} | time: {time.time()-t:.2f}"
                )
            energy.append(avg_E)
            standard_deviation.append(std_E)
            loss_list.append(loss.item())
            i += 1

        print()
    print()
    return energy, standard_deviation, loss_list


def train_vmc(
    model: Model,
    hamiltonian: Hamiltonian,
    nsamples: int = 100,
    iterations: int = 1000,
    lr: float = 0.001,
):
    t = time.time()
    N = model.N
    energy = []
    standard_deviation = []
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    for i in range(iterations):
        optimizer.zero_grad()
        with torch.no_grad():
            samples = model.sample(nsamples)
        logp = model.logp(samples)

        with torch.no_grad():
            eloc = hamiltonian.compute_energy(samples, model.logp)
            Eo = torch.mean(eloc)
            avg_E = torch.mean(eloc).item() / float(N)
            std_E = torch.std(eloc).item() / np.sqrt(float(N))
        loss = torch.mean(logp * (eloc - Eo))

        loss.backward()
        optimizer.step()

        with open('logs_vmc.log', 'a+') as f:
            f.write(f"iter: {i}/{iterations} | energy: {avg_E:.4f} +/- {std_E:.4f} | loss: {loss:0.5f} | time: {time.time()-t:.2f}\n")
            print(fr"iter: {i}/{iterations} | energy: {avg_E:.4f} +/- {std_E:.4f} | loss: {loss:0.5f} | time: {time.time()-t:.2f}")

        energy.append(avg_E)
        standard_deviation.append(std_E)
        loss_list.append(loss.item())
    print()
    return energy, standard_deviation, loss_list


if __name__ == '__main__':
    import os
    import sys
    
    label = sys.argv[1]
    
    # path = '/home/jkambulo/projects/def-rgmelko/jkambulo/data/KZ_Data/12x12'
    # for filename in os.listdir(path):
    #     print(filename)
    #     with np.load(os.path.join(path, filename)) as file:
    #         delta_per_omega_array = file['params']
    #         index = (np.abs(delta_per_omega_array - 1.2)).argmin()
    #         delta_per_omega = delta_per_omega_array[index]
    #         omega = file['rabi_freq']

    #     dataloader = DataLoader(os.path.join(path, filename), index=index)
    #     Lx = 12
    #     Ly = Lx
    #     h = Hamiltonian(omega=omega, rb_per_a=1.15, delta_per_omega=delta_per_omega, Lx=Lx, Ly=Ly)
    #     energy_list = []
    #     standard_deviation_list = []
    #     loss_list = []
    #     for repeat in range(30):
    #         # model = RNN_1D(Lx, Ly, hidden_size=32)
    #         model = RetNet(Lx, Ly, decoder_ffn_embed_dim=200, decoder_layers=3, nheads=3, embedding_dim=12)
    #         energy, standard_deviation, loss = train_data(
    #             model, h, dataloader, epochs=2, batchsize=20, energy_iterations=1, energy_samples=100
    #         )
    #         energy_list.append(energy)
    #         standard_deviation_list.append(standard_deviation)
    #         loss_list.append(loss)
    #     np.savez(
    #         f'sweepdata_energy/{filename.split(".")[0]}_{label}.npz',
    #         energy=np.array(energy_list),
    #         standard_deviation=np.array(standard_deviation_list),
    #         loss_list=np.array(loss_list),
    #     )

    # vmc
    path = '/home/jkambulo/projects/def-rgmelko/jkambulo/data/KZ_Data/16x16'
    for filename in os.listdir(path):
        print(filename)
        with np.load(os.path.join(path, filename)) as file:
            delta_per_omega_array = file['params']
            index = (np.abs(delta_per_omega_array - 1.2)).argmin()
            delta_per_omega = delta_per_omega_array[index]
            omega = file['rabi_freq']
        Lx = 16
        Ly = Lx
        model = RetNet(Lx, Ly, decoder_ffn_embed_dim=300)
        # model = RNN_1D(Lx, Ly, 32, model_type='rnn')
        # dataloader = DataLoader(os.path.join(path, filename), index=index)
        h = Hamiltonian(omega=omega, rb_per_a=1.15, delta_per_omega=delta_per_omega, Lx=Lx, Ly=Ly)
        # energy, standard_deviation, loss = train_data(
        #     model, h, dataloader, epochs=2, batchsize=20, energy_iterations=1, energy_samples=100
        # )
        energy, standard_deviation, loss_list = train_vmc(model, h, nsamples=20, iterations=10000)
        # energy.extend(_energy)
        # standard_deviation.extend(_standard_deviation)
        # loss.extend(_loss_list)
        np.savez(
            f'sweepdata_energy/{filename.split(".")[0]}_{label}.npz',
            energy=np.array(energy),
            standard_deviation=np.array(standard_deviation),
            loss_list=np.array(loss_list),
        )
        break
