import torch
import numpy as np
import torch.nn.functional as F

from models import Model, RNN_1D, Transformer, RetNet
from hamiltonian import Hamiltonian
from dataloader import DataLoader


def train_data(
    model: Model,
    hamiltonian: Hamiltonian,
    dataloader: DataLoader,
    epochs: int = 10,
    batchsize: int = 100,
    energy_samples: int = 100,
    energy_iterations: int = 50,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    nbatches = len(dataloader) // batchsize
    N = model.N
    energy = []
    standard_deviation = []
    loss_list = []
    i = 0
    for epoch in range(1, epochs + 1):
        dataset = dataloader.run(batchsize, workers=4)
        for samples in dataset:
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

            print(
                fr"epoch: {epoch} | iter: {(i+1)%nbatches}/{nbatches} | energy: {avg_E:.4f} +/- {std_E:.4f} | loss: {loss:0.5f}"
            )
            energy.append(avg_E)
            standard_deviation.append(std_E)
            loss_list.append(loss)
            i += 1
        print()
    print()
    return energy, standard_deviation, loss_list


def train_vmc(
    model: Model,
    hamiltonian: Hamiltonian,
    nsamples: int = 100,
    iterations: int = 1000,
):
    N = model.N
    energy = []
    standard_deviation = []
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

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
        loss = torch.mean(logp * eloc)

        loss.backward()
        optimizer.step()

        print(fr"iter: {i}/{iterations} | energy: {avg_E:.4f} +/- {std_E:.4f} | loss: {loss:0.5f}")

        energy.append(avg_E)
        standard_deviation.append(std_E)
        loss_list.append(loss.item())
        print()
    return energy, standard_deviation, loss_list


if __name__ == '__main__':
    import os

    path = '/home/jkambulo/projects/def-rgmelko/jkambulo/Quantum-Computer-Temperature/data/KZ_Data/12x12'
    for file in os.listdir(path):
        print(file)
        index = 24
        with np.load(os.path.join(path, file)) as file:
            delta_per_omega = file['params'][index]
            omega = file['rabi_freq']
            
        dataloader = DataLoader(os.path.join(path, file), index=index)
        Lx = 12
        Ly = Lx
        h = Hamiltonian(omega=omega, rb_per_a=1.15, delta_per_omega=delta_per_omega, Lx=Lx, Ly=Ly)
        energy_list = []
        standard_deviation_list = []
        loss_list = []
        for repeat in range(30):
            model = RetNet(Lx, Ly)
            energy, standard_deviation, loss = train_data(
                model, h, dataloader, epochs=10, batchsize=100, energy_iterations=1, energy_samples=100
            )
            energy_list.append(energy)
            standard_deviation_list.append(standard_deviation)
            loss_list.append(loss)
        np.savez(
            f'sweepdata_energy/{file.split(".")[0]}_training_data.npz',
            energy=np.array(energy_list),
            standard_deviation=np.array(standard_deviation_list),
            loss_list=np.array(loss_list),
        )

    # vmc
    # Lx = 12
    # Ly = Lx
    # model = RetNet(Lx, Ly, decoder_ffn_embed_dim=200)
    # h = Hamiltonian(omega=1, rb_per_a=1.15, delta_per_omega=1.2, Lx=Lx, Ly=Ly)
    # energy, standard_deviation, loss_list = train_vmc(model, h, nsamples=100, iterations=150)
    # np.savez(
    #     f'vmc_training_data.npz', energy=energy, standard_deviation=standard_deviation, loss_list=loss_list
    # )
