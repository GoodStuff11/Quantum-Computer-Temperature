if __name__ == "__main__":
    import sys
    label = sys.argv[1]
    
    import os
    import torch
    import numpy as np
    import torch.nn.functional as F

    from models import RNN_1D, Transformer, RetNet
    from hamiltonian import Hamiltonian
    from dataloader import DataLoader

    from training import train_data, train_vmc
    import json
    
    basefolder = "/home/jkambulo/projects/def-rgmelko/jkambulo/data/qc-temp/"
    folder = "4x4,Rb=1.20,Δ=1.10,β=0.100000_data" #"4x4,Rb=1.20,Δ=1.10,β=10.000_data"
    # path = "/home/jkambulo/projects/def-rgmelko/jkambulo/data/qc-temp/16x16,Rb=1.2,Δ=1.1,β=0.503448275862069_data"
    with open(os.path.join(basefolder, folder, "meta_data.json")) as f:
        dic = json.load(f)
        Lx = dic['nx']
        Ly = dic['ny']
        energy = dic['energy']/(Lx*Ly)
        rb_per_a = dic['Rb_per_a']
        delta_per_omega = dic['Δ_per_Ω']
        omega = dic['Ω']
        print(omega, delta_per_omega, rb_per_a, energy, Lx, Ly)
        
    dataloader = DataLoader(os.path.join(basefolder, folder, 'data.jld'), workers=6)
    h = Hamiltonian(omega=omega, rb_per_a=rb_per_a, delta_per_omega=delta_per_omega, Lx=Lx, Ly=Ly)
    model = RetNet(Lx, Ly, decoder_ffn_embed_dim=200, decoder_layers=3, nheads=3, embedding_dim=12)
    # model = RNN_1D(Lx, Ly, hidden_size=104, model_type='lstm')
    energy, standard_deviation, loss = train_vmc(model, h, nsamples=300, iterations=200, lr=0.0001)
    energy, standard_deviation, loss = train_data(
        model,
        h,
        dataloader,
        epochs=1,
        batchsize=100,
        energy_iterations=1,
        energy_samples=300,
        max_iterations=20_000, 
    )
    # energy.extend(_energy)
    # standard_deviation.extend(_standard_deviation)
    # loss.extend(_loss)

    np.savez(
        f'sweepdata_energy/{folder}_{label}.npz',
        energy=np.array(energy),
        standard_deviation=np.array(standard_deviation),
        loss_list=np.array(loss),
    )

    # vmc
    # path = '/home/jkambulo/projects/def-rgmelko/jkambulo/data/KZ_Data/12x12'
    # for filename in os.listdir(path):
    #     print(filename)
    #     with np.load(os.path.join(path, filename)) as file:
    #         delta_per_omega_array = file['params']
    #         index = (np.abs(delta_per_omega_array - 1.2)).argmin()
    #         delta_per_omega = delta_per_omega_array[index]
    #         omega = file['rabi_freq']
    #     Lx = 12
    #     Ly = Lx
    #     model = RetNet(Lx, Ly, decoder_ffn_embed_dim=300)
    #     # model = RNN_1D(Lx, Ly, 130, model_type='rnn')
    #     # h = Hamiltonian(omega=omega, rb_per_a=1.15, delta_per_omega=delta_per_omega, Lx=Lx, Ly=Ly)
    #     h = Hamiltonian(omega=1, rb_per_a=1.15, delta_per_omega=1.2, Lx=Lx, Ly=Ly)
    #     energy, standard_deviation, loss_list = train_vmc(model, h, nsamples=20, iterations=150)
    #     np.savez(
    #         f'vmc_training_data.npz', energy=energy, standard_deviation=standard_deviation, loss_list=loss_list
    #     )
    #     break
