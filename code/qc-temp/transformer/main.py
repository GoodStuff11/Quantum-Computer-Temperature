if __name__ == "__main__":
    from transformer_model import EncoderOnlyTransformerModel
    from dataloader import QCTempDataset, DataLoader
    import numpy as np
    import torch

    load = False
    dataset = QCTempDataset(
        "/home/jkambulo/projects/def-rgmelko/jkambulo/data/qc-temp",
        size=(4, 4),
        Rb_per_a=1.2,
        delta_per_omega=1.1,
        lattice='SquareLattice',
        query_beta=lambda x: 0 < x < 1,
    )
    dl = DataLoader(dataset, batchsize=100, epochs=1, nbatches=100_000, dataset_index=None)
    if load:
        transformer = EncoderOnlyTransformerModel.load(
            "/home/jkambulo/projects/def-rgmelko/jkambulo/code/qc-temp/transformer/checkpoints/model_moreparams.pt"
        )
    else:
        transformer = EncoderOnlyTransformerModel(
            atom_grid_shape=(4, 4),
            spin_states=2,
            embedding_size=4,
            nhead=4,
            dim_feedforward=1024,
            nlayers=8,
            dropout=0,
            n_phys_params=1,
        )
    transformer.start_training(
        dl,
        checkpoint_path='./checkpoints/model_trained6.pt',
        data_file='./checkpoints/data_trained6.csv',
        log_interval=100,
    )
