import torch
import numpy as np


class Hamiltonian:
    # assumes rectangular lattice with consistent lattice spacing
    def __init__(self, omega, rb_per_a, delta_per_omega, Lx, Ly):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.C = 862690 * 2 * np.pi
        self.omega = omega
        self.rb = (self.C / omega) ** (1 / 6)
        self.a = self.rb / rb_per_a
        self.delta = delta_per_omega * omega
        self.Lx, self.Ly = Lx, Ly
        self.atoms = Lx * Ly

    def compute_energy(self, samples: torch.Tensor, compute_logp, return_Hloc: bool = True):
        """Computes the energy via Monte Carlo sampling

        Args:
            samples (torch.Tensor): Sampled spins with shape (batchsize, Lx*Ly)
            compute_logp (_type_): Function that takes in samples and outputs their corresponding logp
            return_Hloc (bool, optional): Whether to return Hloc tensor as supposed to average energy.
                Defaults to True.

        Returns:
            _type_: _description_
        """

        # print(f'c={C} omega={omega} rb={rb} a={a} delta={delta}')

        def compute_interaction(x: torch.Tensor):
            """Computes sum_ij Vij*ni*nj

            Args:
                x (torch.Tensor): tensor of shape (batchsize, Lx*Ly)

            Returns:
                torch.tensor: tensor of shape (batchsize,)
            """
            x = x.T
            c = torch.arange(self.atoms, device=self.device)
            rows = round(self.Ly)
            d_inv_6 = torch.triu(
                (
                    1
                    / (
                        (c[None] % rows - c[:, None] % rows) ** 2
                        + (
                            torch.div(c[None], rows, rounding_mode='floor')
                            - torch.div(c[:, None], rows, rounding_mode='floor')
                        )
                        ** 2
                    )
                    ** 3
                ),
                diagonal=1,
            )
            i, j = torch.triu_indices(self.atoms, self.atoms, offset=1, device=self.device)
            filter = (x[i] == 1) & (x[j] == 1)
            return torch.sum(d_inv_6[i[:, None] * filter, j[:, None] * filter], axis=0)

        def compute_rabi(samples, batchsize=30):
            x = torch.eye(self.atoms, dtype=torch.int32, device=self.device)  # (atoms, atoms)
            flipped_samples = (x ^ samples.unsqueeze(-1)).reshape(
                len(samples) * self.atoms, self.atoms
            )  # (nsamples*atoms, atoms)
            
            # iterating to not compute so much ram
            flipped_logp = torch.zeros(flipped_samples.shape[0], device=self.device)
            for k in range(flipped_samples.shape[0]//batchsize):
                flipped_logp[k*batchsize:(k+1)*batchsize] = compute_logp(flipped_samples[k*batchsize:(k+1)*batchsize])
            sample_logp = compute_logp(samples)

            return torch.sum(
                torch.exp(
                    0.5
                    * (
                        flipped_logp.reshape(len(samples), self.atoms)
                        - sample_logp.unsqueeze(-1)
                    )
                ),
                axis=1,
            )

        detuning = -self.delta * torch.sum(samples, axis=1)
        interaction = self.C * compute_interaction(samples) / self.a**6
        rabi_energy = -abs(self.omega) / 2 * compute_rabi(samples)  # - omega to make wavefunction real

        if return_Hloc:
            return detuning + interaction + rabi_energy
        energy = torch.mean(detuning + interaction + rabi_energy)
        return energy.item()
