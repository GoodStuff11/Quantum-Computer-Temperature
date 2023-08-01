import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class VariationalMonteCarlo(tf.keras.Model):
    # Constructor
    def __init__(self, Lx, Ly, V, Omega, delta, num_hidden, learning_rate, seed=1234):
        super(VariationalMonteCarlo, self).__init__()

        """ PARAMETERS """
        self.Lx = Lx  # Size along x
        self.Ly = Ly  # Size along y
        self.V = V  # Van der Waals potential
        self.Omega = Omega  # Rabi frequency
        self.delta = delta  # Detuning

        self.N = Lx * Ly  # Number of spins
        self.nh = num_hidden  # Number of hidden units in the RNN
        self.seed = seed  # Seed of random number generator
        self.K = 2  # Dimension of the local Hilbert space

        # Set the seed of the rng
        # tf.random.set_seed(self.seed)

        # Optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate, epsilon=1e-8)

        # Build the model RNN
        # RNN layer: N -> nh
        self.rnn = tf.keras.layers.GRU(
            self.nh,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            return_sequences=True,
            return_state=True,
            stateful=False,
        )

        # Dense layer: nh - > K
        self.dense = tf.keras.layers.Dense(
            self.K, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )

        # Generate the list of bonds for NN,NNN,NNNN on a
        # square lattice with open boundaries
        self.buildlattice()

    @tf.function
    def sample(self, nsamples):
        # Zero initialization for visible and hidden state
        inputs = 0.0 * tf.one_hot(tf.zeros(shape=[nsamples, 1], dtype=tf.int32), depth=self.K)
        hidden_state = tf.zeros(shape=[nsamples, self.nh])

        logP = tf.zeros(
            shape=[
                nsamples,
            ],
            dtype=tf.float32,
        )

        for j in range(self.N):
            # Run a single RNN cell
            rnn_output, hidden_state = self.rnn(inputs, initial_state=hidden_state)
            # Compute log probabilities
            probs = self.dense(rnn_output)
            log_probs = tf.reshape(tf.math.log(1e-10 + probs), [nsamples, self.K])
            # Sample
            sample = tf.random.categorical(log_probs, num_samples=1)
            if j == 0:
                samples = tf.identity(sample)
            else:
                samples = tf.concat([samples, sample], axis=1)
            # Feed result to the next cell
            inputs = tf.one_hot(sample, depth=self.K)
            # add = tf.reduce_sum(log_probs * tf.reshape(inputs, (nsamples, self.K)), axis=1)

            logP = logP + tf.reduce_sum(log_probs * tf.reshape(inputs, (nsamples, self.K)), axis=1)

        return samples, logP

    @tf.function
    def logpsi(self, samples):
        # Shift data
        num_samples = tf.shape(samples)[0]
        data = tf.one_hot(samples[:, 0 : self.N - 1], depth=self.K)

        x0 = 0.0 * tf.one_hot(tf.zeros(shape=[num_samples, 1], dtype=tf.int32), depth=self.K)
        inputs = tf.concat([x0, data], axis=1)

        hidden_state = tf.zeros(shape=[num_samples, self.nh])
        rnn_output, _ = self.rnn(inputs, initial_state=hidden_state)
        probs = self.dense(rnn_output)

        log_probs = tf.reduce_sum(
            tf.multiply(tf.math.log(1e-10 + probs), tf.one_hot(samples, depth=self.K)), axis=2
        )

        return 0.5 * tf.reduce_sum(log_probs, axis=1)

    # @tf.function
    def localenergy(self, samples, logpsi):
        eloc = tf.zeros(shape=[tf.shape(samples)[0]], dtype=tf.float32)

        # Chemical potential
        for j in range(self.N):
            eloc += -self.delta * tf.cast(samples[:, j], tf.float32)
        # NN
        for n in range(len(self.nn)):
            eloc += self.V * tf.cast(samples[:, self.nn[n][0]] * samples[:, self.nn[n][1]], tf.float32)
        for n in range(len(self.nnn)):
            eloc += (self.V / 8.0) * tf.cast(
                samples[:, self.nnn[n][0]] * samples[:, self.nnn[n][1]], tf.float32
            )
        for n in range(len(self.nnnn)):
            eloc += (self.V / 64.0) * tf.cast(
                samples[:, self.nnnn[n][0]] * samples[:, self.nnnn[n][1]], tf.float32
            )

        # Off-diagonal part
        for j in range(self.N):
            flip_samples = np.copy(samples)
            flip_samples[:, j] = 1 - flip_samples[:, j]
            flip_logpsi = self.logpsi(flip_samples)
            eloc += -0.5 * self.Omega * tf.math.exp(flip_logpsi - logpsi)

        return eloc

    """ Generate the square lattice structures """

    def coord_to_site(self, x, y):
        return self.Ly * x + y

    def buildlattice(self):
        self.nn = []
        self.nnn = []
        self.nnnn = []
        for x in range(self.Lx):
            for y in range(self.Ly - 1):
                self.nn.append([self.coord_to_site(x, y), self.coord_to_site(x, y + 1)])
        for y in range(self.Ly):
            for x in range(self.Lx - 1):
                self.nn.append([self.coord_to_site(x, y), self.coord_to_site(x + 1, y)])

        for y in range(self.Ly - 1):
            for x in range(self.Lx - 1):
                self.nnn.append([self.coord_to_site(x, y), self.coord_to_site(x + 1, y + 1)])
                self.nnn.append([self.coord_to_site(x + 1, y), self.coord_to_site(x, y + 1)])

        for y in range(self.Ly):
            for x in range(self.Lx - 2):
                self.nnnn.append([self.coord_to_site(x, y), self.coord_to_site(x + 2, y)])
        for y in range(self.Ly - 2):
            for x in range(self.Lx):
                self.nnnn.append([self.coord_to_site(x, y), self.coord_to_site(x, y + 2)])


def setup_rnn(Lx, Ly):
    # Hamiltonian parameters
    V = 2.31  # Strength of Van der Waals interaction
    Omega = 1.0  # Rabi frequency
    delta = 1.1  # Detuning

    # RNN-VMC parameters
    lr = 0.001  # learning rate of Adam optimizer
    nh = 32  # Number of hidden units in the GRU cell
    seed = 1234  # Seed of RNG

    vmc = VariationalMonteCarlo(Lx, Ly, V, Omega, delta, nh, lr, seed)
    return vmc


def run_training(
    vmc: VariationalMonteCarlo,
    dataset=None,
    epochs=10,
    batchsize=100,
    iterations=1000,
):
    if dataset is None:
        mode = "vmc"
        epochs = 1
        dataset = (vmc.sample(batchsize)[0] for i in range(iterations))
    else:
        mode = "data"
        dataset = dataset.shuffle(1000).batch(batchsize)
        iterations = len(dataset)

    print('start')
    N = vmc.Lx * vmc.Ly
    energy = []
    standard_deviation = []
    loss_list = []
    for epoch in range(1, epochs + 1):
        for i, samples in enumerate(dataset):
            print(samples)
            print(tf.shape(samples))
            # Evaluate the loss function in AD mode
            with tf.GradientTape() as tape:
                logpsi = vmc.logpsi(samples)
                with tape.stop_recording():
                    eloc = vmc.localenergy(samples, logpsi)
                    Eo = tf.reduce_mean(eloc)

                if mode == 'vmc':
                    loss = tf.reduce_mean(
                        2.0 * tf.multiply(logpsi, tf.stop_gradient(eloc))
                        - 2.0 * tf.stop_gradient(Eo) * logpsi
                    )
                else:
                    # print(tf.shape(logpsi), tf.shape(samples))
                    # print(logpsi, samples)
                    loss = - tf.reduce_mean(
                        tf.multiply(tf.expand_dims(logpsi, axis=-1), tf.cast(samples, tf.float32))
                    )

            # Compute the gradients
            gradients = tape.gradient(loss, vmc.trainable_variables)

            # Update the parameters
            vmc.optimizer.apply_gradients(zip(gradients, vmc.trainable_variables))

            energies = eloc.numpy()
            avg_E = np.mean(energies) / float(N)
            std_E = np.std(energies) / np.sqrt(float(N))

            if i % 50 == 0:
                print(
                    fr"epoch: {epoch} | iter: {i+1}/{iterations} | energy: {avg_E:.4f} +/- {std_E:.4f} | loss: {loss:0.5f}"
                )
            energy.append(avg_E)
            standard_deviation.append(std_E)
            loss_list.append(loss)
    print()
    return energy, standard_deviation, loss_list



def setup_rnn(Lx, Ly):
    # Hamiltonian parameters
    V = 2.31  # Strength of Van der Waals interaction
    Omega = 1.0  # Rabi frequency
    delta = 1.1  # Detuning

    # RNN-VMC parameters
    lr = 0.001  # learning rate of Adam optimizer
    nh = 32  # Number of hidden units in the GRU cell
    seed = 1234  # Seed of RNG

    vmc = VariationalMonteCarlo(Lx, Ly, V, Omega, delta, nh, lr, seed)
    return vmc


def run_training(
    vmc: VariationalMonteCarlo,
    dataset=None,
    epochs=10,
    batchsize=100,
    iterations=1000,
):
    if dataset is None:
        mode = "vmc"
        epochs = 1
        dataset = (vmc.sample(batchsize) for i in range(iterations))
    else:
        mode = "data"
        dataset = dataset.batch(batchsize)
        iterations = len(dataset)
        
    print('start')
    N = vmc.Lx * vmc.Ly
    energy = []
    standard_deviation = []
    loss_list = []
    for epoch in range(1, epochs + 1):
        for i, samples in enumerate(dataset):
            # Evaluate the loss function in AD mode
            with tf.GradientTape() as tape:
                logpsi = vmc.logpsi(samples)
                with tape.stop_recording():
                    eloc = vmc.localenergy(samples, logpsi)
                    Eo = tf.reduce_mean(eloc)

                if mode == 'vmc':
                    loss = tf.reduce_mean(
                        2.0 * tf.multiply(logpsi, tf.stop_gradient(eloc))
                        - 2.0 * tf.stop_gradient(Eo) * logpsi
                    )
                else:
                    print(tf.shape(logpsi), tf.shape(samples))
                    print(logpsi, samples)
                    loss = - tf.reduce_mean(
                        tf.multiply(tf.expand_dims(logpsi, axis=-1), tf.cast(samples, tf.float32))
                    )

            # Compute the gradients
            gradients = tape.gradient(loss, vmc.trainable_variables)

            # Update the parameters
            vmc.optimizer.apply_gradients(zip(gradients, vmc.trainable_variables))

            energies = eloc.numpy()
            avg_E = np.mean(energies) / float(N)
            std_E = np.std(energies) / np.sqrt(float(N))

            print(
                fr"epoch: {epoch} | iter: {i}/{iterations} | energy: {avg_E:.4f} +/- {std_E:.4f} | loss: {loss:0.5f}"
            )
            energy.append(avg_E)
            standard_deviation.append(std_E)
            loss_list.append(loss)
    print()
    return energy, standard_deviation, loss_list


if __name__ == '__main__':
    import numpy as np
    import os

    path = "/home/jkambulo/projects/def-rgmelko/jkambulo/Quantum-Computer-Temperature/data/KZ_Data/12x12/"
    sweep_rates = []
    converged_energy = []
    converged_energy_std = []
    converged_loss = []

    all_loss = []
    all_energy = []
    all_energy_std = []
    
    for file in os.listdir(path):
        data = np.load(os.path.join(path, file))
        Lx, Ly = data['rydberg_data'].shape[:2]
        dataset = tf.data.Dataset.from_tensor_slices(data['rydberg_data'][:, :, 0].reshape(Lx * Ly, -1).T.astype(np.int32))

        sweep_rate = data['sweep_rate']

        vmc = setup_rnn(Lx, Ly)
        energy, E_std, loss = run_training(vmc, epochs=50, batchsize=200)
        i = np.argmin(loss)

        converged_energy.append(energy[i])
        converged_energy_std.append(E_std[i])
        converged_loss.append(loss[i])
        sweep_rates.append(sweep_rate)

        all_loss.append(loss)
        all_energy.append(energy)
        all_energy_std.append(E_std)

    np.savez(
        './1Drnn_12x12_data.npz',
        converged_energy=converged_energy,
        converged_energy_std=converged_energy_std,
        converged_loss=converged_loss,
        all_loss=all_loss,
        all_energy=all_energy,
        all_energy_std=all_energy_std,
    )
