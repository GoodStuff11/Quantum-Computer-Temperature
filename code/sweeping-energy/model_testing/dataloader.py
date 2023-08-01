from multiprocessing import Queue, Process
import h5py
import numpy as np
import torch
import pandas as pd
from contextlib import contextmanager


@contextmanager
def fileloader(filename, **kwargs):
    file = None
    data = None
    try:
        if filename.endswith('.jld') or filename.endswith(".h5"):
            file = h5py.File(filename, 'r')['occs']
        elif filename.endswith('.npz'):
            if kwargs is None:
                index = -1
            else:
                index = kwargs['index']
            file = np.load(filename)
            data = file['rydberg_data'][:, :, index]
            data = data.reshape(-1, data.shape[-1])
        yield file if data is None else data
    finally:
        if file is not None:
            file.close()


class DataLoader:
    def __init__(self, data_file, **kwargs):
        self.data_file = data_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with fileloader(self.data_file, **kwargs) as f:
            self.length = f.shape[1]
        
    def __len__(self):
        return self.length

    def run(self, batchsize: int, workers: int = 2, seed: int = 1234, max_iterations: int = np.inf):
        """Generator that goes through all data, shuffles and returns

        Args:
            batchsize (int): Batch size to output
            workers (int, optional): Number of workers to use when reading data, increasing
                will make more take place in the background. Defaults to 2.
            seed (int, optional): Seed to use for random number generation. Defaults to 1234.
            max_iterations (int, optional): Max batches to output before finishing the iteration.
                When -1, goes until all data is read. Defaults to -1.

        Yields:
            torch.tensor: batch with shape (batchsize, columns)
        """
        results = Queue()
        requests = Queue()
        processes = []
        for i in range(workers):
            processes.append(
                Process(target=self.worker, args=(self.data_file, batchsize, results, requests, seed))
            )
            processes[-1].start()
            requests.put(i)

        iterations = min(max_iterations, self.length//batchsize)
        for batch in range(iterations):
            if batch + workers < iterations:
                requests.put(batch)
            yield results.get().to(self.device)

        for p in processes:
            p.terminate()

    @staticmethod
    def worker(data_file, batchsize, results, requests, seed):
        with fileloader(data_file) as f:
            np.random.seed(seed)
            index = np.arange(f.shape[-1])
            np.random.shuffle(index)

        while True:
            i = requests.get()
            sorted_index = np.sort(index[i * batchsize : (i + 1) * batchsize])
            with fileloader(data_file) as f:
                val = torch.from_numpy(f[:, sorted_index].T)
            results.put(val)
