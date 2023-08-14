from multiprocessing import Queue, Process, Pool, set_start_method
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
            file = h5py.File(filename, 'r')
            data = file['occs']
        elif filename.endswith('.npz'):
            index = kwargs.get('index', -1)
            file = np.load(filename)
            data = file['rydberg_data'][:, :, index]
            data = data.reshape(-1, data.shape[-1])
        yield data
    finally:
        if file is not None:
            file.close()


class DataLoader:
    def __init__(self, data_file, workers=2,**kwargs):
        self.kwargs = kwargs
        self.workers = workers
        self.data_file = data_file
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with fileloader(self.data_file, **kwargs) as f:
            self.length = f.shape[1]

    def __len__(self):
        return self.length

    def run(self, batchsize: int, seed: int = 1234, max_iterations: int = np.inf):
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

        # try:
        #     set_start_method('spawn')
        # except RuntimeError:
        #     pass
        
        results = Queue()
        requests = Queue()
        # print('run')
        # with Pool(self.workers) as pool:
        #     for i in range(self.workers):
        #         print('start', i)
        #         pool.apply_async(self.worker, args=(self.data_file, batchsize, results, requests, seed), kwds=self.kwargs)
        #         print('finished', i)
        #         requests.put(i)
                
        #     iterations = min(max_iterations, self.length // batchsize)
        #     for batch in range(iterations):
        #         if batch + self.workers < iterations:
        #             requests.put(batch)
        #         yield results.get().to(self.device)

        processes = []
        for i in range(self.workers):
            print('creating workers', i, self.workers)
            processes.append(
                Process(
                    target=self.worker,
                    args=(self.data_file, batchsize, results, requests, seed),
                    kwargs=self.kwargs,
                )
            )
            print('starting')
            processes[-1].start()
            print('started', i)
            requests.put(i)
    
        iterations = min(max_iterations, self.length // batchsize)
        for batch in range(iterations):
            if batch + self.workers < iterations:
                requests.put(batch)
            print('getting data')
            yield results.get().to(self.device)


        for p in processes:
            p.terminate()


    @staticmethod
    def worker(data_file, batchsize, results, requests, seed, **kwargs):
        print('starting worker')
        with fileloader(data_file, **kwargs) as f:
            np.random.seed(seed)
            index = np.arange(f.shape[-1])
            np.random.shuffle(index)

        while True:
            print('In loop')
            i = requests.get()
            print(f"got {i}")
            sorted_index = np.sort(index[i * batchsize : (i + 1) * batchsize])
            with fileloader(data_file, **kwargs) as f:
                print('load')
                val = torch.from_numpy(f[:, sorted_index].T)
            results.put(val)
            print('put')
