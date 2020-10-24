from torch.utils.data import Sampler
import numpy as np 
class VQA_Sampler(Sampler):
    def __init__(self, source_dt, max_batch_number, batch_size, train, batch_st=None, epoch=None):
        # super(mySampler, self).__init__()
        self.batch_size = batch_size
        self.data = source_dt
        self.data_cnt = len(source_dt)
        self.train = train
        if train:
            if epoch != None:
                self.max_batch_number = int(self.data_cnt * epoch / self.batch_size)
            else:
                self.max_batch_number = max_batch_number
        else:
            assert epoch == None
            val_batch_num = self.data_cnt // self.batch_size
            if self.data_cnt % self.batch_size != 0:
                val_batch_num += 1
            self.max_batch_number = val_batch_num
        if batch_st:
            self.batch_st = batch_st
        else:
            self.batch_st = 0

    def __len__(self):
        return self.max_batch_number

    def __iter__(self):
        batch_cnt = 0
        epoch_cnt = 0
        indices = [i for i in range(self.data_cnt)]
        epoch_indices = []
        batch_size = self.batch_size
        seed = 1333

        while batch_cnt < self.max_batch_number:
            while len(epoch_indices) < batch_size:
                if self.train:
                    np.random.seed(epoch_cnt+seed)
                    epoch_indices = epoch_indices + np.random.permutation(indices).tolist()
                else:
                    epoch_indices = epoch_indices + indices
                    # if len(epoch_indices) == 0:
                    #     epoch_indices = epoch_indices + indices
                    # else:
                    #     break
                #     epoch_indices = epoch_indices + indices
                epoch_cnt += 1
            batch = epoch_indices[:batch_size]
            epoch_indices = epoch_indices[batch_size:]
            if batch_cnt >= self.batch_st:
                yield batch
            batch_cnt += 1