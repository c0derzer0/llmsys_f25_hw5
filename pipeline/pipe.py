from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module
from pipeline import partition
import time

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN ASSIGN5_2_1
    # 
    for k in range(num_batches + num_partitions - 1): # k = 0
        current_clock_cycle = [] 
        for j in range(num_partitions):  # 1
            i = k-j  # 0-1
            if 0 <= i < num_batches:
                current_clock_cycle.append((i, j)) # (0,0)
        yield current_clock_cycle

    # END ASSIGN5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)
        #print("self.partitions, self.devices", self.partitions, self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN ASSIGN5_2_2
        micro_batches = list(torch.split(x, self.split_size, dim=0))
        # print("split_size:", self.split_size, "num_micro_batches:", len(micro_batches), 
        #     "shapes:", [b.shape[0] for b in micro_batches])
        schedules =  _clock_cycles(len(micro_batches), len(self.partitions))
        for k, schedule in enumerate(schedules):
            # if k < 6: print("clock", k, "schedule", schedule)
            # t0 = time.time()
            self.compute(micro_batches, schedule)
            # if k < 6:
            #     print(f"clock {k} dt={time.time()-t0:.4f}s")
        output = torch.cat(micro_batches, dim=0).to(device=self.devices[-1])

        return output
        
        # END ASSIGN5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN ASSIGN5_2_2
        for micro_batch_idx, partition_idx in schedule:
            partition = partitions[partition_idx]
            micro_batch = batches[micro_batch_idx] #.to(devices[partition_idx])
            device = devices[partition_idx]
            #print(f"ENQ mb={micro_batch_idx} part={partition_idx} dev={devices[partition_idx]}")
            self.in_queues[partition_idx].put(Task(lambda p=partition, mb=micro_batch, dev=device: p(mb.to(dev))))
            
        for micro_batch_idx, partition_idx in schedule:
            result = self.out_queues[partition_idx].get()
            #print(f"DEQ mb={micro_batch_idx} part={partition_idx} dev={devices[partition_idx]}")
            if result[0]:
                batches[micro_batch_idx] = result[1][1]
        # END ASSIGN5_2_2

