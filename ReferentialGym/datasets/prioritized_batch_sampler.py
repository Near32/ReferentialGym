from torch.utils.data.sampler import Sampler
from .prioritized_replay_buffer import PrioritizedReplayBuffer


class PrioritizedBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        capacity (int): Size of the dataset being sampled.
        batch_size (int): Size of mini-batch.
        
    Example:
        >>> list(PrioritizedBatchSampler(capacity=10, batch_size=3))
    """

    def __init__(self, capacity: int, batch_size: int) -> None:
        self.capacity = capacity
        self.pb = PrioritizedReplayBuffer(
            capacity=capacity,
            alpha=1.0,
            beta=0.0,
        )

        initial_prio = 1.0
        for idx in range(capacity):
            self.pb.add(exp=idx, priority=initial_prio)

        self.batch_size = batch_size
    
    def update_batch(self, error_batch) -> None:
        #error_batch = 1/(1e-8+error_batch)
        for idx, err in enumerate(error_batch):
            bidx = self.sample[idx][2]
            self.update(idx=bidx, error=err)

    def update(self, idx, error) -> None:
        priority = self.pb.priority(error)
        self.pb.update(idx=idx, priority=priority)

    def __iter__(self):
        self.sample, importanceSamplingWeights = self.pb.sample(batch_size=self.batch_size)
        batch = [s[2] for s in self.sample]
        return batch
            
    def __len__(self):
        return (self.capacity + self.batch_size - 1) // self.batch_size
