import numpy as np
from torch.utils.data.sampler import Sampler
from .prioritized_replay_buffer import PrioritizedReplayBuffer


class PrioritizedSampler(Sampler):
    r"""

    Args:
        capacity (int): Size of the dataset being sampled.
        batch_size (int): Size of mini-batch.
        
    """

    def __init__(self, capacity: int, batch_size: int, logger: object) -> None:
        self.logger = logger
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
        self.iteration = 0 

    def update_batch(self, error_batch) -> None:
        #error_batch = 1/(1e-8+error_batch)
        for idx, err in enumerate(error_batch):
            bidx = self.sample[idx][0]
            self.update(idx=bidx, error=err)

    def update(self, idx, error) -> None:
        priority = self.pb.priority(error)
        self.pb.update(idx=idx, priority=priority)

    def __iter__(self):
        num_samples =(self.capacity + self.batch_size - 1) // self.batch_size
        for _ in range(num_samples):
            self.sample, importanceSamplingWeights = self.pb.sample(batch_size=self.batch_size)
            batch = [s[2] for s in self.sample]

            if self.logger is not None:
                self.iteration += 1
                self.logger.add_histogram(f"Debug/PrioritizedSampler/Priority/Distribution", np.asarray(batch), self.iteration)
                self.logger.add_scalar(f"Debug/PrioritizedSampler/TotalSum", self.pb.total(), self.iteration)
            
                priorities = [s[1] for s in self.sample]
                values = np.asarray(priorities)
                # (batch_size, )

                averaged_value = values.mean()
                std_value = values.std()
                self.logger.add_scalar(f"Debug/PrioritizedSampler/Priority/Mean", averaged_value, self.iteration)
                self.logger.add_scalar(f"Debug/PrioritizedSampler/Priority/Std", std_value, self.iteration)

                median_value = np.nanpercentile(
                    values,
                    q=50,
                    axis=None,
                    interpolation="nearest"
                )
                q1_value = np.nanpercentile(
                    values,
                    q=25,
                    axis=None,
                    interpolation="lower"
                )
                q3_value = np.nanpercentile(
                    values,
                    q=75,
                    axis=None,
                    interpolation="higher"
                )

                self.logger.add_scalar(f"Debug/PrioritizedSampler/Priority/Median", median_value, self.iteration)
                self.logger.add_scalar(f"Debug/PrioritizedSampler/Priority/Q1", q1_value, self.iteration)
                self.logger.add_scalar(f"Debug/PrioritizedSampler/Priority/Q3", q3_value, self.iteration)

            yield from batch

    def __len__(self):
        #return self.batch_size
        #size =(self.capacity + self.batch_size - 1) // self.batch_size
        size = self.capacity
        return size
