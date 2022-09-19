import torch
from torch.utils.data import BatchSampler, Sampler
from typing import List
import gc

from sklearn import linear_model
import random

# TODO: keep on sampling until error is below threshold
class MemoryRequirementEstimator:
    def __init__(self, samples_required: int = 50):
        self.observations = []
        self.current_features = None
        self.regression = None
        self.samples_required = samples_required

    def observe(self, features, batch_size = 1):
        self.current_features = [f * batch_size for f in features]

    def track(self):
        torch.cuda.synchronize()
        total_memory_use = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        self.observations.append((self.current_features, total_memory_use))
        print("Observation:", self.current_features, total_memory_use)

        if len(self.observations) % self.samples_required == 0:
            if self.regression is None:
                self.regression = linear_model.LinearRegression(positive=True)
            self.regression.fit(*zip(*self.observations))
            print("MemoryRequirementEstimator:", self.regression.intercept_, self.regression.coef_)

    def estimate_single(self, features: List[int]):
        if self.regression is None:
            return None
        return self.regression.predict([features])[0]

    def estimate_batch(self, features: List[int], batch_size: int):
        return self.estimate_single([f * batch_size for f in features])


class MaxMemoryInputAwareBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices based on how much memory
    is required. An instance longer than dataset.max_len or shorter than
    dataset.min_len will be filtered out.
    * no bucketing implemented

    :param sampler: Base sampler. Can be any iterable object
    :param memory_size: Size of VRAM in bytes.
    :param drop_last: If `True`, the sampler will drop the last batch if
            its size would be less than `batch_size`
    """

    def __init__(self, sampler: Sampler, memory_size: int, drop_last: bool):
        super().__init__(sampler, memory_size, drop_last)
        self.estimator = MemoryRequirementEstimator()
        self.batch_size = 1
        self.memory_size = memory_size

    def get_features(self, batch):
        src, trg = batch
        if src is None or trg is None:
            return None

        src_len = len(src)
        trg_len = len(trg)
        return [1, src_len, trg_len, src_len ** 2, trg_len ** 2, src_len * trg_len]

    def __iter__(self):
        batch = []
        max_memory = 0
        max_memory_features = None
        last_memory_estimate = 0
        d = self.sampler.data_source
        for idx in self.sampler:
            src, trg = d[idx]  # call __getitem__()
            if src is None:  # drop instance
                continue

            features = self.get_features((src, trg))

            memory_requirement = self.estimator.estimate_single(features)
            # Before we can batch, we need to be able to predict memory usage
            if memory_requirement is None:
                batch.append(idx)
                if random.random() < 0.5: # sample not only 1-size batches
                    self.estimator.observe(features, len(batch)) # TODO wrong, can be previous features
                    yield batch
                    self.estimator.track()
                    batch = []
                continue

            if memory_requirement > max_memory:
                max_memory = memory_requirement
                max_memory_features = features

            # Check if adding this sample makes the batch too big
            memory_estimate = self.estimator.estimate_batch(max_memory_features, len(batch) + 1)
            if memory_estimate < self.memory_size:
                batch.append(idx)
                last_memory_estimate = memory_estimate
            else:
                print("Batch:", len(batch), last_memory_estimate)
                # Scale features linearly by batch size
                self.estimator.observe(max_memory_features, batch_size=len(batch))
                yield batch
                self.estimator.track()

                # Make sure single example is not too big
                if self.estimator.estimate_batch(features, 1) < self.memory_size:
                    batch = [idx]
                    max_memory = memory_requirement
                    max_memory_features = features
                else:
                    batch = []
                    max_memory = 0
                    max_memory_features = None


        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        raise NotImplementedError
