# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Custom implementation of a PyTorch IterableDataset for reading tar files in webdataset format
in a strictly sequential way.

This implementation assumes that:
- The tar files contain shuffle samples.
- The user knows in advance the different files for each sample.
"""

import json
import logging
import os
import random
import tarfile
from dataclasses import dataclass
from glob import glob
from itertools import islice
from typing import IO, Any, Callable, Dict, Generator, List, Mapping, Optional

import numpy as np
import torch
import torch.distributed
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

StateDict = Dict[str, Any]
Sample = Dict[str, Any]

SAMPLE_MAP_FN = {
    "npy": lambda x: np.load(x),
    "pil": lambda x: Image.open(x).convert("RGB"),
    "json": lambda x: json.load(x),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG"}
VIDEO_EXTENSIONS = {".mp4", ".webm"}

logger = logging.getLogger("franca")


def expand_path(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    if os.path.exists(path):
        path = os.path.realpath(path)
    return path


@dataclass
class DistributedInfo:
    """DDP environment information."""

    rank: int
    world_size: int
    local_rank: int
    distributed: bool
    num_nodes: int
    gpu_per_node: int
    node_rank: int


def is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_distributed_info() -> DistributedInfo:
    """
    Get distributed information.

    Returns
    -------
        Distributed information
    """
    if not is_distributed():
        return DistributedInfo(
            rank=0,
            world_size=1,
            local_rank=0,
            distributed=False,
            num_nodes=1,
            gpu_per_node=1,
            node_rank=0,
        )

    gpu_per_node = torch.cuda.device_count()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = rank % gpu_per_node
    num_nodes = world_size // gpu_per_node
    node_rank = rank // gpu_per_node

    return DistributedInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        distributed=True,
        num_nodes=num_nodes,
        gpu_per_node=gpu_per_node,
        node_rank=node_rank,
    )


def distribute_shards(
    shards: List[str],
    num_nodes: int = None,
    num_gpus_per_node: int = None,
    resample: bool = False,
) -> List[List[List[str]]]:
    """
    Distribute shards among workers and nodes.

    Parameters
    ----------
        shards: list of shard filenames
        num_nodes: number of nodes
        num_gpus_per_node: number of gpus per node

    Returns
    -------
        list of shard lists for each GPU -> #nodes, #gpus, [shards]
    """
    if num_nodes is None or num_gpus_per_node is None:
        distributed_info = get_distributed_info()
        num_nodes = distributed_info.num_nodes
        num_gpus_per_node = distributed_info.gpu_per_node

    world_size = num_nodes * num_gpus_per_node

    if resample:
        # Make a copy of the shards list to avoid modifying the original list
        shards = shards.copy()
        # Add some shards to each worker to ensure that each worker has the same number of shards
        num_shards_to_add = (world_size - len(shards) % world_size) % world_size
        if num_shards_to_add > 0:
            # During training to not bias, shuffle should be done before
            # calling distribute_shards
            shards.extend(shards[:num_shards_to_add])

    assert len(shards) % world_size == 0, f"Number of shards ({len(shards)}) must be divisible by the world_size {world_size}"

    # We first cut the shards into num_nodes pieces
    nodes_shards = [list(islice(shards, i, None, num_nodes)) for i in range(num_nodes)]
    # Then we cut each node's shards into num_gpus_per_node pieces
    distributed_shards = [
        [list(islice(node_shards, i, None, num_gpus_per_node)) for i in range(num_gpus_per_node)]
        for node_shards in nodes_shards
    ]

    return distributed_shards


class MultishardStreamer(IterableDataset):
    """
    Iterates over tar files in webdataset format in a strictly sequential way.

    MultishardStreamer groups files with the same key together.

    Parameters
    ----------
        shards_paths (str | List[str]): list of shards to read.
        sample_map: (dict of str -> None | Callable[[IO], Any] | str): transformation to apply to the keys.
            If the mapped value is a string, the sample will be decoded with the corresponding function in SAMPLE_MAP_FN.
            If the mapped value is None, the sample will not be decoded, and its raw bytes will be returned.
            If the mapped value is a callable, a file-like object will be passed to it for decoding.
        exclude_keys (Optional, List[str]): list of keys to exclude from the samples.
        shard_shuffle (bool): shuffle the shards before reading them.
        random_seed (int): seed for the random number generator.
        shard_lengths (int | Dict[str, int]): number of samples in each shard.
            Can be int if shards_path is a single shard.
            Otherwise, it must be a dictionary with the shard name as keys and the number of samples as values.
    """

    def __init__(
        self,
        shard_paths: str | List[str],
        sample_map: Mapping[str, None | str | Callable[[IO], Any]],
        exclude_keys: Optional[List[str]] = None,
        shard_shuffle: bool = False,
        random_seed: int = 0,
        shard_lengths: Optional[int | Dict[str, int]] = None,
    ) -> None:
        self.shard_shuffle = shard_shuffle
        self.random_seed = random_seed
        shard_paths = [shard_paths] if isinstance(shard_paths, str) else shard_paths.copy()
        shard_paths = sorted(shard_paths)
        shard_paths = [expand_path(path) for path in shard_paths]
        if self.shard_shuffle:
            # This shuffle is deterministic because we set the seed
            # It allows having the same order on all GPUs
            random.Random(random_seed).shuffle(shard_paths)
        # we shuffle before setting the attribute
        # in order to make load_state_dict deterministic
        self.shard_paths = shard_paths
        self.shard_lengths = (
            {os.path.basename(pth): shard_lengths for pth in self.shard_paths}
            if isinstance(shard_lengths, int)
            else shard_lengths
        )

        # We get the number of files that define a sample
        self.sample_length = len(sample_map)
        self._input_sample_map = sample_map
        self.exclude_keys = exclude_keys or []

        if is_distributed():
            distributed_info = get_distributed_info()
            # We distribute the shards among the nodes and gpus
            all_shards_for_this_gpu = distribute_shards(
                self.shard_paths,
                resample=True,  # this allows having the same number of shards per gpu.
            )
            # We get the shards of our node
            all_shards_for_this_gpu = all_shards_for_this_gpu[distributed_info.node_rank]
            # We get the shards of our gpu
            self.all_shards_for_this_gpu = all_shards_for_this_gpu[distributed_info.local_rank]
        else:
            # We only split the shards among the workers latter on.
            self.all_shards_for_this_gpu = self.shard_paths

        self._tar_stream = None
        self._this_worker_shards = None
        self._set_sample_map_fn()

        # Which tar file we are reading
        self.tar_index = 0
        # Which file we are reading in the tar file
        self.file_index = 0
        # Which sample we are reading (in the current tar file)
        self.sample_index = 0

        if self.shard_lengths is not None:
            assert len(self.shard_lengths) == len(
                self.shard_paths
            ), f"shard_lengths ({len(self.shard_lengths)}) must have the same length as shards ({len(self.shard_paths)})"
            self._set_length()

    def _set_sample_map_fn(self) -> None:
        """Create the the dictionary of functions to apply to the keys."""
        assert isinstance(self._input_sample_map, dict), "sample_map must be a dictionary"
        self._sample_map_fn = {}
        self._map_ext = {}
        for ext, fn in self._input_sample_map.items():
            if ext == "image":
                _ext = IMAGE_EXTENSIONS
            elif ext == "video":
                _ext = VIDEO_EXTENSIONS
            else:
                _ext = [ext]

            for _e in _ext:
                self._map_ext[_e] = ext
                if isinstance(fn, str):
                    # We map the string to the predefined functions
                    self._sample_map_fn[_e] = SAMPLE_MAP_FN[fn]
                elif fn is None:
                    # We return the bytes
                    self._sample_map_fn[_e] = lambda x: x.read()
                else:
                    self._sample_map_fn[_e] = fn

    @property
    def this_worker_shards(self) -> List[str]:
        """Set the shards used by this worker."""
        if self._this_worker_shards is None:
            self._set_worker_shards()
        return self._this_worker_shards

    def _set_worker_shards(self) -> None:
        """
        Creates a list of shards for this worker.

        1. If DDP we first split the shards among nodes and then GPUs.
        Note that we resample the shards to have the same number of shards per GPU.
        2. We then split the shards among the workers of a DataLoader.
        If called outside of DataLoader, we keep all the shards from __init__
        """
        worker_info = get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        if num_workers > len(self.all_shards_for_this_gpu):
            logger.info(
                f"Number of workers ({num_workers}) is greater than "
                f"the number of shards ({len(self.all_shards_for_this_gpu)})."
            )

        this_worker_shards = []
        for shard in islice(self.all_shards_for_this_gpu, worker_id, None, num_workers):
            this_worker_shards.append(shard)

        # Opens for reading, transparently handles compression, and reads the file strictly sequentially
        self._this_worker_shards = this_worker_shards

    def _set_length(self) -> "MultishardStreamer":
        """Add a __len__ method returning the desired value.

        This does not change the actual number of samples in an epoch.
        PyTorch IterableDataset should not have a __len__ method.
        This is provided only as a workaround for some broken training environments
        that require a __len__ method.
        """
        assert self.shard_lengths is not None, "When calling _set_worker_lengths, shard_lengths must be set."
        # the length of the dataset should be set per GPU and not per DataLoader worker
        self._this_gpu_lengths = [self.shard_lengths[os.path.basename(shard)] for shard in self.all_shards_for_this_gpu]
        self._this_gpu_size = sum(self._this_gpu_lengths)

    def __len__(self) -> int:
        if not hasattr(self, "_this_gpu_size"):
            raise TypeError("object of type 'MultishardStreamer' has no len()")

        return self._this_gpu_size

    def __iter__(self) -> "MultishardStreamer":
        # Allows one single iterator active per instance of the class
        return self

    def _skip_current_stream(self) -> None:
        """Skip the current tar stream."""
        if self._tar_stream is None:
            raise ValueError("No tar stream to skip")
        self._tar_stream.close()
        self._tar_stream = None
        self.tar_index += 1
        self.file_index = 0

    def _tar_stream_next(self) -> tarfile.TarInfo:
        """
        Iterate over the tar stream.

        When we reach the end of the tar stream, we close it and open the next one.
        """
        if self._tar_stream is None:
            if self.tar_index >= len(self.this_worker_shards):
                return None
            self._tar_stream: tarfile.TarFile = tarfile.open(self.this_worker_shards[self.tar_index], "r|*")

        info = self._tar_stream.next()
        if info is None:
            # Close current stream explicitly to avoid waiting for the garbage collector
            self._tar_stream.close()
            # Forces next tar file to be opened and resets counters
            self._tar_stream = None
            self.tar_index += 1
            self.file_index = 0
            # Recursive call to get the next tar file
            return self._tar_stream_next()
        self.file_index += 1
        return info

    def __next__(self) -> Sample:
        sample = {}
        while True:
            info = self._tar_stream_next()
            if info is None:
                raise StopIteration
            if not info.isfile():
                continue

            name = info.name
            key, ext = os.path.splitext(name)
            if ext in self.exclude_keys:
                continue
            if "__key__" in sample and sample["__key__"] != key:
                # Ensure that the key is the same for all the extensions
                raise ValueError(f"Key {key} does not match {sample['__key__']}")
            else:
                sample["__key__"] = key
                sample["__source__"] = self.this_worker_shards[self.tar_index]
            if ext not in self._map_ext.keys():
                raise ValueError(f"File {name} has an invalid extension {ext}")
            f = self._tar_stream.extractfile(info)
            if f is None:
                raise ValueError(f"Error reading file {name}")

            try:
                # Here f is is a file-like object, not directly the byte contents.
                # To have the bytes we need to call f.read()
                assert self._map_ext[ext] not in sample, f"Key {self._map_ext[ext]} already exists in the sample"
                sample[self._map_ext[ext]] = self._sample_map_fn[ext](f)
            except Exception as e:
                raise ValueError(f"Error reading file {name}") from e
            finally:
                f.close()

            # sample is defined when we have seen all the keys from _input_sample_map
            # +2 because we have the __key__ and __source__ keys
            if len(sample) == self.sample_length + 2 and set(sample.keys()).issuperset(self._input_sample_map.keys()):
                # Move the pointer, this sample is done
                self.sample_index += 1
                return sample

    def __getstate__(self) -> StateDict:
        state = self.__dict__.copy()
        if self._this_worker_shards is not None:
            # We have to make an if statement here because __getstate__ is called
            # before __iter__ in the case of a DataLoader.
            # This creates a problem because the DataLoader does not have the worker_info.
            # Thus assigning poorly the shards to the workers.
            state["_this_worker_shards"] = self.this_worker_shards
        state.pop("_tar_stream")
        state.pop("_sample_map_fn")
        # Here we do not close the stream as the StatefulDataLoader queries the
        # state_dict regularly.
        # See the snapshot_every_n_steps parameter in the StatefulDataLoader.
        return state

    def __setstate__(self, state: StateDict) -> None:
        self.__dict__.update(state)
        self._tar_stream = None
        # We skip the data that we have already read
        # Note that here we also already skip to the correct tar file.
        file_index = self.file_index
        self.file_index = 0
        for _ in range(file_index):
            _ = self._tar_stream_next()

    def state_dict(self) -> StateDict:
        """Wrapper for __getstate__ for compatibility with StatefulDataLoader."""
        return self.__getstate__()

    def load_state_dict(self, state: StateDict) -> None:
        """Wrapper for __setstate__ for compatibility with StatefulDataLoader."""
        self.__setstate__(state)


def _make_seed(seed: int, iter_count: int) -> int:
    # NOTE: Tried a few variants (including iter_count << 32), this one worked best.
    return seed + iter_count * 1000


class InfiniteDataset(IterableDataset):
    """Infinite dataset that loops over the input dataset."""

    def __init__(self, dataset: MultishardStreamer, seed: int = 0) -> None:
        self.dataset = dataset
        self._iter_count = 0
        self._seed = seed

    def __iter__(self) -> Generator:
        while True:
            # Reset the dataset to the beginning
            self.dataset.tar_index = 0
            self.dataset.file_index = 0
            self.dataset.sample_index = 0
            # sorting allows making the shuffling deterministic
            _shards = sorted(self.dataset.this_worker_shards.copy())
            # We shuffle the shards for this worker
            random.Random(_make_seed(self._seed, self._iter_count)).shuffle(_shards)
            # We set the shards for this worker
            self.dataset._this_worker_shards = _shards
            iterator = iter(self.dataset)
            for sample in iterator:
                try:
                    yield sample
                except StopIteration:
                    # This is the end of the dataset, we need to reset the iterator
                    break
                except Exception as e:
                    # This is a bug, we should not have any exception here
                    logger.error(f"Caught exception: {type(e).__name__}: {str(e)}, skipping tar stream")
                    self.dataset._skip_current_stream()
            # Increment the iteration count
            self._iter_count += 1

    def __getstate__(self) -> StateDict:
        state = self.dataset.state_dict()
        state["_iter_count"] = self._iter_count
        state["_seed"] = self._seed
        return state

    def __setstate__(self, state: StateDict) -> None:
        self.dataset.load_state_dict(state)
        self._iter_count = state["_iter_count"]
        self._seed = state["_seed"]

    def state_dict(self) -> StateDict:
        """Wrapper for __getstate__ for compatibility with StatefulDataLoader."""
        return self.__getstate__()

    def load_state_dict(self, state: StateDict) -> None:
        """Wrapper for __setstate__ for compatibility with StatefulDataLoader."""
        self.__setstate__(state)


class PILReader:
    """PIL image reader."""

    def __call__(self, path: str) -> Image.Image:
        return Image.open(path).convert("RGB")


def get_laion_dataset(
    root: str,
    transform: Callable[[Image.Image], Tensor],
    target_transform: Optional[Callable[[int], Tensor]] = None,
    infinite: bool = True,
) -> MultishardStreamer:
    """Get the LAION dataset."""
    all_tar_files = glob(os.path.join(expand_path(root), "*.tar"))
    logger.info(f"Found {len(all_tar_files)} tar files")
    transform = transforms.Compose([PILReader(), transform])
    dataset = MultishardStreamer(
        shard_paths=all_tar_files,
        sample_map={"image": transform},
        shard_shuffle=False,  # We do not need to shuffle, LAION is already shuffled
        exclude_keys=[".json", ".txt"],
    )
    if infinite:
        dataset = InfiniteDataset(dataset)
    return dataset
