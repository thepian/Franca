# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import gc
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from gzip import GzipFile
from io import BytesIO
from mmap import ACCESS_READ, mmap
from typing import Any, Callable, List, Optional, Set, Tuple

import numpy as np

# You may need to adjust this import based on your project structure
from franca.data.datasets.extended import ExtendedVisionDataset

_Labels = int

_DEFAULT_MMAP_CACHE_SIZE = 16  # Warning: This can exhaust file descriptors


@dataclass
class _ClassEntry:
    block_offset: int
    maybe_filename: Optional[str] = None


@dataclass
class _Entry:
    class_index: int  # noqa: E701
    start_offset: int
    end_offset: int
    filename: str


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"

    @property
    def length(self) -> int:
        return {
            _Split.TRAIN: 11_797_647,
            _Split.VAL: 561_050,
        }[self]

    def entries_path(self):
        return f"imagenet21kp_{self.value}.txt"


def _get_tarball_path(class_id: str) -> str:
    return f"{class_id}.tar"


def _make_mmap_tarball(tarballs_root: str, mmap_cache_size: int):
    @lru_cache(maxsize=mmap_cache_size)
    def _mmap_tarball(class_id: str) -> mmap:
        tarball_path = _get_tarball_path(class_id)
        tarball_full_path = os.path.join(tarballs_root, tarball_path)
        with open(tarball_full_path) as f:
            return mmap(fileno=f.fileno(), length=0, access=ACCESS_READ)

    return _mmap_tarball


class ImageNet22k(ExtendedVisionDataset):
    _GZIPPED_INDICES: Set[int] = {
        841_545,
        1_304_131,
        2_437_921,
        2_672_079,
        2_795_676,
        2_969_786,
        6_902_965,
        6_903_550,
        6_903_628,
        7_432_557,
        7_432_589,
        7_813_809,
        8_329_633,
        10_296_990,
        10_417_652,
        10_492_265,
        10_598_078,
        10_782_398,
        10_902_612,
        11_203_736,
        11_342_890,
        11_397_596,
        11_589_762,
        11_705_103,
        12_936_875,
        13_289_782,
        13_153_499,  # total number of images in IN-21K dataset (please add this change this number based on what you have)
    }
    Labels = _Labels

    def __init__(
        self,
        *,
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        mmap_cache_size: int = _DEFAULT_MMAP_CACHE_SIZE,
        initialize_only: bool = False,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._gzipped_indices = ImageNet22k._GZIPPED_INDICES
        self._mmap_tarball = _make_mmap_tarball(self._tarballs_root, mmap_cache_size)

        # If we're only initializing (for dump_extra), don't try to load entries
        if initialize_only:
            return

        entries_path = self._get_entries_path(root)
        entries_full_path = os.path.join(self._extra_root, entries_path)

        if not os.path.exists(entries_full_path):
            raise FileNotFoundError(
                f"Entries file not found at {entries_full_path}. "
                f"Please run dataset.dump_extra() first to create necessary index files."
            )

        self._entries = self._load_extra(entries_path)

        class_ids_path = self._get_class_ids_path(root)
        class_ids_full_path = os.path.join(self._extra_root, class_ids_path)

        if not os.path.exists(class_ids_full_path):
            raise FileNotFoundError(
                f"Class IDs file not found at {class_ids_full_path}. "
                f"Please run dataset.dump_extra() first to create necessary index files."
            )

        self._class_ids = self._load_extra(class_ids_path)

    def _get_entries_path(self, root: Optional[str] = None) -> str:
        return "entries.npy"

    def _get_class_ids_path(self, root: Optional[str] = None) -> str:
        return "class-ids.npy"

    def _find_class_ids(self, path: str) -> List[str]:
        class_ids = []

        with os.scandir(path) as entries:
            for entry in entries:
                root, ext = os.path.splitext(entry.name)
                if ext != ".tar":
                    continue
                class_ids.append(root)

        return sorted(class_ids)

    def _generate_blocks_file(self, root: str, class_id: str) -> List[_ClassEntry]:
        """
        Generate a blocks file from a tar file if it doesn't exist.
        This is a fallback for when the .log files don't exist.
        """
        print(f"Generating blocks file for {class_id}")
        tar_file_path = os.path.join(root, f"{class_id}.tar")
        class_entries = []

        try:
            # Read the tar file to extract block offsets and filenames
            with open(tar_file_path, "rb") as f:
                data = f.read()

            # This is a very simplified approach and may not work for all tar files
            # TAR files have 512-byte blocks, where each file is preceded by a header block
            block_size = 512
            pos = 0

            while pos < len(data):
                # Read the header block
                header = data[pos : pos + block_size]

                # Extract filename from header (bytes 0-99)
                filename = header[:100].rstrip(b"\x00").decode("utf-8", errors="ignore")

                if filename:
                    # Extract file size from header (bytes 124-135)
                    size_str = header[124:136].decode("utf-8", errors="ignore").strip("\x00")
                    try:
                        size = int(size_str, 8)  # Size is stored in octal
                    except ValueError:
                        # Skip invalid header
                        pos += block_size
                        continue

                    # Calculate number of blocks (including the header block)
                    num_blocks = (size + block_size - 1) // block_size + 1

                    # Record file entry
                    block_offset = pos // block_size
                    class_entry = _ClassEntry(
                        block_offset,
                        filename if filename != "** Block of NULs **" else None,
                    )
                    class_entries.append(class_entry)

                    # Move to next file
                    pos += num_blocks * block_size
                else:
                    # End of archive (two consecutive zero blocks)
                    if pos + 2 * block_size <= len(data) and all(b == 0 for b in data[pos : pos + 2 * block_size]):
                        break

                    # Add null block
                    block_offset = pos // block_size
                    class_entry = _ClassEntry(block_offset, None)
                    class_entries.append(class_entry)
                    pos += block_size

            # Add final null entry if needed
            if not class_entries or class_entries[-1].maybe_filename is not None:
                block_offset = (pos // block_size) + 1
                class_entries.append(_ClassEntry(block_offset, None))

            return class_entries

        except Exception as e:
            print(f"Error generating blocks for {class_id}: {e}")
            # Return a minimal valid structure
            return [_ClassEntry(0, "dummy.JPEG"), _ClassEntry(1, None)]

    def _load_entries_class_ids(self, root: Optional[str] = None) -> Tuple[List[_Entry], List[str]]:
        root = self.get_root(root)
        entries: List[_Entry] = []
        class_ids = self._find_class_ids(root)

        # Create blocks directory if it doesn't exist
        blocks_dir = os.path.join(root, "blocks")
        os.makedirs(blocks_dir, exist_ok=True)

        for class_index, class_id in enumerate(class_ids):
            log_path = os.path.join(self._extra_root, "blocks", f"{class_id}.log")
            class_entries = []

            try:
                # First try to read from existing log file
                with open(log_path) as f:
                    for line in f:
                        line = line.rstrip()
                        block, filename = line.split(":")
                        block_offset = int(block[6:])
                        filename = filename[1:]

                        maybe_filename = None
                        if filename != "** Block of NULs **":
                            maybe_filename = filename

                        class_entry = _ClassEntry(block_offset, maybe_filename)
                        class_entries.append(class_entry)

            except (OSError, FileNotFoundError):
                # Log file doesn't exist, try to generate it
                class_entries = self._generate_blocks_file(root, class_id)

                # Save the generated log file
                if class_entries:
                    try:
                        with open(log_path, "w") as f:
                            for i, entry in enumerate(class_entries):
                                filename = entry.maybe_filename or "** Block of NULs **"
                                f.write(f"block{entry.block_offset}:{' ' + filename}\n")
                    except Exception as e:
                        print(f"Warning: Couldn't save log file for {class_id}: {e}")

            if not class_entries:
                print(f"Warning: No entries found for class {class_id}")
                continue

            # Make sure the last entry is a null entry
            if class_entries[-1].maybe_filename is not None:
                print(f"Warning: Last entry for {class_id} is not null, adding null entry")
                last_offset = class_entries[-1].block_offset + 1
                class_entries.append(_ClassEntry(last_offset, None))

            # Create entries from class entries
            for class_entry1, class_entry2 in zip(class_entries, class_entries[1:]):
                if class_entry1.maybe_filename is None:
                    continue

                start_offset = 512 * class_entry1.block_offset
                end_offset = 512 * class_entry2.block_offset
                filename = class_entry1.maybe_filename

                # Skip known problematic files
                if filename == "n06470073_47249.JPEG":
                    continue

                entry = _Entry(class_index, start_offset, end_offset, filename)
                entries.append(entry)

        return entries, class_ids

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_root = self._extra_root
        extra_full_path = os.path.join(extra_root, extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_root = self._extra_root
        extra_full_path = os.path.join(extra_root, extra_path)
        os.makedirs(extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _tarballs_root(self) -> str:
        return self.root

    def find_class_id(self, class_index: int) -> str:
        return str(self._class_ids[class_index])

    def get_image_data(self, index: int) -> bytes:
        entry = self._entries[index]
        class_id = entry["class_id"]
        class_mmap = self._mmap_tarball(class_id)

        start_offset, end_offset = entry["start_offset"], entry["end_offset"]
        try:
            mapped_data = class_mmap[start_offset:end_offset]
            data = mapped_data[512:]  # Skip entry header block

            if len(data) >= 2 and tuple(data[:2]) == (0x1F, 0x8B):
                assert index in self._gzipped_indices, f"unexpected gzip header for sample {index}"
                with GzipFile(fileobj=BytesIO(data)) as g:
                    data = g.read()
        except Exception as e:
            raise RuntimeError(f"can not retrieve image data for sample {index} " f'from "{class_id}" tarball') from e

        return data

    def get_target(self, index: int) -> Any:
        return int(self._entries[index]["class_index"])

    def get_targets(self) -> np.ndarray:
        return self._entries["class_index"]

    def get_class_id(self, index: int) -> str:
        return str(self._entries[index]["class_id"])

    def get_class_ids(self) -> np.ndarray:
        return self._entries["class_id"]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self._entries)

    def _dump_entries(self, *args, **kwargs) -> None:
        print("Dumping entries...")
        entries, class_ids = self._load_entries_class_ids(*args, **kwargs)

        # Force garbage collection
        gc.collect()

        max_class_id_length, max_filename_length, max_class_index = -1, -1, -1
        # Process in batches to save memory
        batch_size = 50000
        for i in range(0, len(entries), batch_size):
            end_idx = min(i + batch_size, len(entries))
            batch_entries = entries[i:end_idx]

            for entry in batch_entries:
                class_id = class_ids[entry.class_index]
                max_class_index = max(entry.class_index, max_class_index)
                max_class_id_length = max(len(class_id), max_class_id_length)
                max_filename_length = max(len(entry.filename), max_filename_length)

            print(f"Processed metadata for entries {i} to {end_idx} of {len(entries)}")
            # Force garbage collection
            gc.collect()

        dtype = np.dtype(
            [
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("start_offset", "<u4"),
                ("end_offset", "<u4"),
                ("filename", f"U{max_filename_length}"),
            ]
        )

        # Create the full array but process in batches
        sample_count = len(entries)
        entries_array = np.empty(sample_count, dtype=dtype)

        for i in range(0, sample_count, batch_size):
            end_idx = min(i + batch_size, sample_count)
            batch_entries = entries[i:end_idx]

            for j, entry in enumerate(batch_entries):
                class_index = entry.class_index
                class_id = class_ids[class_index]
                start_offset = entry.start_offset
                end_offset = entry.end_offset
                filename = entry.filename
                entries_array[i + j] = (
                    class_index,
                    class_id,
                    start_offset,
                    end_offset,
                    filename,
                )

            print(f"Processed entries {i} to {end_idx} of {sample_count}")
            # Force garbage collection
            gc.collect()

        entries_path = self._get_entries_path(*args, **kwargs)
        self._save_extra(entries_array, entries_path)
        print(f"Saved {sample_count} entries")

        # Force garbage collection before returning
        del entries_array
        gc.collect()

    def _dump_class_ids(self, *args, **kwargs) -> None:
        print("Dumping class IDs...")
        entries_path = self._get_entries_path(*args, **kwargs)
        entries_array = self._load_extra(entries_path)

        max_class_id_length, max_class_index = -1, -1
        for entry in entries_array:
            class_index, class_id = entry["class_index"], entry["class_id"]
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)

        class_ids_array = np.empty(max_class_index + 1, dtype=f"U{max_class_id_length}")
        for entry in entries_array:
            class_index, class_id = entry["class_index"], entry["class_id"]
            class_ids_array[class_index] = class_id
        class_ids_path = self._get_class_ids_path(*args, **kwargs)
        self._save_extra(class_ids_array, class_ids_path)
        print(f"Saved {len(class_ids_array)} class IDs")

    def _dump_extra(self, *args, **kwargs) -> None:
        self._dump_entries(*args, **kwargs)
        self._dump_class_ids(*args, **kwargs)

    def dump_extra(self, root: Optional[str] = None) -> None:
        print(f"Dumping extra files to {self._extra_root}")
        return self._dump_extra(root)
