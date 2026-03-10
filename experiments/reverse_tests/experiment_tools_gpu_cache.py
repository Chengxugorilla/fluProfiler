import time
from collections import OrderedDict

import numpy as np
import torch

from experiment_tools import (
    default_exp_id,
    find_repo_root,
    fluProfiler_Dataset,
    load_data_and_dataloaders,
    log_epoch_to_file,
    log_metrics_to_tensorboard,
    make_run_dirs,
    setup_optimizer,
    setup_tensorboard_and_logging,
)


DEFAULT_MAX_CACHE_BYTES = 28 * 1024 ** 3


class GpuEmbeddingCache:
    """A simple LRU cache for sequence embeddings on the target device."""

    def __init__(self, cpu_store, device, max_cache_bytes=DEFAULT_MAX_CACHE_BYTES, dtype=torch.float32):
        self.cpu_store = cpu_store
        self.device = device
        self.max_cache_bytes = int(max_cache_bytes)
        self.dtype = dtype

        self.gpu_store = {}
        self.lru = OrderedDict()
        self.current_bytes = 0

        self.hits = 0
        self.misses = 0
        self.copy_count = 0
        self.copied_bytes = 0
        self.evictions = 0
        self.bypass_count = 0

    @property
    def current_items(self):
        return len(self.gpu_store)

    @staticmethod
    def _as_tensor(value):
        if torch.is_tensor(value):
            return value
        return torch.as_tensor(value)

    @staticmethod
    def _tensor_nbytes(tensor):
        return int(tensor.numel() * tensor.element_size())

    def _touch(self, key):
        self.lru.pop(key, None)
        self.lru[key] = None

    def _evict_until_fit(self, required_bytes):
        while self.current_bytes + required_bytes > self.max_cache_bytes and self.lru:
            old_key, _ = self.lru.popitem(last=False)
            old_tensor = self.gpu_store.pop(old_key)
            self.current_bytes -= self._tensor_nbytes(old_tensor)
            self.evictions += 1
            del old_tensor

    def snapshot(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "copy_count": self.copy_count,
            "copied_bytes": self.copied_bytes,
            "evictions": self.evictions,
            "bypass_count": self.bypass_count,
        }

    def delta(self, snapshot):
        delta_stats = {}
        for key, value in self.snapshot().items():
            delta_stats[key] = value - snapshot.get(key, 0)
        return delta_stats

    def get(self, key):
        if key in self.gpu_store:
            self.hits += 1
            self._touch(key)
            return self.gpu_store[key]

        self.misses += 1
        cpu_tensor = self._as_tensor(self.cpu_store[key]).detach()
        device_tensor = cpu_tensor.to(self.device, dtype=self.dtype, non_blocking=True)
        tensor_bytes = self._tensor_nbytes(device_tensor)

        self.copy_count += 1
        self.copied_bytes += tensor_bytes

        if tensor_bytes > self.max_cache_bytes:
            self.bypass_count += 1
            return device_tensor

        self._evict_until_fit(tensor_bytes)
        self.gpu_store[key] = device_tensor
        self.current_bytes += tensor_bytes
        self._touch(key)
        return device_tensor

    def get_many(self, keys):
        return [self.get(key) for key in keys]


def generate_matrix_gpu(matrix_list):
    """Build a padded batch directly on the target device."""
    if not matrix_list:
        raise ValueError("matrix_list must not be empty.")

    batch_size = len(matrix_list)
    device = matrix_list[0].device
    dtype = matrix_list[0].dtype
    embed_dim = matrix_list[0].shape[-1]

    lengths = torch.tensor([mat.shape[0] for mat in matrix_list], device=device, dtype=torch.long)
    max_len = int(lengths.max().item())

    matrix = torch.zeros((batch_size, max_len, embed_dim), device=device, dtype=dtype)
    for idx, mat in enumerate(matrix_list):
        seq_len = mat.shape[0]
        matrix[idx, :seq_len].copy_(mat)

    positions = torch.arange(max_len, device=device).unsqueeze(0)
    mask = (positions < lengths.unsqueeze(1)).to(dtype=dtype)
    return matrix, mask


def _fetch_embeddings(emb_source, keys):
    if hasattr(emb_source, "get_many"):
        return emb_source.get_many(keys)
    return [emb_source[key] for key in keys]


def _prepare_batch(batch, emb_source, device, generate_matrix_fn):
    emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

    matrices_a, masks_a = generate_matrix_fn(_fetch_embeddings(emb_source, emb_file_name_a))
    matrices_b, masks_b = generate_matrix_fn(_fetch_embeddings(emb_source, emb_file_name_b))
    matrices_c, masks_c = generate_matrix_fn(_fetch_embeddings(emb_source, emb_file_name_c))
    matrices_d, masks_d = generate_matrix_fn(_fetch_embeddings(emb_source, emb_file_name_d))

    strainPassCats = strainPassCats.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    return {
        "matrices_a": matrices_a,
        "matrices_b": matrices_b,
        "matrices_c": matrices_c,
        "matrices_d": matrices_d,
        "matrix_attention_masks_a": masks_a,
        "matrix_attention_masks_b": masks_b,
        "matrix_attention_masks_c": masks_c,
        "matrix_attention_masks_d": masks_d,
        "strainPassCats": strainPassCats,
        "labels": labels,
    }


def train_step(model, batch, emb_source, device, generate_matrix_fn=generate_matrix_gpu):
    model_inputs = _prepare_batch(batch, emb_source, device, generate_matrix_fn)
    loss, logits, output = model(**model_inputs)
    return loss


def evaluate_step(model, dataloader, emb_source, device, generate_matrix_fn=generate_matrix_gpu, return_predictions=False):
    try:
        from evaluation.metrics import print_exams
    except ImportError:
        import sys
        sys.path.append("../../../../src/fluprofiler")
        from evaluation.metrics import print_exams

    model.eval()
    prediction_chunks = []
    reference_chunks = []
    loss_ls = []

    with torch.no_grad():
        for batch in dataloader:
            model_inputs = _prepare_batch(batch, emb_source, device, generate_matrix_fn)
            loss, logits, output = model(**model_inputs)

            loss_ls.append(loss.item())
            prediction_chunks.append(output.view(-1).detach())
            reference_chunks.append(model_inputs["labels"].detach())

    prediction_tensor = torch.cat(prediction_chunks, dim=0).float().cpu()
    reference_tensor = torch.cat(reference_chunks, dim=0).float().cpu()

    mae, mse, pearson, spearman, r2 = print_exams(reference_tensor.numpy(), prediction_tensor.numpy())
    pearson_value = float(getattr(pearson, "statistic", pearson[0]))
    spearman_value = float(getattr(spearman, "statistic", spearman[0]))
    metrics = {
        "loss": float(np.mean(loss_ls)) if loss_ls else 0.0,
        "mae": float(mae),
        "mse": float(mse),
        "pearson": pearson_value,
        "spearman": spearman_value,
        "r2": float(r2),
    }

    if return_predictions:
        metrics["predictions"] = prediction_tensor.tolist()
        metrics["references"] = reference_tensor.tolist()

    return metrics


def format_cache_stats(delta_stats, resident_bytes, resident_items):
    requests = delta_stats["hits"] + delta_stats["misses"]
    hit_rate = (delta_stats["hits"] / requests) if requests else 0.0
    resident_gb = resident_bytes / (1024 ** 3)
    copied_gb = delta_stats["copied_bytes"] / (1024 ** 3)
    return (
        f"cache_hit_rate={hit_rate:.4f}, "
        f"cache_hits={delta_stats['hits']}, "
        f"cache_misses={delta_stats['misses']}, "
        f"cpu_to_gpu_copies={delta_stats['copy_count']}, "
        f"copied_gb={copied_gb:.3f}, "
        f"evictions={delta_stats['evictions']}, "
        f"bypass={delta_stats['bypass_count']}, "
        f"resident_items={resident_items}, "
        f"resident_gb={resident_gb:.3f}"
    )


def append_cache_stats_to_log(log_path, epoch, avg_train_step_s, valid_time_s, test_time_s, delta_stats, resident_bytes, resident_items):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    stats_line = format_cache_stats(delta_stats, resident_bytes, resident_items)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"[{current_time}] Epoch {epoch + 1} cache stats, "
            f"avg_train_step_s: {avg_train_step_s:.4f}, "
            f"valid_time_s: {valid_time_s:.2f}, "
            f"test_time_s: {test_time_s:.2f}, "
            f"{stats_line}\n"
        )
import os
import time
from collections import OrderedDict

import numpy as np
import torch

from experiment_tools import (
    default_exp_id,
    find_repo_root,
    load_data_and_dataloaders,
    log_epoch_to_file,
    log_metrics_to_tensorboard,
    make_run_dirs,
    setup_optimizer,
    setup_tensorboard_and_logging,
)


class GpuEmbeddingCache:
    """LRU cache for sequence embeddings resident on GPU."""

    def __init__(self, cpu_store, device, max_cache_bytes):
        if device.type != "cuda":
            raise ValueError("GpuEmbeddingCache only supports CUDA devices.")

        self.cpu_store = cpu_store
        self.device = device
        self.max_cache_bytes = int(max_cache_bytes)
        self.gpu_store = OrderedDict()
        self.current_bytes = 0
        self.reset_stats()

    @staticmethod
    def _tensor_nbytes(tensor):
        return int(tensor.numel() * tensor.element_size())

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.copy_count = 0
        self.copied_bytes = 0
        self.uncached_oversize = 0

    def get_stats(self):
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "copy_count": self.copy_count,
            "copied_bytes": self.copied_bytes,
            "uncached_oversize": self.uncached_oversize,
            "cache_items": len(self.gpu_store),
            "cache_bytes": self.current_bytes,
            "hit_rate": float(self.hits / total) if total else 0.0,
        }

    def _touch(self, key):
        self.gpu_store.move_to_end(key)

    def _evict_until_fit(self, required_bytes):
        while self.current_bytes + required_bytes > self.max_cache_bytes and self.gpu_store:
            old_key, old_tensor = self.gpu_store.popitem(last=False)
            self.current_bytes -= self._tensor_nbytes(old_tensor)
            self.evictions += 1
            del old_tensor

    def get(self, key):
        if key in self.gpu_store:
            self.hits += 1
            self._touch(key)
            return self.gpu_store[key]

        cpu_tensor = self.cpu_store[key]
        gpu_tensor = cpu_tensor.to(self.device, non_blocking=True).detach()
        required_bytes = self._tensor_nbytes(gpu_tensor)

        self.misses += 1
        self.copy_count += 1
        self.copied_bytes += required_bytes

        if required_bytes > self.max_cache_bytes:
            self.uncached_oversize += 1
            return gpu_tensor

        self._evict_until_fit(required_bytes)
        self.gpu_store[key] = gpu_tensor
        self.current_bytes += required_bytes
        return gpu_tensor

    def get_many(self, keys):
        return [self.get(key) for key in keys]


def generate_matrix_on_device(matrix_list, mask_dtype=torch.float32):
    """Build a padded batch directly on the device of matrix_list[0]."""
    if len(matrix_list) == 0:
        raise ValueError("matrix_list must not be empty.")

    device = matrix_list[0].device
    dtype = matrix_list[0].dtype
    batch_size = len(matrix_list)
    embed_dim = int(matrix_list[0].shape[1])

    lengths = torch.tensor(
        [int(matrix.shape[0]) for matrix in matrix_list],
        device=device,
        dtype=torch.long,
    )
    max_len = int(lengths.max().item())

    matrix = torch.zeros((batch_size, max_len, embed_dim), device=device, dtype=dtype)
    for idx, seq_matrix in enumerate(matrix_list):
        matrix[idx, : seq_matrix.shape[0]] = seq_matrix

    positions = torch.arange(max_len, device=device).unsqueeze(0)
    mask = (positions < lengths.unsqueeze(1)).to(dtype=mask_dtype)
    return matrix, mask


def _format_gib(num_bytes):
    return float(num_bytes) / (1024 ** 3)


def _collect_batch_on_gpu(batch, gpu_cache):
    emb_file_name_a, emb_file_name_b, emb_file_name_c, emb_file_name_d, strainPassCats, labels = batch

    matrixs_a, masks_a = generate_matrix_on_device(gpu_cache.get_many(emb_file_name_a))
    matrixs_b, masks_b = generate_matrix_on_device(gpu_cache.get_many(emb_file_name_b))
    matrixs_c, masks_c = generate_matrix_on_device(gpu_cache.get_many(emb_file_name_c))
    matrixs_d, masks_d = generate_matrix_on_device(gpu_cache.get_many(emb_file_name_d))

    return (
        matrixs_a,
        matrixs_b,
        matrixs_c,
        matrixs_d,
        masks_a,
        masks_b,
        masks_c,
        masks_d,
        strainPassCats,
        labels,
    )


def train_step_with_gpu_cache(model, batch, gpu_cache, device):
    batch_build_start = time.perf_counter()
    (
        matrixs_a,
        matrixs_b,
        matrixs_c,
        matrixs_d,
        masks_a,
        masks_b,
        masks_c,
        masks_d,
        strainPassCats,
        labels,
    ) = _collect_batch_on_gpu(batch, gpu_cache)
    strainPassCats = strainPassCats.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    batch_build_time = time.perf_counter() - batch_build_start

    loss, logits, output = model(
        matrices_a=matrixs_a,
        matrices_b=matrixs_b,
        matrices_c=matrixs_c,
        matrices_d=matrixs_d,
        matrix_attention_masks_a=masks_a,
        matrix_attention_masks_b=masks_b,
        matrix_attention_masks_c=masks_c,
        matrix_attention_masks_d=masks_d,
        strainPassCats=strainPassCats,
        labels=labels,
    )
    return loss, {"batch_build_time": batch_build_time}


def evaluate_step_with_gpu_cache(model, dataloader, gpu_cache, device, return_predictions=False):
    try:
        from evaluation.metrics import print_exams
    except ImportError:
        import sys

        sys.path.append("../../../../src/fluprofiler")
        from evaluation.metrics import print_exams

    model.eval()
    prediction_tensors = []
    reference_tensors = []
    loss_ls = []
    batch_build_times = []
    eval_start = time.perf_counter()

    with torch.no_grad():
        for batch in dataloader:
            batch_build_start = time.perf_counter()
            (
                matrixs_a,
                matrixs_b,
                matrixs_c,
                matrixs_d,
                masks_a,
                masks_b,
                masks_c,
                masks_d,
                strainPassCats,
                labels,
            ) = _collect_batch_on_gpu(batch, gpu_cache)
            strainPassCats = strainPassCats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_build_times.append(time.perf_counter() - batch_build_start)

            loss, logits, output = model(
                matrices_a=matrixs_a,
                matrices_b=matrixs_b,
                matrices_c=matrixs_c,
                matrices_d=matrixs_d,
                matrix_attention_masks_a=masks_a,
                matrix_attention_masks_b=masks_b,
                matrix_attention_masks_c=masks_c,
                matrix_attention_masks_d=masks_d,
                strainPassCats=strainPassCats,
                labels=labels,
            )

            loss_ls.append(loss.item())
            prediction_tensors.append(output.view(-1).detach())
            reference_tensors.append(labels.detach())

    if prediction_tensors:
        predictions = torch.cat(prediction_tensors, dim=0).cpu().tolist()
        references = torch.cat(reference_tensors, dim=0).cpu().tolist()
    else:
        predictions = []
        references = []

    mae, mse, pearson, spearman, r2 = print_exams(references, predictions)
    pearson_value = float(getattr(pearson, "statistic", pearson[0]))
    spearman_value = float(getattr(spearman, "statistic", spearman[0]))
    metrics = {
        "loss": float(np.mean(loss_ls)) if loss_ls else 0.0,
        "mae": float(mae),
        "mse": float(mse),
        "pearson": pearson_value,
        "spearman": spearman_value,
        "r2": float(r2),
    }

    if return_predictions:
        metrics["predictions"] = predictions
        metrics["references"] = references

    timing = {
        "eval_time": time.perf_counter() - eval_start,
        "avg_batch_build_time": float(np.mean(batch_build_times)) if batch_build_times else 0.0,
    }
    return metrics, timing


def log_cache_stats_to_file(log_path, epoch, cache_stats, train_timing, valid_timing, test_timing):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            (
                f"[{current_time}] Epoch {epoch + 1} cache stats: "
                f"hit_rate={cache_stats['hit_rate']:.4f}, "
                f"hits={cache_stats['hits']}, misses={cache_stats['misses']}, "
                f"copy_count={cache_stats['copy_count']}, "
                f"copied_gib={_format_gib(cache_stats['copied_bytes']):.3f}, "
                f"cache_items={cache_stats['cache_items']}, "
                f"cache_gib={_format_gib(cache_stats['cache_bytes']):.3f}, "
                f"evictions={cache_stats['evictions']}, "
                f"oversize={cache_stats['uncached_oversize']}, "
                f"train_avg_step_s={train_timing['avg_step_time']:.4f}, "
                f"train_avg_build_s={train_timing['avg_batch_build_time']:.4f}, "
                f"valid_eval_s={valid_timing['eval_time']:.4f}, "
                f"valid_build_s={valid_timing['avg_batch_build_time']:.4f}, "
                f"test_eval_s={test_timing['eval_time']:.4f}, "
                f"test_build_s={test_timing['avg_batch_build_time']:.4f}\n"
            )
        )

