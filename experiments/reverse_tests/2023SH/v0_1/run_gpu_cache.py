import sys
import time

sys.path.append("../../../../src/fluprofiler")
sys.path.append("../../")

from experiment_tools_gpu_cache import (
    GpuEmbeddingCache,
    append_cache_stats_to_log,
    default_exp_id,
    evaluate_step,
    find_repo_root,
    format_cache_stats,
    generate_matrix_gpu,
    load_data_and_dataloaders,
    log_epoch_to_file,
    log_metrics_to_tensorboard,
    make_run_dirs,
    setup_optimizer,
    setup_tensorboard_and_logging,
    train_step,
)
from models.architectures import fluProfiler_Config, fluProfiler_v0_1
from tqdm import tqdm
import os
from pathlib import Path
import json
import torch
import numpy as np
from evaluation.metrics import EarlyStopping
import pickle


_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = find_repo_root(_SCRIPT_PATH)
root_path = str(_REPO_ROOT) + "/"
data_path = root_path + "data/reverse_test/"
season_path = "processed/test_2023SH/"

exp_id = os.environ.get("FLUPROFILER_EXP_ID") or default_exp_id(_SCRIPT_PATH, _REPO_ROOT)
tag = os.environ.get("FLUPROFILER_TAG") or "v0_1_gpu_cache"
run_paths = make_run_dirs(_REPO_ROOT, exp_id=exp_id, tag=tag)

device = torch.device("cuda:1")
cache_gb = float(os.environ.get("FLUPROFILER_CACHE_GB", "28"))
max_cache_bytes = int(cache_gb * 1024 ** 3)

logging_info = setup_tensorboard_and_logging(run_paths, exp_id, season_path, device, tag=tag)
writer = logging_info["writer"]
log_path = logging_info["log_path"]
model_save_path = str(run_paths["checkpoints"]) + "/"

data_loaders = load_data_and_dataloaders(
    data_path=data_path,
    season_path=season_path,
    batch_size=8,
    sample_limit=None,
    use_artificial=False,
)
train_dataloader = data_loaders["train_dataloader"]
valid_dataloader = data_loaders["valid_dataloader"]
test_dataloader = data_loaders["test_dataloader"]
emb_dict = data_loaders["emb_dict"]

emb_cache = GpuEmbeddingCache(
    cpu_store=emb_dict,
    device=device,
    max_cache_bytes=max_cache_bytes,
    dtype=torch.float32,
)

with open(root_path + "configs/config_dict.json", "r") as f:
    config_dict = json.load(f)
with open(root_path + "configs/args.pkl", "rb") as f:
    fluProfiler_args = pickle.load(f)

fluProfiler_config = fluProfiler_Config.from_dict(config_dict)
model = fluProfiler_v0_1(config=fluProfiler_config, args=fluProfiler_args)
model.to(device)

model_structure_path = run_paths["run_root"] / "model_structure.txt"
total_params = sum(param.numel() for param in model.parameters())
trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
model_structure_path.write_text(
    "\n".join(
        [
            f"model_class: {model.__class__.__name__}",
            f"device: {device}",
            f"cache_gb: {cache_gb}",
            f"cache_dtype: float32",
            f"total_params: {total_params}",
            f"trainable_params: {trainable_params}",
            "",
            str(model),
        ]
    ),
    encoding="utf-8",
)
print(f"Model structure saved to: {model_structure_path}")
print(f"GPU cache budget: {cache_gb:.2f} GB")

optimizer = setup_optimizer(model, fluProfiler_args, lr=0.00008)

epochs = 100
num_training_steps = len(train_dataloader) * epochs
progress_bar = tqdm(range(num_training_steps))
early_stopping = EarlyStopping(patience=20, save_dir=model_save_path)

for epoch in range(epochs):
    model.train()
    loss_ls = []
    epoch_cache_snapshot = emb_cache.snapshot()
    epoch_train_step_s = 0.0

    for batch in train_dataloader:
        step_start = time.perf_counter()
        loss = train_step(model, batch, emb_cache, device, generate_matrix_gpu)
        loss.backward()
        loss_ls.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        epoch_train_step_s += time.perf_counter() - step_start

    train_loss = np.mean(loss_ls)
    avg_train_step_s = epoch_train_step_s / max(len(train_dataloader), 1)
    print("train loss :", train_loss)

    valid_start = time.perf_counter()
    valid_metrics = evaluate_step(model, valid_dataloader, emb_cache, device, generate_matrix_gpu)
    valid_time_s = time.perf_counter() - valid_start

    test_start = time.perf_counter()
    test_metrics = evaluate_step(model, test_dataloader, emb_cache, device, generate_matrix_gpu)
    test_time_s = time.perf_counter() - test_start

    epoch_cache_delta = emb_cache.delta(epoch_cache_snapshot)
    cache_stats_str = format_cache_stats(
        epoch_cache_delta,
        resident_bytes=emb_cache.current_bytes,
        resident_items=emb_cache.current_items,
    )
    print(cache_stats_str)

    log_metrics_to_tensorboard(writer, epoch, train_loss, valid_metrics, test_metrics)

    early_stopping(valid_metrics["mse"], model)
    if early_stopping.early_stop:
        print("Early stopping")
        append_cache_stats_to_log(
            log_path,
            epoch,
            avg_train_step_s,
            valid_time_s,
            test_time_s,
            epoch_cache_delta,
            emb_cache.current_bytes,
            emb_cache.current_items,
        )
        writer.close()
        break

    log_epoch_to_file(log_path, epoch, epochs, train_loss, valid_metrics, test_metrics)
    append_cache_stats_to_log(
        log_path,
        epoch,
        avg_train_step_s,
        valid_time_s,
        test_time_s,
        epoch_cache_delta,
        emb_cache.current_bytes,
        emb_cache.current_items,
    )

else:
    writer.close()
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("../../../../src/fluprofiler")
sys.path.append("../../")

from experiment_tools_gpu_cache import (
    GpuEmbeddingCache,
    default_exp_id,
    evaluate_step_with_gpu_cache,
    find_repo_root,
    load_data_and_dataloaders,
    log_cache_stats_to_file,
    log_epoch_to_file,
    log_metrics_to_tensorboard,
    make_run_dirs,
    setup_optimizer,
    setup_tensorboard_and_logging,
    train_step_with_gpu_cache,
)
from evaluation.metrics import EarlyStopping
from models.architectures import fluProfiler_Config, fluProfiler_v0_1


_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = find_repo_root(_SCRIPT_PATH)
root_path = str(_REPO_ROOT) + "/"
data_path = root_path + "data/reverse_test/"
season_path = "processed/test_2023SH/"

exp_id = os.environ.get("FLUPROFILER_EXP_ID") or default_exp_id(_SCRIPT_PATH, _REPO_ROOT)
tag = os.environ.get("FLUPROFILER_TAG") or "v0_1_gpu_cache"
run_paths = make_run_dirs(_REPO_ROOT, exp_id=exp_id, tag=tag)

device = torch.device("cuda:1")
cache_gib = float(os.environ.get("FLUPROFILER_MAX_CACHE_GIB", "28"))
max_cache_bytes = int(cache_gib * (1024 ** 3))

logging_info = setup_tensorboard_and_logging(run_paths, exp_id, season_path, device, tag=tag)
writer = logging_info["writer"]
log_path = logging_info["log_path"]
model_save_path = str(run_paths["checkpoints"]) + "/"

data_loaders = load_data_and_dataloaders(
    data_path=data_path,
    season_path=season_path,
    batch_size=8,
    sample_limit=None,
    use_artificial=False,
)
train_dataloader = data_loaders["train_dataloader"]
valid_dataloader = data_loaders["valid_dataloader"]
test_dataloader = data_loaders["test_dataloader"]
emb_dict = data_loaders["emb_dict"]

with open(root_path + "configs/config_dict.json", "r", encoding="utf-8") as f:
    config_dict = json.load(f)
with open(root_path + "configs/args.pkl", "rb") as f:
    fluProfiler_args = pickle.load(f)

fluProfiler_config = fluProfiler_Config.from_dict(config_dict)
model = fluProfiler_v0_1(config=fluProfiler_config, args=fluProfiler_args)
model.to(device)

gpu_cache = GpuEmbeddingCache(
    cpu_store=emb_dict,
    device=device,
    max_cache_bytes=max_cache_bytes,
)

model_structure_path = run_paths["run_root"] / "model_structure.txt"
total_params = sum(param.numel() for param in model.parameters())
trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
model_structure_path.write_text(
    "\n".join(
        [
            f"model_class: {model.__class__.__name__}",
            f"device: {device}",
            f"cache_budget_gib: {cache_gib}",
            f"total_params: {total_params}",
            f"trainable_params: {trainable_params}",
            "",
            str(model),
        ]
    ),
    encoding="utf-8",
)
print(f"Model structure saved to: {model_structure_path}")
print(f"GPU cache budget: {cache_gib:.2f} GiB")

optimizer = setup_optimizer(model, fluProfiler_args, lr=0.00008)

epochs = 100
num_training_steps = len(train_dataloader) * epochs
progress_bar = tqdm(range(num_training_steps))
early_stopping = EarlyStopping(patience=20, save_dir=model_save_path)

for epoch in range(epochs):
    model.train()
    loss_ls = []
    train_step_times = []
    train_build_times = []
    gpu_cache.reset_stats()

    for batch in train_dataloader:
        step_start = time.perf_counter()
        loss, step_stats = train_step_with_gpu_cache(model, batch, gpu_cache, device)
        loss.backward()
        loss_ls.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        train_step_times.append(time.perf_counter() - step_start)
        train_build_times.append(step_stats["batch_build_time"])

    train_loss = np.mean(loss_ls)
    print("train loss :", train_loss)

    valid_metrics, valid_timing = evaluate_step_with_gpu_cache(model, valid_dataloader, gpu_cache, device)
    test_metrics, test_timing = evaluate_step_with_gpu_cache(model, test_dataloader, gpu_cache, device)

    log_metrics_to_tensorboard(writer, epoch, train_loss, valid_metrics, test_metrics)

    cache_stats = gpu_cache.get_stats()
    train_timing = {
        "avg_step_time": float(np.mean(train_step_times)) if train_step_times else 0.0,
        "avg_batch_build_time": float(np.mean(train_build_times)) if train_build_times else 0.0,
    }
    print(
        "cache stats:",
        f"hit_rate={cache_stats['hit_rate']:.4f}",
        f"hits={cache_stats['hits']}",
        f"misses={cache_stats['misses']}",
        f"copy_count={cache_stats['copy_count']}",
        f"copied_gib={cache_stats['copied_bytes'] / (1024 ** 3):.3f}",
        f"avg_step_s={train_timing['avg_step_time']:.4f}",
        f"avg_build_s={train_timing['avg_batch_build_time']:.4f}",
    )

    early_stopping(valid_metrics["mse"], model)
    if early_stopping.early_stop:
        print("Early stopping")
        log_epoch_to_file(log_path, epoch, epochs, train_loss, valid_metrics, test_metrics)
        log_cache_stats_to_file(log_path, epoch, cache_stats, train_timing, valid_timing, test_timing)
        writer.close()
        break

    log_epoch_to_file(log_path, epoch, epochs, train_loss, valid_metrics, test_metrics)
    log_cache_stats_to_file(log_path, epoch, cache_stats, train_timing, valid_timing, test_timing)

else:
    writer.close()

