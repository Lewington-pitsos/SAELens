import time
import gc
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.cache_activations_runner import CacheActivationsRunnerConfig

path = 'research/data'

context_size = 1024
n_batches_in_buffer = 10
store_batch_size = 128
device = "cuda:0"
total_training_tokens = 10_000

cfg = CacheActivationsRunnerConfig(
    new_cached_activations_path=path,
    cached_activations_path=path,
    # new_cached_activations_path=cached_activations_fixture_path,
    # Pick a tiny model to make this easier.
    model_name="gelu-1l",
    ## MLP Layer 0 ##
    hook_name="blocks.0.hook_mlp_out",
    hook_layer=0,
    d_in=512,
    dataset_path="NeelNanda/c4-tokenized-2b",
    context_size=context_size,  # Speed things up.
    is_dataset_tokenized=True,
    prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
    training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
    train_batch_size_tokens=4096,
    # Loss Function
    ## Reconstruction Coefficient.
    # Buffer details won't matter in we cache / shuffle our activations ahead of time.
    n_batches_in_buffer=n_batches_in_buffer,
    store_batch_size_prompts=store_batch_size,
    normalize_activations="none",
    #
    shuffle_every_n_buffers=2,
    n_shuffles_with_last_section=1,
    n_shuffles_in_entire_dir=1,
    n_shuffles_final=1,
    # Misc
    device=device,
    seed=42,
    dtype="float16",
)


model = HookedTransformer.from_pretrained(cfg.model_name)
activations_store = ActivationsStore.from_config(model, cfg)
n_batches = 1000 + 4


with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=n_batches),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
) as p:
    start = time.time()
    for i in range(n_batches):
        batch = activations_store.next_batch()
        p.step()

    end = time.time()
    duration = end - start
    print(f"Time taken to get {n_batches} batches: {duration:.2f} seconds")
    print(f"Batches Per Second: {n_batches / duration:.2f} batches per second")
    print(f"Samples Per Second: {n_batches * store_batch_size / duration:,} samples per second")
    print(f"Tokens Per Second: {n_batches * store_batch_size * context_size / duration:,} tokens per second")


print(p.key_averages().table(sort_by="cpu_time_total", row_limit=10))