from sae_lens.cache_activations_runner import CacheActivationsRunner, CacheActivationsRunnerConfig

path = 'research/data'

context_size = 1024
n_batches_in_buffer = 10
store_batch_size = 128
device = "cuda:0"
total_training_tokens = 40_000 * context_size

cfg = CacheActivationsRunnerConfig(
    new_cached_activations_path=path,
    # cached_activations_path=path
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
    dtype="float32",
)

runner = CacheActivationsRunner(cfg)
runner.run()