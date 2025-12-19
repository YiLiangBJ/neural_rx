# Configuration Files Comparison

This document summarizes the parameter names and differences across the .cfg files in the `config/` folder of the repository.

## Summary

- All config files share the same main sections: `global`, `system`, `neural_receiver`, `training`, `evaluation`, `baseline`.
- Two PRB/bandwidth related fields exist in most configs:
  - `n_size_bwp` — the PRB/grid size used in training / default workflows.
  - `n_size_bwp_eval` — the PRB/grid size used for evaluation (inference) runs.

## Key findings

1. `Parameters` behavior

- In `utils/parameters.py`, `Parameters.__init__` overwrites several fields when `training==False` (evaluation mode). Among them:
  - `self.n_size_bwp = self.n_size_bwp_eval`
  - Several other fields are also switched to their `_eval` counterparts (e.g. `channel_type`, `max_ut_velocity`, etc.).

- Consequently:
  - Training runs (created with `Parameters(..., training=True)`) use `n_size_bwp`.
  - Evaluation runs (created with `Parameters(..., training=False)`) use `n_size_bwp_eval`.

2. `train_neural_rx.py` vs `evaluate.py`

- `train_neural_rx.py` constructs `Parameters(config_name, system='nrx', training=True)` and therefore uses `n_size_bwp` during training.
- `evaluate.py` constructs a short-lived dummy `Parameters(config_name, training=True, system='dummy')` only to load labels/paths, but when running the actual evaluation it creates `Parameters(config_name, training=False, num_tx_eval=..., system='nrx')`. The latter is used to build transmitters, resource grids and channel: therefore evaluation uses `n_size_bwp_eval`.

3. Which config files differ from the common set

The majority of `.cfg` files use the shared parameter set. The following files contain extra options beyond the common parameters:

- `nrx_large_var_mcs.cfg`
  - `[training]` extras: `double_readout`, `mcs_training_snr_db_offset`

- `nrx_large_var_mcs_64qam_masking.cfg`
  - `[neural_receiver]` extras: `mcs_var_mcs_masking`
  - `[training]` extras: `double_readout`, `mcs_training_snr_db_offset`

- `nrx_rt_var_mcs.cfg`
  - `[training]` extras: `double_readout`, `mcs_training_snr_db_offset`

- `nrx_site_specific*.cfg` (site-specific variants)
  - `[training]` extras: `random_subsampling` (dataset related)
  - `[evaluation]` extras: `random_subsampling_eval`

All other config files in `config/` match the common parameter set.

## Practical implications and suggestions

- If you want identical behavior for training and evaluation, set `n_size_bwp_eval` equal to `n_size_bwp` in the chosen config file.
- If you modify `n_size_bwp_eval` (e.g. increase PRBs for evaluation), these changes automatically affect transmitters, resource grids, and channel initialization because `Parameters` assigns `self.n_size_bwp = self.n_size_bwp_eval` in evaluation mode.
- Training loops that perform internal evaluations may create `Parameters(..., training=False)` for the evaluation step; check `utils/training_loop.py` if you need strict control over which grid size is used inside training-time evaluations.

## Files scanned

All `.cfg` files under `config/` were scanned. They include (examples):
```
config/e2e_baseline.cfg
config/e2e_large.cfg
config/e2e_rt.cfg
config/nrx_large.cfg
config/nrx_large_64qam.cfg
config/nrx_large_qpsk.cfg
config/nrx_large_var_mcs.cfg
config/nrx_large_var_mcs_64qam_masking.cfg
config/nrx_rt.cfg
config/nrx_rt_64qam.cfg
config/nrx_rt_qpsk.cfg
config/nrx_rt_var_mcs.cfg
config/nrx_site_specific*.cfg
```

## Where to place breakpoints / checks

- `utils/parameters.py`: lines near the block that overwrites eval fields (search for "Overwrite channel and PRBs in inference mode"). This is the authoritative place where `n_size_bwp` gets replaced by `n_size_bwp_eval` in evaluation mode.
- `scripts/train_neural_rx.py`: where `Parameters(..., training=True)` is created.
- `scripts/evaluate.py`: where `Parameters(..., training=False)` is created inside the evaluation loop.

---

If you want, I can:
- Export a CSV/table with every config file and its exact parameter values, or
- Add a short note in `utils/parameters.py` as a comment warning about the runtime overwrite behavior (I can prepare a small patch and run tests if you want).

