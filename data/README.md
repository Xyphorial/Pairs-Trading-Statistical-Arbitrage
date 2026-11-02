# Historical adjusted-close input

`adjusted_close.csv` from the original research bundle is **not included** in this public GitHub version. The full price history may have market-data redistribution restrictions.

To reproduce with updated publicly available quotes, see the data source instructions in the root README and use `project.py --download`. The constituent snapshot in `universe_snapshot.csv` is a saved current-membership snapshot, not historical point-in-time index membership. This causes survivorship bias.

The archive of published results is preserved in `results/real_data/`, and was produced using the original historical input. Re-downloading historical quotes may yield different results because data corrections, ticker coverage and index constituents change over time. See `data/provenance.json` and `results/real_data/run_metadata.json`.
