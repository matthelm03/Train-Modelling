# Dataset

This project uses **PKP Intercity delays**, version 2, by **Marek Kostrz**:
[Zenodo record](https://zenodo.org/records/23283780),
[DOI: 10.5281/zenodo.23283780](https://doi.org/10.5281/zenodo.23283780).
The source lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
This attribution describes the external dataset, not a license grant for this project's code.

From the repository root, run:

```sh
python scripts/download_data.py
```

The script downloads the approximately 65 MB source archive and verifies its
published MD5 checksum, then extracts and verifies these SHA-256 checksums:

- `train_delays_tabular.parquet`: 558,334 source stop records and 57 columns.
- `train_runs.csv`: service dates used to construct chronological splits.
- `SOURCE_README.md`: the publisher's original documentation.

Exact paths, checksums and the source version are recorded in [source.json](source.json).
Downloaded files are ignored by Git. If you already have the original archive:

```sh
python scripts/download_data.py --archive path/to/pkp_intercity_delays_dataset.zip
```

Preprocessing removes exact duplicates, rows without finite targets or previous
delays, and rows flagged as having an imputed previous departure delay. It shifts
weather within each journey and fits imputation, scaling and category mappings
on training rows only. The arrival-delay target is reconstructed as
`prev_delay_departure_min + delta_delay`.

The experiment creates its own chronological splits. It does not use the
publisher's benchmark split assignments, so scores are not directly comparable
with results using those assignments.
