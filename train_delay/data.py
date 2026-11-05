from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader
from torch_geometric.data import Data


class DelayData:
    """Clean the source data and build stop-history batches for all three models."""

    def __init__(self, data_dir, history=24, batch_size=512, seed=42, gap_days=1):
        data_dir = Path(data_dir)
        if not all((data_dir / name).is_file() for name in ("train_delays_tabular.parquet", "train_runs.csv")):
            raise FileNotFoundError("Dataset missing. Run: python scripts/download_data.py")
        torch.set_num_threads(8)
        self.history, self.batch_size, self.seed = history, batch_size, seed
        self.gap_days = gap_days
        raw = pd.read_parquet(data_dir / "train_delays_tabular.parquet")
        self.df = raw.drop_duplicates().copy()
        self.df["source_row_id"] = self.df.index
        self.df["arrival_delay_min"] = self.df.prev_delay_departure_min + self.df.delta_delay  # Arrival-delay target (minutes).
        # Remove missing/infinite labels and estimated previous delays.
        valid = np.isfinite(self.df[["arrival_delay_min", "prev_delay_departure_min"]]).all(axis=1)    
        self.df = self.df.loc[valid & self.df.is_prev_departure_delay_imputed.eq(0)]
        # Add each journey's service date.
        run_dates = pd.read_csv(data_dir / "train_runs.csv", usecols=["id", "date"], parse_dates=["date"])
        self.df = self.df.merge(run_dates.rename(columns={"id": "run_id"}), on="run_id", how="left", validate="many_to_one")
        self.df = self.df.sort_values(["date", "run_id", "stop_order"], ignore_index=True)
        weather = ["temperature_2m", "snowfall", "snow_depth"]
        # Use weather from the preceding retained stop in the same journey.
        self.df = self.df.join(self.df.groupby("run_id")[weather].shift().add_prefix("previous_stop_"))

        # Reserve later dates for validation/testing, with unused days between splits.
        dates = np.sort(self.df.date.unique())
        validation_cut, test_cut = dates[int(len(dates) * .7)], dates[int(len(dates) * .85)]
        gap = pd.Timedelta(days=gap_days)
        masks = {"train": self.df.date < validation_cut,
                 "val": (self.df.date >= validation_cut + gap) & (self.df.date < test_cut),
                 "test": self.df.date >= test_cut + gap}
        self.rows = {split: np.flatnonzero(mask) for split, mask in masks.items()}
        self.summary = pd.DataFrame([
            dict(Split=split, Start=self.df.date.iloc[ids].min().date(), End=self.df.date.iloc[ids].max().date(),
                 Journeys=self.df.run_id.iloc[ids].nunique(), Stops=len(ids)) for split, ids in self.rows.items()
        ])
        self.source_stats = dict(source_rows=len(raw), duplicates=int(raw.duplicated().sum()), usable_rows=len(self.df))
        del raw

        self.numeric = [
            "prev_delay_departure_min", "stop_order", "distance_from_start_km", "is_international",
            "arrival_hour_sin", "arrival_hour_cos", "departure_hour_sin", "departure_hour_cos",
            "day_of_week", "month_sin", "month_cos", "is_track_closure", "track_closure_length",
            "is_speed_warning", "speed_warning_length", "passenger_volume_rank", "num_platforms",
            "num_platform_tracks", "intersecting_lines_count", "is_node", "is_passing_loop",
            *[f"previous_stop_{column}" for column in weather],
        ]
        # Category labels, such as station and route IDs.
        self.categories = ["station_id", "segment_id", "category_code", "day_category"]

        def prepare(values, transformer, dtype=torch.float32):
            # Learn preprocessing settings from training rows only, then apply to all rows.
            return torch.tensor(transformer.fit(values.iloc[self.rows["train"]]).transform(values), dtype=dtype)

        # Fill missing numbers with training medians, add missing-value flags, then scale.
        self.preprocessor = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), StandardScaler())
        self.x = prepare(self.df[self.numeric].replace([np.inf, -np.inf], np.nan), self.preprocessor)
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.codes = prepare(self.df[self.categories].astype("string").fillna("unknown"), self.encoder, torch.long) + 1  # 0 = unseen category.
        self.category_sizes = [len(values) + 1 for values in self.encoder.categories_]

        self.y = torch.tensor(self.df.arrival_delay_min.to_numpy(), dtype=torch.float32)  # Actual delay (minutes).
        self.baseline = torch.tensor(self.df.prev_delay_departure_min.to_numpy(), dtype=torch.float32)  # Baseline (minutes).
        # Use training statistics to scale past delays and the correction to the baseline.
        train_y = self.y[self.rows["train"]]
        self.history_mean, self.history_scale = train_y.mean(), max(train_y.std(unbiased=False).item(), 1.)
        self.scale = max((train_y - self.baseline[self.rows["train"]]).std(unbiased=False).item(), 1.)

        # Count available stops within each journey.
        self.history_lengths = torch.tensor(self.df.groupby("run_id", sort=False).cumcount().to_numpy() + 1)
        # Only connect consecutive stops with matching route endpoints.
        segment_start = self.df.segment_id.astype(str).str.split("_").str[0].astype(int)
        self.adjacent = torch.tensor((self.df.run_id.eq(self.df.run_id.shift()) & self.df.stop_order.diff().eq(1)
                                & segment_start.eq(self.df.station_id.shift())).to_numpy())

    def make_batch(self, row_ids):
        targets = torch.tensor(row_ids, dtype=torch.long)
        # Take up to HISTORY stops ending at each target and flatten the windows.
        lengths, steps = self.history_lengths[targets].clamp(max=self.history), torch.arange(self.history)
        ids = (targets[:, None] - lengths[:, None] + 1 + steps)[steps < lengths[:, None]]
        ptr = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])  # Window boundaries.
        batch = torch.repeat_interleave(torch.arange(len(targets)), lengths)  # Window ID for each node.
        # Add normalised past delays and an observed-delay flag.
        history = torch.column_stack([(self.y[ids] - self.history_mean) / self.history_scale, torch.ones(len(ids))])
        history[ptr[1:] - 1] = 0  # Hide the target delay and flag.
        edges = torch.arange(len(ids) - 1)[batch[:-1].eq(batch[1:]) & self.adjacent[ids[1:]]]  # Edges within each window.
        return Data(x=torch.column_stack([self.x[ids], history]), categories=self.codes[ids],
                    edge_index=torch.stack([edges, edges + 1]), batch=batch, ptr=ptr,
                    y=self.y[targets], baseline=self.baseline[targets], row_id=targets)

    def loader(self, split, shuffle=False):
        return DataLoader(self.rows[split], batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.make_batch,
                          generator=torch.Generator().manual_seed(self.seed))
