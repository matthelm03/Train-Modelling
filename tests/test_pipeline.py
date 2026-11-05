"""Small offline checks for the data boundary and complete training/export path."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_delay.data import DelayData
from train_delay.models import DelayModel
from train_delay.training import CONFIGS, run_experiment


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="train-delay-tests-")
        self.root = Path(self.temporary.name).resolve()
        if self.root.parent != Path(tempfile.gettempdir()).resolve():
            raise RuntimeError("Unexpected temporary directory")
        self.addCleanup(self.temporary.cleanup)
        settings = json.loads((Path(__file__).resolve().parents[1] / "results/settings.json").read_text())
        fields = [name for name in settings["features"] if not name.startswith("previous_stop_")]
        records, journeys = [], []
        for day, date in enumerate(pd.date_range("2026-03-01", periods=20)):
            journeys.append(dict(id=day + 1, date=date))
            for stop in range(4):
                row = {name: 1 for name in fields}
                row.update(run_id=day + 1, stop_order=stop + 2, station_id=stop + 2,
                           segment_id=f"{stop + 1}_{stop + 2}", category_code="NEW" if day >= 14 else "IC",
                           day_category="weekday", prev_delay_departure_min=5., delta_delay=1.,
                           is_prev_departure_delay_imputed=0, temperature_2m=10., snowfall=0., snow_depth=0.,
                           distance_from_start_km=10. if day < 14 else 1000.)
                if (day, stop) in ((4, 1), (18, 0)):
                    row["distance_from_start_km"] = np.nan
                records.append(row)
        pd.DataFrame(records).to_parquet(self.root / "train_delays_tabular.parquet", index=False)
        pd.DataFrame(journeys).to_csv(self.root / "train_runs.csv", index=False)
        self.data = DelayData(self.root, history=3, batch_size=16)

    def test_preprocessing_uses_only_training_dates(self):
        data = self.data
        distance = data.numeric.index("distance_from_start_km")
        self.assertEqual(data.preprocessor[0].statistics_[distance], 10.)
        self.assertEqual(data.preprocessor[1].mean_[distance], 10.)
        self.assertTrue((data.codes[data.rows["test"], 2] == 0).all())
        for first, second in (("train", "val"), ("val", "test")):
            a, b = (data.df.iloc[data.rows[name]] for name in (first, second))
            self.assertGreaterEqual((b.date.min() - a.date.max()).days, 2)
            self.assertFalse(set(a.run_id) & set(b.run_id))

    def test_history_hides_its_target_and_keeps_windows_separate(self):
        data = self.data
        targets = [0, 3, 4, 7]
        batch = data.make_batch(targets)
        self.assertTrue((batch.x[batch.ptr[1:] - 1, -2:] == 0).all())
        self.assertTrue((batch.batch[batch.edge_index[0]] == batch.batch[batch.edge_index[1]]).all())
        torch.testing.assert_close(batch.ptr.diff(), torch.tensor([1, 3, 1, 3]))
        before = data.make_batch([7]).x.clone()
        data.y[7] += 1000
        torch.testing.assert_close(data.make_batch([7]).x, before, rtol=0, atol=0)

    def test_models_start_at_the_previous_delay_baseline(self):
        batch = self.data.make_batch([0, 3, 4, 7])
        for name, config in CONFIGS.items():
            model = DelayModel(name, config, self.data).eval()
            with torch.no_grad():
                correction = model(batch)
            torch.testing.assert_close(correction, torch.zeros(4), rtol=0, atol=0)

    def test_one_epoch_runs_and_exports_correct_metrics(self):
        output = self.root / "run"
        results, history = run_experiment(self.data, output, epochs=1)
        self.assertEqual(len(results), 4)
        self.assertEqual(len(history), 3)
        predictions = pd.read_csv(output / "predictions.csv")
        actual = predictions.Actual.to_numpy(dtype=np.float32)
        for name in (*CONFIGS, "Baseline"):
            values = predictions[name].to_numpy(dtype=np.float32)
            error = values.astype(float) - actual
            row = results.set_index("Model").loc[name]
            self.assertAlmostEqual(row.MAE, np.abs(values - actual).mean(), places=6)
            self.assertAlmostEqual(row.RMSE, np.sqrt(((values - actual) ** 2).mean()), places=6)
            self.assertAlmostEqual(row.P90_AE, np.percentile(np.abs(error), 90), places=6)
            self.assertAlmostEqual(row.Mean_bias, error.mean(), places=6)
        for name in CONFIGS:
            checkpoint = torch.load(output / f"{name}.pt", weights_only=True)
            self.assertEqual(checkpoint["best_epoch"], 1)
        for name in ("mae.png", "rmse.png", "validation_curves.png", "settings.json", "data_split.csv"):
            self.assertTrue((output / name).is_file())


if __name__ == "__main__":
    unittest.main()
