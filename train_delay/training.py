import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

from .models import DelayModel


CONFIGS = {
    "LSTM": dict(lr=.003, hidden=32, layers=2, dropout=.1),
    "Transformer": dict(lr=.003, hidden=32, layers=2, dropout=.1),
    "GNN": dict(lr=.001, hidden=16, layers=1, dropout=0.),
}

@torch.no_grad()
def evaluate(data, split, model=None):
    values = data.baseline[data.rows[split]]
    if model is not None:
        model.eval()
        # Convert predicted corrections to minutes and add the previous-delay baseline.
        values = torch.cat([batch.baseline + data.scale * model(batch) for batch in data.loader(split)])
    actual, values = data.y[data.rows[split]].numpy(), values.numpy()
    error = values.astype(float) - actual  # Negative = underprediction.
    scores = dict(MAE=mean_absolute_error(actual, values),
                  RMSE=np.sqrt(mean_squared_error(actual, values)), R2=r2_score(actual, values),
                  P90_AE=float(np.percentile(np.abs(error), 90)), Mean_bias=float(error.mean()))
    return values, scores


def fit(data, name, config, output, epochs=20, patience=4):
    random.seed(data.seed)
    np.random.seed(data.seed)
    torch.manual_seed(data.seed)
    model = DelayModel(name, config, data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=.0001)
    # Lower the learning rate when validation MAE stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2,
                    threshold=1e-4, threshold_mode="abs", min_lr=1e-6)
    best, best_epoch, history = float("inf"), 0, []
    train_loader = data.loader("train", shuffle=True)
    for epoch in range(1, epochs + 1):
        model.train()
        absolute_error = 0.
        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            # Learn the correction to the baseline, measuring training error in minutes.
            error = (model(batch) - (batch.y - batch.baseline) / data.scale) * data.scale
            loss = error.abs().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            absolute_error += error.detach().abs().sum().item()
            if step % 100 == 0:
                (output / "progress.json").write_text(json.dumps(dict(model=name, epoch=epoch,
                    batch=step, batches=len(train_loader))))
        _, scores = evaluate(data, "val", model)
        score = scores["MAE"]
        history.append(dict(Model=name, Epoch=epoch, Train_MAE=absolute_error / len(data.rows["train"]),
                            Validation_MAE=score, Learning_rate=optimizer.param_groups[0]["lr"]))
        # Choose checkpoints using validation MAE; test data is kept out of this decision.
        if score < best - 1e-5:
            best, best_epoch, state = score, epoch, copy.deepcopy(model.state_dict())
            torch.save(dict(state_dict=state, config=config, best_epoch=best_epoch), output / f"{name}.pt")
        scheduler.step(score)
        pd.DataFrame(history).to_csv(output / f"{name}_history.csv", index=False)
        print(f"{name} epoch {epoch}: validation MAE {score:.4f}; best epoch {best_epoch}", flush=True)
        if epoch >= 5 and epoch - best_epoch >= patience:
            break
    model.load_state_dict(state)  # Best validation checkpoint.
    return model, history


def plot_results(results, history, validation_baseline, output):
    # Use separate scales for MAE and RMSE; the labels show the actual differences.
    for metric in ["MAE", "RMSE"]:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        ax.scatter(results.Model, results[metric], s=60)
        for position, value in enumerate(results[metric]):
            ax.annotate(f"{value:.4f}", (position, value), xytext=(0, 9), textcoords="offset points", ha="center")
        ax.set(title=f"Future-date test {metric} (lower is better)", ylabel=f"{metric} (minutes)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.margins(y=.3)
        ax.grid(axis="y", alpha=.2)
        fig.savefig(output / f"{metric.lower()}.png")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    history.pivot(index="Epoch", columns="Model", values="Validation_MAE").plot(ax=ax)
    ax.axhline(validation_baseline["MAE"], color="grey", linestyle="--", label="Baseline")
    ax.set(title="Validation learning curves", ylabel="MAE (minutes)")
    ax.legend()
    fig.savefig(output / "validation_curves.png")
    plt.close(fig)


def run_experiment(data, output, epochs=20, patience=4):
    """Train, score and export one chronological comparison."""
    output = Path(output)
    models, histories = {}, []
    output.mkdir(parents=True, exist_ok=True)
    for name, config in CONFIGS.items():
        models[name], history = fit(data, name, config, output, epochs, patience)
        histories.extend(history)
    history = pd.DataFrame(histories)

    predictions, scores = {}, []
    # Compare all models and the baseline on exactly the same held-out stops.
    for name, model in {**models, "Baseline": None}.items():
        predictions[name], metrics = evaluate(data, "test", model)
        scores.append(dict(Model=name, **metrics))
    results = pd.DataFrame(scores)
    print(results.round(4).to_string(index=False))
    predictions = pd.DataFrame(predictions).assign(
        row_id=data.rows["test"], source_row_id=data.df.source_row_id.iloc[data.rows["test"]].to_numpy(),
        run_id=data.df.run_id.iloc[data.rows["test"]].to_numpy(), date=data.df.date.iloc[data.rows["test"]].to_numpy(),
        Actual=data.y[data.rows["test"]].numpy())
    splits = data.df[["source_row_id", "run_id", "date"]].copy()
    splits["Split"] = "gap"
    for split, ids in data.rows.items():
        splits.loc[ids, "Split"] = split
    # Save scores, predictions and split details so results can be checked later.
    for name, table in {"comparison": results, "history": history, "predictions": predictions,
                        "data_split": splits, "split_summary": data.summary}.items():
        table.to_csv(output / f"{name}.csv", index=False)
    settings = dict(seed=data.seed, history=data.history, batch_size=data.batch_size, epochs=epochs, patience=patience,
                    gap_days=data.gap_days, configs=CONFIGS, features=data.numeric + data.categories,
                    source="https://zenodo.org/records/23283780")
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    _, validation_baseline = evaluate(data, "val")
    plot_results(results, history, validation_baseline, output)
    return results, history
