"""Fetch the versioned Zenodo dataset and verify its archive and extracted files."""
import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


def digest(path, algorithm="sha256"):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def prepare_data(destination, archive=None):
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "data/source.json").read_text(encoding="utf-8"))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in source["files"].items():
        path = destination / name
        if path.exists() and digest(path) != expected:
            raise ValueError(f"Existing {path} differs from the source. Choose an empty --data-dir.")
    if all((destination / name).exists() for name in source["files"]):
        print("Dataset files already present; all checksums verified.")
        return

    downloaded = archive is None
    if downloaded:
        archive = destination / ".dataset-download.zip"
        request = urllib.request.Request(source["archive"]["url"], headers={"User-Agent": "train-delay-prediction"})
        print(f"Downloading {source['archive']['size'] / 1e6:.1f} MB from Zenodo...", flush=True)
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
    archive = Path(archive)
    algorithm, expected = source["archive"]["checksum"].split(":", 1)
    if archive.stat().st_size != source["archive"]["size"] or digest(archive, algorithm) != expected:
        raise ValueError("Archive verification failed; obtain a fresh copy of the versioned dataset.")

    with zipfile.ZipFile(archive) as zipped:
        # Extract only the three named files; ignore all other archive paths.
        for name, member in source["members"].items():
            temporary = destination / (name + ".part")
            with zipped.open(member) as input_file, temporary.open("wb") as output:
                shutil.copyfileobj(input_file, output)
            if digest(temporary) != source["files"][name]:
                temporary.unlink()
                raise ValueError(f"Extracted checksum mismatch: {name}")
            temporary.replace(destination / name)
            print(f"Verified {name}", flush=True)
    if downloaded:
        archive.unlink()


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--archive", type=Path, help="Use an already downloaded source ZIP instead of the network.")
    args = parser.parse_args()
    prepare_data(args.data_dir, args.archive)
