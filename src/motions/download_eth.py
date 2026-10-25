"""Download and verify the official ETH Walking Pedestrians annotations."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = (
    "https://ethz.ch/content/dam/ethz/special-interest/itet/cvl/vision-dam/"
    "documents/ewap_dataset_light.tgz"
)
SHA256 = "3e057d7c8631c9b5ae638b0d74d454d7be567532a51cd5e06f0323bfe863ce98"


def download(output: str | Path) -> Path:
    output = Path(output).resolve()
    target = output / "ewap_dataset"
    if (target / "seq_eth" / "obsmat.txt").exists() and (
        target / "seq_hotel" / "obsmat.txt"
    ).exists():
        return target
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tgz") as temporary:
        urllib.request.urlretrieve(URL, temporary.name)
        digest = hashlib.sha256(Path(temporary.name).read_bytes()).hexdigest()
        if digest != SHA256:
            raise RuntimeError(f"ETH archive checksum mismatch: {digest}")
        with tarfile.open(temporary.name, "r:gz") as archive:
            for member in archive.getmembers():
                destination = (output / member.name).resolve()
                if output not in destination.parents and destination != output:
                    raise RuntimeError(f"unsafe archive path: {member.name}")
                if not (member.isfile() or member.isdir()):
                    raise RuntimeError(f"unsupported archive member: {member.name}")
            archive.extractall(output)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official ETH trajectory annotations")
    parser.add_argument("--output", default="data/real")
    args = parser.parse_args()
    print(download(args.output))


if __name__ == "__main__":
    main()
