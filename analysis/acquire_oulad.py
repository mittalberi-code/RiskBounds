"""Fetch and safely extract the official OULAD release from the approved UCI host."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
HOST = "archive.ics.uci.edu"
EXPECTED = {"assessments.csv", "courses.csv", "studentAssessment.csv", "studentInfo.csv",
            "studentRegistration.csv", "studentVle.csv", "vle.csv", "OULAD.names"}


class SameHostRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urlsplit(urljoin(req.full_url, newurl))
        if target.scheme != "https" or target.hostname != HOST:
            raise OSError(f"redirect outside approved HTTPS host refused: {target.hostname}")
        return super().redirect_request(req, fp, code, msg, headers, target.geturl())


def validate_members(names: list[str]) -> None:
    if set(names) != EXPECTED or len(names) != len(EXPECTED):
        raise ValueError(f"unexpected archive member set: {sorted(names)}")
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or len(p.parts) != 1 or ".." in p.parts or "\\" in name:
            raise ValueError(f"unsafe archive path: {name}")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(archive: Path, timeout: int = 30) -> dict:
    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(archive.suffix + ".part")
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {archive}")
    if part.exists():
        raise FileExistsError(f"partial download exists; inspect before retry: {part}")
    request = Request(URL, headers={"User-Agent": "academic-risk-landmark-study/1.0"})
    opener = build_opener(SameHostRedirect())
    with opener.open(request, timeout=timeout) as response:
        final = urlsplit(response.geturl())
        if response.status != 200 or final.scheme != "https" or final.hostname != HOST:
            raise OSError(f"unexpected HTTP response or destination: {response.status} {final.hostname}")
        expected_size = response.headers.get("Content-Length")
        received = 0
        with part.open("xb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                received += len(chunk)
            f.flush()
            os.fsync(f.fileno())
    if expected_size is not None and received != int(expected_size):
        raise OSError(f"truncated download: expected {expected_size} bytes, received {received}")
    if received < 1_000_000:
        raise OSError(f"archive response too small to be the OULAD release: {received} bytes")
    try:
        with zipfile.ZipFile(part) as zf:
            validate_members(zf.namelist())
            corrupt = zf.testzip()
            if corrupt:
                raise OSError(f"ZIP CRC failed at {corrupt}")
    except zipfile.BadZipFile as exc:
        raise OSError("download is not a valid ZIP (possible HTML/error response)") from exc
    os.replace(part, archive)
    return {"url": URL, "retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bytes": archive.stat().st_size, "sha256": hash_file(archive)}


def extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        validate_members(zf.namelist())
        corrupt = zf.testzip()
        if corrupt:
            raise OSError(f"ZIP CRC failed at {corrupt}")
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite extraction directory: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="oulad-extract-", dir=destination.parent) as tmp:
            stage = Path(tmp)
            for name in zf.namelist():
                target = stage / name
                with zf.open(name) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            # Atomic publication of the complete verified directory.
            os.replace(stage, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/oulad.zip"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/raw/extracted"))
    args = parser.parse_args()
    metadata = fetch(args.archive)
    extract(args.archive, args.extract_dir)
    print(f"Acquired {metadata['bytes']} bytes; SHA256 {metadata['sha256']}; extracted verified OULAD tables.")


if __name__ == "__main__":
    main()
