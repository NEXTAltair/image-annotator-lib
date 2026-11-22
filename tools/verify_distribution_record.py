"""インストール済み distribution のファイルが RECORD の hash と一致するか検証する (標準ライブラリのみ)。

onnxruntime と onnxruntime-gpu のように同じ配置先へファイルを書く distribution を同時に入れると、
片方を削除してもう片方の共有ファイルまで消える・上書きされる。残す distribution の完全性は
``pip show`` や import の成否ではなく、RECORD に記録された hash との照合で確かめる (Issue #175)。

使い方 (検証したい環境の Python で実行する)::

    python tools/verify_distribution_record.py onnxruntime-gpu
    python tools/verify_distribution_record.py onnxruntime

終了コード: 0=完全 / 1=不一致・欠落あり / 2=未インストール・引数不正。
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys
from importlib import metadata


def _record_digest(path: str, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")


def _hashed_entries(distribution: metadata.Distribution) -> list[tuple[str, str, str]]:
    """RECORD から (相対パス, hash 方式, hash 値) を読む。

    ``Distribution.files`` は存在しないファイルを黙って除外する (Python 3.13 で確認)。削除された
    ファイルこそ検出したいので、RECORD を直接読む。
    """
    text = distribution.read_text("RECORD") or ""
    entries = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) >= 2 and "=" in row[1]:
            mode, value = row[1].split("=", 1)
            entries.append((row[0], mode, value))
    return entries


def verify_distribution(distribution: metadata.Distribution) -> list[str]:
    """RECORD に hash がある全ファイルを照合し、問題のあるファイルの説明を返す。

    Args:
        distribution: 検証する distribution。

    Returns:
        ``missing: <path>`` または ``mismatch: <path>`` の一覧。空なら完全。
    """
    problems: list[str] = []
    for relative, mode, value in _hashed_entries(distribution):
        located = str(distribution.locate_file(relative))
        try:
            actual = _record_digest(located, mode)
        except FileNotFoundError:
            problems.append(f"missing: {relative}")
            continue
        if actual != value:
            problems.append(f"mismatch: {relative}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    try:
        distribution = metadata.distribution(argv[0])
    except metadata.PackageNotFoundError:
        print(f"not installed: {argv[0]}")
        return 2
    problems = verify_distribution(distribution)
    for problem in problems:
        print(problem)
    checked = len(_hashed_entries(distribution))
    print(f"{argv[0]} {distribution.version}: {checked} files checked, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
