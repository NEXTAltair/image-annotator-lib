---
type: Runbook
title: onnxruntime の CPU 版と GPU 版が混在した環境の修復
description: onnxruntime と onnxruntime-gpu が同じ環境に入って CUDA 指定でも CPU 実行になる状態の診断・修復・完全性検証の手順
status: Accepted
timestamp: 2026-11-22
tags: [onnx, dependencies, cuda, troubleshooting]
depends_on: [onnxruntime, onnxruntime-gpu, uv]
---
# onnxruntime の CPU 版と GPU 版が混在した環境の修復

`onnxruntime` と `onnxruntime-gpu` は、どちらも同じ `onnxruntime` モジュールのファイルを配置する。
同時に入れると後から書かれたファイルが残り、CUDA 版を入れていても CPU 版のバイナリが読み込まれることがある
([Issue #175](https://github.com/NEXTAltair/image-annotator-lib/issues/175))。
公式も、同じ環境には CPU 版と GPU 版の片方だけを入れるよう案内している
([Install ONNX Runtime](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime))。

このライブラリの依存定義は、macOS だけ CPU 版、それ以外は GPU 版を要求する (環境 marker で排他)。
通常の依存同期では混在しない。以下は、混在してしまった既存の環境を直す手順である。

## 1. 症状と診断

次のいずれかなら混在を疑う。

- ログに `ONNX要求プロバイダー: ['CUDAExecutionProvider', ...]` の後で
  `session は [...'CPUExecutionProvider'] で作成されたため CPU で実行されます` と出る
- `onnxruntime.get_available_providers()` に `CUDAExecutionProvider` が無いのに、`onnxruntime-gpu` が入っている

対象環境の Python で、両 distribution の有無と provider 一覧を確認する。

```bash
# Linux (環境の python を指定する)
uv pip list --python .venv/bin/python | grep -i onnxruntime
.venv/bin/python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

```powershell
# Windows PowerShell
uv pip list --python .venv\Scripts\python.exe | Select-String onnxruntime
.venv\Scripts\python.exe -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

`onnxruntime` と `onnxruntime-gpu` の両方が一覧に出たら混在している。

## 2. なぜ「片方を消すだけ」では直らないか

二つの distribution は同じパス (`onnxruntime/capi/...` など) にファイルを持つ。

- 後から入れた方が先のファイルを上書きする (RECORD には先の hash が残るので、先の distribution は不一致になる)
- 片方を uninstall すると、その RECORD に載っている共有パスが削除され、残した方のファイルも消える

そのため、片方を消して `import onnxruntime` が通っても正常とは言えない。必ず、残す distribution を
入れ直し、RECORD の hash と照合する (手順 4)。

## 3. 修復

共有の開発 venv に対しては、並列で他の `uv` 操作を走らせない (`uv sync` / `uv pip` は直列で実行する)。
Windows では、アプリ (GUI) が `onnxruntime` の `.pyd` を掴んでいると削除に失敗して中途半端な状態になる。
アプリと、同じ venv を使うすべての Python プロセスを終了してから行う。

残す distribution は、このライブラリの依存定義に合わせる (macOS は `onnxruntime`、それ以外は `onnxruntime-gpu`)。

```bash
# Linux: 両方を削除し、残す版を入れ直す
uv pip uninstall --python .venv/bin/python onnxruntime onnxruntime-gpu
uv pip install --python .venv/bin/python --reinstall "onnxruntime-gpu>=1.16.0"
```

```powershell
# Windows PowerShell
uv pip uninstall --python .venv\Scripts\python.exe onnxruntime onnxruntime-gpu
uv pip install --python .venv\Scripts\python.exe --reinstall "onnxruntime-gpu>=1.16.0"
```

LoRAIro のように lockfile 管理の環境では、両方を `uv pip uninstall` した後に `uv sync --dev` で
lockfile どおりの版に戻す。`uv sync` だけでは、メタデータ上は入っているように見える片方のファイルが
欠けていても再導入されないことがあるため、先に両方を削除する。

`pip` を使う環境は `python -m pip uninstall -y onnxruntime onnxruntime-gpu` の後に
`python -m pip install --force-reinstall onnxruntime-gpu` とする。

## 4. 完全性の検証

`tools/verify_distribution_record.py` は、distribution の RECORD に記録された hash と実ファイルを照合する
(標準ライブラリのみ)。検証したい環境の Python で実行する。

```bash
.venv/bin/python tools/verify_distribution_record.py onnxruntime-gpu
```

```powershell
.venv\Scripts\python.exe tools\verify_distribution_record.py onnxruntime-gpu
```

- 終了コード 0 かつ `0 problems` なら、RECORD の全ファイルが一致している
- `missing:` は他方の uninstall でファイルが消えた状態、`mismatch:` は他方に上書きされた状態。手順 3 をやり直す
- もう片方 (例: `onnxruntime`) が `not installed` (終了コード 2) になっていることも確認する

最後に、`CUDAExecutionProvider` が提供されていることを確かめる。ただしこれは「その版のビルドが CUDA EP を含む」
ことしか示さない。実際に GPU で session が作れるか・推論が通るかは、CUDA ドライバと
CUDA / cuDNN の互換が要るため、GPU のある環境で別途確認する。

```bash
.venv/bin/python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

## 検証した範囲

- 同じパスに別内容のファイルを置く 2 つの distribution を使い捨て venv に導入し、上記の症状
  (後から入れた方が勝つ、片方の削除だけで他方の共有ファイルが消える) と、手順 3・4 で完全に戻ることを
  Linux で確認した。実物の onnxruntime の wheel での再現は行っていない。
- Windows PowerShell のコマンドは、パス区切りとインタプリタの場所以外は Linux と同じ操作である。
  Windows 上での実行は、確認 Issue に分離した実機確認で行う。
