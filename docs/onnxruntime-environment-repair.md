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

- ログに WARNING として
  `ONNXモデル '...': cuda を要求しましたが、session は ['CPUExecutionProvider'] で作成されたため CPU で実行されます`
  が出る (INFO のロード成功ログにも `実際のプロバイダー: [...]` が出る。要求した provider のログは DEBUG のみ)
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

**共有の開発 venv では実行しない。** 先に手順を使い捨ての venv で確認し、その後で対象の環境に適用する。
並列で他の `uv` 操作を走らせない (`uv sync` / `uv pip` は直列で実行する)。
Windows では、アプリ (GUI) が `onnxruntime` の `.pyd` を掴んでいると削除に失敗して中途半端な状態になる。
アプリと、同じ venv を使うすべての Python プロセスを終了してから行う。

残す distribution は、このライブラリの依存定義に合わせる (macOS は `onnxruntime`、それ以外は `onnxruntime-gpu`)。
どの場合も最初に、両方の distribution を削除する (片方だけ消さない。手順 2)。

```bash
# Linux
uv pip uninstall --python .venv/bin/python onnxruntime onnxruntime-gpu
```

```powershell
# Windows PowerShell
uv pip uninstall --python .venv\Scripts\python.exe onnxruntime onnxruntime-gpu
```

### lockfile 管理の環境 (LoRAIro など): `uv sync` を使う

両方を削除した後、lockfile どおりの版を入れ直す。lockfile は更新しない (`--frozen`)。

```bash
uv sync --frozen --dev
```

`uv sync` は、メタデータ上は入っているように見える distribution のファイルが欠けていても再導入しない
ことがある。そのため、先に両方を削除してから同期する。

### lockfile の無い環境: 残す distribution だけを `--reinstall-package` で入れ直す

版は、削除前に `uv pip list` で確認した現在の版に固定する。

```bash
# Linux
uv pip install --python .venv/bin/python --reinstall-package onnxruntime-gpu "onnxruntime-gpu==<現在の版>"
```

```powershell
# Windows PowerShell
uv pip install --python .venv\Scripts\python.exe --reinstall-package onnxruntime-gpu "onnxruntime-gpu==<現在の版>"
```

`--reinstall` (package 指定なし) は使わない。解決した全 package を再導入し、無関係な依存まで最新へ
上げてしまう (使い捨て venv で、`--reinstall` が `unrelated` を 1.0 から 2.0 へ上げ、`--reinstall-package` は
上げないことを確認した)。`pip` の場合も、依存を動かさないよう
`python -m pip install --force-reinstall --no-deps "onnxruntime-gpu==<現在の版>"` とする (pip の手順は未検証)。

## 4. 完全性の検証

`tools/verify_distribution_record.py` は、distribution の RECORD に記録された hash と実ファイルを照合する
(標準ライブラリのみ)。検証したい環境の Python で実行する。以下のパスは、image-annotator-lib の
リポジトリ直下から実行する前提である。

```bash
.venv/bin/python tools/verify_distribution_record.py onnxruntime-gpu
```

```powershell
.venv\Scripts\python.exe tools\verify_distribution_record.py onnxruntime-gpu
```

LoRAIro のリポジトリ直下から実行する場合は、サブモジュール内のスクリプトを指定する。

```bash
.venv/bin/python local_packages/image-annotator-lib/tools/verify_distribution_record.py onnxruntime-gpu
```

```powershell
.venv\Scripts\python.exe local_packages\image-annotator-lib\tools\verify_distribution_record.py onnxruntime-gpu
```

- 終了コード 0 かつ `0 problems` なら、RECORD の全ファイルが一致している。RECORD が無い、または hash 付きの項目が
  0 件で照合できなかった場合は、問題ありとして終了コード 1 になる
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
  (後から入れた方が勝つ、片方の削除だけで他方の共有ファイルが消える) と、両方を削除してから
  `--reinstall-package` で入れ直すと完全に戻ることを Linux で確認した。実物の onnxruntime の wheel での
  再現は行っていない。
- `uv sync --frozen --dev` による復旧と、`pip` の手順は実行していない (共有の lockfile・環境に触れるため)。
- Windows PowerShell のコマンドは、パス区切りとインタプリタの場所以外は Linux と同じ操作である。
  Windows 上での実行は、確認 Issue に分離した実機確認で行う。
