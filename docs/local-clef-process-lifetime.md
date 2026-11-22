---
type: Reference
title: ローカル Clef server の所有権と回収
description: llama-server 子プロセスの終了責任と、所有プロセスが異常終了したときの OS による回収
status: Accepted
timestamp: 2026-11-22
tags: [decision, local, clef, lifecycle]
depends_on: [psutil]
---

# ローカル Clef server の所有権と回収

[ADR 0011](decisions/0011-typed-clef-decision-api.md) の managed local runtime が持つ
「所有する server」の意味と、終了責任の分担を説明する。

## 誰が終了責任を持つか

| 場面 | 責任を持つもの | 動作 |
| --- | --- | --- |
| 設定変更・推論失敗・`shutdown_local_runtime()`・通常終了 | server を起動したプロセスの runtime | terminate、応答しなければ kill、回収、一時キー削除 |
| 所有プロセスの強制終了・クラッシュ・`os._exit`・`exec` | OS | server を強制終了する (OS ごとの例外は下記)。Python の後始末は走らない |
| fork した子プロセス | 子自身が起動した server だけ | 親の server とキーには触れない。子の終了で親の server は止まらない |

所有者は「プロセス」であり、判定を呼んだスレッドやクライアントではない。worker thread が終わっても
server は残り、次の判定で再利用される。LoRAIro などの利用側は PID を管理しない。

## OS ごとの仕掛け

### Linux

- 短い Python の launcher が `PR_SET_PDEATHSIG(SIGKILL)` を設定し、設定済みを所有側の起動スレッドへ報告する。
  起動スレッドの返信を受けてから llama-server を exec する。PID は変わらない。
- 返信が届くのは、設定の時点で起動スレッドが生きていた場合だけである。所有プロセスが設定より前に
  終了・exec した場合は返信が届かず (close-on-exec の pipe が閉じる)、server は起動しない。
- この signal は子を作ったスレッドの終了で届く。そのため server は、プロセスの終了まで残る専用の
  daemon thread からだけ起動する。**このスレッドを終わらせたり、別のスレッドから直接起動したりしない。**
- 届くのは server 本体だけで、server が作った孫プロセスには届かない。
- set-user-ID、set-group-ID、capability 付きの実行ファイルでは kernel が設定を消すため、回収されない。
- `sys.executable` が通常の Python であることを前提にする。
- インタプリタの終了処理の後半 (`sys.is_finalizing()` が真) では起動スレッドが止まるため、起動せず
  `configuration` エラーにする。

### Windows

- server を停止状態で起動し、`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` 付きの Job Object に入れてから再開する。
  登録または再開に失敗したら kill し、判定は `configuration` エラーになる。
- Job は所有プロセスごとに 1 つで、継承不可の handle を終了まで保持する。**途中で閉じない。**
  所有プロセスが終わると OS が handle を閉じ、server とその子孫が停止する。
- 生成から登録までの間に所有プロセスが強制終了されると、停止状態の server が残り得る。
  実行前なのでモデルは読み込まれていない。

### その他の OS

異常終了時の回収は無い。通常終了時の後始末だけが働く。

## 対象外

- アイドル時の自動停止。メモリを先に解放する場合は `shutdown_local_runtime()` を呼ぶ。
- 強制終了後に残る一時キーのディレクトリの削除。キーは停止した server 専用で、再利用されない。
