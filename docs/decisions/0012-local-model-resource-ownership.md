---
type: ADR
title: "Local Model Context Ownership and Resource Budgets"
status: Accepted
timestamp: 2026-10-10
tags: [models, memory, lifecycle]
---
# ADR 0012: Local Model Context Ownership and Resource Budgets

## Context

[Issue #165](https://github.com/NEXTAltair/image-annotator-lib/issues/165) removes
the GPU-to-CPU round trip at every successful Transformers annotation exit.
Retaining weights requires admission and release to distinguish an executing
owner from an idle cached model. A RAM-only LRU cannot govern GPU residency.
The initial VRAM repair exposed these shared ownership and accounting gaps;
[Issue #174](https://github.com/NEXTAltair/image-annotator-lib/issues/174) compares
owner leases with exclusive local execution. The user selected owner leases.

## Decision

Reuse LoaderBase's state, LRU and release callbacks. Protect resource transitions
with a standard-library reentrant lock; inference does not hold that lock.
The local runner also creates cached annotator instances under this lock.

Each context reserves its model name and owner before loading. Entry, reuse,
failed entry, and exit balance a lease count. A superclass context method call
is part of its outer method dispatch; a separate nested `with` acquires another
lease. The owner remains pinned until the last matching exit. Different active
owners, and concurrent entry of the same owner on another thread, are rejected
with ModelLoadError. This does not promise concurrent inference safety.

All built-in local frameworks inherit LocalModelAnnotator and decorate their
context methods with `local_model_enter` and `local_model_exit`. A concrete
subclass overriding those methods must apply the same decorators so preparation
and cleanup remain inside the shared contract. WebAPI contexts are unaffected.

Automatic RAM and VRAM eviction selects only idle models. An idle same-name owner
can be replaced; a rejected active replacement cannot run cleanup for the
incumbent. Public release requests during use are deferred to final exit.
Device migration is allowed during the owning preparation or final exit and
rejected while the model is being used in a context body.

Ordinary nested context exceptions defer cleanup until final exit. A model's own
OOM can invalidate its unusable generation immediately, without consuming the
outstanding leases. Re-entry registers its release callback again. Allocator
cleanup pending after OOM is distinct from an explicit release request: final
exit may clean unused blocks while retaining a newly recovered generation.

Model-size estimates remain in `_MEMORY_USAGE` for compatibility. The separate
host ledger governs host cache admission. Transformers weights transferred by
the loader to CUDA are excluded from the host weight cache; host staging still
passes the physical available-RAM check. CPU weights and backends without proof
of GPU-only weight placement retain their full host estimate. An `on_cuda` label
alone is insufficient proof: ONNX may use CPU providers and TensorFlow may place
weights differently from the requested device.

CPU offload and GPU restoration replace the transitioning model's host footprint
instead of adding its size a second time. Host LRU victims must have reclaimable
host weight bytes. GPU LRU victims must be idle and on the target GPU. Unused
PyTorch allocator blocks are trimmed before comparing driver free VRAM to weight
size. Known capacity shortage is reported after idle eviction; a valid CPU cache
can continue through the existing restoration fallback.

## Rationale

The alternative exclusive execution contract would reject overlapping public
contexts and still require resource-specific budgets. Leases preserve reusable
models while expressing precisely when their references may be released. The
existing LRU and callbacks remain the source of release behavior; no new runtime
dependency is required. Python's
[RLock](https://docs.python.org/3/library/threading.html#rlock-objects) permits
the existing nested loader/facade calls to share one transition lock.

## Consequences

Repeated successful Transformers chunks reuse the same weights on their effective
device. Normal Pipeline/CLIP CPU offload and TensorFlow final release continue,
but only at final context exit. ModelLoad.release_model_components returns the
original dictionary while release is deferred, allowing the final callback to
clear actual aliases instead of only changing accounting.

Budgets estimate model weights. They do not bound inference intermediates or
prove every CUDA allocation will succeed. Actual GPU performance and allocation
peaks require hardware measurement; the regression suite replaces downloads and
device probes while exercising real ownership, LRU and accounting paths.
TensorFlow batching and runtime placement policy remain tracked separately in
[Issue #172](https://github.com/NEXTAltair/image-annotator-lib/issues/172).
