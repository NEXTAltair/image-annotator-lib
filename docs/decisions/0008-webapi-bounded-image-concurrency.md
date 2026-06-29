---
type: ADR
title: "WebAPI Bounded Image Concurrency"
status: Accepted
timestamp: 2026-06-03
tags: [webapi, performance]
depends_on: [openai-api]
---
# ADR 0008: WebAPI Bounded Image Concurrency

## Context

WebAPI inference preprocesses an image batch quickly, then sends one provider request per image.
The previous implementation awaited each image request sequentially inside one model invocation. In GUI
batch annotation this made progress appear stalled during the model inference phase, and total runtime
became the sum of all provider round trips and retry budgets.

## Decision

Run per-image WebAPI requests concurrently with an `asyncio.Semaphore` limit. The default limit is `3`,
and callers may override it with `max_concurrency` on `ProviderManager` or `WebApiAnnotator`.

The same bounded behavior applies to the OpenAI Moderations special path so rating-only WebAPI models do
not remain serial.

## Consequences

- Batch latency improves when provider requests are I/O-bound.
- Result pHash keys, per-image error outcomes, and input-order result construction are preserved.
- Provider rate limits are protected by a conservative default instead of unbounded `gather`.
- `max_concurrency=1` remains available for serial behavior when a provider or caller requires it.
