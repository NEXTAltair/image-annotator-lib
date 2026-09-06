---
type: Reference
title: Read-only model configuration policy
status: Accepted
tags: [model-registry, validation]
---

# Read-only configuration initialization

Set the supported environment variable `IMAGE_ANNOTATOR_CONFIG_READ_ONLY=1`
**before importing** `image_annotator_lib` when automatic configuration writes are
not permitted. `true` and `yes` are also accepted, case-insensitively. Unset/default
preserves existing automatic template creation and persistence behavior.

```python
import os
os.environ["IMAGE_ANNOTATOR_CONFIG_READ_ONLY"] = "1"
from image_annotator_lib import list_annotator_info
models = list_annotator_info()
```

Public type imports remain available without configuration. Explicit registry initialization
or `list_annotator_info()` requires the existing system model configuration at the normal
`SYSTEM_CONFIG_PATH` (the import-time CWD's `config/annotator_config.toml`). It loads
that file without creating its directory, copying the template, or writing any
configuration. Missing/unreadable/invalid required configuration (including non-table model entries) raises public
`ReadOnlyConfigError`. Prepare configuration with write permission using the usual
writable initialization, then retry. The exception carries `details.config_path`
and `details.action`; callers may map it to a precondition error.

The policy is enforced inside `ModelConfigRegistry` at explicit public initialization/list calls.
If the file disappears between existence validation and reading, loading fails
without copying a replacement. `save_system_config`, `save_user_config`, and
`save_runtime_cache` reject persistence while the policy is enabled, including
registries initialized before the policy was enabled. Existing optional user and
runtime-cache files are read normally; malformed non-table user overrides are ignored,
preserving valid system model entries. Model inference/downloads are separate APIs;
this policy does not authorize or make those operations read-only.

The environment variable name is also exported as `CONFIG_READ_ONLY_ENV`, and
`ReadOnlyConfigError` is exported at the package root. Set the literal environment
name first when a cold import itself must be protected; importing a submodule also
executes the package initializer.

Host applications temporarily enabling the policy must serialize that process-wide
environment scope against other library initialization/persistence, restore its
previous value on success or failure, and avoid interleaving asynchronous scopes.
Diagnostic logging is separate and may still create log directories/files. No API
keys or provider settings are changed by this policy.
