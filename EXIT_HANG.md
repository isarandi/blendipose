# Exit hang on bpy 4.x (fixed; do not reintroduce)

On bpy 4.5.x the process can hang forever *after* Python finalization: bpy's C++ static
destructor `~GlobalState()` calls `pthread_cond_destroy()` on a condvar that bpy's own
`delayed_close_thread_run()` worker thread is still waiting on, and glibc blocks until the
waiter drains — which never happens. Not reproducible on bpy 5.0.1 (fixed/defused upstream);
numba is *not* involved (early analysis misattributed it).

The hang only triggers if the `bpy` module object stays alive through interpreter teardown, so
bpy's proper thread-stopping cleanup never runs. Evaluating a typing generic like
`Sequence[SomeClass]` at import time does exactly that: `typing._tp_cache` keeps a strong
reference to the class, which pins its module's globals — and thus `bpy` — via
`SomeClass.method.__globals__`.

```python
# Minimal repro (bpy 4.5.x): prints "done", then hangs forever.
from typing import Sequence
import bpy

class Colors:
    def foo(self): ...

X = Sequence[Colors]
print("done", flush=True)
```

**The fix in this repo:** `from __future__ import annotations` in every blendify module, with
module-level type aliases and their imports behind `if TYPE_CHECKING:`. Keep it that way — an
import-time typing generic (including `@dataclass` field annotations, which are evaluated at
class creation) in any bpy-importing module can re-trigger the hang on bpy 4.x. Any other
durable strong reference to such a class (an `lru_cache`, a global registry) can too.

A full writeup with gdb stacks, the ablation matrix, and the history of the earlier wrong
"numba deadlock" analysis is in git history: `hang_fix_writeup.md`, deleted by the same commit
that introduced this summary.
