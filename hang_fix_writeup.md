# Fix: Process hang on exit when using blendify with bpy

## Problem

When bpy (Blender Python module) is used together with blendify-style code, the process can hang
indefinitely on exit. Originally observed in combination with numba (hence the historical name
"numba bug"), but numba turned out to be incidental — see "History: the earlier (wrong) analysis"
below. A minimal trigger on bpy 4.5.x:

```python
# trigger.py (inside any module, including __main__)
from typing import Sequence
import bpy

class Colors:          # any class defined in a module that imports bpy
    def foo(self):
        pass

X = Sequence[Colors]   # creating the generic alias is the trigger
print("done")          # prints, but the process never exits
```

The hang is deterministic. Tested on bpy 4.5.11, Python 3.11.14, Linux/glibc. It does **not**
reproduce on bpy 5.0.1 — apparently fixed or defused upstream in Blender 5.0.

## Root cause

Two independent facts combine:

### 1. bpy has a shutdown bug (the actual hang)

A gdb backtrace of the hung process shows that Python interpreter finalization **completes
successfully** — `main()` returns 0 and the process enters libc's `exit()`, which runs C++ static
destructors. There it blocks:

```
Thread 1 (main):
#0  futex_wait (...)                                    <- blocked forever
#2  __GI___pthread_cond_destroy (cond=...g_state+112)
#3  (anonymous namespace)::GlobalState::~GlobalState()  <- bpy/__init__.so static destructor
#4  __run_exit_handlers (run_dtors=true)
#5  __GI_exit

Thread 53:
#3  __pthread_cond_wait_common (cond=...g_state+112)    <- still waiting on the SAME condvar
#5  delayed_close_thread_run()                          <- bpy's own background worker
```

bpy spawns a background worker thread `delayed_close_thread_run()` that waits on a condition
variable in a file-local `GlobalState`. At process exit, `~GlobalState()` (a C++ static destructor)
calls `pthread_cond_destroy()` on that condvar **while the worker is still waiting on it**. glibc's
`pthread_cond_destroy` blocks until all waiters have drained; this waiter never wakes, so the
process hangs forever in a futex wait. The destructor should signal shutdown and join the worker
first (or skip destruction at process exit).

There is no lock-vs-lock deadlock between libraries, no GC frame, no weakref callback, and no numba
frame anywhere in the blocking relationship. It is a single-library bug in bpy's teardown. The
remaining ~80 threads of the hung process are idle OpenBLAS and TBB pool workers, uninvolved.

### 2. typing generics pin the bpy module alive (why annotations mattered)

Whether the hang triggers depends on whether bpy's *proper* cleanup (which stops the worker) runs
during Python finalization. That in turn depends on whether the `bpy` module object gets torn down
during finalization — and typing generics can prevent exactly that.

Evaluating `Sequence[Colors]` at module scope stores the alias in `typing._tp_cache` (an
`lru_cache` inside the typing module) with a **strong** reference to `Colors`. Measured directly:
zero new weakrefs are created on the class; its refcount rises by 2; the sole referrer of the alias
is an `_lru_cache_wrapper`; and the alias survives `del X` with identity intact (`Sequence[Colors]
is Sequence[Colors]` after deleting all user references).

The pinning chain is:

```
typing._tp_cache  →  Colors  →  Colors.foo.__globals__  →  trigger module dict  →  bpy module
     (strong)                                                                        object
```

If the module defining `Colors` also contains `import bpy`, this chain keeps the bpy module object
reachable through interpreter teardown, bpy's thread-stopping cleanup never runs, and the process
reaches the C++ static destructors with the worker thread still parked on the condvar → hang.

### Conditions (verified by ablation, each result deterministic across repeated runs)

| Variant                                                          | Result |
|------------------------------------------------------------------|--------|
| Full original repro (numba jit called, ABC class, package import)| HANG   |
| No generic alias (control)                                       | clean  |
| Alias on a plain **non-ABC** class                               | HANG   |
| ABC + alias, but `import bpy` moved to a *different* module      | clean  |
| Alias `del`'d — survives only inside `typing._tp_cache`          | HANG   |
| numba imported, `@njit` defined but **not called**               | clean  |
| **No numba at all** (numpy + package import only)                | HANG   |
| Bare `import repro_pkg`, nothing else in the script              | HANG   |
| Everything in `__main__`, no package, no numba                   | HANG   |

So the only real requirements are: (a) a typing generic parametrized with a class, (b) defined in a
module whose globals (transitively, via the class's methods' `__globals__`) reference the `bpy`
module, (c) on an affected bpy version (4.5.x; not 5.0.1). ABC inheritance, numba, and
package-vs-`__main__` are all irrelevant — see the history section for how they got blamed.

More generally, typing generics are just one common way to create the pinning chain. *Any* durable
strong reference to a class defined in a bpy-importing module — an `lru_cache`, a global registry, a
logging handler holding a bound method, etc. — could trigger the same hang on bpy 4.x.

## History: the earlier (wrong) analysis

An earlier version of this document attributed the hang to a "three-way deadlock during interpreter
shutdown" between numba's LLVM thread pool, bpy's C++ destructors, and ABC weakref callbacks
supposedly registered by `_GenericAlias` creation. Stack traces later disproved all three legs:
finalization completes before the hang; `_GenericAlias` creation registers no weakrefs (it holds
strong references); and numba appears nowhere in the blocked stacks.

The numba misattribution is instructive. All original ablations kept `import numba` and the
`@numba.njit` decorator in place and only toggled whether the function was *called*. It turns out
that importing numba and defining a jit function **suppresses** the hang (clean exit), while
calling the function perturbs teardown ordering enough to **unmask** it again. Within that slice of
the configuration space, "called vs. not called" flips the hang, so numba looked causal — but the
hang reproduces with numba entirely absent. Lesson: when ablating trigger conditions, also test the
corners where the suspected component is fully removed, and confirm the mechanism with a stack
trace of the actual blocked state rather than inferring it from which ablations flip the symptom.

(Caveat: the original tests ran on bpy 4.5.7 with an older numpy; the re-investigation used the
same env after drift to bpy 4.5.11 / numpy 2.4.2. The exact condition set may have differed then,
but the blocked-stack mechanism is unambiguous.)

## Failed approach: `atexit.register(os._exit, 0)`

Before the proper fix, a brute-force workaround was tried in `blendipose/__init__.py`:

```python
import atexit, os
atexit.register(os._exit, 0)
```

This skips the rest of process teardown, avoiding the hang. But `os._exit` bypasses all cleanup:

- **stdout/stderr buffers are not flushed** — unflushed `print()` output is silently lost (this
  was discovered when stdout appeared to be "swallowed" after importing blendipose).
- **Other atexit handlers don't run** — anything registered by libraries or user code is skipped.
- **`__del__` destructors don't run** — open file handles, temp file cleanup, etc.
- **Coverage data is lost** — `coverage.py` writes results in an atexit handler.

Flushing all IO objects before `os._exit` reduces the damage but it remains a nuclear option. The
workaround has been removed now that the proper fix is in place. (Note that with the corrected
mechanism in mind, a targeted variant — an atexit handler that runs *after* bpy's cleanup — would
not help either: the hang happens after all of Python finalization, in C++ static destructors.)

## The fix in blendipose: lazy annotations via PEP 563

Add `from __future__ import annotations` to all modules that have typing generics in their
annotations, and move module-level type aliases behind `if TYPE_CHECKING:`. This makes annotations
lazy strings, so `Union[Light, str]` is stored as the string `"Union[Light, str]"` and no
`_GenericAlias` is created at import time — hence no `_tp_cache` entry, no pinning chain, and bpy
gets torn down normally during finalization.

This applies to all annotation contexts: function/method parameters, return types, variable
annotations, and `@dataclass` field annotations. Notably, `@dataclass` evaluates field annotations
at class creation time to build `__init__`, so without the future import, fields like
`extra_nodes: Tuple[bpy.types.ShaderNode]` create `_GenericAlias` objects at import time.

Module-level type alias *assignments* (not annotations) execute regardless of the future import and
must be guarded behind `if TYPE_CHECKING:`.

### Files changed

**`blendify/internal/types.py`**
```python
# Added at top:
from __future__ import annotations
from typing import TYPE_CHECKING

# Wrapped all type aliases behind guard:
if TYPE_CHECKING:
    from typing import Union, Tuple
    import bpy
    import numpy as np
    BlenderGroup = Union[bpy.types.Collection, bpy.types.Object]
    Vector2d = Union[np.ndarray, Tuple[float, float]]
    # ... etc
```

**All files importing from `internal/types.py`** (14 files):
```python
# Pattern applied to each:
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..internal.types import Vector3d, RotationParams  # etc.
```

Files: `internal/positionable.py`, `cameras/base.py`, `cameras/common.py`, `lights/base.py`,
`lights/common.py`, `lights/area.py`, `lights/collection.py`, `colors/base.py`,
`colors/common.py`, `renderables/primitives.py`, `renderables/collection.py`,
`renderables/base.py`, `renderables/mesh.py`, `utils/image.py`, `utils/pointcloud.py`, `scene.py`.

**`renderables/pointcloud.py`** — key trigger found via bisecting: `@dataclass` fields with
`Tuple[bpy.types.ShaderNode]` and `Optional[bpy.types.Image]` annotations created `_GenericAlias`
objects at class creation time. Fixed with the future import.

**`internal/parser.py`** — bare `bpy.types.Object` in signatures; bare class references don't
create `_GenericAlias`, but the future import was added for uniformity.

**Additional files** with typing generics in annotations: `colors/texture.py`, `materials/base.py`.

**`blendipose/__init__.py`**: removed the `atexit.register(os._exit)` workaround.

The fix was verified by clean exits with bpy imported and numba jit functions executed. With the
corrected mechanism understood, the fix's effectiveness has a crisper explanation: what matters is
that no `_tp_cache` entry pins a blendify class (whose modules all transitively reference bpy)
across interpreter teardown.

Note the fix's limits: it removes the *typing* pinning chain from blendipose's own modules. Any
other mechanism that keeps the bpy module alive at exit (caches or registries in user code holding
blendipose/bpy-adjacent objects) could re-trigger the hang on bpy 4.x.

## Upstream fixes

- **Blender (the actual bug).** `~GlobalState()` in bpy's `__init__.so` must not call
  `pthread_cond_destroy` on a condvar with a live waiter; it should signal shutdown and join
  `delayed_close_thread_run()` first, or skip destruction at process exit. The bug does not
  reproduce on bpy 5.0.1, so it appears already fixed or defused there; worth reporting against
  4.5 LTS with the stack trace above if 4.5 support matters. No existing upstream report was found.
- **Upgrading bpy.** `pyproject.toml` already allows `bpy>=3.6,<6.0`; moving to bpy 5.x makes the
  hang moot (verify the full blendipose import against 5.x when upgrading).
- **CPython / typing.** `typing._tp_cache` holding strong references to parametrized classes is a
  known sharp edge: it silently extends the lifetime of user classes (and, transitively, their
  modules and everything those modules import) to interpreter teardown. PEP 649/749 (Python 3.14+)
  makes lazy annotations the default, which removes the most common way this cache gets populated.

## Lessons for library authors

**Type annotations create runtime objects with surprising lifetimes.** Until lazy annotations are
the default (Python 3.14+), `def foo(x: Union[MyClass, str])` at module scope executes
`Union.__getitem__`, which caches the result in `typing._tp_cache` with a strong reference to
`MyClass`. Through `MyClass`'s methods' `__globals__`, that pins the whole defining module and
every module it imports — including C extension modules — until interpreter teardown. If any such
extension misbehaves when it is still alive at process exit (as bpy 4.5 does), an innocent-looking
annotation becomes a hang.

**The safe default: `from __future__ import annotations` everywhere,** plus `if TYPE_CHECKING:`
for module-level aliases and their imports. It costs nothing — annotations still work for type
checkers and IDEs — and keeps annotations inert at runtime.

**Don't trust a deadlock story without a stack trace.** The original analysis inferred a plausible
mechanism from which ablations flipped the symptom, and got every component wrong. One
`gdb -p <pid> -batch -ex "thread apply all bt"` of the hung process identified the real culprit in
minutes. For ptrace-restricted systems (`yama/ptrace_scope=1`), have the target opt in:
`ctypes.CDLL("libc.so.6").prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)` (PR_SET_PTRACER_ANY).

**Ablate to zero.** When testing whether a component is required for a bug, include variants where
it is completely absent, not just disabled. Teardown-ordering bugs are especially prone to masking
effects where an unrelated import flips the symptom.

**Don't use `os._exit` as a workaround for shutdown bugs.** It trades a visible bug (hang) for
invisible ones (lost output, skipped cleanup, lost coverage data).

## Reproduction

Minimal (bpy 4.5.x, no other packages needed):

```python
# test_hang.py — run directly; prints "done", then hangs forever
from typing import Sequence
import bpy

class Colors:
    def foo(self):
        pass

X = Sequence[Colors]
print("done", flush=True)
```

Control: delete the `X = Sequence[Colors]` line → clean exit. Moving `import bpy` into a different
module from `Colors` → clean exit. Environment: bpy 4.5.11, Python 3.11.14, glibc 2.39
(micromamba env `blender_bpy`); clean on bpy 5.0.1 (env `blender`).
