# blendipose exit hang (formerly "numba bug")

**Status: FIXED in blendipose.** See `hang_fix_writeup.md` for full details.

The fix was applying `from __future__ import annotations` (PEP 563) to all blendify modules
that had typing generics referencing local classes, and moving type alias imports behind
`if TYPE_CHECKING:`. This prevents `_GenericAlias` objects from being created at import time.

**Note on the name:** later re-investigation (2026-07-21, with stack traces of the hung process
and a condition-ablation matrix) showed that numba is *not* involved. The hang is a bpy shutdown
bug: bpy's C++ static destructor destroys a condition variable that bpy's own
`delayed_close_thread_run()` worker thread is still waiting on, so `pthread_cond_destroy` blocks
forever. The typing generics mattered because `typing._tp_cache` holds a strong reference to the
parametrized class, which transitively (via the class's methods' `__globals__` → module dict) keeps
the `bpy` module object alive through interpreter teardown, preventing bpy's proper thread shutdown
from running before process exit. The PEP 563 fix works because it removes this pinning chain.

The bug does not reproduce on bpy 5.0.1; it appears fixed or defused upstream in Blender 5.0.
