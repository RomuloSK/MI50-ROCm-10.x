# Contributing

Contributions should keep the MI50/gfx906 support contract explicit and
reproducible.

## Before opening a pull request

1. Run `python -m pytest -q` on the host.
2. Run `bash -n` over the changed shell scripts.
3. If the change affects a build, packaging or runtime path, run the relevant
   validator and record the command and result in the pull request.
4. Never use `HSA_OVERRIDE_GFX_VERSION` or `ROCR_OVERRIDE_GFX_VERSION` to make
   a newer ISA appear to be gfx906.
5. Keep GPU execution claims marked `GPU-test-pending` until a real MI50 has
   exercised the changed path.

## Patch queue rules

Place patches in the subsystem directory that owns the affected source and
keep one behavior change per patch. Update the appropriate lock file with the
patch digest and record:

- the exact pinned source revision;
- a minimal reproducer or static assertion;
- why the change is safe for gfx906;
- fallback behavior when the target data is unavailable; and
- host-only and hardware test evidence.

Do not commit generated build trees, downloaded source checkouts, wheels,
archives, credentials or hardware logs. Use the ignored output directories or
GitHub Releases for those artifacts.
