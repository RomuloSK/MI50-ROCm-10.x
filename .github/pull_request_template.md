## Summary

<!-- What changed, and which component owns the change? -->

## Verification

- [ ] `python -m pytest -q`
- [ ] `bash -n` passed for changed shell scripts
- [ ] Relevant artifact/runtime validator was run
- [ ] GPU-dependent results are labeled `GPU-test-pending` unless tested on a real MI50
- [ ] No ISA override is used to masquerade as gfx906

## Compatibility and risk

<!-- Describe target data, fallback behavior, regressions, and recovery path. -->
