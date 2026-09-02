# tests/ — Testing Philosophy

This suite protects the **scientific correctness** of a binding-assay fitting
toolkit. A test earns its place by catching a plausible math or logic bug that
would otherwise ship silently. Use these principles to decide *what* to test and
to write tests that are meaningful rather than ceremonial.

## What's worth testing — the three pillars

Most tests should serve one of these:

1. **Scientific validity** — does the math do the right thing? Forward-model
   values at known inputs, conservation / mass-balance laws, limiting cases
   (no-guest → DBA, zero concentration, saturation), and parameter recovery:
   fit synthetic data with known ground truth and assert the fitted parameters
   return within tolerance.
2. **Data integrity** — do I/O and transforms preserve values? Round-trip a
   write→read, a unit conversion, a reshape; assert what went in comes out.
3. **Fail-fast contracts** — do invalid inputs raise immediately with a clear
   message instead of silently producing garbage? (Wrong units, missing
   conditions, mismatched shapes, non-positive constants.)

If a proposed test doesn't serve one of these, question whether it's worth adding.

## How to write a meaningful test

- **Test a property, not an implementation.** Assert an invariant, a limit, a
  conservation law, or recovery-within-tolerance — not that a line of code
  returns what that same line of code computes.
- **Never check source against itself.** Computing the expected value with the
  same formula the code uses (or calling the same method on both sides of an
  assertion) is a tautology that passes even when the code is wrong. Derive the
  expected value an **independent** way: solve the equilibrium by a different
  method, hand-compute a small case, or use a closed-form limit.
- **Prefer analytical / direct checks over fit-then-compare.** A direct model
  call with an independently-computed expected value is faster and stronger than
  running the optimizer and inspecting its output. Reserve full multi-start fits
  for genuine end-to-end recovery tests — one or two representative cases, not
  every parameter combination.
- **For fits, assert recovery within a tolerance band, not bit-equality.**
  Convention: ~10 % for clean synthetic data, ~25 % for ~5 % Gaussian noise.
- **One test, one property.** Each test pins a single invariant or behavior. If
  two tests catch the same class of bug, keep the stronger one.
- **Pin observed behavior for edge cases — don't assume it.** For degenerate or
  boundary inputs (NaN, empty array, single point, underdetermined fit), run it
  once, see what actually happens, and assert that with a comment on *why* it's
  correct.
- **Reproduce a bug as a failing test first**, then make it pass.

## What to avoid

- Safety-blanket tests with no logic beneath them: constructor-sets-attribute,
  trivial getters, `repr`, default-value echoes, config dicts that merely
  round-trip. Skip these unless one guards a specific past regression.
- Sprawling `parametrize` matrices that drive one code path many times. Cover the
  nominal case, the boundaries (zero / edge), and one pathological case; justify
  anything beyond that.
- Slow tests where a smaller input or an analytical value verifies the same math.
  A slow suite is run less often, so bugs survive longer.

## Repo-specific truths that make tests meaningful

- **Reuse the synthetic ground truth.** `tests/conftest.py` is the single source
  of known-truth parameters, data generators, recovery bounds, and the tolerance
  helper. Build on it; don't re-roll constants or synthetic data per file.
- **Fits are deterministic by seeding** (the global RNG is seeded per test), so
  treat fit results as reproducible — but assert the recovered *physics*, never
  exact optimizer output.
- **Signal coefficients are degenerate.** In DBA/IDA only `Ka` is identifiable
  (`I0`, `I_dye_free`, `I_dye_bound` trade off). Test `Ka` recovery and signal
  reconstruction — never the individual signal coefficients.
- **GUI tests never pop a window.** A widget test that needs real geometry
  (`tabRect()`, child positions, `isVisible()`) has to be laid out, and Qt only
  computes that on `show()`. Set `WA_DontShowOnScreen` before showing: the
  layout resolves, nothing appears on screen, and the test still runs under the
  platform's native style. A visible window interrupts whoever is running the
  suite, and needs a display server that CI will not have.
- Layout: `tests/unit/` is the fast layer, `tests/unit/gui/` holds PyQt6 widget
  tests, `tests/integration/` runs real fits (the slow part).

## Running

```bash
uv run pytest                  # full suite
uv run pytest tests/unit       # fast layer only (skips the integration fits)
uv run pytest -k <pattern>     # iterate on a subset
uv run pytest -x               # stop on first failure
```
