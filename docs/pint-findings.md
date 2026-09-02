# Pint in SupraSimFit — the practical reference

_A conclusive, repo-specific guide to using Pint here. Read this before touching
anything that carries a physical unit (concentrations, binding constants,
signal). Every claim below was verified against the code and by running Pint in
the repo venv (pint 0.25.2)._

## TL;DR — the ten rules

1. **One registry.** Import `ureg`, `Q_`, `Quantity` from `core/units.py`. Never
   construct another `UnitRegistry`.
2. **Canonical internal units: `M` (concentration), `1/M` (binding constant),
   `au` (signal).** Store and compute in these.
3. **Quantities at the edges, floats in the core.** GUI/IO/config speak
   `Quantity`; the optimizer and forward models speak plain `float`/`ndarray`.
4. **Convert with an explicit target: `q.to('M').magnitude`. Never
   `q.to_base_units()`** — molar's SI base is mol/m³, so it is 1000× off.
5. **`au` has its own dimension `[signal]`.** It is *not* dimensionless. That is
   what keeps `au/M` distinct from `1/M`.
6. **Classify parameters by `ParamKind` (and unit), never by dimensionality
   guesswork.** The registry declares each parameter's kind.
7. **Validate units at each boundary crossing (once per fit), never inside the
   per-iteration kernel.** Boundary checks are free; per-call `@wraps` is ~13× slower.
8. **Serialize magnitude + explicit unit token** (`parameter_units`,
   `condition_units`). Reconstruct with `Q_(value, token)`.
9. **Format for display with Pint** (`~P`/`~H`); only wrap what Pint 0.25 can't do
   (negative-exponent reciprocals, domain math labels).
10. **No silent fallbacks.** A missing/unknown unit surfaces (raise or warn),
    never a silent default.

---

## 1. The shared registry — `core/units.py`

```python
import pint
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
Quantity = pint.Quantity

ureg.define('micromolar = 1e-6 * molar = uM = µM')
ureg.define('nanomolar = 1e-9 * molar = nM')
ureg.define('picomolar = 1e-12 * molar = pM')
ureg.define('millimolar = 1e-3 * molar = mM')
ureg.define('au = [signal] = a.u.')      # signal has its OWN dimension

pint.set_application_registry(ureg)      # pickling / cross-module safety
```

- **Single registry.** Quantities from two different registries raise on any
  mixed operation (`ValueError`/`DimensionalityError`). Everything imports the one
  in `core/units.py`.
- **`set_application_registry(ureg)`** makes this the process-wide default so
  pickled Quantities (multiprocessing, caches) reconstruct against a registry that
  knows `au`, `µM`, … Cheap insurance; keep it.
- **Explicit prefixed units.** Defining `µM`/`nM`/`mM`/`pM` ourselves is what makes
  `Q_(10,'µM').to('M')` correct and unambiguous. (Generic Pint would still parse
  `µM` via SI prefixes, but the explicit definitions are the contract.)

### Why `au = [signal]` and not `au = []`

`au` is an *arbitrary* fluorescence unit. Pint has no built-in fluorescence unit
(`fluorescence`, `RFU`, `absorbance`, `OD` are all undefined; `AU`=absorbance and
`count` exist but are **dimensionless**), so a built-in cannot help. If `au` were
dimensionless (`au = []`, the old definition), then a signal coefficient in
`au/M` would be **dimensionally identical to a binding constant in `1/M`**:

```python
Q_(3, 'au/M').to('1/M')     # old (au=[]) → silently returns 3/molar  ← WRONG
Q_(3, 'au/M').to('1/M')     # now (au=[signal]) → raises DimensionalityError ✓
```

Giving `au` its own base dimension `[signal]` makes the collision impossible and
makes the 4-parameter signal law dimensionally checkable:

```
I_tot = I0[au] + I_dye_free[au/M]·[D][M] + I_dye_bound[au/M]·[HD][M]  →  [signal]
```

`I0` is a signal **offset** (`au`); the coefficients are `au/M`. The definition
`au = [signal] = a.u.` keeps the display symbol `a.u.` (`~P → 'a.u.'`) and the
serialization tokens (`'au'`, `'au / molar'`) **unchanged**, so no data migration.

> Note: `au` shadows Pint's built-in `astronomical_unit` (symbol `au`). That is
> intentional and harmless in this domain. Do **not** rely on the string
> `'a.u.'` as a *parse* token — `ureg('1 a.u.')` mis-parses (the `.`-split reads
> `a·u` = year·atomic-mass-unit). Construct with the token `'au'`; `'a.u.'` is a
> display alias only.

---

## 2. The canonical normalization boundary

Units are attached/validated at the edges and stripped to floats for the numeric
core. The boundaries, in code:

| Boundary | Where | What happens |
|---|---|---|
| GUI numeric input | `gui/widgets/assay_config_panel.py` `_UnitWidget` | display unit → base `Quantity` via `Q_(1, label).to(base).magnitude` (cached float), `value()` returns `Q_(x, base_unit)` |
| File data unit | `gui/widgets/data_panel.py` "Imported Unit" combo → `MeasurementSet.set_concentrations(Q_(vals, unit))` → `.to('M')` | user-declared unit converted to M |
| File reader (in-band unit) | `jasco_reader.py` `XUNITS`, `txt.py` `# units:` header | reader converts to M or tags `df.attrs['concentration_unit']` |
| Assay construction | `core/assays/base.py` `__post_init__`, subclass `__init__` | `x_data.to('M')`, `y_data.to('au')`, conditions `.to('1/M')/.to('M')` — validated, clear errors |
| Params → floats | `core/pipeline/fit_pipeline.py` `fit_assay` | bounds `.to(canonical).magnitude` per parameter, then scipy runs on floats |
| Floats → result | `_wrap_params_as_quantities`, `FitResult` | floats re-wrapped as `Q_(value, registry_unit)` |
| Result → disk | `FitResult.to_dict` | magnitude + `parameter_units`/`condition_units` tokens |
| Result/label → display | `gui/session.py`, `gui/plotting/labels.py`, `fit_summary_widget.py` | Pint `~P`/`~H` formatting |

**The rule:** cross a boundary → attach or validate the unit there. Inside the
boundary (the fit loop, the model math) everything is a validated float.

---

## 3. Float-only core, and why (the benchmark)

`core/models/` and `core/optimizer/` operate on plain floats. Forward models
(`dba_signal`, `ida_signal`, …) take float params + a float `x` array and return
`Q_(result, 'au')`. The optimizer (`multistart_minimize`) never sees a Quantity.

This is not just style — it is required for speed, and it is *safe* because the
boundary already validated the units. Measured (10 000 calls, 20-point arrays):

| Pattern | µs/call | vs bare |
|---|---|---|
| bare-float forward model | 3.2 | 1× |
| `@ureg.wraps` on the per-iteration model | 41 | **12.7× slower** |
| validate dimensionality **once** at the boundary, then bare loop | 3.2 | **1.01×** |

Units are invariant across optimizer iterations, so re-checking them every call
buys nothing. **Do not** push Quantities into scipy or `@wraps` the forward
model; **do** validate once where floats are produced.

---

## 4. Converting: the one correct idiom

```python
conc.to('M').magnitude        # ✓ float in canonical M
Q_(10, 'µM').to('M').magnitude   # 1e-05  ✓
```

**Never `to_base_units()`** for concentration:

```python
Q_(1, 'M').to_base_units()    # 999.999… mol/m³  — 1000× + float noise  ✗
```

`.to('target')` also *validates*: a dimensionally wrong input raises
`DimensionalityError` (fail-fast). Use `.m_as('M')` as the terse equivalent of
`.to('M').magnitude` if you prefer.

---

## 5. Runtime unit-awareness — validate at boundaries

The whole app is unit-checked at runtime, but only at boundary crossings:

- **`base.py __post_init__`** — `x_data` must be `[concentration]`, `y_data` must
  be `[signal]`; a wrong unit raises a domain-clear `ValueError`. (With
  `au=[signal]`, a dimensionless `y_data` now fails here instead of being
  silently accepted.)
- **Assay subclasses** — each condition (`Ka_dye`, `h0`, `d0`, `g0`,
  `fixed_conc`) is `.to(...)`-validated (raises `pint.DimensionalityError` on a
  wrong dimension; this is the established, tested contract).
- **`fit_assay`** — each bound is converted to its parameter's canonical unit
  before `.magnitude`, so a custom bound supplied in a compatible-but-non-canonical
  unit (a Ka in `MM⁻¹`, a concentration in `µM`) converts correctly instead of
  having its face value taken, and a wrong-dimension bound fails fast.
- **Output** — `forward_model` returns `Q_(result, 'au')`; `residuals =
  y_data − forward_model` is `[signal]−[signal]`, so a wrong-dimension model
  output would raise. Results are re-wrapped with registry units.

`@ureg.check`/`@ureg.wraps` are the right *tools* for boundary functions — just
never on the per-iteration kernel.

---

## 6. Parameter identity — `ParamKind` + `[signal]` (the "both" design)

Dimensional analysis alone cannot name a quantity ("dimensionless" covers ratios,
percentages, and — historically — `au`). We use two mutually-reinforcing
mechanisms, cross-checked by a test so neither drifts:

- **`au = [signal]`** gives signal quantities a distinct dimension, so
  `_scale_factor` (`core/optimizer/scaling.py`) classifies a unit component by
  *dimension* (`[signal]` → `s_max`, `[concentration]` → `c_max`, genuine
  dimensionless → left at 1.0) — no `.dimensionless` guesswork, no private-API
  fragility of intent.
- **`ParamKind`** (`core/assays/registry.py`: `BINDING_CONSTANT`,
  `SIGNAL_OFFSET`, `SIGNAL_COEFFICIENT`) is the explicit label. Each registry
  entry declares `param_kinds`; `KIND_UNIT` maps a kind to its canonical unit.

`tests/unit/test_registry_units.py` asserts, for every assay: every parameter has
a kind; each kind is dimensionally consistent with its declared unit; and
`BINDING_CONSTANT ⟺ log_scale_keys`. A future mislabeled parameter fails the
suite instead of silently mis-scaling a fit.

---

## 7. Formatting for display and export

Use Pint's formatters; keep only the small, genuinely-needed wrappers.

```python
f'{Q_(1.68e7, "1/M"):~P}'                 # '1.68×10⁷ 1/M'   (pretty unicode)
f'{Q_(4.3e-6, "M"):.3g~P}'                # '4.3×10⁻⁶ M'
f'{Q_(v, "dimensionless"):.3g~H}'         # magnitude-only HTML: '1.23×10⁶'
fmt_unit_html('1/M')  # 'M<sup>−1</sup>'  (labels.py: Pint ~H + reciprocal fix)
fmt_unit_pretty('1/M')  # 'M⁻¹'
```

- `~` abbreviated, `P` pretty-unicode, `H` HTML, `L`/`Lx` LaTeX/siunitx. Combine
  (`~P`, `.3g~H`).
- **Magnitude-only:** format a *dimensionless* Quantity (`Q_(v,'dimensionless')`)
  — you get the number with HTML/unicode superscripts and no unit. Prefer this to
  string-splitting a formatted Quantity (which is fragile).
- **Legitimately custom** (keep): `labels.py` `fmt_unit_html/pretty` post-process
  `1/M → M⁻¹` (Pint 0.25 has no negative-exponent modifier); `PARAM_LABELS` math
  subscripts (`K<sub>a(D)</sub>`); axis-tick `×10ⁿ` scientific notation.
- Pint's `~P` renders `au`'s symbol as `a.u.` and micromolar as `uM` (ASCII).
  When you need to match a GUI token (`µM`), compare **pint-unit identity**, not
  the formatted string: `next(u for u in UNITS if Q_(1, u).units == q.units)`.
- **Source a displayed unit from the value, not a registry re-lookup.** The fit
  summary and TXT export read `str(value.units)` off each parameter Quantity, so
  the label is right even for an assay type absent from the registry.
- The registry stores the y-axis unit as the `a.u.` alias for raw display; the
  `labels.py` formatters normalise it to `au` first — pint parses the dotted
  `a.u.` as `year·amu`, so a raw `a.u.` would garble to `u a`.

---

## 8. Serialization

`FitResult.to_dict`/`from_dict` store **magnitude + explicit unit token**:

```json
{ "parameters": {"Ka_guest": 1680000.0},
  "parameter_units": {"Ka_guest": "1 / molar"},
  "conditions": {"h0": 4.3e-06}, "condition_units": {"h0": "molar"} }
```

- Reconstruct with `Q_(value, token)`. Tokens round-trip because we own the
  registry (`'au'`, `'au / molar'`, `'1 / molar'` all parse back).
- `parameter_units` **and** `condition_units` are both stored, so parameters *and*
  conditions come back as Quantities (a re-imported result still prints its
  condition units in the TXT report).
- A missing unit is **not** silently defaulted to dimensionless — `from_dict`
  **raises** (the GUI import path surfaces it). Reported spread takes each unit
  from the fitted parameter's own `Quantity` (`summarize_parameters`), never
  from a registry re-lookup.
- `x_fit`/`y_fit` and `custom_bounds` also store their unit tokens (older files
  fall back to the M/au convention), so the whole payload is self-describing.
- `Infinity`/`NaN` are still emitted for failed/linear fits — the app's own
  importer (`json.loads`) reads them and there is no external-consumer need.
- Pint's `to_tuple()`/`from_tuple()` is a registry-independent alternative worth
  knowing, but the readable token scheme is preferred here and needs no change.

---

## 9. `wraps` / `check` — where they belong

- `@ureg.wraps(ret, args)` converts inputs to declared units and strips to
  magnitudes; `@ureg.check('[dim]', …)` enforces dimensionality without
  converting. Both parse/convert **on every call**.
- Use them (or plain `.to()`/`.check()`) on **boundary helpers** called once per
  fit. **Never** on `dba_signal`/`forward_model` or anything scipy calls in the
  loop (measured 12.7× hit, §3).

## 10. Contexts — not needed

Contexts are for cross-dimensional conversions needing extra physics (e.g.
wavelength↔frequency). Every conversion here is a same-dimension scale change
(µM↔M) — plain `.to()` suffices. Only introduce a context if a genuine
cross-dimension conversion (e.g. Ka↔ΔG) is ever added.

---

## 11. Self-describing I/O and the unit contract

`load_measurements(path)` returns `concentration` in **M** and `signal` in **au**
*unless a reader declares otherwise*:

- **JASCO** converts using its `XUNITS` token.
- **TXT** writer emits `# units: concentration=M, signal=au`; `TxtReader` parses a
  declared unit and tags `df.attrs['concentration_unit']`;
  `MeasurementSet.from_dataframe` converts a non-M grid to M.
- Plain CSV/XLSX/TXT with no in-band unit are **assumed M**. A file actually in
  another unit is corrected at the GUI boundary (the "Imported Unit" selector,
  which converts via Pint).
- **Unit tokens are not guessed from column names.** Heuristic label parsing
  (`conc_M`, `[µM]`) is itself a fragile classifier that could introduce the very
  silent errors this design removes — so it is deliberately excluded. Declaration
  is explicit (header/`XUNITS`/GUI) or the value is M by contract.

`read_raw_concentrations(path)` returns a **`Quantity`** (magnitude + unit
together), not a bare array plus a separate unit string — so the two cannot drift.

**Readers declare; every consumer must honor.** A declared unit is worthless if a
consumer ignores it (the re-audit found three that did, each a silent 1e6–1e9
error). The normalization point is `MeasurementSet.from_dataframe` (attr → M);
`extract_concentrations_from_file`, batch import (`load_measurements_multi`
converts each file *before* `pd.concat` drops per-file attrs), and the DataPanel
loader all route through the declared unit — none assumes M behind a reader that
declared otherwise. A concentration unit outside the GUI's nM/µM/mM/M set is
converted to M with a notice, never silently reinterpreted.

---

## Do / Don't (grounded in real code)

| Do | Don't |
|---|---|
| `q.to('M').magnitude` | `q.to_base_units().magnitude` (mol/m³) |
| `Q_(x, 'au')` for signal | assume `au` is dimensionless |
| classify by `ParamKind`/dimension | branch on `.dimensionless` to mean "signal" |
| validate units once at the boundary | `@wraps` the per-iteration forward model |
| `Q_(v,'dimensionless'):.3g~H` for magnitude | `formatted.rsplit(' ',1)[0]` to strip a unit |
| store magnitude + unit token | store magnitude and rely on positional/implicit unit |
| raise/warn on unknown unit | `dict.get(unit, some_default_scale)` |
| compare `Q_(1,u).units == q.units` to map to a GUI token | compare `f'{q.units:~P}'` to a GUI string |

## Anti-pattern checklist

- Second `UnitRegistry` anywhere.
- `.to_base_units()` on a concentration.
- `au` treated as dimensionless / `au/M` converted to `1/M`.
- Dimensionality used as *semantic identity*.
- A hardcoded scale/conversion factor that duplicates or bypasses Pint.
- A silent unit default (`.get(unit, default)`, `else 'dimensionless'`).
- Quantities inside the optimizer / per-iteration `@wraps`.
- `np.asarray(quantity)` (drops units silently) — use `.magnitude`.
- Guessing a unit from a column name.
