"""Scientist-facing descriptions of fitted parameters and their bounds.

Each entry maps a parameter key (as used in ``ASSAY_REGISTRY``) to a
``(title, html_body)`` tuple rendered by
:class:`gui.widgets.info_button.InfoButton`. Content is written to help a
scientist decide what to adjust when a fit is poor, not to prove the
theory from first principles.
"""

from __future__ import annotations

_SCI_NOTE_HTML = """
<p><b>Scientific Notation</b></p>
<p><code>1.000e+12</code> means <i>1.0&nbsp;&times;&nbsp;10<sup>12</sup></i>
&mdash; the number after <code>e</code> is the power of 10. Negative
exponents are small values: <code>1.0e-06</code> is 0.000 001.</p>
"""


def _with_sci_note(body: str) -> str:
    return body + _SCI_NOTE_HTML


_KA_DYE_HTML = _with_sci_note("""
<h3>K<sub>a,dye</sub> &mdash; Host&ndash;Dye Association Constant</h3>

<p><b>What It Means Physically</b></p>
<p>Equilibrium constant for the reaction
<i>H + D &#x21CC; HD</i>, in units of M<sup>&minus;1</sup>. A larger
K<sub>a,dye</sub> means the host binds the dye more tightly. Typical
supramolecular host&ndash;dye pairs range from
10<sup>3</sup>&nbsp;M<sup>&minus;1</sup> (weak) to
10<sup>8</sup>&nbsp;M<sup>&minus;1</sup> (very tight).</p>

<p><b>Where It Is Fit</b></p>
<p>K<sub>a,dye</sub> is a free parameter <b>only in the Direct Binding
Assays</b> (DBA Host&rarr;Dye and DBA Dye&rarr;Host). In IDA and GDA it
is a <i>known</i> input that you supply in the Assay Configuration
panel &mdash; those assays fit K<sub>a,guest</sub> instead.</p>

<p><b>What It Controls in the Fit</b></p>
<p>Sets the curvature and midpoint of the DBA titration curve. Too
small &rarr; the curve looks linear (no saturation); too large &rarr;
the dye is fully complexed from the first point.</p>

<p><b>Bounds Guidance</b></p>
<ul>
  <li>Defaults cover roughly 10<sup>&minus;8</sup>&ndash;10<sup>12</sup>
      M<sup>&minus;1</sup> in log space &mdash; wide enough for any
      realistic host&ndash;dye system.</li>
  <li>If the optimiser consistently hits a bound, widen that side;
      don&rsquo;t clamp it artificially.</li>
  <li>If you already know K<sub>a,dye</sub> within an order of
      magnitude, tighten the bounds to that window &mdash; this often
      fixes convergence in noisy DBA data.</li>
  <li>Fits &ldquo;walking away&rdquo; to implausibly large values
      usually indicate a saturation plateau shorter than expected;
      re-examine [H]<sub>0</sub> and [D]<sub>0</sub>.</li>
</ul>
""")

_KA_GUEST_HTML = _with_sci_note("""
<h3>K<sub>a,guest</sub> &mdash; Host&ndash;Guest Association Constant</h3>

<p><b>What It Means Physically</b></p>
<p>Equilibrium constant for <i>H + G &#x21CC; HG</i> in
M<sup>&minus;1</sup>. The scientific target of IDA and GDA experiments.</p>

<p><b>What It Controls in the Fit</b></p>
<p>In IDA/GDA, it sets how aggressively guest displaces dye from the
pre-formed HD complex. High K<sub>a,guest</sub> shifts the displacement
curve to lower guest concentrations.</p>

<p><b>Bounds Guidance</b></p>
<ul>
  <li>Sampled in log space, so even the default
      10<sup>&minus;8</sup>&ndash;10<sup>12</sup>&nbsp;M<sup>&minus;1</sup>
      range is cheap to search.</li>
  <li>If the fit fails to converge, check first that
      K<sub>a,guest</sub> &sdot; [H]<sub>0</sub> is of order 1 at the
      titration midpoint; otherwise the displacement window is too
      narrow for meaningful discrimination.</li>
  <li>A flat residual with K<sub>a,guest</sub> pinned at the lower
      bound means guest is too weak to displace the dye under your
      conditions &mdash; decrease [H]<sub>0</sub> or use a weaker
      indicator.</li>
</ul>
""")

_I0_HTML = _with_sci_note("""
<h3>I<sub>0</sub> &mdash; Baseline Signal</h3>

<p><b>What It Means Physically</b></p>
<p>The constant term in the signal model: solvent background, detector
offset, and any non-dye contribution to the measured signal. Units:
a.u. (arbitrary signal units).</p>

<p><b>What It Controls in the Fit</b></p>
<p>Shifts the whole fitted curve vertically. Because I<sub>0</sub>
enters additively, it is <b>structurally degenerate</b> with
I<sub>dye,free</sub>&sdot;[D] when [D]<sub>0</sub> is fixed (DBA). Only
the combination I<sub>0</sub> + I<sub>dye,free</sub>&sdot;[D]<sub>0</sub>
is physically identifiable in that case.</p>

<p><b>Bounds Guidance</b></p>
<ul>
  <li>Use the dye-alone calibration intercept as a prior if available
      (&ldquo;Load Dye-Alone&rdquo; in this panel).</li>
  <li>Default bounds are
      0&ndash;10<sup>8</sup>&nbsp;a.u. &mdash; the lower bound is
      strictly non-negative. If your raw signal has been
      background-subtracted and can legitimately go below zero, lower
      the bound manually.</li>
  <li>Don&rsquo;t over-tighten I<sub>0</sub> on its own &mdash; pinning
      it can force the optimiser to compensate through the other
      signal coefficients and give a deceptive fit.</li>
</ul>
""")

_I_DYE_FREE_HTML = _with_sci_note("""
<h3>I<sub>dye,free</sub> &mdash; Free-Dye Signal Coefficient</h3>

<p><b>What It Means Physically</b></p>
<p>Molar response of the instrument to free dye: how much signal is
produced per 1&nbsp;M of uncomplexed dye. Units: a.u.&nbsp;M<sup>&minus;1</sup>.
Equal to the slope of a &ldquo;Dye Alone&rdquo; calibration.</p>

<p><b>What It Controls in the Fit</b></p>
<p>Scales the part of the signal coming from free dye. Combined with
I<sub>dye,bound</sub>, it sets the overall dynamic range of the
titration.</p>

<p><b>Bounds Guidance</b></p>
<ul>
  <li>The best prior is a dye-alone fit &mdash; use the
      &ldquo;Load Dye-Alone&rdquo; button to constrain this parameter
      automatically.</li>
  <li>If you suspect the fit is collapsing into a degenerate solution
      (very large I<sub>dye,bound</sub>, near-zero
      I<sub>dye,free</sub>), tightening this bound is the single most
      effective intervention.</li>
  <li>Must stay &ge;&nbsp;0 for any physically meaningful instrument.</li>
</ul>
""")

_I_DYE_BOUND_HTML = _with_sci_note("""
<h3>I<sub>dye,bound</sub> &mdash; Host&ndash;Dye Complex Signal Coefficient</h3>

<p><b>What It Means Physically</b></p>
<p>Molar response of the instrument to the host&ndash;dye complex, in
a.u.&nbsp;M<sup>&minus;1</sup>. Often very different from
I<sub>dye,free</sub> because binding changes the dye&rsquo;s environment
(quantum yield, &lambda;<sub>max</sub>, protonation state, etc.).</p>

<p><b>What It Controls in the Fit</b></p>
<p>Sets the saturation level of the titration. If your titration curve
has reached a clear plateau, I<sub>dye,bound</sub> is the easiest
parameter to recover reliably.</p>

<p><b>Bounds Guidance</b></p>
<ul>
  <li>If your signal <i>decreases</i> upon binding (turn-off sensor),
      remember this coefficient can legitimately be smaller than
      I<sub>dye,free</sub>.</li>
  <li>When a fit returns a plateau that doesn&rsquo;t match the data,
      the problem is usually I<sub>dye,bound</sub> hitting an upper
      bound &mdash; widen it before blaming K<sub>a</sub>.</li>
  <li>In DBA, its absolute value is under-determined; trust the
      difference (I<sub>dye,bound</sub>&nbsp;&minus;&nbsp;I<sub>dye,free</sub>)
      rather than either coefficient alone.</li>
</ul>
""")

_SLOPE_HTML = _with_sci_note("""
<h3>Slope &mdash; Dye-Alone Calibration Slope</h3>

<p><b>What It Means Physically</b></p>
<p>Instrument response per unit dye concentration in the absence of
host, in a.u.&nbsp;M<sup>&minus;1</sup>. Numerically equal to
I<sub>dye,free</sub> in the dye-alone assay.</p>

<p><b>What It Controls in the Fit</b></p>
<p>Sets the steepness of the linear calibration
<i>S = slope &middot; [D] + intercept</i>.</p>

<p><b>Bounds Guidance</b></p>
<ul>
  <li>If the fit slope is negative and your dye gets brighter with
      concentration, recheck your data: something is wrong (inverted
      sign, mis-ordered columns).</li>
  <li>Defaults cover a very wide positive range &mdash; you rarely need
      to adjust them for dye-alone calibrations.</li>
</ul>
""")

_INTERCEPT_HTML = _with_sci_note("""
<h3>Intercept &mdash; Dye-Alone Calibration Intercept</h3>

<p><b>What It Means Physically</b></p>
<p>The extrapolated signal at zero dye concentration in a
dye-alone experiment: solvent background, autofluorescence, and
instrument offset rolled into one term, in a.u.</p>

<p><b>What It Controls in the Fit</b></p>
<p>Shifts the calibration line vertically. Gets propagated into
I<sub>0</sub> of later IDA/DBA fits when you use dye-alone priors.</p>

<p><b>Bounds Guidance</b></p>
<ul>
  <li>Allow the intercept to go negative if your data could be
      background-subtracted.</li>
  <li>A very large |intercept| in a dye-alone fit usually means a
      systematic baseline issue &mdash; investigate before using the
      result downstream.</li>
</ul>
""")

_KA_HG_HTML = _with_sci_note("""
<h3>K<sub>a(HG)</sub> &mdash; First Stepwise Association Constant</h3>
<p><b>What It Means</b></p>
<p>Association constant for the first binding step <i>H + G &#x21CC; HG</i>
(M<sup>&minus;1</sup>). Shared by the 1:2 (HG<sub>2</sub>) and 2:1
(H<sub>2</sub>G) models as the first host&ndash;guest contact.</p>
<p><b>Identifiability</b></p>
<p>Sampled on a log scale over a wide range. It trades off with the
second-step constant &mdash; judge confidence from the reported range across
the accepted fits, not the estimate alone.</p>
""")

_KA_HG2_HTML = _with_sci_note("""
<h3>K<sub>a(HG&#x2082;)</sub> &mdash; Second Stepwise Constant (1:2)</h3>
<p>Association constant for <i>HG + G &#x21CC; HG<sub>2</sub></i>
(M<sup>&minus;1</sup>), the binding of a second guest. The ratio
K<sub>a(HG&#x2082;)</sub>/K<sub>a(HG)</sub> reports cooperativity relative to
the statistical expectation. Strongly correlated with K<sub>a(HG)</sub>
&mdash; read the pair's spread.</p>
""")

_KA_H2G_HTML = _with_sci_note("""
<h3>K<sub>a(H&#x2082;G)</sub> &mdash; Second Stepwise Constant (2:1)</h3>
<p>Association constant for <i>HG + H &#x21CC; H<sub>2</sub>G</i>
(M<sup>&minus;1</sup>), the binding of a second host to the same guest.
Strongly correlated with K<sub>a(HG)</sub> &mdash; read the reported range
rather than the point value.</p>
""")

_I_G_HTML = _with_sci_note("""
<h3>I<sub>G</sub> &mdash; Free-Guest Signal Coefficient</h3>
<p>Per-molar signal contribution of unbound guest (a.u./M). Like all the
linear signal coefficients it is only weakly identifiable on its own;
constrain it with a calibration if a fit looks unstable.</p>
""")

_I_H_HTML = _with_sci_note("""
<h3>I<sub>H</sub> &mdash; Free-Host Signal Coefficient</h3>
<p>Per-molar signal contribution of unbound host (a.u./M). <b>Pinned to zero
by default</b> &mdash; most hosts are optically silent. Widen its bounds in
this panel to fit it.</p>
""")

_I_HG_HTML = _with_sci_note("""
<h3>I<sub>HG</sub> &mdash; 1:1 Complex Signal Coefficient</h3>
<p>Per-molar signal contribution of the HG complex (a.u./M).</p>
""")

_I_HG2_HTML = _with_sci_note("""
<h3>I<sub>HG&#x2082;</sub> &mdash; 1:2 Complex Signal Coefficient</h3>
<p>Per-molar signal contribution of the HG<sub>2</sub> complex (a.u./M).</p>
""")

_I_H2G_HTML = _with_sci_note("""
<h3>I<sub>H&#x2082;G</sub> &mdash; 2:1 Complex Signal Coefficient</h3>
<p>Per-molar signal contribution of the H<sub>2</sub>G complex (a.u./M).</p>
""")

PARAMETER_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    'Ka_dye': ('K\u2090,dye \u2014 Host\u2013Dye Association Constant', _KA_DYE_HTML),
    'Ka_guest': ('K\u2090,guest \u2014 Host\u2013Guest Association Constant', _KA_GUEST_HTML),
    'I0': ('I\u2080 \u2014 Baseline Signal', _I0_HTML),
    'I_dye_free': ('I_dye,free \u2014 Free-Dye Signal Coefficient', _I_DYE_FREE_HTML),
    'I_dye_bound': ('I_dye,bound \u2014 Bound-Dye Signal Coefficient', _I_DYE_BOUND_HTML),
    'slope': ('Slope \u2014 Dye-Alone Calibration Slope', _SLOPE_HTML),
    'intercept': ('Intercept \u2014 Dye-Alone Calibration Intercept', _INTERCEPT_HTML),
    'Ka_HG': ('Kₐ(HG) — First Stepwise Association Constant', _KA_HG_HTML),
    'Ka_HG2': ('Kₐ(HG₂) — Second Stepwise Constant (1:2)', _KA_HG2_HTML),
    'Ka_H2G': ('Kₐ(H₂G) — Second Stepwise Constant (2:1)', _KA_H2G_HTML),
    'I_G': ('I_G — Free-Guest Signal Coefficient', _I_G_HTML),
    'I_H': ('I_H — Free-Host Signal Coefficient', _I_H_HTML),
    'I_HG': ('I_HG — 1:1 Complex Signal Coefficient', _I_HG_HTML),
    'I_HG2': ('I_HG₂ — 1:2 Complex Signal Coefficient', _I_HG2_HTML),
    'I_H2G': ('I_H₂G — 2:1 Complex Signal Coefficient', _I_H2G_HTML),
}
