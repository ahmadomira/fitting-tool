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
<h3>K<sub>a,dye</sub> &mdash; Host&ndash;Dye Affinity</h3>
<p>Association constant for H + D &#x21CC; HD, in M<sup>&minus;1</sup>.
It is fitted in the two 1:1 direct-binding assays and supplied as a known
condition in IDA/GDA.</p>
<p>Sample the binding transition to resolve affinity. Nearly linear or
nearly stoichiometric curves may fit well while leaving K poorly determined.
A value at a bound calls for checking concentrations, the sampled range and
independent information before changing that bound.</p>
<p>Starts are sampled logarithmically by default. Bounds restrict the
search; they do not establish precision.</p>
""")

_KA_GUEST_HTML = _with_sci_note("""
<h3>K<sub>a,guest</sub> &mdash; Host&ndash;Guest Affinity</h3>
<p>Association constant for H + G &#x21CC; HG, in M<sup>&minus;1</sup>.
IDA varies guest concentration; GDA varies dye concentration at fixed guest.
Both require a known dye affinity.</p>
<p>Measure the competition transition and check the supplied concentrations
and dye affinity. Without guest in GDA, or without a signal change between
free and bound dye, the curve contains no guest-affinity information.</p>
<p>Logarithmic starts explore a wide range. A narrow accepted-fit range
can reflect bounds or limited sampling; it is not a confidence interval.</p>
""")

_I0_HTML = _with_sci_note("""
<h3>I<sub>0</sub> &mdash; Background Signal</h3>
<p>Constant additive background in a.u. It need not equal the first
measured signal, which can already contain dye or complex fluorescence.</p>
<p>At fixed dye total (Host&rarr;Dye and IDA), the curve determines
I<sub>0</sub> + I<sub>dye,free</sub>[D]<sub>0</sub>, rather than separating
all raw signal responses. Dye&rarr;Host and GDA can separate them with an
informative concentration range.</p>
<p>A matched dye-only intercept can constrain this value. Default bounds
are nonnegative; allow negative values for an appropriate
background-subtracted signal. Bounds add assumptions unless independently
supported.</p>
""")

_I_DYE_FREE_HTML = _with_sci_note("""
<h3>I<sub>dye,free</sub> &mdash; Free-Dye Response</h3>
<p>Signal per molar concentration of unbound dye, in a.u./M.
A dye-only calibration slope measures this response under the same
medium and optical settings.</p>
<p>At fixed total dye (Host&rarr;Dye and IDA), it cannot be separated from
background and bound-dye response by that curve alone. In Dye&rarr;Host
and GDA, the dye-excess tail helps determine it.</p>
<p>Use matched calibration information for bounds. Nonnegative values suit
unsubtracted intensity; signed difference channels may need signed bounds.</p>
""")

_I_DYE_BOUND_HTML = _with_sci_note("""
<h3>I<sub>dye,bound</sub> &mdash; Bound-Dye Response</h3>
<p>Signal per molar concentration of HD complex, in a.u./M. This is the
whole complex response; the binding contrast is its difference from
I<sub>dye,free</sub>. A bound response below the free-dye response means
binding quenches the dye; the titration trend depends on assay direction.</p>
<p>Host&rarr;Dye and IDA can identify that contrast but cannot separate
all individual signal parameters without extra information. Dye&rarr;Host
and GDA can distinguish them in an informative curve. A plateau alone does
not determine the absolute response without background information.</p>
<p>Dye-only calibration supplies no bound-dye response. Check the signal
convention and experimental evidence when setting these bounds.</p>
""")

_SLOPE_HTML = _with_sci_note("""
<h3>Slope &mdash; Dye-Only Response</h3>
<p>Free-dye signal per molar concentration, in a.u./M, from
<i>S = slope &middot; [D] + intercept</i>. It can inform
I<sub>dye,free</sub> in a binding experiment with matched medium and optics.</p>
<p>Use at least two distinct concentrations and check that the response is
linear over the intended range. An unexpected slope sign warrants checking
the signal convention, data columns and optical effects.</p>
<p>Dye-only fitting uses ordinary linear regression; the nonlinear
optimizer bounds do not constrain this fit.</p>
""")

_INTERCEPT_HTML = _with_sci_note("""
<h3>Intercept &mdash; Dye-Only Background</h3>
<p>Predicted signal at zero dye, in a.u. A measured blank helps constrain
it; background subtraction can produce a negative value.</p>
<p>With matched medium and optics, it can inform I<sub>0</sub> in a later
binding fit. Loading a calibration applies a chosen bound margin; it does
not propagate the calibration's uncertainty. Ordinary linear regression
fits this intercept without the nonlinear optimizer bounds.</p>
""")

_KA_HG_HTML = _with_sci_note("""
<h3>K<sub>a(HG)</sub> &mdash; First Stepwise Affinity</h3>
<p>Association constant for H + G &#x21CC; HG, in M<sup>&minus;1</sup>,
in both stepwise models.</p>
<p>The two positive stepwise constants can be distinguished by an ideal
complete curve with binding-dependent signal. A limited concentration range
or noise can still make them strongly correlated. Sample substantial HG
population and both binding regimes; accepted-start ranges describe the
search pool, not confidence.</p>
""")

_KA_HG2_HTML = _with_sci_note("""
<h3>K<sub>a(HG&#x2082;)</sub> &mdash; Second Guest Affinity</h3>
<p>Stepwise association constant for HG + G &#x21CC; HG<sub>2</sub>, in
M<sup>&minus;1</sup>. The cumulative formation constant is the product
K<sub>a(HG)</sub>K<sub>a(HG&#x2082;)</sub>, in M<sup>&minus;2</sup>.</p>
<p>For two identical independent sites, K<sub>a(HG)</sub> =
4K<sub>a(HG&#x2082;)</sub>. Interpreting deviations as cooperativity requires
that site model; unequal sites can produce a similar ratio.</p>
<p>Include formation of HG and HG<sub>2</sub> to distinguish the two
constants in noisy data. Search-pool ranges are not confidence intervals.</p>
""")

_KA_H2G_HTML = _with_sci_note("""
<h3>K<sub>a(H&#x2082;G)</sub> &mdash; Second Host Affinity</h3>
<p>Stepwise association constant for HG + H &#x21CC; H<sub>2</sub>G, in
M<sup>&minus;1</sup>. Its product with K<sub>a(HG)</sub> is the cumulative
formation constant, in M<sup>&minus;2</sup>.</p>
<p>Early guest additions can form H<sub>2</sub>G; excess guest favors HG.
Include both regions, since high-guest data alone may poorly constrain this
constant. Practical correlation with the first constant does not imply an
exact ambiguity of the ideal model.</p>
""")

_I_G_HTML = _with_sci_note("""
<h3>I<sub>G</sub> &mdash; Free-Guest Response</h3>
<p>Signal per molar concentration of unbound guest, in a.u./M. Guest-excess
data help determine it; a matched guest-only calibration supplies additional
information. A narrow range or noise can weaken its estimate.</p>
""")

_I_H_HTML = _with_sci_note("""
<h3>I<sub>H</sub> &mdash; Free-Host Response</h3>
<p>Signal per molar concentration of unbound host, in a.u./M.
<b>Fixed at zero by default</b>, which assumes a dark host.</p>
<p>Widening its bounds allows fitting, but at one fixed host total it
cannot be separated from background and the other host-containing species
responses. A matched host-only calibration or shared experiments at several
host totals can add the missing information.</p>
""")

_I_HG_HTML = _with_sci_note("""
<h3>I<sub>HG</sub> &mdash; 1:1 Complex Response</h3>
<p>Signal per molar concentration of HG, in a.u./M. Sample a region with
appreciable HG to estimate it. If free-host response is also fitted at one
host total, the individual host-containing species responses remain
inseparable without extra information.</p>
""")

_I_HG2_HTML = _with_sci_note("""
<h3>I<sub>HG&#x2082;</sub> &mdash; 1:2 Complex Response</h3>
<p>Signal per molar concentration of HG<sub>2</sub> complexes, in a.u./M.
Each complex is counted once; the two guests do not add another factor of
two to this coefficient. Data with appreciable HG<sub>2</sub> help determine
its contribution.</p>
""")

_I_H2G_HTML = _with_sci_note("""
<h3>I<sub>H&#x2082;G</sub> &mdash; 2:1 Complex Response</h3>
<p>Signal per molar concentration of H<sub>2</sub>G complexes, in a.u./M.
Each complex is counted once; the two hosts do not add another factor of
two to this coefficient. Include the early guest region where
H<sub>2</sub>G is appreciable.</p>
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
