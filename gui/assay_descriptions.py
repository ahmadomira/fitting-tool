"""Scientist-facing descriptions of each supported assay type.

Each entry maps an ``AssayType.name`` (enum member name) to a
``(title, html_body)`` tuple. The HTML body is rendered by
:class:`gui.widgets.info_button.InfoButton` inside a QDialog.

Content is targeted at a practising scientist, not a mathematician:
physics first, the model second, then practical tuning guidance.
"""

from __future__ import annotations

_GDA_HTML = """
<h3>Guest Displacement Assay (GDA)</h3>
<p>Add dye to a host&ndash;guest mixture. The dye competes for host and
releases guest. Supply the fixed host and guest totals and an independently
measured host&ndash;dye association constant.</p>
<p>The model includes H + D &#x21CC; HD and H + G &#x21CC; HG, both 1:1.
It fits K<sub>a,guest</sub> (M<sup>&minus;1</sup>), background I<sub>0</sub>
(a.u.), and free/bound dye responses (a.u./M):</p>
<p><i>S = I<sub>0</sub> + I<sub>dye,free</sub>[D] +
I<sub>dye,bound</sub>[HD]</i>.</p>
<p>With known concentrations, nonzero guest and a signal change on binding,
an ideal complete curve can distinguish all four parameters. Real data
need the competition transition and a dye-excess tail; a good fit alone
does not establish precision. Errors in the supplied dye affinity affect
the guest estimate.</p>
<p>Use concentrations after mixing. Host and guest totals must remain
fixed; this assay does not model their dilution during additions.</p>
"""

_IDA_HTML = """
<h3>Indicator Displacement Assay (IDA)</h3>
<p>Add guest to a host&ndash;dye mixture. Guest competes for host and
releases dye. Supply fixed host and dye totals and the independently
measured host&ndash;dye association constant.</p>
<p>The model includes competing 1:1 HD and HG complexes and fits
K<sub>a,guest</sub> (M<sup>&minus;1</sup>), background I<sub>0</sub> (a.u.),
and free/bound dye responses (a.u./M):</p>
<p><i>S = I<sub>0</sub> + I<sub>dye,free</sub>[D] +
I<sub>dye,bound</sub>[HD]</i>.</p>
<p>Because total dye is fixed, individual background and dye responses
cannot all be separated by this curve alone. The curve can identify the
affinity, the combined background I<sub>0</sub> +
I<sub>dye,free</sub>[D]<sub>0</sub>, and the bound-minus-free response
when binding changes the signal. A matched dye calibration adds information
about the individual responses.</p>
<p>Include partial displacement and the displaced endpoint. Keep host and
dye totals fixed after mixing; this assay does not model their dilution.</p>
"""

_DBA_HtoD_HTML = """
<h3>Direct Binding &mdash; Host Into Dye</h3>
<p>Vary total host while keeping total dye fixed. The 1:1 model is
H + D &#x21CC; HD, with K<sub>a,dye</sub> = [HD]/([H][D]) in
M<sup>&minus;1</sup>.</p>
<p><i>S = I<sub>0</sub> + I<sub>dye,free</sub>[D] +
I<sub>dye,bound</sub>[HD]</i>, where background is in a.u. and
species responses are in a.u./M.</p>
<p>The curve can distinguish affinity, combined background
I<sub>0</sub> + I<sub>dye,free</sub>[D]<sub>0</sub>, and the
bound-minus-free response. It cannot separate all three raw signal
parameters without additional information. A matched dye-only calibration
can supply background and free-dye response.</p>
<p>Measure the unbound region, binding transition and approach to saturation.
Equal free/bound dye responses hide binding. Keep dye concentration fixed
after mixing; adding host stock can otherwise dilute it.</p>
"""

_DBA_DtoH_HTML = """
<h3>Direct Binding &mdash; Dye Into Host</h3>
<p>Vary total dye while keeping total host fixed. The model contains one
1:1 host&ndash;dye complex and fits K<sub>a,dye</sub> (M<sup>&minus;1</sup>),
background I<sub>0</sub> (a.u.), and free/bound dye responses (a.u./M).</p>
<p><i>S = I<sub>0</sub> + I<sub>dye,free</sub>[D] +
I<sub>dye,bound</sub>[HD]</i>.</p>
<p>With known host concentration and different free/bound dye responses,
an ideal complete curve can distinguish all four parameters. Real data
need binding curvature, a blank and the dye-excess linear tail. Weak
curvature or noise can still leave affinity poorly determined.</p>
<p>Excess free dye can keep increasing the signal after host occupancy
saturates. Keep host concentration fixed after mixing; this assay does
not model dilution during dye additions.</p>
"""

_DYE_ALONE_HTML = """
<h3>Dye Alone &mdash; Linear Calibration</h3>
<p>Measure dye without host over the concentration range of interest.
The fit is <i>S = slope &middot; [D] + intercept</i>: slope is free-dye
response (a.u./M), and intercept is the zero-dye background (a.u.).
At least two distinct concentrations are required.</p>
<p>Use matched solvent, temperature and optical settings before transferring
these values to a binding experiment. The calibration supplies no affinity
or bound-dye response. Curvature can indicate that a linear response model
is unsuitable over the chosen range.</p>
<p><b>Load Dye-Alone</b> in Parameter Bounds uses the fitted slope and
intercept to restrict I<sub>dye,free</sub> and I<sub>0</sub> in
DBA/IDA/GDA fits. Those bounds are a chosen tolerance around calibration
values; they do not propagate calibration uncertainty. Verify that the
calibration is transferable before applying tight bounds.</p>
"""

_DBA_HG2_HTML = """
<h3>Stepwise 1:2 Host&ndash;Guest Binding (HG2)</h3>
<p>Add guest to a fixed host total. The two steps are
H + G &#x21CC; HG and HG + G &#x21CC; HG<sub>2</sub>.
Both fitted association constants are stepwise values in M<sup>&minus;1</sup>.</p>
<p><i>S = I<sub>0</sub> + I<sub>G</sub>[G] + I<sub>H</sub>[H] +
I<sub>HG</sub>[HG] + I<sub>HG&#x2082;</sub>[HG<sub>2</sub>]</i>.
I<sub>0</sub> is in a.u.; each species response is in a.u./M of that species.</p>
<p>I<sub>H</sub> is fixed at zero by default. With that known response,
positive constants and binding-dependent signal, an ideal complete curve
can distinguish the six remaining parameters. Freeing I<sub>H</sub> makes
individual background/host-containing species responses inseparable at one
host total.</p>
<p>In real data, sample both binding steps and appreciable HG population;
otherwise the two constants may be strongly correlated. Accepted-fit ranges
describe the search results, not confidence intervals. Keep the host total
fixed after mixing; dilution is not modeled.</p>
"""

_DBA_H2G_HTML = """
<h3>Stepwise 2:1 Host&ndash;Guest Binding (H2G)</h3>
<p>Add guest to a fixed host total. The steps are
H + G &#x21CC; HG and HG + H &#x21CC; H<sub>2</sub>G.
Both fitted association constants are stepwise values in M<sup>&minus;1</sup>.
At high guest concentration, HG replaces H<sub>2</sub>G.</p>
<p><i>S = I<sub>0</sub> + I<sub>G</sub>[G] + I<sub>H</sub>[H] +
I<sub>HG</sub>[HG] + I<sub>H&#x2082;G</sub>[H<sub>2</sub>G]</i>.
I<sub>0</sub> is in a.u.; each species response is in a.u./M of that species.</p>
<p>I<sub>H</sub> is fixed at zero by default. With that known response,
positive constants and binding-dependent signal, an ideal complete curve
can distinguish the six remaining parameters. Freeing I<sub>H</sub> makes
individual background/host-containing species responses inseparable at one
host total.</p>
<p>Include the early H<sub>2</sub>G-rich region and its conversion to HG;
high-guest data alone can poorly constrain the second step. Accepted-fit
ranges describe the search results, not confidence intervals. Keep host
total fixed after mixing; dilution is not modeled.</p>
"""

ASSAY_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    'GDA': ('Guest Displacement Assay (GDA)', _GDA_HTML),
    'IDA': ('Indicator Displacement Assay (IDA)', _IDA_HTML),
    'DBA_HtoD': ('Direct Binding Assay (Host \u2192 Dye)', _DBA_HtoD_HTML),
    'DBA_DtoH': ('Direct Binding Assay (Dye \u2192 Host)', _DBA_DtoH_HTML),
    'DYE_ALONE': ('Dye Alone (Linear Calibration)', _DYE_ALONE_HTML),
    'DBA_HG2': ('Stepwise 1:2 Host\u2013Guest Binding (HG2)', _DBA_HG2_HTML),
    'DBA_H2G': ('Stepwise 2:1 Host\u2013Guest Binding (H2G)', _DBA_H2G_HTML),
}
