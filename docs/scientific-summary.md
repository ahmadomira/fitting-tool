# Scientific reference for SupraSimFit

This reference covers all seven implemented assays. Literature establishes the reaction and response models; the derivations below establish their mathematical consequences. Algorithms, defaults, and reporting policies are separately identified as application choices. A successful fit alone does not establish the chemical model.

## 1. Scope, notation, units, and experimental conditions

The observable is a scalar signal $F(x)$, normally fluorescence intensity in instrumental arbitrary units (au), measured at known **analytical total titrant concentrations** $x$. Uppercase $H_T,D_T,G_T$ denote post-mixing totals of host, dye, and guest. Lowercase $h,d,g$ denote their free concentrations; $c=[HD]$ denotes concentration of the molecular complex. Concentrations are internally M = mol/L. Free and total titrant are different when binding depletes the free pool.

Every fitted association constant $K_a$ is a conditional concentration constant in M⁻¹; $K_d=1/K_a$ has units M. These are not dimensionless thermodynamic constants. Concentration mass action assumes that activity coefficients are effectively constant under the experiment's solvent, temperature, pH, and ionic strength. A reported logarithm means $\log_{10}[K_a/(1\,\mathrm{M}^{-1})]$.

| Registered assay | Independent variable $x$ | Fixed conditions, treated as known | Ordered fitted parameters |
|---|---|---|---|
| `DBA_HtoD` | $H_T$ | $D_T=D_0$, named `fixed_conc` | `Ka_dye`, `I0`, `I_dye_free`, `I_dye_bound` |
| `DBA_DtoH` | $D_T$ | $H_T=H_0$, named `fixed_conc` | Same four parameters |
| IDA | $G_T$ | `h0`=$H_T$, `d0`=$D_T$, `Ka_dye` | `Ka_guest`, `I0`, `I_dye_free`, `I_dye_bound` |
| GDA | $D_T$ | `h0`=$H_T$, `g0`=$G_T$, `Ka_dye` | Same four parameters |
| `DYE_ALONE` | $D_T$ | No binding conditions | slope, intercept |
| `DBA_HG2` | $G_T$ | `h0`=$H_T$ | `Ka_HG`, `Ka_HG2`, `I0`, `I_G`, `I_H`, `I_HG`, `I_HG2` |
| `DBA_H2G` | $G_T$ | `h0`=$H_T$ | `Ka_HG`, `Ka_H2G`, `I0`, `I_G`, `I_H`, `I_HG`, `I_H2G` |

Fitting and simulation share the seven model implementations listed above.

Every baseline $b$ (`I0` or calibration intercept) has units au. Every species response $s$ or $e$ (`I_*` or calibration slope) has units au/M **per mole of that species**. Concentrations are finite and nonnegative. The equilibrium models use finite positive association constants; zero constants define useful no-binding/reduced-model limits supported by low-level solvers. Assay wrappers require positive fixed binding-partner totals and positive known dye affinity; GDA permits zero fixed guest, making guest affinity unobservable.

Registry defaults restrict affinities to $10^{-8}$–$10^{12}$ M⁻¹, brightness coefficients to 0–$10^{12}$ au/M, and `I0` to 0–$10^8$ au. Stepwise `I_H` is fixed to zero by equal default bounds. Dye calibration has registry intercept bounds −$10^6$–$10^6$ au, but its usual closed-form fit is unconstrained ordinary least squares. These are project defaults, not literature-derived physical ranges or uncertainty intervals. Overrides and processed difference signals can require signed coefficients; finite negative observations are accepted.

All binding models assume equilibrium, the specified stoichiometry, component conservation, and no omitted competing species, protonation changes, aggregation, precipitation, or kinetic trapping. They contain no time coordinate, rate constants, or kinetic initial conditions. Preforming a complex does not change the equilibrium predicted from the same final totals.

Scalar fixed totals describe independently prepared samples or additions that preserve those totals adequately. In ordinary serial additions, each total is its amount divided by $V_0+v$, including added amount, where $V_0$ is initial volume and $v$ cumulative added volume. Current wrappers do not accept per-point diluted fixed totals. Multiplying signal by a volume factor alone does not restore the equilibrium. The intended preparation protocol remains an experimental question; no automatic dilution correction is claimed.

## 2. Signal law and dye calibration

For direct and competitive assays, only free and host-bound dye contribute to the concentration-dependent signal:

$$
F=b+s_fd+s_bc=b+s_fD_T+\Delta c,\qquad \Delta=s_b-s_f. \tag{1}
$$

Here $b=\mathrm{I0}$, $s_f=\mathrm{I\_dye\_free}$, $s_b=\mathrm{I\_dye\_bound}$, and $c=[HD]$. Host, guest, and host–guest complex are optically silent in this response convention. This is an additive intensity law, not a normalized bound fraction. DeJaco et al., [DOI:10.1016/j.bpj.2023.03.002](https://doi.org/10.1016/j.bpj.2023.03.002), [manuscript p.2, Eqs.(1)–(4)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=935038), support the free/bound response construction; their separate DNA binding model is not adopted. The linked NIST copy is a submitted manuscript; cited pages use its pagination.

Constant coefficients require stable settings and an optical-linear regime. Significant inner-filter effects, concentration quenching, detector saturation, bleaching, or changing geometry invalidate this simple response. Absorbance, anisotropy, polarization, normalized ratios, and chemical shifts need their corresponding measurement equations; the application does not convert them into Eq.(1).

`DYE_ALONE` sets host to zero:

$$
F(x)=b+s_fx. \tag{2}
$$

At least two distinct concentrations are needed to determine slope and intercept. The line measures free-dye response and background under the calibration conditions, neither affinity nor bound-dye brightness. Calibration curvature indicates model failure; a fitted line does not correct inner-filter effects. Transfer requires matching conditions. The application's default calibration windows use a half-width of 20% of the fitted value's absolute magnitude, with a 0.2 au or au/M half-width at zero, and clip both endpoints to be nonnegative. Negative fitted coefficients therefore produce a zero bound pair. These are heuristics, not confidence limits, and require reconsideration for signed difference signals.

## 3. Direct 1:1 binding: both DBA directions

$$
H+D\rightleftharpoons HD,\quad c=Khd,\quad H_T=h+c,\quad D_T=d+c, \tag{3}
$$

where $K=\mathrm{Ka\_dye}$. Substitution yields

$$
c^2-(H_T+D_T+K^{-1})c+H_TD_T=0.
$$

For $S=H_T+D_T+K^{-1}$, the physical branch is

$$
c=\frac{S-\sqrt{S^2-4H_TD_T}}2
 =\frac{2H_TD_T}{S+\sqrt{S^2-4H_TD_T}},\quad
h=H_T-c,\quad d=D_T-c. \tag{4}
$$

The smaller root obeys $0\le c\le\min(H_T,D_T)$. The rationalized form avoids cancellation for small complex populations. Production evaluation also rationalizes free-minority concentration when strong binding would lose precision in the free remainder. This changes numerical evaluation, not Eq.(3).

The total-concentration quadratic and both titration orientations appear in Olson et al., [DOI:10.1128/AAC.01499-05](https://doi.org/10.1128/AAC.01499-05), fluorescence-titration methods and Figs.3–4. Equation (4) is independently derived from Eq.(3); textbook corroboration is Steed & Atwood, *Supramolecular Chemistry*, 3rd ed., [p.20, Eqs.(1.11)–(1.13)](https://catalogimages.wiley.com/images/db/pdf/9781119582519.excerpt.pdf).

`DBA_HtoD` uses $H_T=x,D_T=D_0$. At zero host, $F=b+s_fD_0$; at excess host, $F\to b+s_bD_0$. Signal rises for brighter bound dye and falls for quenched bound dye. `DBA_DtoH` uses $H_T=H_0,D_T=x$. At zero dye, $F=b$; at high dye,

$$
F\sim b+s_fx+\Delta H_0.
$$

Thus raw dye-titration signal generally has a linear tail even when the host is saturated. Free host must not replace free dye in its response.

Independent checks: zero either total or $K\to0$ gives $c\to0$; weak binding gives $c\sim KH_TD_T$; $K\to\infty$ gives $c\to\min(H_T,D_T)$. Half occupancy of a fixed partner of total $C_0$ occurs at **total** titrant $K^{-1}+C_0/2$. Occupancy is exactly hyperbolic in free titrant; replacing free titrant by total titrant in that hyperbola requires negligible depletion. For $K=5\times10^5$ M⁻¹, $H_T=2$ µM and $D_T=3$ µM, independently chosen equilibrium species are $h=1,d=2,c=1$ µM. With $b=5$ au, $s_f=2\times10^6,s_b=8\times10^6$ au/M, Eq.(1) gives 17 au.

## 4. Mutually exclusive competition: IDA and GDA

Both assays implement two 1:1 reactions without a ternary complex:

$$
H+D\rightleftharpoons HD,\qquad H+G\rightleftharpoons HG.
$$

Let $a=\mathrm{Ka\_dye}$ be known, $k=\mathrm{Ka\_guest}$ be fitted, $c=[HD]$, and $y=[HG]$. Mass action and conservation give

$$
d=\frac{D_T}{1+ah},\quad c=\frac{ahD_T}{1+ah},\qquad
g=\frac{G_T}{1+kh},\quad y=\frac{khG_T}{1+kh}, \tag{5}
$$

$$
R(h)=h+\frac{ahD_T}{1+ah}+\frac{khG_T}{1+kh}-H_T=0. \tag{6}
$$

The balances are $H_T=h+c+y,D_T=d+c,G_T=g+y$. Multiplication of Eq.(6) gives

$$
ak h^3+[a+k+ak(D_T+G_T-H_T)]h^2
+[1+aD_T+kG_T-(a+k)H_T]h-H_T=0. \tag{7}
$$

Every term has units M. The root is unique on $0\le h\le H_T$: the endpoint signs bracket zero and

$$
R'(h)=1+\frac{aD_T}{(1+ah)^2}+\frac{kG_T}{(1+kh)^2}>0.
$$

The application solves this monotone balance numerically. Signal follows Eq.(1).

The primary GDA derivation is Sinn, Krämer & Biedermann, [DOI:10.1039/D0CC01841D](https://doi.org/10.1039/D0CC01841D), [p.6621, Eqs.(2)–(6)](https://publikationen.bibliothek.kit.edu/1000122219/83747544), and [ESI p.S7, Eqs.S6–S14](https://www.rsc.org/suppdata/d0/cc/d0cc01841d/d0cc01841d1.pdf). IDA is treated by Hargrove et al., [DOI:10.1039/B9NJ00498J](https://doi.org/10.1039/B9NJ00498J), indicator-displacement section, author-manuscript pp.8–10, Eqs.(27)–(41). The [Hargrove author manuscript](https://www.researchgate.net/publication/44665195_Algorithms_for_the_determination_of_binding_constants_and_enantiomeric_excess_in_complex_host_guest_equilibria_using_optical_measurements) was inspected because publisher access was unavailable; some equation images were absent from text extraction, but the displayed cubic coefficients and signal were legible. The GDA article and ESI were read in full text. Equations (5)–(7) also follow independently from the balances.

IDA increases guest total, reducing HD and approaching $F=b+s_fD_T$ at excess guest. Signal decreases for brighter bound dye and increases for quenched bound dye. GDA instead increases dye total, displacing guest from an initially dye-free host–guest mixture. At large dye total, $HD\to H_T$ and $F\sim b+s_fD_T+\Delta H_T$. GDA does not describe adding guest to host–dye. Its usefulness relative to IDA depends on the actual affinities, concentration range, solubility, and optical contrast.

Zero guest total or guest affinity reduces to direct dye binding. Zero dye total removes the modeled concentration-dependent signal. An independent anchor has $a=5\times10^5,k=2\times10^6$ M⁻¹ and $h=1,d=2,g=3,c=1,y=6$ µM, hence $H_T=8,D_T=3,G_T=9$ µM. With $b=7$ au and $s_f=2\times10^6,s_b=5\times10^6$ au/M, the signal is 16 au.

Published competition models include different dye stoichiometries, including the cited ESI's separate 1:2 variant. These require different balances and are not the implemented IDA/GDA model. Supplied dye affinity must apply to the 1:1 model under the experiment's conditions. Its uncertainty is not propagated by the current fit.

## 5. Sequential 1:2 and 2:1 binding

These models fit macroscopic stepwise constants. $K_1=\mathrm{Ka\_HG}$ and $K_2=\mathrm{Ka\_HG2}$ or $\mathrm{Ka\_H2G}$ each have units M⁻¹. The derived cumulative constant $\beta_2=K_1K_2$ has units M⁻²; it is **not** the second fitted parameter.

The literature basis is Thordarson, [DOI:10.1039/C0CS00062K](https://doi.org/10.1039/C0CS00062K), pp.1310–1312, §§3.4–3.6, Charts 2–3 and Eqs.(16)–(26), with the [publisher correction](https://www.rsc.org/suppdata/cs/c0/c0cs00062k/addition.htm). The [review PDF](https://vingaarden.mono.net/upl/website/dokumenter/bestemK.pdf) was checked as extracted text; its chart equations were not visually accessible. The publisher correction addresses Eqs.(10)–(13), total-concentration notation, and Fig.6. The older PDF also has an inconsistent free-host denominator in Eq.(4) and an inverted interaction-parameter definition in Appendix A. The balances and conventions below are independently derived from the reactions, not transcribed from those expressions.

### `DBA_HG2`: one host, up to two guests

$$
H+G\rightleftharpoons HG,\qquad HG+G\rightleftharpoons HG_2,
$$

$$
c_1=[HG]=K_1hg,\quad c_2=[HG_2]=\beta_2hg^2,\quad
H_T=h+c_1+c_2,\quad G_T=g+c_1+2c_2. \tag{8}
$$

With $Q(g)=1+K_1g+\beta_2g^2$,

$$
h=\frac{H_T}{Q(g)},\quad c_1=\frac{H_TK_1g}{Q(g)},\quad
c_2=\frac{H_T\beta_2g^2}{Q(g)},\qquad
G_T=g+H_T\frac{K_1g+2\beta_2g^2}{Q(g)}. \tag{9}
$$

Equivalently,

$$
\beta_2g^3+[K_1+\beta_2(2H_T-G_T)]g^2
+[1+K_1(H_T-G_T)]g-G_T=0.
$$

### `DBA_H2G`: up to two hosts, one guest

$$
H+G\rightleftharpoons HG,\qquad H+HG\rightleftharpoons H_2G,
$$

$$
c_1=[HG]=K_1hg,\quad c_2=[H_2G]=\beta_2h^2g,\quad
H_T=h+c_1+2c_2,\quad G_T=g+c_1+c_2. \tag{10}
$$

With $Q(h)=1+K_1h+\beta_2h^2$,

$$
g=\frac{G_T}{Q(h)},\quad c_1=\frac{G_TK_1h}{Q(h)},\quad
c_2=\frac{G_T\beta_2h^2}{Q(h)},\qquad
H_T=h+G_T\frac{K_1h+2\beta_2h^2}{Q(h)}. \tag{11}
$$

Its cubic follows from Eq.(9)'s cubic by exchanging $g,h$ and $H_T,G_T$. Both implemented experiments vary **guest total at fixed host total**; a shared solver must preserve those meanings and the conservation multiplicities.

### Unique root, response, checks, and variants

Both balances have the generic form

$$
L_T=z+R_T\frac{K_1z+2\beta_2z^2}{Q(z)},\quad
Q(z)=1+K_1z+\beta_2z^2,
$$

with $(z,R_T,L_T)=(g,H_T,G_T)$ for HG2 and $(h,G_T,H_T)$ for H2G. The derivative of the right side is

$$
1+R_T\frac{K_1+4\beta_2z+K_1\beta_2z^2}{Q(z)^2}\ge1,
$$

giving a unique physical root in $[0,L_T]$. Balance terms have units M; $Q$ is dimensionless. This establishes the branch independently of the numerical algorithm.

The signal is

$$
F=b+e_Hh+e_Gg+e_1c_1+e_2c_2, \tag{12}
$$

where $e_H=\mathrm{I\_H},e_G=\mathrm{I\_G},e_1=\mathrm{I\_HG}$, and $e_2=\mathrm{I\_HG2}$ or $\mathrm{I\_H2G}$. Response is per complex: a factor of two in conservation is **not** an extra signal multiplier. At zero guest, $F=b+e_HH_T$. The same optical-linearity conditions as Eq.(1) apply.

Setting $K_2=0$ reduces to 1:1 binding; $K_1=0$ with finite $K_2$ removes both complexes. At excess guest with finite positive constants, HG2 gives $c_2\to H_T,c_1\to0,g\sim G_T-2H_T$. H2G gives $c_1\to H_T,c_2\to0,g\sim G_T-H_T$: excess guest removes the doubly hosted species. Both raw signals can retain a free-guest linear tail. [Stepwise regression tests](../tests/unit/test_audit_stepwise.py) contain independent manufactured-species expectations, dimensional rescaling, conservation, limits, and numerical stress cases.

For two identical independent sites of microscopic affinity $k_s$, $Q(z)=(1+k_sz)^2$ implies $K_1=2k_s,K_2=k_s/2$, hence $K_1=4K_2$. The cooperativity ratio relative to this statistical reference is $\alpha=4K_2/K_1$. Two different independent sites instead give $K_1=k_A+k_B,\beta_2=k_Ak_B$; $\alpha\le1$ can therefore arise from heterogeneity without negative cooperativity. See Hibbert & Thordarson, [DOI:10.1039/C6CC03888C](https://doi.org/10.1039/C6CC03888C), Eqs.(5)–(7), Fig.3/Table1. The application fits unrestricted macroscopic constants and does not identify a unique microscopic cooperative mechanism.

## 6. Structural and practical identifiability

**Structural identifiability** concerns what an ideal exact curve determines within a specified model; **practical identifiability** concerns finite noisy data. An exact symmetry persists with perfect measurements. Poor scaling, optimizer failure, and small sensitivities are separate issues. General profile-likelihood distinctions follow Raue et al., [DOI:10.1093/bioinformatics/btp358](https://doi.org/10.1093/bioinformatics/btp358), §§2.1–2.3 and 4. Assay-specific conclusions below are independent derivations.

### Exact symmetries and observable combinations

With fixed dye $D_0$, Eq.(1) becomes $F=A+\Delta c$, where $A=b+s_fD_0$. For any admissible $q$ in au/M,

$$
(b,s_f,s_b)\mapsto(b-qD_0,s_f+q,s_b+q) \tag{13}
$$

leaves the curve unchanged. This applies to **`DBA_HtoD` and IDA**. Their signal identifies $A,\Delta$, not all three raw coefficients. `DBA_DtoH` and GDA vary dye total and do not have this fixed-dye symmetry.

At fixed $H_T$, the stepwise signal reduces to

$$
F=B+e_GG_T+D_1c_1+D_2c_2,\quad
B=b+e_HH_T,\quad D_1=e_1-e_H-e_G, \tag{14}
$$

with $D_2=e_2-e_H-2e_G$ for HG2 and $D_2=e_2-2e_H-e_G$ for H2G. Its exact symmetry is

$$
b\mapsto b-qH_T,\quad e_H\mapsto e_H+q,\quad
e_1\mapsto e_1+q,\quad e_2\mapsto e_2+n_Hq, \tag{15}
$$

where $n_H=1$ for HG2 and 2 for H2G; $e_G$ is unchanged. Fixing `I_H`=0 selects a convention or supplies external information; it does not demonstrate that the host is experimentally dark.

| Assay | Ideal identifiable quantities under the conditions below | Essential exceptions |
|---|---|---|
| `DBA_HtoD` | $K,A,\Delta$ | Zero contrast or zero fixed dye hides affinity; Eq.(13) remains |
| `DBA_DtoH` | $K,b,s_f,s_b$ | Zero contrast or zero host hides affinity |
| IDA | $k,A,\Delta$, conditional on $a,H_T,D_T$ | Zero contrast or missing partners hides affinity; Eq.(13) remains |
| GDA | $k,b,s_f,s_b$, conditional on $a,H_T,G_T$ | Zero contrast, guest, host, or dye affinity hides guest affinity |
| `DYE_ALONE` | $b,s_f$ | Identical repeated concentrations identify only one line value |
| `DBA_HG2` | $K_1,K_2,B,e_G,D_1,D_2$ | $D_1=D_2=0$ hides both affinities; Eq.(15) remains for raw coefficients |
| `DBA_H2G` | Same six combinations | Same exceptions; reduced-model boundaries need separate analysis |

Binding statements assume finite positive constants, positive required fixed totals, exactly known conditions, and an ideal analytic curve on an open concentration interval. They are **global statements for this macroscopic model under those assumptions**, not promises of precision from limited measurements.

For the single-affinity assays, constructive arguments suffice. HtoD endpoints determine $A,\Delta$, then $c(x)$ determines $K=c/[(H_T-c)(D_T-c)]$. IDA's zero-guest complex is known from dye affinity; its excess-guest endpoint and contrast recover $c$, then $h=c/[a(D_T-c)]$, $y=H_T-h-c$, and $k=y/[h(G_T-y)]$ at interior points. DtoH/GDA use zero-dye intercept and high-dye slope/intercept to recover $b,s_f,\Delta$, followed by the constant. These ideal limiting signals can be inaccessible or noisy experimentally.

For stepwise models, [Appendix A](#appendix-a-global-identifiability-of-the-stepwise-models) proves that equality of the entire nonconstant reduced signal forces the same positive macroscopic constants, including the identical-site cancellation case; raw coefficients retain Eq.(15). This conclusion follows from the rational parameterizations, not numerical rank.

If concentration calibration has an unknown common scale $\lambda$, scaling all totals/free species by $\lambda$, all association constants by $1/\lambda$, and all response coefficients by $1/\lambda$ leaves signal unchanged. Resolving this ambiguity requires an independent concentration or affinity anchor. In competitive assays, the transformation must also rescale `Ka_dye`; holding an independently known absolute dye affinity fixed breaks this particular symmetry. Dye affinity determined using the same uncalibrated concentration scale is not necessarily an independent anchor. Microscopic site labels and combinations of heterogeneity/cooperativity may also be indistinguishable despite unique macroscopic constants.

### Finite-data limitations and experimental design

Weak binding makes response approximately proportional to an affinity–brightness product. Narrow ranges, nearly constant occupancy, weak contrast, and missing baselines/tails allow large parameter changes within noise. An unobservable stepwise intermediate can make a cumulative constant or other combinations much better constrained than separate constants. Excess guest also removes H2G and reduces sensitivity to the second step. These are practical limitations or limiting approximations, not a universal exact degeneracy between two positive step constants.

Improve information by covering binding transitions and relevant baselines/tails, varying known fixed totals in additional experiments, measuring applicable free-dye/host responses, calibrating concentrations and dye affinity, or adding independently informative channels. Current replica pooling is not a joint multi-condition fit. Bounds, fixed values, and any external priors add information not supplied by the curve. Report observable combinations where raw coefficients are ambiguous; high $R^2$ or narrow bounds do not establish precise affinities, stoichiometry, or microscopic cooperativity.

## 7. Fitting, simulation, and uncertainty in the application

With $n$ observations and parameter vector $\theta$, residuals are $r_i=F_i^{obs}-F(x_i;\theta)$ in au:

$$
\mathrm{SSE}=\sum_{i=1}^n r_i^2,\quad
\mathrm{RMSE}=\sqrt{\mathrm{SSE}/n},\quad
R^2=1-\frac{\mathrm{SSE}}{\sum_i(F_i^{obs}-\overline F^{obs})^2}. \tag{16}
$$

SSE has units au², RMSE au, and $R^2$ is dimensionless. For constant observations $R^2$ is mathematically undefined; the application's value 0 is a sentinel. Unweighted SSE is a maximum-likelihood objective for independent equal-variance Gaussian signal errors with known concentrations. Fitting raw data avoids an imposed transformed residual but does not establish this noise assumption. Heteroscedasticity, correlated drift, concentration errors, and uncertain fixed dye affinity are not modeled or propagated.

Binding fits use bounded multistart L-BFGS-B. Affinity **starts** are sampled logarithmically; optimization uses linearly scaled constants. With positive concentration and signal scales $C_*,F_*$, dimensionless coordinates are $KC_*,b/F_*,sC_*/F_*$, and the objective is divided by $F_*^2$. This preserves mathematical minimizers without guaranteeing conditioning, convergence, or global optimality, and cannot remove structural null directions.

Quality filters use configured $R^2$ and optional relative RMSE thresholds; optimizer success alone does not determine acceptance. The default representative is an actual accepted parameter vector with highest $R^2$, and its predicted curve. Min/max, means/SD, medians/MAD, and central percentiles describe accepted optimized solutions. They are **not bootstrap samples, posterior samples, or calibrated confidence intervals**. Per-replica fitting pools those solutions, weighting replicas by their number of accepted entries. Its spread mixes search and replica variability rather than estimating a shared parameter's uncertainty through a joint likelihood.

Dye-only fitting usually uses one unconstrained ordinary-least-squares line. The optional replica filter uses median/MAD scores. At zero MAD its current policy gives zero scores, including majority ties with a differing value; a justified noise/resolution floor remains an unresolved policy choice.

Simulation evaluates the same physical equations and adds independent Gaussian noise with standard deviation `noise_frac` times the noiseless signal span. For a flat signal, the scale falls back to maximum absolute signal, then 1 au if zero. This is a synthetic design convention, not a measured detector-noise law. Production-generated synthetic data test integration/recovery; independent equations, species anchors, and conservation checks validate forward mathematics.

## 8. Scope and evidence limits

The equations, physical-root arguments, exact signal symmetries, and hand-computed examples above define the mathematical reference. Independent regression checks exercise conservation, dimensional scaling, and boundary cases; fitting or simulation generated by the production model alone cannot establish its correctness. Stable numerical evaluation does not guarantee reliable results for every extreme input.

Applicability to real chemistry remains conditional on equilibrium, stoichiometry, optical linearity, and concentration accuracy. No bundled experimental binding dataset has verified publication-linked reference parameters. The cited stepwise convention sources are reviews; the macroscopic equations do not establish a unique microscopic binding mechanism. Neither these derivations nor a successful numerical fit provide calibrated uncertainty for arbitrary experimental data.

## Appendix A. Global identifiability of the stepwise models

Assume the complete exact analytic response curve, known fixed host total $H=H_T>0$, finite positive macroscopic $K_1,K_2$, and at least one nonzero contrast $D_1,D_2$ from Eq.(14). Here $t=G_T$; all free and total concentrations are in M. Equality on an open positive concentration interval extends along the physical analytic branch. Limits below establish uniqueness; they do not prescribe extrapolation of noisy observations.

The rational-map method follows Sendra and Winkler, *Computation of the Degree of Rational Maps Between Curves*, ISSAC 2001, [author manuscript, §2 Lemmas 1–2 and §3 Lemma 4/Theorem 5](https://www3.risc.jku.at/publications/download/risc_225/Nr.15_final.pdf). Its six-page full text establishes field-extension degree, multiplicativity, and one-variable rational-function degree. Bibliographic placeholders in that manuscript are not relied upon. The application of these facts to the two binding models is the independent derivation below.

In this appendix only, let $a=K_1>0,k=K_2>0$ in M⁻¹ and $d=ak=\beta_2$ in M⁻²; this local $d$ does not denote free dye. In both assays, $F(0)=B$ and $\lim F(t)/t=e_G$. Consequently identical entire response curves have the same $B,e_G$ and identical residual $r(t)=F(t)-B-e_Gt$. Equations (8)–(14) give the following rational parameterizations:

$$
\begin{aligned}
\mathrm{HG2}:\quad Q(z)&=1+az+dz^2,\quad z=g\ge0,\\
t(z)&=z+H\frac{az+2dz^2}{Q(z)},\\
r(z)&=H\frac{D_1az+D_2dz^2}{Q(z)};                           \tag{A1}
\end{aligned}
$$

$$
\begin{aligned}
\mathrm{H2G}:\quad z&=h\in(0,H],\\
t(z)&=\frac{(H-z)(1+az+akz^2)}{az(1+2kz)},\\
r(z)&=\frac{(H-z)(D_1+D_2kz)}{1+2kz}.                      \tag{A2}
\end{aligned}
$$

The domain extends to complex $z$ solely for a rational-identity argument; negative or complex poles are not physical equilibria.

**Why a rational inverse exists.** The degree of a reduced rational function is the larger numerator/denominator degree. When $a\ne4k$, $t(z)$ has degree 3 in both (A1) and (A2), while every nonzero residual has degree 1 or 2. The field-extension degree of the pair $(t,r)$ divides the degrees of both coordinate functions, by multiplicativity. Their greatest common divisor is one, so the pair is a proper rational parameterization: the free concentration is a rational function of $(t,r)$. These field facts are the brief method imported from Sendra–Winkler; the degrees and cancellations are independently checked here.

The only reduction of degree of $t$ in the positive parameter domain is $a=4k$. For HG2 the binding polynomial then equals $(1+2kz)^2$. For H2G the numerator cancels $1+2kz$. In both cases $t$ has degree 2. If $D_2=2D_1\ne0$, the residual has degree 1 and properness again follows. If this response cancellation does not hold and the residual has degree 1, the same argument applies. Otherwise its degree is 2. A nonproper pair of degree-2 coordinates would force $r$ to be a degree-1 rational function of $t$. That is impossible: in HG2, $t\to\infty$ at both $z=\infty$ and $z=-1/(2k)$, but $r$ is finite at the former and has a pole at the latter; in H2G, $t\to\infty$ at $z=0$ and $z=\infty$, but $r$ is finite at zero and divergent at infinity in this degree-2 noncancelled case. A rational function of $t$ cannot have two different limiting values at the same $t=\infty$. Thus the pair is proper also at $a=4k$, unless both contrasts vanish.

Two proper parameterizations of the same curve are related by an invertible rational change of their free-concentration coordinate. Its degree and its inverse's degree multiply to one, so it is a fractional linear (Möbius) map. Equality of the physical response curves guarantees the same rational curve and chooses corresponding physical branches. We can now eliminate every nontrivial admissible map.

**HG2.** Let alternative free guest be $z'=M(z)$. The physical endpoints require $M(0)=0$ and $M(\infty)=\infty$. Hence $M(z)=sz$. Both models satisfy $t=z+2H+o(1)$ as $z\to\infty$, so $s=1$. With the free guest unchanged,

$$
t-z=Hz\,Q'(z)/Q(z)
$$

determines $Q$ uniquely: equality of logarithmic derivatives gives a constant ratio of the two polynomials, and $Q(0)=1$ fixes that ratio to one. Thus $a,d$, and therefore $K_1,K_2$, coincide. Equality of the residual numerator then determines $D_1,D_2$.

**H2G.** Here the physical endpoint conditions are $M(0)=0$ and $M(H)=H$. For $a\ne4k$, the poles of $t(z)$ are exactly $0,-1/(2k),\infty$, all simple. A Möbius reparameterization preserves degree, so a degree-3 model cannot be equivalent to a degree-2 model. The pole zero is already fixed by the physical infinite-guest limit. If infinity maps to infinity, the map is linear and fixing $H$ forces identity. The only remaining pole permutation exchanges infinity and the negative pole; then

$$
M(z)=\frac{(1+2kH)z}{1+2kz},\qquad
M(\infty)=\frac{1+2kH}{2k}>0.
$$

But that image would have to equal the alternative negative pole $-1/(2k')<0$. This is impossible for positive $k'$. At $a=4k$, the only poles are zero and infinity, so fixing zero forces infinity to infinity, again giving identity. Once $z$ is identical, the coefficient of $1/z$ in $t(z)$ near zero is $H/a$, fixing $a$; the remaining pole, or the degree-2 condition, fixes $k$. Since $c_2/c_1=kz$ varies, the two contrast coefficients are uniquely determined as well.

Therefore neither full finite-positive stepwise model has an additional global macroscopic-affinity ambiguity when at least one contrast is nonzero, after removing the optional host-response gauge. This is an ideal-model proof. It supplies no lower bound on the size of detectable curvature and does not extend to zero constants, unknown concentration scales, omitted species, normalized observables or finite noisy sampling.
