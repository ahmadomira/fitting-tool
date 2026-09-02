# Meta Data

**The concentrations in all data files are in Molar**

## DBA System Information (host-to-dye)
- $[D_0]$ = $151 \times 10^{-6}$ $M$
- $[H_0]$ = $840 \times 10^{-6}$ $M$

## GDA System Information
- $[H_0]$ (CB7) = $50 \times 10^{-6}$ $M$
- $[G_0]$ (nButanol):  $292 \times 10^{-6}$ $M$
- $K_{a(Dye)}$ = $33 \times 10^3$ $M^{-1}$

**Dye-Alone Measurement (for this GDA System):**
- $I_0$ = 29
- $I_{dye, free}$ = $3.52 \times 10^6$ $M^{-1}$
- Dye type: TNS

Note: The values of $I_0$ and $I_{dye, free}$ can be passed to the optimizer as bounds (value ± tolerance) to constrain the fit. This is useful because of the inherent degeneracy in the parameters, which can lead to multiple sets of parameters that fit the data equally well. By constraining the fit with known values, we can reduce the parameter space and improve the reliability of the fitted parameters.

## IDA System Information
- $K_{a(Dye)}$ = $1.68 \times 10^7$ $M^{-1}$
- $[H_0]$ = $4.3 \times 10^{-6}$ $M$
- $[D_0]$ = $6 \times 10^{-6}$ $M$