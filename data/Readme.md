# Meta Data

Text-data concentrations use M (mol/L) by default. Instrument formats can
declare their own units; inspect the reader and file metadata.
The values below are retained project metadata, not independently verified
published reference parameters. Their source, preparation/dilution protocol,
and uncertainty are not documented in the supplied files.

## DBA System Information (host-to-dye)
- $[D_0]$ = $151 \times 10^{-6}$ $M$
- $[H_0]$ = $840 \times 10^{-6}$ $M$

## GDA System Information
- $[H_0]$ (CB7) = $50 \times 10^{-6}$ $M$
- $[G_0]$ (nButanol):  $292 \times 10^{-6}$ $M$
- $K_{a(Dye)}$ = $33 \times 10^3$ $M^{-1}$

**Dye-Alone Measurement (for this GDA System):**
- $I_0$ = 29 au
- $I_{dye, free}$ = $3.52 \times 10^6$ au/M
- Dye type: TNS

Calibration values can constrain a fit when medium and optical settings match.
Dye-only measurements determine free-dye response and baseline, not bound-dye
response. Chosen tolerance bounds supply external information and are not
propagated confidence intervals. GDA does not have the fixed-dye signal
ambiguity of IDA/HtoD, although finite data can still constrain its parameters
poorly. See the [scientific reference](../docs/scientific-summary.md).

## IDA System Information
- $K_{a(Dye)}$ = $1.68 \times 10^7$ $M^{-1}$
- $[H_0]$ = $4.3 \times 10^{-6}$ $M$
- $[D_0]$ = $6 \times 10^{-6}$ $M$
