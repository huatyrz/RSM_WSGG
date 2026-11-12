## Publication
**High-Temperature NO-LIF Measurements in Laminar Premixed NH3/H2/N2-Air Counterflow Flames**
## Authors
M. Richter, R.S. Barlow, J.R. Dawson, A. Dreizler, D. Geyer
## Journal
Proceedings of the Combustion Institute 42
## Data level
The HDF5 file contains processed publication data. It is not a raw-data archive. The experimental NO profiles are calibrated mole-fraction profiles obtained from NO-LIF measurements and are reported in ppm. Experimental NO data are limited to positions where `T >= 1200 K`.
## Licence
This dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
## File structure
```text
/conditions
    Investigated flame conditions.

/cases/case_XX/experimental/NO
    Experimental NO profiles, temperature data, uncertainties, and correction functions.

/cases/case_XX/experimental/rayleigh
    Rayleigh scattering profiles used for thermochemical-state evaluation.

/cases/case_XX/simulation/<mechanism>
    One-dimensional simulation results for each chemical-kinetic mechanism.
```
## Details
### Experimental - NO
#### Published / masked data, limited to `T >= 1200 K`

| Dataset | Description |
|---|---|
| `x` | Flame-fixed spatial coordinate in mm. |
| `T` | Temperature corresponding to the published NO profile. The temperature was obtained from Rayleigh-based fitting of STA simulation results and interpolated onto the same flame-fixed `x` grid as the NO profile. |
| `NO` | Calibrated NO mole fraction in ppm. |
| `NO_standard_deviation` | Shot-to-shot standard deviation of the calibrated NO mole fraction in ppm. |
| `NO_positive_err` / `NO_negative_err` | Combined uncertainty bounds used for quantitative comparison in ppm. |
#### Correction function, full unmasked grid

| Dataset | Description |
|---|---|
| `x_full` | Unmasked flame-fixed spatial coordinate in mm. |
| `T_full` | Temperature on the full correction-function grid in K. |
| `gamma_C` | Total correction function accounting for number density, Boltzmann fraction, line-overlap integral, and collisional quenching. |
| `gamma_n` | Number-density contribution to `gamma_C`. |
| `gamma_spec` | Spectroscopic contribution to `gamma_C`. |
### Experimental - Rayleigh

| Dataset | Description |
|---|---|
| `x` | Flame-fixed spatial coordinate in mm. |
| `RAY` | Processed Rayleigh scattering signal after flame-front alignment and energy normalization. Unit: a.u. |
| `RAY_standard_deviation` | Shot-to-shot standard deviation of the Rayleigh scattering signal. Unit: a.u. |
### Simulation
One-dimensional simulations were performed using three chemical-kinetic mechanisms: NAK [1], STA [2], and HAN [3].

| Dataset / group | Description |
|---|---|
| `x` | Flame-fixed spatial coordinate in mm. |
| `NO` | Simulated NO mole fraction converted to ppm. |
| `T` | Temperature in K. |
| `major` | Major-species mole fractions: `NH3`, `H2`, `N2`, `O2`, `H2O`. |
| `minor` | Minor-species mole fractions: `NH2`, `NH`, `OH`, `O`, `H`. |
## Further comments
### Calibration note
The NO mole-fraction profiles are provided as processed, calibrated data. The NO-LIF calibration factor `C_C` was determined position-wise along the laser beam by linear regression of the flat-flame NO-addition calibration signal as a function of the NO mole fraction. `C_C` is not included as a separate dataset.
### Units
Units are stored as HDF5 attributes for each dataset.
## References
[1] H. Nakamura et al., Kinetic modeling of ammonia/air weak flames in a micro flow reactor with a controlled temperature profile, *Combustion and Flame* 185 (2017) 16-27.
[2] A. Stagni et al., Low- and intermediate-temperature ammonia/hydrogen oxidation in a flow reactor: Experiments and a wide-range kinetic modeling, *Chemical Engineering Journal* 471 (2023) 144577.
[3] X. Han et al., Experimental and kinetic modeling study of laminar burning velocities of NH3/syngas/air premixed flames, *Combustion and Flame* 213 (2020) 1-13.
