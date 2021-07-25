# VT1210
Code used in the analysis of VT 1210+4956. 

Contents:

**Radio modeling**
- **radio_modeling.py**
  - MCMC fitting of radio spectra in both follow-up epochs, Monte Carlo estimation of parameter distributions
- VT1210_epoch(1/2)_photometry.csv
  - Input files to radio_modeling.py. Contains frequencies, fluxes, uncertainties from sub-band imaging of both radio epochs.
- Granot_triangle_epoch(1/2).png
  - Output plot of radio_modeling.py. Corner plot of parameters from radio spectrum model (eq. 1 of Granot & Sari 2002). 
- Plots of the posterior distributions of peak (flux/frequency) in epoch (1/2), R, B, U, n, and v_radio assuming 
  - (stat): epsilon_e = epsilon_B = 0.1, f = uniform(0.1,0.5) 
  - (sys): epsilon_e = 0.1, epsilon_B = log-uniform(1e-3, 0.1), f = uniform(0.1,0.5)
- VT1210_subplot_nolabels_log_SED.png
  - Fitted radio spectra of VT 1210+4956. Used in Fig. 1 of the paper.
- **free_free_absorption_test.py**
  - MCMC test to show that the lowest frequency observations of VT 1210+4956 are not consistent with free-free absorption. 
- VT1210-free-free-abs-corner.png
  - Output plot of free_free_absorption_test.py. Corner plot of best-fit free-free absorption parameters (the emission measure and temperature of absorbing gas and the spectral index and scale amplitude of the synchrotron source being absorbed). 
- VT1210_FFA_spectrum_fit.png
  - Output plot of free_free_absorption_test.py. Note that the data are not sufficient to constrain the properties of the hypothetical free-free absorbing gas. Various (degenerate) combinations of emission measures and temperatures can reproduce the peak, and high frequency emission but not the low frequency emission. 


**Optical modeling**
- **optical_modeling.py**
  - MCMC fitting of emission lines in the LRIS followup spectrum. 
- lrisC0477_g1.spec
  - Input file to optical_modeling.py. The optical spectrum of VT 1210+4956 reduced with LPIPE. 
- C0477_corner.png
  - Output plot of optical_modeling.py. Corner plot of parameters in posterior distribution. 
    - Column labels left to right are: 'Zero point offset', 'Wavelength shift', 'Continuum slope', 'Halpha broad mu', 'Halpha broad A', 'Halpha broad sigma', 'Halpha narrow A', 'NII 6548 A', 'NII 6583 A' where:
      - Zero point offset (10^-17 erg/s/cm^2/angstrom) and Continuum slope (dimensionless) are a locally linear model for the underlying continuum
      - Wavelength shift (angstroms) is an overall offset in the wavelength direction to account for e.g. wavelength calibration errors
      - mu (angstroms) is the central wavelength of the Gaussian profile for the relevant line. If not specified, mu is assumed to be the rest air wavelength of the line.
      - sigma (angstroms) is the Gaussian standard deviation. If not specified, sigma is assumed to be the instrumental resolution (7 angstrom FWHM for our observing settings)
      - A (10^-17 erg/s/cm^2/angstrom) is the Gaussian amplitude
- C0477_fitted_spectrum_plot.png
  - Fitted optical spectra of VT 1210+4956. Used in Fig. 2 of the paper.

**X-ray modeling**
- **X_ray_lightcurve.py**
  - Code for plotting the MAXI lightcurve and effective area. Additional MCMC consistency check that the short observed burst duration is not due to the effective area. Calculation of the total corrected energy emitted during the burst.
- X_ray_counts.csv
  - Input file for X_ray_lightcurve.py. Counts per cm^2 per second in the 3 MAXI bands (2-4 keV, 4-10 keV, 10-20 keV) corrected for the effective area function at the center of the constant flux localization region given in https://gcn.gsfc.nasa.gov/gcn3/16686.gcn3
- (GCN/VT)_pos_effective_area.fits
  - Input files for X_ray_lightcurve.py. Effective area curves generated with mxscancur. Used to convert lightcurves normalized for the GCN position to the VT 1210 position normalization. See https://heasarc.gsfc.nasa.gov/docs/maxi/analysis/maxi_software_usage_20181025a.pdf for documentation
- (GCN/VT)_ position_(Model/corner)_plot_4to10keV.png
  - Output files of X_ray_lightcurve.py. Plots showing the results of the MCMC test confirming that the burst is shorter than the transient duration of the MAXI GSC.
- X_ray_lightcurve_(GCN/VT)_position.png
  - Output file of X_ray_lightcurve.py. The effective area + galactic extinction corrected lightcurve for MAXI GRB 140814A at the position given in the MAXI GCN and at the position of VT 1210+4956. The plot at the position of VT 1210+4956 is used in Fig. 3 of the paper.


