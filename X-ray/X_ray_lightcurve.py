#X-ray lightcurve
import numpy as np 
import matplotlib.pyplot as plt
from astropy.table import Table
import pandas as pd
import emcee
import corner
plt.ion()

#Read in lightcurves in counts/cm2/s corrected for GCN position effective area
counts = pd.read_csv('X_ray_counts.csv')

##########     GCN position luminosity lightcurve 2-4 keV
#from webPIMMS: 1 count per s per cm2 is 9.06e-9 erg/cm2/s
# Assuming galactic nH of 2.1e20 (appropriate for MAXI transient coords)

def cpas_to_lum_2to4keV(cpas): 
	'''converts 2-4 keV counts per cm2 per s to luminosity in erg/s'''
	return cpas*9.06e-9*2.621e54 #erg / s

seconds_since_t0_2to4keV = np.array(counts['Time_relative_to_t0'])[:-1] #Last value is nan
cpas_2to4keV = np.array(counts['Count_rate_2-4keV'])[:-1] #Counts per cm2 per s
cpas_2to4keV_err = np.array(counts['Count_rate_err_2-4keV'])[:-1]
lightcurve_2to4keV = cpas_to_lum_2to4keV(cpas_2to4keV)/1e46 #Normalize to 1e46 erg/s
lightcurve_2to4keV_err = cpas_to_lum_2to4keV(cpas_2to4keV_err)/1e46


##########     GCN position luminosity lightcurve 4-10 keV
#from webPIMMS: 1 count per s per cm2 is 1.38e-8 erg/cm2/s
# Assuming galactic nH of 2.1e20 (appropriate for MAXI transient coords)
# Assuming redshift of 0.03

def cpas_to_lum_4to10keV(cpas):
	return cpas*1.38e-8*2.621e54 #erg / s

seconds_since_t0_4to10keV = np.array(counts['Time_relative_to_t0']) 
cpas_4to10keV = np.array(counts['Count_rate_4-10keV']) #Counts per cm2 per s
cpas_4to10keV_err = np.array(counts['Count_rate_err_4-10keV'])
lightcurve_4to10keV = cpas_to_lum_4to10keV(cpas_4to10keV)/1e46
lightcurve_4to10keV_err = cpas_to_lum_4to10keV(cpas_4to10keV_err)/1e46

'''
##########     GCN position luminosity lightcurve 10-20 keV
#All nondetections

seconds_since_t0_10to20keV = np.array(counts['Time_relative_to_t0'])
cpas_10to20keV = np.array(counts['Count_rate_10-20keV']) #Counts per cm2 per s
cpas_10to20keV_err = np.array(counts['Count_rate_err_10-20keV'])

#from webPIMMS: 1 count per s per cm2 is 3.345e-8 erg/cm2/s
# Assuming galactic nH of 2.1e20 (appropriate for MAXI transient coords)
# Assuming redshift of 0.03

def cpas_to_lum_10to20keV(cpas):
	return cpas*3.348e-8*2.621e54 #erg/s

lightcurve_10to20keV = cpas_to_lum_10to20keV(cpas_10to20keV)/1e46
lightcurve_10to20keV_err = cpas_to_lum_10to20keV(cpas_10to20keV_err)/1e46
'''

#Correct for differing position

GCN_pos = Table.read('GCN_pos_effective_area.fits').to_pandas()
VT_pos = Table.read('VT_pos_effective_area.fits').to_pandas()

tmax_GCN = 4.6131421e8 + 2.35
GCN_time_zero = np.array(GCN_pos['TIME']) - tmax_GCN
GCN_area = np.array(GCN_pos['AREA'])

VT_time_zero = np.array(VT_pos['TIME']) - tmax_GCN 
VT_area = np.array(VT_pos['AREA'])

plt.figure()
plt.plot(GCN_time_zero, GCN_area, label = 'GCN_area')
plt.plot(VT_time_zero, VT_area, label = 'VT_area')
plt.plot(GCN_time_zero, GCN_area/VT_area, label = 'GCN_area / VT_area')
plt.xlim([-50,50])
plt.ylim([-1,5])
plt.legend()
plt.xlabel('Time relative to GCN effective area peak')
plt.ylabel('Effective area (cm^2), ratio')


'''
Need correction factors for the following 3s bins (-15 is the average of -16.5 to -13.5)

In [173]: seconds_since_t0_4to10keV
Out[173]: [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]

In [221]: seconds_since_t0_2to4keV
Out[221]: [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12]
'''
area_ratio = GCN_area / VT_area
correction_factors = []
GCN_area_correction = []
VT_area_correction = []
bin_radius = 1.5 #seconds, half of the bin width

for bin_center in seconds_since_t0_4to10keV:
	tmin, tmax = bin_center - bin_radius, bin_center + bin_radius
	relevant_times = np.where((GCN_time_zero > tmin) & (GCN_time_zero < tmax))
	correction_factors.append(np.average(area_ratio[relevant_times]))
	GCN_area_correction.append(np.average(GCN_area[relevant_times]))
	VT_area_correction.append(np.average(VT_area[relevant_times]))


correction_factors_4to10keV = np.array(correction_factors)
correction_factors_10to20keV = np.array(correction_factors)
correction_factors_2to4keV = correction_factors_4to10keV[:-1]

plt.figure()
plt.title('Correction factors per bin')
plt.scatter(seconds_since_t0_4to10keV, correction_factors, label = 'Avg correction factors')
plt.xlabel('Seconds since t0')
plt.show()


#Apply corrections for VT position vs GCN position
lightcurve_2to4keV_corrected, lightcurve_2to4keV_err_corrected = lightcurve_2to4keV * correction_factors_2to4keV, lightcurve_2to4keV_err * correction_factors_2to4keV
lightcurve_4to10keV_corrected, lightcurve_4to10keV_err_corrected = lightcurve_4to10keV * correction_factors_4to10keV, lightcurve_4to10keV_err * correction_factors_4to10keV
#lightcurve_10to20keV_corrected, lightcurve_10to20keV_err_corrected = lightcurve_10to20keV * correction_factors_10to20keV, lightcurve_10to20keV_err * correction_factors_10to20keV


##################### Plot calibrated lightcurve ---- VT position
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(1,1,1)
ax2 = ax1.twinx()

significant_2to4keV = []
for i in range(len(seconds_since_t0_2to4keV))[1:]:
	t = seconds_since_t0_2to4keV[i]
	flux = lightcurve_2to4keV_corrected[i]
	err = lightcurve_2to4keV_err_corrected[i]
	#if flux > err:
	if True: #Do not cut on flux > err
		alpha = 1
		significant_2to4keV.append(i)
	else:
		alpha = 0.2
	if i == 5:
		ax1.errorbar(t, flux, err, marker = 's', color = 'darkorange',label = 'Corrected 2-4 keV luminosity', fmt = 'H', alpha = alpha)
	else:
		ax1.errorbar(t, flux, err, marker = 's', color = 'darkorange', fmt = 'H', alpha = alpha)

significant_4to10keV = []
for i in range(len(seconds_since_t0_4to10keV))[1:]:
	t = seconds_since_t0_4to10keV[i]
	flux = lightcurve_4to10keV_corrected[i]
	err = lightcurve_4to10keV_err_corrected[i]
	#if flux > err:
	if True:
		alpha = 1
		significant_4to10keV.append(i)
	else:
		alpha = 0.2
	if i == 5:
		ax1.errorbar(t, flux, err, marker = 's', color = 'blue',label = 'Corrected 4-10 keV luminosity', fmt = 'H', alpha = alpha)
	else:
		ax1.errorbar(t, flux, err, marker = 's', color = 'blue', fmt = 'H', alpha = alpha)

'''
significant_10to20keV = []
for i in range(len(seconds_since_t0_10to20keV)):
	t = seconds_since_t0_10to20keV[i]
	flux = lightcurve_10to20keV_corrected[i]
	err = lightcurve_10to20keV_err_corrected[i]
	#if flux > err:
	if True:
		alpha = 1
		significant_10to20keV.append(i)
	else:
		alpha = 0.2
	if i == 5:
		ax1.errorbar(t, flux, err, marker = 's', color = 'black',label = 'Corrected 10-20 keV', fmt = 'H', alpha = alpha)
	else:
		ax1.errorbar(t, flux, err, marker = 's', color = 'black', fmt = 'H', alpha = alpha)
'''

ax1.plot(np.array(seconds_since_t0_2to4keV)[np.array(significant_2to4keV)],np.array(lightcurve_2to4keV_corrected)[np.array(significant_2to4keV)], color = 'darkorange')
ax1.plot(np.array(seconds_since_t0_4to10keV)[np.array(significant_4to10keV)],np.array(lightcurve_4to10keV_corrected)[np.array(significant_4to10keV)], color = 'blue')
#ax1.plot(np.array(seconds_since_t0_10to20keV)[np.array(significant_10to20keV)],np.array(lightcurve_10to20keV_corrected)[np.array(significant_10to20keV)], color = 'black')
ax1.axhline(0, ls = 'dashed', color = 'gray')
ax2.plot(VT_time_zero, VT_area, label = 'MAXI sensitivity at\nVT 1210+4956 position', alpha = 0.5)
ax1.set_ylim([-1,6])
ax2.set_ylim([-1,6])
plt.xlim([-20,20])
ax1.legend()
ax2.legend(loc = 2)
ax1.set_xlabel('Time relative to UT 07:12:23 on Aug 14, 2014 (s)')
ax1.set_ylabel('X-ray luminosity (10$^{46}$ erg/s)')
ax2.set_ylabel('MAXI effective area (cm$^{2}$)')
plt.savefig('X_ray_lightcurve_VT_position.png')




##################### Plot flux calibrated lightcurve ---- GCN position

fig2 = plt.figure()
ax1a = fig2.add_subplot(1,1,1)
ax2a = ax1a.twinx()


VT_significant_2to4keV = []
for i in range(len(seconds_since_t0_2to4keV)):
	t = seconds_since_t0_2to4keV[i]
	flux = lightcurve_2to4keV[i]
	err = lightcurve_2to4keV_err[i]
	#if flux > err:
	if True:
		alpha = 1
		VT_significant_2to4keV.append(i)
	else:
		alpha = 0.2
	if i == 5:
		ax1a.errorbar(t, flux, err, marker = 's', color = 'purple',label = 'Corrected 2-4 keV', fmt = 'H', alpha = alpha)
	else:
		ax1a.errorbar(t, flux, err, marker = 's', color = 'purple', fmt = 'H', alpha = alpha)

VT_significant_4to10keV = []
for i in range(len(seconds_since_t0_4to10keV)):
	t = seconds_since_t0_4to10keV[i]
	flux = lightcurve_4to10keV[i]
	err = lightcurve_4to10keV_err[i]
	#if flux > err:
	if True:
		alpha = 1
		VT_significant_4to10keV.append(i)
	else:
		alpha = 0.2
	if i == 5:
		ax1a.errorbar(t, flux, err, marker = 's', color = 'green',label = 'Corrected 4-10 keV', fmt = 'H', alpha = alpha)
	else:
		ax1a.errorbar(t, flux, err, marker = 's', color = 'green', fmt = 'H', alpha = alpha)

ax1a.plot(np.array(seconds_since_t0_2to4keV)[np.array(VT_significant_2to4keV)],np.array(lightcurve_2to4keV)[np.array(VT_significant_2to4keV)], color = 'purple', alpha = 0.5)
ax1a.plot(np.array(seconds_since_t0_4to10keV)[np.array(VT_significant_4to10keV)],np.array(lightcurve_4to10keV)[np.array(VT_significant_4to10keV)], color = 'green', alpha = 0.5)
ax1a.axhline(0, ls = 'dashed', color = 'gray')
ax2a.plot(GCN_time_zero, GCN_area, label = 'Effective area at GCN position', alpha = 0.5, color = 'orange')
ax1a.set_ylim([-1,8])
ax2a.set_ylim([-1,8])
plt.xlim([-20,20])
ax1a.legend()
ax2a.legend(loc = 2)
ax1a.set_xlabel('Time relative to UT 07:12:23 on Aug 14, 2014 (s)')
ax1a.set_ylabel('X-ray luminosity (10$^{46}$ erg/s)')
ax2a.set_ylabel('Effective area (cm$^{2}$)')
plt.savefig('X_ray_lightcurve_GCN_position.jpg')



########  Attempt to fit a tophat function with varying lengths
#Test if the observed lightcurve is consistent with being flat over the duration of the transient.

#Get the non effective area corrected "raw" lightcurve 
raw_time = np.array([-15,-12,-9,-6,-3,0,3,6,9,12,15])
raw_lc_2to4keV = np.array(counts['Uncorrected_count_rate_2-4keV'])
raw_lc_2to4keV_err = np.array(counts['Uncorrected_count_rate_2-4keV_err'])
raw_lc_4to10keV = np.array(counts['Uncorrected_count_rate_4-10keV'])
raw_lc_4to10keV_err = np.array(counts['Uncorrected_count_rate_4-10keV_err'])
norm_GCN_area = np.array(GCN_area_correction)/np.max(GCN_area_correction)
norm_VT_area = np.array(VT_area_correction)/np.max(VT_area_correction)


def modulated_tophat(width, height, time_bins, effective_area, plot = False, effective_area_time = None):
	time_bins = np.array(time_bins)
	tmin, tmax = -width/2., width/2.
	y = []
	if plot == False:
		for i,x in enumerate(time_bins):
			if x > tmin and x < tmax:
				y.append(height * effective_area[i])
			else:
				y.append(0)
	else:
		radius = 0.5
		for i, t in enumerate(time_bins):
			if t > tmin and t < tmax:
				relevant_times = np.where(np.abs(effective_area_time - t) < radius)
				relevant_area = np.mean(effective_area[relevant_times])
				y.append(height * relevant_area)
			else:
				y.append(0)
	return time_bins, np.array(y)


def log_likelihood(theta, x, y, yerr, time_bins, effective_area):
	width, height = theta
	model = modulated_tophat(width, height,time_bins, effective_area)
	sigma2 = yerr ** 2 
	return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_prior(theta):
	width, height = theta
	if 0 < width < 100. and 0 < height < 10.:
		return 0.0
	return -np.inf

def log_probability(theta, x, y, yerr, time_bins, effective_area):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood(theta, x, y, yerr, time_bins, effective_area)



#Run simulation at GCN position
pos = np.array([15.,2.]) + np.random.randn(32, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(raw_time, raw_lc_4to10keV, raw_lc_4to10keV_err, raw_time, norm_GCN_area))
sampler.run_mcmc(pos, 20000, progress=True)

#Make walker plot
fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["width", "height"]
for i in range(ndim):
	ax = axes[i]
	ax.plot(samples[:, :, i], "k", alpha=0.3)
	ax.set_xlim(0, len(samples))
	ax.set_ylabel(labels[i])
	ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

#Flatten samples
flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)


fig = corner.corner(flat_samples, labels=labels)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    print txt

plt.savefig('GCN_position_corner_plot_4to10keV.png')


#Plot of data / models for 4-10 keV
plt.figure()
plt.errorbar(raw_time, raw_lc_4to10keV, raw_lc_4to10keV_err, color = 'green',fmt = 'H', label = '4-10 keV raw lightcurve')
plt.xlabel('Time relative to UT 07:12:23 on Aug 14, 2014 (s)')
plt.ylabel('X-ray flux (counts/s)')
plt.axhline(0, color = 'gray', ls = 'dashed')
plt.xlim([-30,30])
plt.ylim([-0.4,5])

#x,y = modulated_tophat(15., 2.0, 0., raw_time, norm_GCN_area)
inds = np.random.randint(len(flat_samples), size=500)
for ind in inds:
	width, height = flat_samples[ind]
	if width > 40:
		color = 'purple'
		alpha = 1
	else:
		color = 'gray'
		alpha = 0.1
	x,y = modulated_tophat(width, height, np.linspace(-30,30,1000), GCN_area/np.max(GCN_area), plot = True, effective_area_time = GCN_time_zero)
	plt.plot(x,y, alpha = alpha, color = 'gray')

plt.plot(GCN_time_zero, GCN_area, label = 'A$_{eff}$ assuming\nconstant flux (cm$^{2}$)', alpha = 0.5, color = 'orange')
plt.legend()
plt.savefig('GCN_position_Model_plot_4to10keV.png')



#MCMC for 4-10 keV VT 1210 position
pos = np.array([15.,2.]) + np.random.randn(32, 2)
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(raw_time, raw_lc_4to10keV, raw_lc_4to10keV_err, raw_time, norm_VT_area))
sampler.run_mcmc(pos, 20000, progress=True)

#Make walker plot
fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["width", "height"]
for i in range(ndim):
	ax = axes[i]
	ax.plot(samples[:, :, i], "k", alpha=0.3)
	ax.set_xlim(0, len(samples))
	ax.set_ylabel(labels[i])
	ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

#Flatten samples
flat_samples = sampler.get_chain(discard=1000, thin=1, flat=True)#sampler.get_chain(discard=1000, thin=15, flat=True)


fig = corner.corner(flat_samples, labels=labels)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    print txt

plt.savefig('VT_position_Corner_plot_4to10keV.png')


#Plot of data / models for 4-10 keV
plt.figure()
plt.errorbar(raw_time, raw_lc_4to10keV, raw_lc_4to10keV_err, color = 'green',fmt = 'H', label = '4-10 keV raw lightcurve')
plt.xlabel('Time relative to UT 07:12:23 on Aug 14, 2014 (s)')
plt.ylabel('X-ray flux (counts/s)')
plt.axhline(0, color = 'gray', ls = 'dashed')
plt.xlim([-30,30])
plt.ylim([-0.4,5])

inds = np.random.randint(len(flat_samples), size=500)
for ind in inds:
	width, height = flat_samples[ind]
	if width > 40:
		color = 'purple'
		alpha = 1
	else:
		color = 'gray'
		alpha = 0.1
	x,y = modulated_tophat(width, height, np.linspace(-30,30,1000), VT_area/np.max(VT_area), plot = True, effective_area_time = VT_time_zero)
	plt.plot(x,y, alpha = alpha, color = 'gray')

plt.plot(VT_time_zero, VT_area, label = 'A$_{eff}$ assuming\nVT 1210+4956 position (cm$^{2}$)', alpha = 0.5, color = 'orange')
plt.legend()

plt.savefig('VT_position_Model_plot_4to10keV.png')

'''
# Test if a variable center of the burst matters
def modulated_tophat(width, height, center, time_bins, effective_area):
	time_bins = np.array(time_bins)
	tmin, tmax = center-width/2., center+width/2.
	y = []
	for i,x in enumerate(time_bins):
		if x > tmin and x < tmax:
			y.append(height * effective_area[i])
		else:
			y.append(0)
	return time_bins, np.array(y)


def log_likelihood(theta, x, y, yerr, time_bins, effective_area):
	width, height, center = theta
	model = modulated_tophat(width, height, center, time_bins, effective_area)
	sigma2 = yerr ** 2 
	return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_prior(theta):
	width, height, center = theta
	if 0 < width < 100. and 0 < height < 10. and -10 < center < 10:
		return 0.0
	return -np.inf

def log_probability(theta, x, y, yerr, time_bins, effective_area):
	lp = log_prior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + log_likelihood(theta, x, y, yerr, time_bins, effective_area)



#MCMC for 4-10 keV
pos = np.array([15.,2.,0.]) + np.random.randn(32, 3)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(raw_time, raw_lc_4to10keV, raw_lc_4to10keV_err, raw_time, norm_GCN_area))
sampler.run_mcmc(pos, 20000, progress=True)

#Make walker plot
fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["width", "height", "center"]
for i in range(ndim):
	ax = axes[i]
	ax.plot(samples[:, :, i], "k", alpha=0.3)
	ax.set_xlim(0, len(samples))
	ax.set_ylabel(labels[i])
	ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")

#Flatten samples
flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)


fig = corner.corner(flat_samples, labels=labels)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    print txt

plt.savefig('Corner_plot_4to10keV.png')


#Plot MCMC models for variable center
plt.figure()
plt.errorbar(raw_time, raw_lc_4to10keV, raw_lc_4to10keV_err, color = 'green',fmt = 'H', label = '4-10 keV raw lightcurve')
plt.xlabel('Time relative to UT 07:12:23 on Aug 14, 2014 (s)')
plt.ylabel('X-ray flux / effective area (counts/s/cm$^{2}$)')
plt.axhline(0, color = 'gray', ls = 'dashed')
plt.xlim([-30,30])
plt.ylim([-0.4,5])

inds = np.random.randint(len(flat_samples), size=500)
for ind in inds:
	width, height, center = flat_samples[ind]
	x,y = modulated_tophat(width, height, center, raw_time, norm_GCN_area)
	plt.plot(x,y, alpha = 0.1, color = 'gray')

x,y = modulated_tophat(100,1.5,0,raw_time, norm_GCN_area)
plt.plot(x,y, color = 'purple', label = '100s flat burst')
plt.plot(GCN_time_zero, GCN_area, label = 'A$_{eff}$ assuming\nconstant flux (cm$^{2}$)', alpha = 0.5, color = 'orange')
plt.legend()

plt.savefig('GCN_position_Model_plot_4to10keV.png')


#Results 4-10 keV (GCN position):
#\mathrm{width} = 15.640_{-3.102}^{3.022}
#\mathrm{height} = 1.512_{-0.422}^{0.412}
#\mathrm{center} = -0.459_{-1.440}^{1.541}
'''


###################### Total energy
total_energy_2to4keV = 0
var_2to4keV = 0
min_detected_time_bin_2to4keV = 100000
max_detected_time_bin_2to4keV = -100000
for i,lum in enumerate(lightcurve_2to4keV_corrected):
	err = lightcurve_2to4keV_err_corrected[i]
	if lum-err > 0:
		total_energy_2to4keV+=lum*3 #3s bins
		var_2to4keV+=err**2
		if seconds_since_t0_2to4keV[i] < min_detected_time_bin_2to4keV:
			min_detected_time_bin_2to4keV = seconds_since_t0_2to4keV[i]
			print seconds_since_t0_2to4keV[i]
		if seconds_since_t0_2to4keV[i] > max_detected_time_bin_2to4keV:
			max_detected_time_bin_2to4keV = seconds_since_t0_2to4keV[i]
			print seconds_since_t0_2to4keV[i]
print 'Total energy 2-4 keV (erg/s):', total_energy_2to4keV, '+/-', np.sqrt(var_2to4keV) 
print 'Burst duration:', max_detected_time_bin_2to4keV - min_detected_time_bin_2to4keV

total_energy_4to10keV = 0
var_4to10keV = 0
min_detected_time_bin_4to10keV = 100000
max_detected_time_bin_4to10keV = -100000
for i,lum in enumerate(lightcurve_4to10keV_corrected):
	err = lightcurve_4to10keV_err_corrected[i]
	if lum-err > 0:
		total_energy_4to10keV+=lum*3 #3s bins
		var_4to10keV+=err**2
		if seconds_since_t0_4to10keV[i] < min_detected_time_bin_4to10keV:
			min_detected_time_bin_4to10keV = seconds_since_t0_4to10keV[i]
			print seconds_since_t0_4to10keV[i]
		if seconds_since_t0_4to10keV[i] > max_detected_time_bin_4to10keV:
			max_detected_time_bin_4to10keV = seconds_since_t0_4to10keV[i]
			print seconds_since_t0_4to10keV[i]
print 'Total energy 4-10 keV (erg/s):', total_energy_4to10keV, '+/-', np.sqrt(var_4to10keV) 
print 'Burst duration:', max_detected_time_bin_4to10keV - min_detected_time_bin_4to10keV












