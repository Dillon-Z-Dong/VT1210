#Parse imfit logs and plot spectrum
from parse import *
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as op
from matplotlib import rc
import emcee
import corner
import astropy.units as u
plt.ion()


#Epoch 1 photometry
epoch1 = pd.read_csv('VT1210_epoch1_photometry.csv')
freqs, fluxes, errs = [np.array(x) for x in [epoch1['Freq_GHz'], epoch1['Flux_mJy'], epoch1['Flux_stat_unc_mJy']]]



def calc_tau(EM, T, nu):
	'''
	EM in pc cm^-6
	T in K
	nu in GHz
	returns eq 4.60 of https://www.cv.nrao.edu/~sransom/web/Ch4.html
	'''
	return 3.28e-7 * (T/1e4)**(-1.35) * (nu)**(-2.1) * EM

def plot_data_and_model(theta, model_freq_limits = (0.5,20), data = (freqs,fluxes,errs), plot_stretch_high = 2.0, plot_stretch_low = 10., samples = None):
	'''
	Assumes an underlying synchrotron flux A * nu^-alpha
	Then attenuates it at each frequency by e^-tau, where tau = -kappa_nu * s, 
	and kappa_nu is given by equation 4B4 of https://www.cv.nrao.edu/course/astr534/FreeFreeEmission.html
	
	EM has units of pc cm^-6
	T has units of K
	alpha is dimensionless
	A has units of mJy/GHz
	'''
	EM, T, alpha, A = theta
	#Plot data
	plt.figure()
	plt.errorbar(data[0],data[1],data[2], fmt = 'H', label = 'Data', color = 'green')
	plt.xlim([model_freq_limits[0],model_freq_limits[1]])
	plt.ylim([np.min(data[1])/plot_stretch_low, np.max(data[1])*plot_stretch_high])
	plt.loglog()
	
	#Generate synchrotron spectrum
	nu = np.linspace(model_freq_limits[0],model_freq_limits[1],1000) #GHz
	flux = A*nu**(-alpha)
	plt.plot(nu,flux, label = 'Unattenuated synchrotron')
	tau = calc_tau(EM, T, nu)
	flux_attenuated = flux * np.exp(-tau)
	
	L1 = []
	L2 = []
	if np.max(samples) != None:
		for i in range(100):
			idx = np.random.choice(range(len(samples)))
			EM_rand, T_rand, alpha_rand, A_rand = samples[idx]
			flux_rand = A*nu**(-alpha_rand)
			tau_rand = calc_tau(EM_rand, T_rand, nu)
			flux_attenuated_rand = flux_rand * np.exp(-tau_rand)
			plt.plot(nu,flux_attenuated_rand, color = 'k', alpha = 0.03)

			flux_L1_rand = A*(1.5)**(-alpha_rand)
			flux_L2_rand = A*(1.14)**(-alpha_rand)
			flux_attenuated_L1_rand = flux_L1_rand * np.exp(-calc_tau(EM_rand, T_rand, 1.5))
			L1.append(flux_attenuated_L1_rand)
			flux_attenuated_L2_rand = flux_L2_rand * np.exp(-calc_tau(EM_rand, T_rand, 1.14))
			L2.append(flux_attenuated_L2_rand)

	print '1.5 GHz flux (mJy) (16th, 50th, 84th):', np.percentile(L1,[16,50,84])
	print '1.14 GHz flux (mJy) (16th, 50th, 84th):', np.percentile(L2,[16,50,84])

	plt.plot(nu, flux_attenuated, color = 'orange', label = 'Free-free attenuated')
	plt.xlabel('Freq (GHz)')
	plt.ylabel('Flux (mJy)')
	plt.legend()
	plt.savefig('VT1210_FFA_spectrum_fit.png')
	plt.show()
	return 


############

def lnlike(theta, nu, S, Serr):
	EM, T, alpha, A = theta
	flux = A*nu**(-alpha)
	tau = calc_tau(EM, T, nu)
	model = flux * np.exp(-tau)
	inv_sigma2 = 1.0/(Serr**2)
	return -0.5*(np.sum((S-model)**2*inv_sigma2 - np.log(inv_sigma2)))


nll = lambda *args: -lnlike(*args)
initial_guess = [1e9, 1e5, 1.0, 20] #EM, T, alpha, A
ndim, nwalkers = 4, 300
pos = [np.array(initial_guess) + 1e-3*np.random.randn(ndim)*np.array(initial_guess) for i in range(nwalkers)]

#MCMC
def lnprior(theta):
	EM, T, alpha, A = theta
	if 1e2 < EM < 1e14 and 1e3 < T < 1e8 and 0.5 < alpha < 1.5 and 1. < A < 100.: #Extremely wide range of allowed parameters.
		return 0.0    
	return -np.inf

def lnprob(theta, x, y, yerr):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, x, y, yerr)

try:
	samples
except:
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(freqs,fluxes,errs))
	sampler.run_mcmc(pos, 5000)
	samples = sampler.chain[:, 4000:, :].reshape((-1, ndim))


try:
	corner_fig 
except:
	plt.figure(figsize = (10,10))
	corner_fig = corner.corner(samples, labels=[r'EM (pc cm^{-6})', r"T (K)", r'alpha', r'A (mJy/GHz)'])
	plt.savefig('VT1210-free-free-abs-corner.png')
	plt.show()


median_theta = np.median(samples,axis = 0)
med = np.median(samples,axis = 0)
plus = np.percentile(samples,84, axis = 0) - med
minus = med - np.percentile(samples,16, axis = 0) 

for i, val in enumerate(['EM','T','alpha','A']):
	median, p, m = ['{:e}'.format(s) for s in [med[i],plus[i],minus[i]]]
	print val, '     ', median, '+', p, '-', m


plot_data_and_model(median_theta, samples = samples, data = (freqs,fluxes,errs)) 
