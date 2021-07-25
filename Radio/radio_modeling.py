#Code to do MCMC fitting of epoch 1 and 2 radio spectra. 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import emcee
import corner
plt.ion()

###############################################################################################
################################    Get fluxes         ######################################## 
###############################################################################################

#Epoch 1 photometry
epoch1 = pd.read_csv('VT1210_epoch1_photometry.csv')
freqs, fluxes, errs = [np.array(x) for x in [epoch1['Freq_GHz'], epoch1['Flux_mJy'], epoch1['Flux_stat_unc_mJy']]]

#Epoch 2 photometry
epoch2 = pd.read_csv('VT1210_epoch2_photometry.csv')
freq2, flux2, err2 = [np.array(x) for x in [epoch2['Freq_GHz'], epoch2['Flux_mJy'], epoch2['Flux_stat_unc_mJy']]]

###############################################################################################
########################        Define plotting functions         #############################
###############################################################################################

def format_exp(value,tick_number):
	return '%.1f' % value#10**value

def format_exp_x(value,tick_number):
    return '%1i' % value

def plt_errorbar_log(freq,flux,err,name,epoch = 1, label = '',plt_model = False, plot_samples=None,legend = True, save = True, show = True, extra_points = None):
    fig,ax = plt.subplots(1,1,figsize=(8,5))
    arrow = u'$\u2193$'
    plt.errorbar(freq,flux,err,fmt='p',color = 'green',label = 'May 30, 2018', markersize = 4, alpha = 1)
    plt.errorbar([3.0],[2.7],[0.6], marker = 's', label = 'Nov 20, 2017', color = '#E4883F') #plot VLASS detection

    if extra_points != None:
        for i, tup in enumerate(extra_points):
            label2,freqs2,fluxes2,errs2,symb2,color2 = tup
            plt.errorbar(freqs2,fluxes2,errs2,fmt = 'H',marker = symb2, color = color2, markersize = 4, label = label2, alpha = 0.8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_exp_x))
    ax.xaxis.set_major_locator(plt.FixedLocator([1.,2.,3.,4.,5.,6.,8.,10.,12.,15.,18.])) 
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_exp))
    ax.yaxis.set_major_locator(plt.FixedLocator(range(1,8,2)))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    
    ax.tick_params(axis='both', which='major', labelsize=14, length = 6, width = 1)
    ax.tick_params(axis='both', which='minor', length = 3, width = 1)

    ax.set_xlabel('Frequency (GHz)', fontsize = 20)
    ax.set_ylabel('Flux (mJy)', fontsize = 20)

    
    ax.grid()
    ax.set_xlim(0.8,18)
    plt.tight_layout()


    if plt_model == True:
        if epoch == 1:
            result = op.minimize(nll, initial_guess, args=(freq,flux,err))
            ml_model = calc_model(result['x'])
            ax.plot(median_model[0],median_model[1],color = 'r',lw=2,alpha=0.9,label = 'Max Likelihood model')
        elif epoch == 2:
            result = op.minimize(nll, initial_guess, args=(freq2,flux2,err2))
            print result['x']
            ml_model = calc_model(result['x'])
            ax.plot(ml_model[0],ml_model[1],color = 'r',lw=2,alpha=0.9,label = 'Max Likelihood model')
        
        elif epoch == 'both':
            colors = ['green','purple'] #Epoch 1, Epoch 2
            for i, samples in enumerate(plot_samples):
                for j, samp in enumerate(samples[np.random.randint(len(samples),size=1000)]):
                    nu, Fnu, pf, fpf = calc_model(samp)
                    if j==0:
                        freq = nu
                        flux = Fnu
                    else:
                        flux = np.vstack([flux,Fnu])

                p16,p50,p84 = np.percentile(flux,[16,50,84],axis = 0)
                ax.plot(freq, p50, color = colors[i], lw = 2, alpha = 0.8)
                ax.fill_between(freq, p16, p84, alpha = 0.2, color = colors[i])
        

    
    plt.scatter([1.5],[0.41],marker = arrow,color = '#377eb8',label = 'Apr 1997 -- 3$\sigma$ upper limit', s = 500)#Plot FIRST upper limit
    

    if legend == True:
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0,2,1,3]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 4, fontsize = 16)

    if save == True:
        plt.savefig(name+'_log_SED.png')
    if show == True:
        plt.show()

###############################################################################################
#################################    Do Epoch 1 MCMC fit   #################################### 
###############################################################################################

#Model spectrum
#Using equation 1 of Granot & Sari 2002
def calc_model(theta,nu_min=0.8,nu_max=19.0, beta1 = 5./2, resolution = 1000):
    nu_b, F_nu_b_ext, s, beta2 = theta
    nu = np.linspace(nu_min,nu_max,resolution)
    F_nu = F_nu_b_ext * ((nu/nu_b)**(-s*beta1)+(nu/nu_b)**(-s*beta2))**(-1./s)
    peak_freq_GHz = nu[np.argmax(F_nu)]
    peak_flux_mJy = np.max(F_nu)
    return (nu,F_nu,peak_freq_GHz,peak_flux_mJy)

def lnlike(theta, nu, S, Serr, beta1 = 5./2):
    nu_b, F_nu_b_ext, s, beta2 = theta
    model = F_nu_b_ext * ((nu/nu_b)**(-s*beta1)+(nu/nu_b)**(-s*beta2))**(-1./s)
    inv_sigma2 = 1.0/(Serr**2)
    return -0.5*(np.sum((S-model)**2*inv_sigma2 - np.log(inv_sigma2)))


nll = lambda *args: -lnlike(*args)
initial_guess = [3.0,13.0,0.7,-0.8]

def lnprior(theta):
    nu_b, F_nu_b_ext, s, beta2 = theta
    if 2.0 < nu_b < 5.0 and 8. < F_nu_b_ext < 15. and 0.1 < s < 2. and -1.5 < beta2 < -0.7:
        return 0.0
    return -np.inf


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


ndim, nwalkers = 4, 300
pos = [np.array(initial_guess) + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]


try:
    samples
except:
    print 'Fitting Epoch 1'
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(freqs,fluxes,errs))
    sampler.run_mcmc(pos, 5000)
    samples = sampler.chain[:, 3000:, :].reshape((-1, ndim))


fig = corner.corner(samples, labels=[r'$\nu_b$ (GHz)', r"$F_{\nu_b,ext}$ (mJy)", r's', r'$\beta_2$'])

for ax in fig.get_axes():
    ax.tick_params(which = 'major', axis = 'both',labelsize=16, direction='out',length=6)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
fig.savefig("Granot_triangle_epoch1.png")


nu_b_mcmc, F_nu_b_ext_mcmc, s_mcmc, beta2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print 'nu_b_mcmc:', nu_b_mcmc
print 'F_nu_b_ext_mcmc:', F_nu_b_ext_mcmc
print 's_mcmc:', s_mcmc
print 'beta2_mcmc', beta2_mcmc



###############################################################################################
#################################    Do Epoch 2 MCMC fit   #################################### 
###############################################################################################

try:
    samples2
except:
    print 'Fitting Epoch 2'
    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(freq2,flux2,err2))
    sampler2.run_mcmc(pos, 5000)
    samples2 = sampler2.chain[:, 3000:, :].reshape((-1, ndim))


fig = corner.corner(samples, labels=[r'$\nu_b$ (GHz)', r"$F_{\nu_b,ext}$ (mJy)", r's', r'$\beta_2$'])

for ax in fig.get_axes():
    ax.tick_params(which = 'major', axis = 'both',labelsize=16, direction='out',length=6)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)
fig.savefig("Granot_triangle_epoch2.png")


nu_b_mcmc2, F_nu_b_ext_mcmc2, s_mcmc2, beta2_mcmc2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples2, [16, 50, 84],axis=0)))

print 'nu_b_mcmc:', nu_b_mcmc2
print 'F_nu_b_ext_mcmc:', F_nu_b_ext_mcmc2
print 's_mcmc:', s_mcmc2
print 'beta2_mcmc', beta2_mcmc2
plt.show()


###############################################################################################
####################################    Run plot command   #################################### 
###############################################################################################
E1 = ('Epoch 1.1, Nov 20. 2017',[2.220,2.732,3.244,3.756],[1.606,2.087,3.627,3.691],[0.20,0.23,0.23,0.27],'D','k')
followup2 = [('April 30 - May 8, 2019', freq2, flux2, err2, '*','purple')]
plt_errorbar_log(freqs,fluxes,errs, epoch = 'both', name = 'VT1210_subplot_nolabels', label = 'May 30. 2018', plt_model = True, save = True, show = True, plot_samples = [samples,samples2], extra_points = followup2)

###############################################################################################
###################   Calculate physical quantities in both epochs   ##########################
###############################################################################################

def calc_radius(peak_freq,peak_flux,dist_Mpc,epsilon_e = 0.1,epsilon_B = 0.1, f = 0.5, p = 3):
    '''
    Returns the equipartition radius (in cm) of a given model
    Using equation 10 of Ho et al.
    '''
    if p != 3:
        print 'WARNING: the radius formula is not correct for p != 3'
    peak_freq_5GHz, peak_flux_Jy = peak_freq/5., peak_flux/1000.

    return 8.8*10**15 * (epsilon_e/epsilon_B)**(-1./19) * (f/0.5)**(-1./19) * (peak_flux_Jy)**(9./19) * dist_Mpc**(18./19) * peak_freq_5GHz**(-1.)

def calc_B(peak_freq,peak_flux,dist_Mpc,epsilon_e = 0.1,epsilon_B = 0.1, f = 0.5, p = 3):
    '''
    Returns the B field (in Gauss) of a given model
    Using equation 11 of Ho et al.
    '''
    if p != 3:
        print 'WARNING: the B field formula is not correct for p != 3'
    peak_freq_5GHz, peak_flux_Jy = peak_freq/5., peak_flux/1000.

    return 0.58 * (epsilon_e/epsilon_B)**(-4./19) * (f/0.5)**(-4./19) * (peak_flux_Jy)**(-2./19) * dist_Mpc**(-4./19) * peak_freq_5GHz

def calc_energy(peak_freq,peak_flux,dist_Mpc,epsilon_e = 0.1,epsilon_B = 0.1, f = 0.5, p = 3):
    '''
    Returns the equipartition energy (in erg) of a given model
    Using equation 12 of Ho et al.
    '''
    if p != 3:
        print 'WARNING: the energy formula is not correct for p != 3'
    peak_freq_5GHz, peak_flux_Jy = peak_freq/5., peak_flux/1000.

    return 1.9e46 * (1./epsilon_B) * (epsilon_e/epsilon_B)**(-11./19) * (f/0.5)**(8./19) * (peak_flux_Jy)**(23./19) * dist_Mpc**(46./19) * peak_freq_5GHz**(-1.)


#Get distribution of observed peak frequencies, fluxes
dist = 148. #Mpc
try:
    assert(len(peak_flux_epoch1) > 0)
except:
    peak_freq_epoch1 = []
    peak_flux_epoch1 = []
    for i in range(len(samples)):
        junkfreq,junkflux,peak_freq,peak_flux = calc_model(samples[i],nu_min = 3.0,nu_max = 6.0)
        peak_freq_epoch1.append(peak_freq)
        peak_flux_epoch1.append(peak_flux)
    peak_freq_epoch1, peak_flux_epoch1 = [np.array(x) for x in [peak_freq_epoch1,peak_flux_epoch1]]
    
plt.figure()
plt.hist(peak_freq_epoch1, bins = 300)
plt.title('Peak freq, median ' + str(np.median(peak_freq_epoch1)) + '+/-' + str(np.std(peak_freq_epoch1)))
plt.savefig('Peak_freq_epoch1.png')
plt.show()
plt.figure()
plt.hist(peak_flux_epoch1, bins = 300)
plt.title('Peak flux, median ' + str(np.median(peak_flux_epoch1)) + '+/-' + str(np.std(peak_flux_epoch1)))
plt.savefig('Peak_flux_epoch1.png')
plt.show()

try:
    assert(len(peak_flux_epoch2) > 0)
except:
    peak_freq_epoch2 = []
    peak_flux_epoch2 = []
    for i in range(len(samples2)):
        junkfreq,junkflux,peak_freq,peak_flux = calc_model(samples2[i],nu_min = 3.0,nu_max = 6.0)
        peak_freq_epoch2.append(peak_freq)
        peak_flux_epoch2.append(peak_flux)    
    peak_freq_epoch2, peak_flux_epoch2 = [np.array(x) for x in [peak_freq_epoch2,peak_flux_epoch2]]
    
plt.figure()
plt.hist(peak_freq_epoch2, bins = 300)
plt.title('Peak freq, median ' + str(np.median(peak_freq_epoch2)) + '+/-' + str(np.std(peak_freq_epoch2)))
plt.savefig('Peak_freq_epoch2.png')
plt.show()
plt.figure()
plt.hist(peak_flux_epoch2, bins = 300)
plt.title('Peak flux, median ' + str(np.median(peak_flux_epoch2)) + '+/-' + str(np.std(peak_flux_epoch2)))
plt.savefig('Peak_flux_epoch2.png')
plt.show()


a,b,c = np.percentile(peak_freq_epoch1, [16,50,84])
print 'Peak freq epoch 1:', b, c-b, b-a

a,b,c = np.percentile(peak_freq_epoch2, [16,50,84])
print 'Peak freq epoch 2:', b, c-b, b-a

a,b,c = np.percentile(peak_flux_epoch1, [16,50,84])
print 'Peak flux epoch 1:', b, c-b, b-a

a,b,c = np.percentile(peak_flux_epoch2, [16,50,84])
print 'Peak flux epoch 2:', b, c-b, b-a



##### Estimation of R, B, U with default epsilon_B = epsilon_e = 0.1, f = 0.5 (stat) and range of epsilon_B, f but epsilon_e = 0.1 (sys)
dist_Mpc = 148.
log10_random_epsilon_B = np.random.uniform(-3,-1, len(peak_freq_epoch1))
random_epsilon_B = 10**(log10_random_epsilon_B)
random_f = np.random.uniform(0.1,0.5,len(peak_freq_epoch1))

radius_dist_stat_epoch1 = calc_radius(peak_freq_epoch1,peak_flux_epoch1,dist_Mpc,epsilon_e = 0.1,epsilon_B = 0.1, f = random_f, p = 3)
B_dist_stat_epoch1 = calc_B(peak_freq_epoch1,peak_flux_epoch1,dist_Mpc,epsilon_e = 0.1,epsilon_B = 0.1, f = random_f, p = 3)
energy_dist_stat_epoch1 = calc_energy(peak_freq_epoch1,peak_flux_epoch1,dist_Mpc,epsilon_e = 0.1,epsilon_B = 0.1, f = random_f, p = 3)

plt.figure()
plt.hist(radius_dist_stat_epoch1, bins = 1000)
plt.title('Epoch 1 Radius distribution (stat errors) R/1e16 cm = \n'+str(np.median(radius_dist_stat_epoch1/1e16))+' +/- '+str(np.std(radius_dist_stat_epoch1)/1e16))
plt.savefig('radius_dist_stat_epoch1.png')

plt.figure()
plt.hist(B_dist_stat_epoch1, bins = 1000)
plt.title('Epoch 1 B field distribution (stat errors) B Gauss = \n'+str(np.median(B_dist_stat_epoch1))+' +/- '+str(np.std(B_dist_stat_epoch1)))
plt.savefig('B_dist_stat_epoch1.png')

plt.figure()
plt.hist(energy_dist_stat_epoch1, bins = 1000)
plt.title('Epoch 1 Energy distribution (stat errors) E/1e49 erg = \n'+str(np.median(energy_dist_stat_epoch1/1e49))+' +/- '+str(np.std(energy_dist_stat_epoch1)/1e49))
plt.savefig('Epoch 1 energy_dist_stat_epoch1.png')


a,b,c = np.percentile(radius_dist_stat_epoch1, [16,50,84])
print 'radius_dist_stat_epoch1 (1e16 cm):', b/1e16, (c-b)/1e16, (b-a)/1e16

a,b,c = np.percentile(B_dist_stat_epoch1, [16,50,84])
print 'B_dist_stat_epoch1 (G):', b, (c-b), (b-a)

a,b,c = np.percentile(energy_dist_stat_epoch1, [16,50,84])
print 'energy_dist_stat_epoch1 (1e49 erg):', b/1e49, (c-b)/1e49, (b-a)/1e49


radius_dist_sys_epoch1 = calc_radius(peak_freq_epoch1,peak_flux_epoch1,dist_Mpc,epsilon_e = 0.1,epsilon_B = random_epsilon_B, f = random_f, p = 3)
B_dist_sys_epoch1 = calc_B(peak_freq_epoch1,peak_flux_epoch1,dist_Mpc,epsilon_e = 0.1,epsilon_B = random_epsilon_B, f = random_f, p = 3)
energy_dist_sys_epoch1 = calc_energy(peak_freq_epoch1,peak_flux_epoch1,dist_Mpc,epsilon_e = 0.1,epsilon_B = random_epsilon_B, f = random_f, p = 3)

plt.figure()
plt.hist(radius_dist_sys_epoch1, bins = 1000)
plt.title('Epoch 1 Radius distribution (sys errors) R/1e16 cm = \n'+str(np.median(radius_dist_sys_epoch1/1e16))+' +/- '+str(np.std(radius_dist_sys_epoch1)/1e16))
plt.savefig('radius_dist_sys_epoch1.png')

plt.figure()
plt.hist(B_dist_sys_epoch1, bins = 1000)
plt.title('Epoch 1 B field distribution (sys errors) B Gauss = \n'+str(np.median(B_dist_sys_epoch1))+' +/- '+str(np.std(B_dist_sys_epoch1)))
plt.savefig('B_dist_sys_epoch1.png')

plt.figure()
plt.hist(energy_dist_sys_epoch1, bins = 1000)
plt.title('Epoch 1 Energy distribution (sys errors) E/1e49 erg = \n'+str(np.median(energy_dist_sys_epoch1/1e49))+' +/- '+str(np.std(energy_dist_sys_epoch1)/1e49))
plt.semilogx()
plt.savefig('energy_dist_sys_epoch1.png')


a,b,c = np.percentile(radius_dist_sys_epoch1, [16,50,84])
print 'radius_dist_sys_epoch1 (1e16 cm):', b/1e16, (c-b)/1e16, (b-a)/1e16

a,b,c = np.percentile(B_dist_sys_epoch1, [16,50,84])
print 'B_dist_sys_epoch1 (G):', b, (c-b), (b-a)

a,b,c = np.percentile(energy_dist_sys_epoch1, [16,50,84])
print 'energy_dist_sys_epoch1 (1e49 erg):', b/1e49, (c-b)/1e49, (b-a)/1e49


########## Epoch 2 -- keeping the same epsilon_B and f vectors (so self consistent with epoch 1)

radius_dist_stat_epoch2 = calc_radius(peak_freq_epoch2,peak_flux_epoch2,dist_Mpc,epsilon_e = 0.1,epsilon_B = 0.1, f = random_f, p = 3)
B_dist_stat_epoch2 = calc_B(peak_freq_epoch2,peak_flux_epoch2,dist_Mpc,epsilon_e = 0.1,epsilon_B = 0.1, f = random_f, p = 3)
energy_dist_stat_epoch2 = calc_energy(peak_freq_epoch2,peak_flux_epoch2,dist_Mpc,epsilon_e = 0.1,epsilon_B = 0.1, f = random_f, p = 3)

plt.figure()
plt.hist(radius_dist_stat_epoch2, bins = 1000)
plt.title('Epoch 1 Radius distribution (stat errors) R/1e16 cm = \n'+str(np.median(radius_dist_stat_epoch2/1e16))+' +/- '+str(np.std(radius_dist_stat_epoch2)/1e16))
plt.savefig('radius_dist_stat_epoch2.png')

plt.figure()
plt.hist(B_dist_stat_epoch2, bins = 1000)
plt.title('Epoch 1 B field distribution (stat errors) B Gauss = \n'+str(np.median(B_dist_stat_epoch2))+' +/- '+str(np.std(B_dist_stat_epoch2)))
plt.savefig('B_dist_stat_epoch2.png')

plt.figure()
plt.hist(energy_dist_stat_epoch2, bins = 1000)
plt.title('Epoch 1 Energy distribution (stat errors) E/1e49 erg = \n'+str(np.median(energy_dist_stat_epoch2/1e49))+' +/- '+str(np.std(energy_dist_stat_epoch2)/1e49))
plt.savefig('Epoch 1 energy_dist_stat_epoch2.png')

a,b,c = np.percentile(radius_dist_stat_epoch2, [16,50,84])
print 'radius_dist_stat_epoch2 (1e16 cm):', b/1e16, (c-b)/1e16, (b-a)/1e16

a,b,c = np.percentile(B_dist_stat_epoch2, [16,50,84])
print 'B_dist_stat_epoch2 (G):', b, (c-b), (b-a)

a,b,c = np.percentile(energy_dist_stat_epoch2, [16,50,84])
print 'energy_dist_stat_epoch2 (1e49 erg):', b/1e49, (c-b)/1e49, (b-a)/1e49


radius_dist_sys_epoch2 = calc_radius(peak_freq_epoch2,peak_flux_epoch2,dist_Mpc,epsilon_e = 0.1,epsilon_B = random_epsilon_B, f = random_f, p = 3)
B_dist_sys_epoch2 = calc_B(peak_freq_epoch2,peak_flux_epoch2,dist_Mpc,epsilon_e = 0.1,epsilon_B = random_epsilon_B, f = random_f, p = 3)
energy_dist_sys_epoch2 = calc_energy(peak_freq_epoch2,peak_flux_epoch2,dist_Mpc,epsilon_e = 0.1,epsilon_B = random_epsilon_B, f = random_f, p = 3)

plt.figure()
plt.hist(radius_dist_sys_epoch2, bins = 1000)
plt.title('Epoch 1 Radius distribution (sys errors) R/1e16 cm = \n'+str(np.median(radius_dist_sys_epoch2/1e16))+' +/- '+str(np.std(radius_dist_sys_epoch2)/1e16))
plt.savefig('radius_dist_sys_epoch2.png')

plt.figure()
plt.hist(B_dist_sys_epoch2, bins = 1000)
plt.title('Epoch 1 B field distribution (sys errors) B Gauss = \n'+str(np.median(B_dist_sys_epoch2))+' +/- '+str(np.std(B_dist_sys_epoch2)))
plt.savefig('B_dist_sys_epoch2.png')

plt.figure()
plt.hist(energy_dist_sys_epoch2, bins = 1000)
plt.title('Epoch 1 Energy distribution (sys errors) E/1e49 erg = \n'+str(np.median(energy_dist_sys_epoch2/1e49))+' +/- '+str(np.std(energy_dist_sys_epoch2)/1e49))
plt.semilogx()
plt.savefig('energy_dist_sys_epoch2.png')


a,b,c = np.percentile(radius_dist_sys_epoch2, [16,50,84])
print 'radius_dist_sys_epoch2 (1e16 cm):', b/1e16, (c-b)/1e16, (b-a)/1e16

a,b,c = np.percentile(B_dist_sys_epoch2, [16,50,84])
print 'B_dist_sys_epoch2 (G):', b, (c-b), (b-a)

a,b,c = np.percentile(energy_dist_sys_epoch2, [16,50,84])
print 'energy_dist_sys_epoch2 (1e49 erg):', b/1e49, (c-b)/1e49, (b-a)/1e49

###### Velocity estimation between Epoch 1 and 2
import astropy.units as u
deltaT_epoch1_vs_2 = 323*u.day

velocity_dist_stat = (((radius_dist_stat_epoch2 - radius_dist_stat_epoch1)*u.cm) / (deltaT_epoch1_vs_2)).to(u.km*u.s**-1)

plt.figure()
plt.hist(velocity_dist_stat.value, bins = 1000)
plt.xlabel('Velocity (km/s)')
plt.title('Velocity between Epoch 1 and 2 (km/s) stat errors \n' + str(np.median(velocity_dist_stat.value)) + '+/-' + str(np.std(velocity_dist_stat.value)))
plt.savefig('velocity_dist_stat_between_epoch1_and_2.png')

a,b,c = np.percentile(velocity_dist_stat, [16,50,84])
print 'velocity_dist_stat (km/s):', b, (c-b), (b-a)


velocity_dist_sys = (((radius_dist_sys_epoch2 - radius_dist_sys_epoch1)*u.cm) / (deltaT_epoch1_vs_2)).to(u.km*u.s**-1)

plt.figure()
plt.hist(velocity_dist_sys.value, bins = 1000)
plt.xlabel('Velocity (km/s)')
plt.title('Velocity between Epoch 1 and 2 (km/s) sys errors \n' + str(np.median(velocity_dist_sys.value)) + '+/-' + str(np.std(velocity_dist_sys.value)))
plt.savefig('velocity_dist_sys_between_epoch1_and_2.png')


a,b,c = np.percentile(velocity_dist_sys, [16,50,84])
print 'velocity_dist_sys (km/s):', b, (c-b), (b-a)


#Use H alpha velocity instead
Ha_velocity_dist_stat = np.random.normal(1345,60,len(radius_dist_stat_epoch1))


####### Density estimation between Epoch 1 and 2
def calc_density(peak_freq,peak_flux,dist_Mpc, velocity, epsilon_e = 0.1,epsilon_B = 0.1, f = 0.5, p = 3):
    peak_freq_5GHz, peak_flux_Jy = peak_freq/5., peak_flux/1000.
    peak_luminosity_1e28erg_s_Hz = 11.96 * 1e-2 * peak_flux_Jy * (dist_Mpc)**2
    velocity = velocity/1000. #normalize to 1000 km/s
    return 3.9e6 * (epsilon_B/0.1)**(-1) * (epsilon_e/epsilon_B)**(-8./19) * (f/0.2)**(-8./19) * (peak_luminosity_1e28erg_s_Hz)**(-4./19) * peak_freq_5GHz**(2.) * (velocity)**(-2.)

plt.figure()
n_epoch1_stat = calc_density(peak_freq_epoch1,peak_flux_epoch1,dist_Mpc,velocity = Ha_velocity_dist_stat, epsilon_e = 0.1,epsilon_B = 0.1, f = random_f, p = 3)
plt.hist(n_epoch1_stat, bins = 1000)
plt.title('Density using shock jump condition (10^6 cm^-3) \n'+str(np.median(n_epoch1_stat)/1e6) + '+' + str(np.percentile(n_epoch1_stat,86)/1e6 - np.median(n_epoch1_stat)/1e6) + '-' + str(np.median(n_epoch1_stat)/1e6 - np.percentile(n_epoch1_stat,16)/1e6))
plt.semilogx()
plt.savefig('density_distribution_stat.png')

a,b,c = np.percentile(n_epoch1_stat, [16,50,84])
print 'density distribution stat (1e6 cm^-3):', b, (c-b), (b-a)


n_epoch1_sys = calc_density(peak_freq_epoch1,peak_flux_epoch1,dist_Mpc,velocity = Ha_velocity_dist_stat, epsilon_e = 0.1,epsilon_B = random_epsilon_B, f = random_f, p = 3)
plt.figure()
plt.hist(n_epoch1_sys, bins = 1000)
plt.title('Density using shock jump condition (10^6 cm^-3) \n'+str(np.median(n_epoch1_sys)/1e6) + '+' + str(np.percentile(n_epoch1_sys,86)/1e6 - np.median(n_epoch1_sys)/1e6) + '-' + str(np.median(n_epoch1_sys)/1e6 - np.percentile(n_epoch1_sys,16)/1e6))
plt.semilogx()
plt.savefig('density_distribution_sys.png')

a,b,c = np.percentile(n_epoch1_sys, [16,50,84])
print 'density distribution sys (1e6 cm^-3):', b, (c-b), (b-a)


