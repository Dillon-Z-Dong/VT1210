import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import emcee
import corner
from IPython.display import display, Math
from matplotlib import rc
import matplotlib
from collections import OrderedDict


spec_C477 = pd.read_csv('lrisC0477_g1.spec',sep='\s+',skiprows= 117, names = ['wavelen','flux','sky_flux','flux_unc','xpixel','ypixel','response'])
wavelen_C477 = np.array(spec_C477['wavelen'])
flux_C477 = np.array(spec_C477['flux'])
flux_unc_C477 = np.array(spec_C477['flux_unc'])
z_C477 = 0.03472


def x_clip(xmin,xmax,wavelen):
    clip = np.where((xmin < wavelen) & (wavelen < xmax))
    return (clip, wavelen[clip])

def gaussian(x, mu, A, sigma):
    #Does not take into account any offsets!
    return (x,A*(1./(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5 * np.power((x - mu)/sigma, 2.)))

def integrated_flux(x, mu, A, sigma):
    #gaussian(...) returns a discreet wavelength array "lambda", and a model flux array "f_lambda"
    lamb, f_lamb = gaussian(x,mu,A,sigma)
    dlamb = lamb[1]-lamb[0]
    return np.sum(dlamb * f_lamb)

def velocity(rest_lambda, sigma):
    #This is the FWHM velocity
    return (sigma / rest_lambda) * 3e5 * 2 * np.sqrt(2*np.log(2))


def plot(wavelen,flux,flux_err,redshift, models = [], shift = 0.0, model_offset = 0.0, model_slope = 0.0, rest_lambda_min = 6500., rest_lambda_max = 6600., ymin = -0.5,ymax = 15., filename = 'test', save = False, labels = None, plot_uncertainties = False, velocity_plot_central_wavelength = None):#,line_axv = 6563.):
    '''
    If plotting shaded uncertainty regions, need to have models in the form [[(median mu,median A,median sigma,color),(median mu, 84th A, 84th sigma), (median mu, 16th A, 16th sigma)]]
    Note the double list
    '''
    fig,ax1 = plt.subplots(figsize = (8,7))
    ax1.set_xlabel('Rest wavelength '+u' ($\mathring{A}$)', fontsize = 15)
    ax1.set_xlim(rest_lambda_min,rest_lambda_max)
    ax1.xaxis.set_minor_locator(plt.MultipleLocator(5))
    
    if plot_uncertainties == True:
        ax1.errorbar(wavelen/(1+redshift) + shift, flux*1e17, flux_err*1e17, color = 'black', alpha = 0.25, label = 'Data')
        ax1.set_ylim(ymin,ymax)
        ax1.set_ylabel('Flux '+u'($10^{-17} $'+' erg '+u' s$^{-1}$'+u' cm$^{-2}$'+u' $\mathring{A}$)', fontsize = 15)
        
        if velocity_plot_central_wavelength != None:
            if type(velocity_plot_central_wavelength) == bool:
                velocity_plot_central_wavelength = int(np.average(rest_lambda_min, rest_lambda_max))
            ax2 = ax1.twiny()
            ax2.set_ylim(ymin,ymax)
            ax2.set_xlim((rest_lambda_min-velocity_plot_central_wavelength)*3e5/(velocity_plot_central_wavelength),(rest_lambda_max-velocity_plot_central_wavelength)*3e5/(velocity_plot_central_wavelength))
            ax2.set_xlabel('Velocity relative to '+str(velocity_plot_central_wavelength)+u' $\mathring{A}$ (km/s)', fontsize = 15)
            ax2.xaxis.set_minor_locator(plt.MultipleLocator(100))

        if len(models) > 0 and type(models) == list:
            median_model_curves = []
            p84th_model_curves = []
            p16th_model_curves = []
            x = x_clip(xmin = rest_lambda_min,xmax = rest_lambda_max,wavelen = wavelen)[1]
            for i, tup_list in enumerate(models):
                median,plus,minus = tup_list
                
                mu,A,sigma,color = median 
                mu84,A84,sigma84, color = plus
                mu16,A16,sigma16, color = minus
                
                median_model = gaussian(x = x+shift, mu = mu, A = A, sigma = sigma)#*1e17
                model84 = gaussian(x = x+shift, mu = mu84, A = A84, sigma = sigma84)#*1e17
                model16 = gaussian(x = x+shift, mu = mu16, A = A16, sigma = sigma16)#*1e17

                ax1.plot(median_model[0],(median_model[1]+model_offset + (x+shift)*model_slope)*1e17, linestyle = '-.', alpha = 0.9, color = color, label = labels[i])
                ax1.fill_between(median_model[0], (model84[1]+model_offset + (x+shift)*model_slope)*1e17, (model16[1]+model_offset + (x+shift)*model_slope)*1e17, color = color, alpha = 0.25)
                
                median_model_curves.append(median_model[1])
                p84th_model_curves.append(model84[1])
                p16th_model_curves.append(model16[1])
                
            ax1.plot(median_model[0],(sum(median_model_curves) + model_offset + (x+shift)*model_slope)*1e17, label = 'Sum', linestyle = '--', color = 'black')
            ax1.fill_between(median_model[0], (sum(p84th_model_curves) + model_offset + (x+shift)*model_slope)*1e17, (sum(p16th_model_curves) + model_offset + (x+shift)*model_slope)*1e17, color = 'black', alpha = 0.25)
            
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        if save == True:
            plt.savefig(filename + '_fitted_spectrum_plot.png')
            plt.savefig(filename + '_fitted_spectrum_plot.eps')
        plt.show()
    else:
        ax1.plot(wavelen/(1+redshift) + shift, flux*1e17,color = 'black',alpha = 0.25, label = 'Data')
        ax1.set_ylim(ymin,ymax) 
        ax1.set_ylabel('Flux '+u'($10^{-17} $'+' erg '+u' s$^{-1}$'+u' cm$^{-2}$'+u' $\mathring{A}$)', fontsize = 15)

        if velocity_plot_central_wavelength != None:
            if type(velocity_plot_central_wavelength) == bool:
                velocity_plot_central_wavelength = int(np.average(rest_lambda_min, rest_lambda_max))
            ax2 = ax1.twiny()
            #ax2.plot(((wavelen/(1+redshift) + shift)-velocity_plot_central_wavelength)*3e5/(velocity_plot_central_wavelength), flux*1e17, flux_err*1e17, color = 'black', alpha = 0.5, label = 'Data')
            ax2.set_ylim(ymin,ymax)
            ax2.set_xlim((rest_lambda_min-velocity_plot_central_wavelength)*3e5/(velocity_plot_central_wavelength),(rest_lambda_max-velocity_plot_central_wavelength)*3e5/(velocity_plot_central_wavelength))
            ax2.set_xlabel('Velocity relative to '+str(velocity_plot_central_wavelength)+u' $\mathring{A}$ (km/s)', fontsize = 15)
            ax2.xaxis.set_minor_locator(plt.MultipleLocator(100))

        if len(models) > 0:
            model_curves = []
            for i, tup in enumerate(models):
                mu,A,sigma,color = tup 
                x = x_clip(xmin = rest_lambda_min,xmax = rest_lambda_max,wavelen = wavelen)[1]
                model = gaussian(x = x+shift, mu = mu, A = A, sigma = sigma)
                ax1.plot(model[0],(model[1]+model_offset + (x+shift)*model_slope)*1e17, linestyle = '-.', alpha = 0.9, color = color, label = labels[i])
                model_curves.append(model[1])

            ax1.plot(model[0],1e17*(sum(model_curves) + model_offset + (x+shift)*model_slope), label = 'Sum', linestyle = '--', color = 'black')
        ax1.legend()
        if save == True:
            plt.savefig(filename + '_fitted_spectrum_plot.png')
        plt.show()
            
def vac_to_air(vac):
    return vac/(1.0 + 2.735182e-4 + 131.4182/vac**2 + 2.76249e8/vac**4)


def estimate_offset_and_slope(wavelen,flux,region_min,region_max,emission_min,emission_max):
    continuum = np.where(((region_min < wavelen) & (wavelen < emission_min)) | ((emission_max < wavelen) & (wavelen < region_max)))
    x = wavelen[continuum]
    y = flux[continuum]
    return x,y


def fit(xmin, xmax, wavelen, flux, flux_err, redshift, dist_Mpc, lines = {}, offset_guess = 0.6e-17, slope_guess = -1e-22, shift_guess = 0.1,  instrumental_sigma = 3.0, A_max = 1e-14, nwalkers = 100, filename = 'test', save = False, make_plot = False, ymin = None, ymax = None, colors = ['green','blue','purple','purple'], noise_buffer = 30., continuum_subtract = False, velocity_plot_central_wavelength = None, relax_broad_line_pos = False, fix_OIII_relative = False):
    '''
    Varies parameters per model: (red/blueshift, sigma, A) and for the whole set (offset) so that it best fits the data
    '''
    ###################################   Preprocessing   ####################################
    #-->  old code:  theta must be a 1D array of the form [offset, rest_lambda1, mu1, A, sigma, rest_lambda2, mu2, A2, sigma2, ...]
    #new code: pass in names = ['Offset', 'Overall_shift', 'Ha_broad_mu', 'Ha_broad_sigma', etc.] [offset, overall_shift]
    
    #example: lines = {'Ha_broad': (6562.8, ['mu','A','sigma'],[6562.8, 1.5e-16, 11.]), 'Ha_narrow':(6562.8,['A'],[7.5e-17]), ... etc}
    
    ########## Initial guess parameters
    theta_guess = [offset_guess, shift_guess, slope_guess]
    rest_lambdas = []
    linenames = sorted(lines.keys())
    varnames = ['Offset','Shift', 'Slope']
    line_class = []
    for line in linenames:
        rest_lambda, name, guess = lines[line]
        
        rest_lambdas.append(rest_lambda)
        varnames += [line + '_' + var for var in name]
        theta_guess += list(guess)
        if len(name) == 1:
            line_class.append('narrow')
        elif len(name) == 2:
            line_class.append('broad_fixed_pos')
        else:
            line_class.append('broad')
    print 'Fitting parameters: ', varnames
    theta_guess = np.array(theta_guess)
    print 'Theta_guess:', theta_guess
    print 'linenames:', linenames
    print 'line_class:', line_class
    

    ######### Array dimensions
    ndim = len(theta_guess) 
    #nlines = len(rest_lambdas)
    #nlines = ndim // 3 #floor division (similar to integer division. Shouldn't break with python 3.)

    ######### x, y, yerr arrays
    clip, x = x_clip(xmin,xmax,wavelen/(1+redshift))
    y = flux[clip]
    yerr = flux_err[clip]*2.0 #errors are almost certainly underestimated and correlated.

    def log_likelihood(theta, x, y, yerr):
        offset = theta[0]
        shift = theta[1]
        slope = theta[2]
        model = np.zeros_like(x) + offset + x*slope #initial y model
        read_pos = 3 #position down the 1D theta vector that you've read to. Starts with 3 b/c of offset, shift, and slope
        for i, line in enumerate(linenames):
            if line_class[i] == 'narrow':
                mu = rest_lambdas[i]
                sigma = instrumental_sigma
                A = theta[read_pos]
                model += gaussian(x+shift, mu, A, sigma)[1]
                read_pos += 1
            elif line_class[i] == 'broad_fixed_pos':
                A, sigma = theta[read_pos:read_pos+2]
                mu = rest_lambdas[i]
                model += gaussian(x+shift, mu, A, sigma)[1]
                read_pos += 2
            elif line_class[i] == 'broad':
                mu, A, sigma = theta[read_pos:read_pos+3]
                model += gaussian(x+shift, mu, A, sigma)[1]
                read_pos += 3
        sigma2 = yerr**2
        return -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))


    def log_prior(theta):
        likelihood = 0.0
        offset = theta[0]
        shift = theta[1]
        slope = theta[2]
        if (np.abs(offset)/np.abs(offset_guess)) > 5.: #or (np.abs(offset)/np.abs(offset_guess)) < 0.2: #or offset < 0.0:  
            #print 'Offset guess is too small or something is going wrong!!'
            #print offset, offset_guess, np.abs(offset)/np.abs(offset_guess)
            #print 'Offset guess:', offset_guess, 'Offset:', offset
            return -np.inf
        
        if np.abs(shift) > 5.:
            #print 'Fitted shift is above 5 angstroms!'
            return -np.inf

        if (np.abs(slope)/np.abs(slope_guess)) > 5.: #or (np.abs(slope)/np.abs(slope_guess)) < 0.2:
            #print 'Slope guess is way off or something has gone wrong!!'
            #print slope, slope_guess
            return -np.inf

        read_pos = 3 #position down the 1D theta vector that you've read to
        for i, line in enumerate(linenames):
            if line_class[i] == 'narrow':
                A = theta[read_pos]
                rest_lambda = rest_lambdas[i]
                if 0.0 < A < A_max:
                    pass
                else:
                    return -np.inf
                read_pos += 1

            elif line_class[i] == 'broad_fixed_pos':
                A, sigma = theta[read_pos:read_pos+2]
                rest_lambda = rest_lambdas[i]
                if 0.0 < A < A_max and instrumental_sigma < sigma < 0.03 * rest_lambda:
                    pass
                else:
                    return -np.inf
                read_pos += 2
                
            elif line_class[i] == 'broad':
                mu, A, sigma = theta[read_pos:read_pos+3]
                mu += shift
                rest_lambda = rest_lambdas[i]
                read_pos += 3

                if relax_broad_line_pos == True:
                    max_wavelength_shift = 50.
                else:
                    max_wavelength_shift = 10.
                
                if np.abs(mu - rest_lambda) < max_wavelength_shift and 0.0 < A < A_max and instrumental_sigma < sigma < 0.015 * rest_lambda:
                    pass
                else:
                    return -np.inf      

        return likelihood


    def log_probability(theta, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)

    theta_guess = np.array(theta_guess)
    pos = [theta_guess + 1e-4 * theta_guess * np.random.randn(ndim) for i in range(nwalkers)]
    print 'Initial pos of walker 0:', pos[0]
    print 'Walker 0 / initial_guess:', pos[0]/theta_guess

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
    niter = 5000
    print 'Running MCMC for ', niter, 'iterations'
    sampler.run_mcmc(pos, niter)#, progress=True);
    samples = sampler.chain[:, int(0.9*niter):, :].reshape((-1, ndim)) #clip first 80% of sample chain

    #Corner plot
    varnames = [r'Zero point offset', r'Wavelength shift', 'Continuum slope', r'H$\alpha$ broad $\mu$', r'H$\alpha$ broad $A$', r'H$\alpha$ broad $\sigma$', r'H$\alpha$ narrow $A$', r'NII 6548 $A$', r'NII 6583 $A$']
    fig = corner.corner(samples,range = [0.999]*ndim, labels = varnames, figsize = (60,60))
    plt.savefig(filename + '_corner.png')
    plt.show()
    plt.close(fig)
    

    #Corner plot of useful params
    #useful_params = np.where([('mu' not in ___) & ('narrow sigma' not in ___) for ___ in labels]) #mu since I force the mean to be within 4 angstrom of the center for broad lines, and 2 angstrom for narrow lines. narrow sigma because i force it to be 2.5 to 3.5
    #useful_labels = np.array(labels)[useful_params]
    #ndim_useful = len(useful_labels)
    #useful_samples = sampler.chain[:, int(0.8*niter):, useful_params].reshape((-1, ndim_useful))
    #fig = corner.corner(useful_samples,range = [0.999]*ndim_useful, labels = useful_labels, figsize = (40,40))
    #plt.savefig(filename + '_useful_params_corner.png')
    #plt.close(fig)

    #Parameter evolution plot
    fig, axes = plt.subplots(ndim, figsize=(10, 2*ndim+1), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:,i],'k',alpha = 0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(varnames[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    plt.savefig(filename + '_parameter_evol.png')
    plt.close(fig)

    #Percentiles
    medians = []
    p84th = []
    p16th = []
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print varnames[i], '------', '{:0.4e}'.format(mcmc[1]), '-', '{:0.3e}'.format(q[0],3), '+', '{:0.3e}'.format(q[1],3)
        medians.append(mcmc[1])
        p16th.append(mcmc[0])
        p84th.append(mcmc[2])

    #Median line fluxes and velocities
    line_fluxes_and_velocities = {}
    median_offset = medians[0]
    median_shift = medians[1]
    median_slope = medians[2]
    read_pos = 3
    for i, line in enumerate(linenames):
        if line_class[i] == 'narrow':
            mu = rest_lambdas[i]
            sigma = instrumental_sigma
            median_A = medians[read_pos]
            p16th_A = p16th[read_pos]
            p84th_A = p84th[read_pos]
            
            line_fluxes = np.array([integrated_flux(x+median_shift, mu,median_A,sigma), integrated_flux(x+median_shift,mu,p84th_A,sigma)-integrated_flux(x+median_shift, mu,median_A,sigma), integrated_flux(x+median_shift, mu,median_A,sigma)-integrated_flux(x+median_shift,mu,p16th_A,sigma)])
            line_velocities = np.array([velocity(mu, sigma)])
            line_fluxes_and_velocities[line] = {'Assumed rest lambda:': rest_lambdas[i], 'flux (erg/s/cm2)': line_fluxes, 'Luminosity': 1.196e50 * dist_Mpc**2 * line_fluxes,'velocity': line_velocities}

            read_pos += 1

        elif line_class[i] == 'broad_fixed_pos':
            mu = rest_lambdas[i]
            median_A, median_sigma = medians[read_pos:read_pos+2]
            p84th_A, p84th_sigma = p84th[read_pos:read_pos+2]
            p16th_A, p16th_sigma = p16th[read_pos:read_pos+2]

            line_fluxes = np.array([integrated_flux(x+median_shift, mu,median_A,median_sigma), integrated_flux(x+median_shift,mu,p84th_A,p84th_sigma)-integrated_flux(x+median_shift, mu,median_A,median_sigma), integrated_flux(x+median_shift, mu,median_A,median_sigma)-integrated_flux(x+median_shift,mu,p16th_A,p16th_sigma)])
            line_velocities = np.array([velocity(mu, median_sigma), velocity(mu,p84th_sigma)-velocity(mu, median_sigma), -velocity(mu,p16th_sigma)+velocity(mu, median_sigma)])
            line_fluxes_and_velocities[line] = {'Assumed rest lambda:': rest_lambdas[i], 'flux (erg/s/cm2)': line_fluxes, 'Luminosity': 1.196e50 * dist_Mpc**2 * line_fluxes,'velocity': line_velocities}
            
            read_pos += 2
            
        elif line_class[i] == 'broad':
            median_mu, median_A, median_sigma = medians[read_pos:read_pos+3]
            p84th_mu, p84th_A, p84th_sigma = p84th[read_pos:read_pos+3]
            p16th_mu, p16th_A, p16th_sigma = p16th[read_pos:read_pos+3]
            line_fluxes = np.array([integrated_flux(x+median_shift, median_mu,median_A,median_sigma), integrated_flux(x+median_shift,median_mu,p84th_A,p84th_sigma)-integrated_flux(x+median_shift, median_mu,median_A,median_sigma), integrated_flux(x+median_shift, median_mu,median_A,median_sigma)-integrated_flux(x+median_shift,median_mu,p16th_A,p16th_sigma)])
            line_velocities = np.array([velocity(median_mu, median_sigma), velocity(median_mu,p84th_sigma)-velocity(median_mu, median_sigma), -velocity(median_mu,p16th_sigma)+velocity(median_mu, median_sigma)])
            line_fluxes_and_velocities[line] = {'Assumed rest lambda:': rest_lambdas[i], 'flux (erg/s/cm2)': line_fluxes, 'Luminosity': 1.196e50 * dist_Mpc**2 * line_fluxes,'velocity': line_velocities}
            
            read_pos += 3
        

    ftxt = open(filename+'_fit_parameters.txt','w')
    for key in line_fluxes_and_velocities:
        print
        print
        print key, ':'
        ftxt.write(key+':'+'\n')
        for key2 in line_fluxes_and_velocities[key]:
            print '    ', key2, line_fluxes_and_velocities[key][key2]
            ftxt.write('    '+str(key2)+' '+str(line_fluxes_and_velocities[key][key2]))
            ftxt.write('\n')
    ftxt.close()


    #Make plot of the median fit
    if make_plot == True:
        offset = medians[0]
        shift = medians[1]
        slope = medians[2]
        colors = colors
        fitted_models = []
        read_pos = 3
        for i, line in enumerate(linenames):
            if line_class[i] == 'narrow':
                mu = rest_lambdas[i]
                sigma = instrumental_sigma
                median_A = medians[read_pos]
                p16th_A = p16th[read_pos]    
                p84th_A = p84th[read_pos]
                
                fitted_models.append([(mu, median_A, sigma, colors[i]),(mu, p84th_A, sigma, colors[i]),(mu, p16th_A, sigma, colors[i])])
                read_pos += 1

            elif line_class[i] == 'broad_fixed_pos':
                mu = rest_lambdas[i]
                median_A, median_sigma = medians[read_pos:read_pos+2]
                p84th_A, p84th_sigma = p84th[read_pos:read_pos+2]
                p16th_A, p16th_sigma = p16th[read_pos:read_pos+2]

                fitted_models.append([(mu, median_A, median_sigma, colors[i]),(mu, p84th_A, p84th_sigma, colors[i]),(mu, p16th_A, p16th_sigma, colors[i])])
                read_pos += 2
            
            elif line_class[i] == 'broad':
                median_mu, median_A, median_sigma = medians[read_pos:read_pos+3]
                p84th_mu, p84th_A, p84th_sigma = p84th[read_pos:read_pos+3]
                p16th_mu, p16th_A, p16th_sigma = p16th[read_pos:read_pos+3]
                #fitted_models.append((median_mu, median_A, median_sigma, colors[i]))
                fitted_models.append([(median_mu, median_A, median_sigma, colors[i]),(median_mu, p84th_A, p84th_sigma, colors[i]),(median_mu, p16th_A, p16th_sigma, colors[i])])
                
                read_pos += 3

        
        #Final plot -- special casing some label names
        for i, linename in enumerate(linenames):
            if 'NII' in linename:
                linenames[i] = 'N II 6548, 6583'
        plot(wavelen,flux,flux_err,redshift, models = fitted_models, shift = shift, model_offset = offset, model_slope = slope,rest_lambda_min = xmin,rest_lambda_max = xmax,ymin = ymin, ymax = ymax, filename = filename, save = True, labels = linenames, plot_uncertainties = True, velocity_plot_central_wavelength = velocity_plot_central_wavelength)


    #Compute the BIC of this model
    #n = len(x)
    #k = ndim
    #theta_median = np.median(samples,axis=0)
    #lnL = log_likelihood(theta_median,x,y,yerr)
    #BIC = np.log(n)*k - 2*lnL
    #print 
    #print 'n,k,lnL:', n, k, lnL
    #print 'BIC:', BIC

    return line_fluxes_and_velocities, medians, varnames, sampler



#Fit C0477
#Initial guess plot
#plot(wavelen_C477,flux_C477,flux_unc_C477,z_C477,models = [(6563.,1.2e-16,3.,'green'),(6559,2.6e-16,11.,'blue'),(6549.,2e-17,3.,'purple'),(6583.,2.5e-17,3.,'purple')], model_offset = 0.3e-17,rest_lambda_min = 6490.,rest_lambda_max = 6630.,ymin = 0., ymax = 3.5)

plt.figure(figsize = (7,7))
x,y = estimate_offset_and_slope(wavelen_C477/(1+z_C477),flux_C477,6460,6690,6500,6610) #Appropriate for H alpha
#x,y = estimate_offset_and_slope(wavelen_C477/(1+z_C477),flux_C477,4800,4850,4870,4900) #Appropriate for H beta (narrow)
m,b = np.polyfit(x,y,1)
plt.plot(x,y)
plt.plot(x,m*x + b)
#'''
C0477_linedict = {'H'+r'$\alpha$'+' broad': (6562.8, ['mu','A','sigma'],[6559., 2.6e-16, 11.]), 'H'+r'$\alpha$'+' narrow':(6562.8,['A'],[7.5e-17]), 'NII 6548':(6548.05,['A'],[1e-17]), 'NII 6583':(6583.45, ['A'], [3e-17])}
line_fluxes, medians, varnames, sampler = fit(velocity_plot_central_wavelength = 6562.8,dist_Mpc = 148, xmin = 6490., xmax = 6630., wavelen = wavelen_C477, flux = flux_C477, flux_err = flux_unc_C477, redshift = z_C477, lines = C0477_linedict, shift_guess = 0.1, instrumental_sigma = 2.5, A_max = 1e-15, filename = 'C0477', save = True, make_plot = True, ymin = 0., ymax = 3.5, offset_guess = b, slope_guess = m)
#'''
'''
C0477_linedict = {'H'+r'$\beta$'+' narrow':(4861.35,['A'],[1e-17])}
line_fluxes, medians, varnames, sampler = fit(velocity_plot_central_wavelength = 4861.35,dist_Mpc = 148, xmin = 4840., xmax = 4880., wavelen = wavelen_C477, flux = flux_C477, flux_err = flux_unc_C477, redshift = z_C477, lines = C0477_linedict, shift_guess = 0.1, instrumental_sigma = 2.5, A_max = 1e-15, filename = 'C0477_Hb', save = True, make_plot = True, ymin = 0., ymax = 1.0, offset_guess = b, slope_guess = m)
'''

'''
#fit SII
C0477_linedict = {'SII_6716': (6716, ['A'],[2.6e-16]), 'SII_6731': (6731, ['A'],[2.6e-16])}
line_fluxes, medians, varnames, sampler = fit(velocity_plot_central_wavelength = 6725,dist_Mpc = 148, xmin = 6700., xmax = 6740., wavelen = wavelen_C477, flux = flux_C477, flux_err = flux_unc_C477, redshift = z_C477, lines = C0477_linedict, shift_guess = 0.1, instrumental_sigma = 2.5, A_max = 1e-15, filename = 'C0477_SII', save = True, make_plot = True, ymin = 0., ymax = 3.5, offset_guess = b, slope_guess = m)
A_6716 = sampler.chain[:,:,-2]
A_6731 = sampler.chain[:,:,-1]
r = A_6716/A_6731
p16,p50,p84= np.percentile(r,[16,50,84])
print 'SII ratio = ', p50, '+', p84-p50, '-', p50-p16

def ne(r_SII):
    ne = (627.1 * (r_SII) - (0.4315*2107))/(0.4315-r_SII)
    return ne

p16,p50,p84= np.percentile(ne(r),[16,50,84])
print 'n_e = ', p50, '+', p84-p50, '-', p50-p16
'''


