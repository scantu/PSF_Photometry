
# coding: utf-8

# In[1]:

# get_ipython().magic('matplotlib inline')


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as s
import corner
import emcee
import time
from math import log, exp
# import sys


# Read in the mock data and define variables.  Below that are the true values when the mock data was created and rotated.

# In[3]:

df = pd.read_csv('./gruicen.txt', sep='\s+', header=None)


x = df['ra_r']
y = df['dec_r']

rh_true = .75
rot_true = np.pi / 180. * (90. - 41.)
q_true = .5
gamma_true = 4.


# In[4]:

# def model1(x, y, rh, q, rot, gammma):
def model1(x, y, rh, q):
    # model = (pow((np.pi * q * rh**2) / (-2 + 5), -1) *
    #             pow(1. + ((x * np.cos(rot) + y * np.sin(rot))**2 +
    #                      (-x * np.sin(rot) + y * np.cos(rot))**2 / q / q)
    #    / rh / rh, -5 / 2.))
    model = pow(np.pi * q * rh**2, -1) * \
        pow(1. + (x * x + y * y / q / q) / rh**2, -2.)
    return model


# Setting up the space and prior

# In[5]:

nwalkers = 300
ndim = 4
n = 100
# pos_min = np.array([.001, 0., 0., 2.])  # rh 1 pc - 10kpc
# pos_max = np.array([10., 1., np.pi, 6.])  # q 0-1
pos_min = np.array([.001, 0.])
pos_max = np.array([100., 1.])


psize = pos_max - pos_min
p0 = [pos_max - psize * np.random.rand(ndim) for i in range(nwalkers)]
#p0 = [psize/2 + 1.e-4*np.random.rand(ndim) for i in range(nwalkers)]


def lnprior(theta):
    #    rh, q, rot, gamma = theta
    rh, q = theta
    # if 0.001 < rh < 10. and 0.0 < q < 1.0 and 0. < rot < np.pi and 2. < gamma < 6.:
    if 0.001 < rh < 100. and 0.0 < q < 1.0:
        return 0.0
    return -np.inf


# In[6]:

def lnlike(theta, x, y):
    #    rh, q, rot, gamma = theta
    rh, q = theta
    #rh = np.log10(rh)
    model = model1(x, y, rh, q)
    return np.sum(np.log(model))


# In[7]:

def lnprob(theta, x, y):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y)


# In[8]:


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y),
                                threads=8)
#sampler1 = emcee.EnsembleSampler(nwalkers, ndim,lnprob, args=(x,y), threads=ndim)


# In[9]:


time0 = time.time()
# burnin phase
print('feel the burn')
pos, prob, state = sampler.run_mcmc(p0, n)
#pos1,prob1,state1 = sampler1.run_mcmc(pos,n)

sampler.reset()
# sampler1.reset()
#savarr = open("burnchain.dat",'w')
# savarr.write(sampler.flatchain)
time1 = time.time()
print(time1 - time0)


# In[10]:

time0 = time.time()
# perform MCMC
print('time for MCHAMMER')
pos, prob, state = sampler.run_mcmc(pos, 50)
# for i, result in enumerate((sampler.sample(pos, iterations=n/8))):
#    pos, prob = i, result

#   if (i+1) % 100 == 0:

#      print("{0:5.1%}".format(float(i) / n),end='\r')
time1 = time.time()
print(time1 - time0)


# In[21]:

samples = sampler.flatchain
sampler_chain = sampler.chain
# samples.shape
#emcee_trace = sampler.chain[:, n:, :].reshape(-1, ndim).T
#plot_MCMC_results(x, y, emcee_trace)
#samples = emcee_trace
rh_med = s.median(samples[:, 0])
q_med = s.median(samples[:, 1])
rh_std = s.stdev(samples[:, 0])
q_std = s.stdev(samples[:, 1])
# rot_med = s.median(samples[:, 2])
# rot_std = s.stdev(samples[:, 2])
# gamma_med = s.median(samples[:, 3])
# gamma_std = s.median(samples[:, 3])

# print('halflight=', rh_med, 'q=', q_med, 'pos angle=',
#       rot_med, 'gamma=', gamma_med, rh_std, q_std, rot_std, gamma_std)
print(rh_med, rh_std, q_med, q_std)

# In[12]:
"""
fig = plt.figure()
#ax = plt.add_subplot(111)
for rh, q, rot, gamma in samples[np.random.randint(len(samples), size=100)]:
    plt.scatter(((x * np.cos(rot_true) + y * np.sin(rot_true))**2 +
                 (-x * np.sin(rot_true) + y * np.cos(rot_true))**2
                 / q_true / q_true), model1(x, y, rh, q, rot, gamma),
                color="k", marker='.', edgecolors='none')


plt.scatter(((x * np.cos(rot_true) + y * np.sin(rot_true))**2 +
             (-x * np.sin(rot_true) + y * np.cos(rot_true))**2 / q_true / q_true),
            model1(x, y, rh_true, q_true, rot_true, gamma_true), color="r",
            marker='.', edgecolors='none')
#pl.errorbar(x, y, yerr=yerr, fmt=".k")

plt.yscale('log')
plt.xscale('log')
plt.xlim([10**(-3), 10**1])
plt.ylim([10**(-2), 10**2])

plt.show()


# In[13]:
"""
print("Mean acceptance fraction: {0:.3f}"
      .format(np.mean(sampler.acceptance_fraction)))


# In[14]:

plt.plot(sampler_chain[:, :, 0].T, '-', color='k', alpha=0.3)

# plt.plot(samples1)
plt.show()


# In[15]:

func = emcee.autocorr.function(sampler_chain[:, :, 0].T)
plt.plot(func, color='k', alpha=.3)
plt.show()


# In[16]:

mcmc_time = emcee.autocorr.integrated_time(samples)


# In[17]:

print(mcmc_time)


# In[18]:

plt.plot(mcmc_time, 'ok')


# In[19]:

#fig = corner.corner(samples, bins=100, labels=["$rh$", "$q$", "$rot$", "$gamma$"])
fig = corner.corner(samples, bins=20, labels=["$rh$", "$q$", "$rot$", "$gamma$"], quantiles=[.16, .5, .84],
                    show_titles=True)


# In[20]:

res = plt.plot(sampler.chain[:, :, 0].T, '-', color='k', alpha=0.3)
plt.plot(rh_true, color='blue')


# # In[ ]:

# # Create some convenience routines for plotting

# def compute_sigma_level(trace1, trace2, nbins=20):
# #    From a set of traces, bin by number of standard deviations
#     L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
#     L[L == 0] = 1E-16
#     logL = np.log(L)

#     shape = L.shape
#     L = L.ravel()

#     # obtain the indices to sort and unsort the flattened array
#     i_sort = np.argsort(L)[::-1]
#     i_unsort = np.argsort(i_sort)

#     L_cumsum = L[i_sort].cumsum()
#     L_cumsum /= L_cumsum[-1]

#     xbins = 0.5 * (xbins[1:] + xbins[:-1])
#     ybins = 0.5 * (ybins[1:] + ybins[:-1])

#     return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


# def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
# #    Plot traces and contours
#     xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
#     ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
#     if scatter:
#         ax.plot(trace[0], trace[1], ',k', alpha=0.1)
#     ax.set_xlabel(r'$\alpha$')
#     ax.set_ylabel(r'$\beta$')


# def plot_MCMC_model(ax, xdata, ydata, trace):
# #    Plot the linear model and 2sigma contours
#     ax.plot(xdata, ydata, 'ok')

#     alpha, beta = trace[:2]
#     xfit = np.linspace(-20, 120, 10)
#     yfit = alpha[:, None] + beta[:, None] * xfit
#     mu = yfit.mean(0)
#     sig = 2 * yfit.std(0)

#     ax.plot(xfit, mu, '-k')
#     ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

#     ax.set_xlabel('x')
#     ax.set_ylabel('y')


# def plot_MCMC_results(xdata, ydata, trace, colors='k'):
# #    Plot both the trace and the model together
#     fig, ax = plt.subplots(1, 2, figsize=(10, 4))
#     plot_MCMC_trace(ax[0], xdata, ydata, trace, True, colors=colors)
#     plot_MCMC_model(ax[1], xdata, ydata, trace)


# # In[ ]:

# import scipy.optimize as opt
# nll = lambda *args: -lnlike(*args)
# result = opt.minimize(nll, [rh_true + .1, rot_true + .2, q_true - .2, gamma_true + .5],
#                       args=(x, y))
# print(result['x'])


# # In[ ]:

# p0 = [result['x'] + .1 + 1.e-4 *
#       np.random.randn(ndim) for i in range(nwalkers)]


# # In[ ]:

# sampler1 = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
#                                  args=(x, y), threads=8)
# pos, prob, state = sampler1.run_mcmc(p0, 100)


# # In[ ]:

# res = plt.plot(sampler1.chain[:, :, 0].T, '-', color='k', alpha=0.3)


# # In[27]:

# mtrace = sampler.chain[:, :, :]
# plt.plot(mtrace[:, :, 0].T)

# plt.show()


# # In[28]:

# if mtrace.all() == sampler_chain[:, :, 0].T.all():
#     print('good job')


# # In[29]:

# get_ipython().run_cell_magic('file', 'gelmanRubinTest.py',
#                              'def gelman_rubin(mtrace):\n    Rhat = {}\n    for var in range(len(mtrace[0,0,:])):\n        x = np.array(mtrace[:,:,var].T)\n        num_samples = x.shape[1]\n\n        # Calculate between-chain variance\n        B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)\n\n        # Calculate within-chain variance\n        W = np.mean(np.var(x, axis=1, ddof=1), axis=0)\n\n        # Estimate of marginal posterior variance\n        Vhat = W * (num_samples - 1) / num_samples + B / num_samples\n\n        Rhat[var] = np.sqrt(Vhat / W)\n\n    return Rhat\n    ')


# # In[30]:

# gelman_rubin(mtrace)


# # In[31]:

# np.shape(mtrace)


# # In[32]:

# len(mtrace[0, 0, :])


# # In[ ]:
