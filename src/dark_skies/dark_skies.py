import sys
import csv
import math
import numpy
import os
import random
from multiprocessing import Pool
#import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.integrate import simps

import time

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

def sort_compare(x1, x2):
        if(x1[0] > x2[0]):
            return 1
        elif (x1[0] < x2[0]):
            return -1
        else:
            if x1[1] > x2[1]:
                return 1
            elif x1[1] < x2[1]:
                return -1
        return 0

def sort_key(x1):
    return (x1[0], x1[1])

def sort_predictions(a):
    return a[1]

def sort_likelihood(a):
    return a[1]

def sort_radius(a):
    return a[0]

cols = ['dist:double:84', 'theta:double:72', 'e_tan:double:50']
col1s = ['dist', 'theta', 'e_tan']

dir = "Test_Skies"
file_prefix = "Test"
#dir = "Train_Skies"
#file_prefix = "Training"
total_done = 0
global total_done, x_y_skip, number_random
x_y_skip = 1
total = (4200 / x_y_skip) * (4200 / x_y_skip)
number_random = 2

model_file = ""

def myround(x, base=5):
    return int(base * round(float(x) / base))

def gauss(x, A=1, mu=1, sigma=1):
    """
    Evaluate Gaussian.
    
    Parameters
    ----------
    A : float
        Amplitude.
    mu : float
        Mean.
    std : float
        Standard deviation.

    """
    return numpy.real(A * numpy.exp(-(x - mu)**2 / (2 * sigma**2)))

def twod_gauss(x, y, A, mu=1, sigma=1):
    return numpy.real(A* numpy.exp(-((x - mu)**2 + (y - mu)**2)/ (2* sigma**2)))

#def int_gauss(y, R, fwhm):
#    erf_input = (numpy.sqrt(4*numpy.log(2*R**2) - 4*numpy.log(2*y**2))/fwhm)
#    x_int = (numpy.sqrt(numpy.log(2))*erf(erf_input))/(numpy.sqrt(math.pi)*fwhm*2*(4*y**2/fwhm**2))
#    return x_int


def x0_y0_polynomial_model(r, a1, a2, a3):
    return 1.0/(a3*numpy.power(r, 3)+a2*numpy.power(r, 2)+a1*r+1)

def x0_y0_model(r, n, C, A):
#    return gauss(r, C, 0, A)
#    return ((C/(numpy.power((1 + (r)/A), n))))
#    return numpy.piecewise(r, [r< A, r >= A], [1.0, lambda r: ((1.0/(numpy.power((1 + (r-A)/C), n))))])
#    tmp = numpy.piecewise(r, [r < A, r >= A], [1.0, lambda r: numpy.exp(-1*numpy.power(((r-A)), n))])
#    return tmp
#    const = 1.0/numpy.exp(-1*numpy.power(0, n))
    return C*numpy.exp(-1*numpy.power((r[:])/A, n))
#    return 1.0/((r[:]/C)*(1+r[:]/C)**2)
#    return const * numpy.exp((-1 * (r[:] + C) / n))

def x0_y0_polynomial_likelihood(data, params):
    e_tan_perfect, x, y, theta_perfect, e1, e2 = data
    x0, y0, a1, a2, a3 = params
    rs = []
    thetas = []
    for i in xrange(len(x0)):
        rs.append(numpy.sqrt(numpy.power(x[:] - x0[i], 2) + numpy.power(y[:] - y0[i], 2)))
        thetas.append(numpy.arctan2( (y[:] - y0[i]), (x[:] - x0[i])))
    vals = []
    e_tan_model = []
    for i in xrange(len(rs)):
        val = x0_y0_polynomial_model(rs[i], a1[i], a2[i], a3[i])
        e_tan_model.append(val)
    e1_model = None
    e2_model = None
    for i in xrange(len(rs)):
        theta = thetas[i]
        if e1_model is None:
            e1_model = -1*e_tan_model[i][:] * numpy.cos(2.0 * theta[:])
        else:
            e1_model += -1*e_tan_model[i][:] * numpy.cos(2.0 * theta[:])
        if e2_model is None:
            e2_model = -1*e_tan_model[i][:] * numpy.sin(2.0 * theta[:])
        else:
            e2_model += -1*e_tan_model[i][:] * numpy.sin(2.0 * theta[:])
    diff = numpy.power((e1_model[:] - e1[:]), 2) + numpy.power((e2_model[:] - e2[:]), 2)
#        diff = numpy.power(val[:] - e_tan[:], 2)
    vals.append(numpy.log(numpy.exp(-0.5*(diff))))
#    vals.append(numpy.log(gauss(numpy.power((e1_model[:] - e1[:]), 2), 1, 0.0, 0.2) * gauss(numpy.power((e2_model[:]-e2[:]), 2), 1, 0.0, 0.2)))#numpy.exp(-0.5*(diff))*p_rad))
#    test = -numpy.power((e1_model[:] - e1[:]), 2) / (2. * 0.2**2) * -numpy.power((e2_model[:] - e2[:]), 2) / (2. * 0.2**2)
    log_likelihood = None
    for i in xrange(len(vals)):
        log_likelihood = numpy.sum(vals[i])
    return log_likelihood

def x0_y0_prior(data, params):
    e_tan_perfect, x, y, theta_perfect, e1, e2 = data
    x0, y0, n, C, a = params
    priors = []
#    for i in xrange(len(x0)):
#        norm = (math.pi*erf(4200.0/(numpy.sqrt(2.0)*a[i]))**2)/4.0
#        priors.append(numpy.log(norm))
#        rs = numpy.sqrt(numpy.power(x[:] - x0[i], 2) + numpy.power(y[:] - y0[i], 2))
#        theta = numpy.arctan2( (y[:] - y0[i]), (x[:] - x0[i]))
#        e_tans = -1*(e1[:]*numpy.cos(2*theta[:])+e2[:]*numpy.sin(2*theta[:]))
        # the prior_probs is the P(D|Gaussian distributed e_tans), we want the other thing
        # P(D|NOT Gaussian distributed e_tans), or 1.0 - 
#        filtered_e_tans = []
#        for j, r in enumerate(rs):
#            if r < 500:
#                filtered_e_tans.append(e_tans[j])
#        gausses = gauss(numpy.asarray(filtered_e_tans), 1.0, 0.268259881753,  0.274968778304)
#        total_prior_prob = numpy.sum(numpy.log(gausses))
#        prior_probs = numpy.log(1.0-gauss(e_tans, 1.0, 0.0, 0.22305784))
#        total_prior_prob = numpy.sum(prior_probs)#prior_probs)
#        mean = numpy.mean(e_tans)
#        total_prior_prob = numpy.log(1.0-gauss(mean, 1.0, 0.0, 0.22305784))
#        total_prior_prob = numpy.log(numpy.sum(e_tans)/float(len(e_tans)))
#        priors.append(total_prior_prob)
#    if len(x0) > 1:
#        p_rad = 1.0
#        for i in xrange(len(x0)):
#            for j in xrange(i+1, len(x0)):
#                rad = numpy.sqrt(numpy.power(x0[i] - x0[j], 2) + numpy.power(y0[i] - y0[j], 2))
#                if rad < 500:
#                    p_rad *= 1E-20
#                if len(x0) == 2:
#                    p_rad *= gauss(rad, 1.0, 2100.0, 863.0)
#                else:
#                    p_rad *= gauss(rad, 1.0, 2100.0, 974.0)
#        priors.append(numpy.log(p_rad))
    return priors

def x0_y0_likelihood(data, params):
    e_tan_perfect, x, y, theta_perfect, e1_tmp, e2_tmp = data
    e1 = numpy.copy(e1_tmp)
    e2 = numpy.copy(e2_tmp)
    x0, y0, n, C, a = params
    priors = []
#    priors = x0_y0_prior(data, params)
    rs = []
    thetas = []
    e_tans = []
    for i in xrange(len(x0)):
        rs.append(numpy.sqrt(numpy.power(x[:] - x0[i], 2) + numpy.power(y[:] - y0[i], 2)))
        thetas.append(numpy.arctan2( (y[:] - y0[i]), (x[:] - x0[i])))
    vals = []
    e_tan_model = []
    e1_model = None
    e2_model = None
    max_rs = []
    for i in xrange(len(rs)):
        val = x0_y0_model(rs[i], n[i], C[i], a[i])
        vals.append(val)
#        e_tan_model.append(val)
#        theta = thetas[i]
#        e_tan_measured = -1 * (e1[:] * numpy.cos(2 * theta[:]) + e2 * numpy.sin(2 * theta[:]))
#        e1_model = -1*e_tan_model[i][:] * numpy.cos(2.0 * theta[:])
#        e2_model = -1*e_tan_model[i][:] * numpy.sin(2.0 * theta[:])
#        diff = numpy.power((e1_model[:] - e1[:]), 2) + numpy.power((e2_model[:] - e2[:]), 2)
#        vals.append(-0.5*diff)
#        vals.append(numpy.log(numpy.exp(-0.5*(diff))))
        # subtract off the e's from this model, to fit the next one
#        e1 -= e1_model
#        e2 -= e2_model

#    for i in xrange(len(rs)):
#        theta = thetas[i]
#        e_tan_measured = -1 * (e1[:] * numpy.cos(2 * theta[:]) + e2 * numpy.sin(2 * theta[:]))
#        if e1_model is None:
#            e1_model = -1*e_tan_model[i][:] * numpy.cos(2.0 * theta[:])
#        else:
#            e1_model += -1*e_tan_model[i][:] * numpy.cos(2.0 * theta[:])
#        if e2_model is None:
#            e2_model = -1*e_tan_model[i][:] * numpy.sin(2.0 * theta[:])
#        else:
#            e2_model += -1*e_tan_model[i][:] * numpy.sin(2.0 * theta[:])
##    diff = numpy.power(e_tan_model[i][:]-numpy.abs(e_tan_measured[:]), 2)
#    diff = numpy.power((e1_model[:] - e1[:]), 2) + numpy.power((e2_model[:] - e2[:]), 2)
#    vals.append(numpy.log(numpy.exp(-0.5*(diff))))
    log_likelihood = None
    for i in xrange(len(vals)):
        log_likelihood = numpy.sum(vals[i]) + numpy.sum(priors)
#    log_likelihood = numpy.sum(priors)#+numpy.sum(vals[i])
    return log_likelihood, priors

def x0_y0_likelihood_additive(data, params):
    e_tan_perfect, x, y, theta_perfect, e1, e2 = data
    x0, y0, n, C, a = params
    priors = []
#    priors = x0_y0_prior(data, params)
    rs = []
    thetas = []
    e_tans = []
    for i in xrange(len(x0)):
        rs.append(numpy.sqrt(numpy.power(x[:] - x0[i], 2) + numpy.power(y[:] - y0[i], 2)))
        thetas.append(numpy.arctan2( (y[:] - y0[i]), (x[:] - x0[i])))
    vals = []
    # the model values of e_tan
    e_tan_model = []
    # norms are the integral of the model 
    norms = []
    for i in xrange(len(rs)):
        val = x0_y0_model(rs[i], n[i], C[i], a[i])
        e_tan_model.append(val)
#        r_samps = []
#        for j in xrange(1000):
#            r_samps.append(numpy.sqrt((numpy.random.uniform(0,4200)-x0[i])**2 + (numpy.random.uniform(0, 4200)-y0[i])**2))
#        r_samps = numpy.asarray(r_samps)
#        mod_samps= x0_y0_model(r_samps, n[i], C[i], a[i])
#        integral = numpy.sum(mod_samps)/float(len(r_samps))
##        norm = (math.pi*erf(4200.0/(numpy.sqrt(2.0)*a[i]))**2)/4.0
#        norms.append(integral)

    e1_model = None
    e2_model = None
    e1_tmp = e1[:]
    e2_tmp = e2[:]
    for i in xrange(len(rs)):
        theta = thetas[i]
        e_tan_measured = -1 * (e1[:] * numpy.cos(2 * theta[:]) + e2 * numpy.sin(2 * theta[:]))
        if e1_model is None:
            e1_model = -1*e_tan_model[i][:] * numpy.cos(2.0 * theta[:])
        else:
            e1_model += -1*e_tan_model[i][:] * numpy.cos(2.0 * theta[:])
        if e2_model is None:
            e2_model = -1*e_tan_model[i][:] * numpy.sin(2.0 * theta[:])
        else:
            e2_model += -1*e_tan_model[i][:] * numpy.sin(2.0 * theta[:])
#    diff = numpy.power(e_tan_model[i][:]-numpy.abs(e_tan_measured[:]), 2)
    diff = numpy.power((e1_model[:] - e1[:]), 2) + numpy.power((e2_model[:] - e2[:]), 2)
    vals.append(numpy.log(numpy.exp(-0.5*(diff))))
    log_likelihood = None
    for i in xrange(len(vals)):
        log_likelihood = numpy.sum(vals[i])
#    log_likelihood = numpy.sum(priors)#+numpy.sum(vals[i])
    return log_likelihood, priors


def x0_y0_polynomial_next(params):
    x0, y0, a1, a2, a3 = params
    new_x0s = x0[:] 
    for i in xrange(len(x0)):
        new_x0 = None
        while new_x0 is None or new_x0 > 4200 or new_x0 < 0:
            new_x0 = x0[i] + numpy.random.normal(0, 210) #.uniform (-210, 210)
        new_x0s[i] = new_x0
    
    new_y0s = y0[:]
    for i in xrange(len(y0)):
        new_y0 = None
        while new_y0 is None or new_y0 > 4200 or new_y0 < 0:
            new_y0 = y0[i] + numpy.random.normal(0, 210) #uniform(-210, 210)
        new_y0s[i] = new_y0
 
    new_a1s = a1[:]
    for i in xrange(len(a1)):
        new_a1 = None
        while new_a1 is None or new_a1 < 0 or new_a1 > 100:
            new_a1 = a1[i] + numpy.random.normal(0, 1.0) #uniform(-0.1, 0.1)
        new_a1s[i] = new_a1

    new_a2s = a2[:]
    for i in xrange(len(a2)):
        new_a2 = None
        while new_a2 is None or new_a2 < 0 or new_a2 > 100:
            new_a2 = a2[i] + numpy.random.normal(0, 1.0) #uniform(-0.1, 0.1)
        new_a2s[i] = new_a2

    new_a3s = a3[:]
    for i in xrange(len(a3)):
        new_a3 = None
        while new_a3 is None or new_a3 < 0 or new_a3 > 100:
            new_a3 = a3[i] + numpy.random.normal(0, 1.0) #uniform(-0.1, 0.1)
        new_a3s[i] = new_a3
    return (new_x0s, new_y0s, new_a1s, new_a2s, new_a3s)

def x0_y0_next(params, fixed_states):
    x0, y0, n, C, A = params
    new_x0s = x0[:] 
    for i in xrange(len(x0)):
        if len(fixed_states) != 0 and len(fixed_states) > i:
            new_x0 = fixed_states[i][0]
        else:
            new_x0 = None
        while new_x0 is None or new_x0 > 4200 or new_x0 < 0:
            new_x0 = x0[i] + numpy.random.normal(0, 200) #.uniform (-210, 210)
        new_x0s[i] = new_x0
    
    new_y0s = y0[:]
    for i in xrange(len(y0)):
        if len(fixed_states) != 0 and len(fixed_states) > i:
            new_y0 = fixed_states[i][1]
        else:
            new_y0 = None
        while new_y0 is None or new_y0 > 4200 or new_y0 < 0:
            new_y0 = y0[i] + numpy.random.normal(0, 200) #uniform(-210, 210)
        new_y0s[i] = new_y0
    
    new_ns = n[:]
    for i in xrange(len(n)):
        # greatly constrain models with more parameters
        if len(fixed_states) != 0 and len(fixed_states) > i:
            new_n = fixed_states[i][2]
        else:
            new_n = None
        while new_n is None or new_n < 0.0 or new_n > 0.30 :
            new_n = n[i] + numpy.random.normal(0, 0.01) #uniform(-0.01, 0.01)
        new_ns[i] = new_n
    new_Cs = C[:]
    for i in xrange(len(C)):
        if len(fixed_states) != 0 and len(fixed_states) > i:
            new_C = fixed_states[i][3]
        else:
            new_C = None
        while new_C is None or new_C < 0.01 or new_C > 1.0:
            new_C = C[i] + numpy.random.normal(0, 0.01) #uniform(-0.1, 0.1)
        new_Cs[i] = new_C
    new_As = A[:]
    for i in xrange(len(A)):
        if len(fixed_states) != 0 and len(fixed_states) > i:
            new_A = fixed_states[i][4]
        else:
            new_A = None
        while new_A is None or new_A < 75.0 or new_A > 750.0:
            new_A = A[i] + numpy.random.normal(0, 42.0) #uniform(-0.1, 0.1)
#            new_A = A[i] + numpy.random.normal(0, 1.0) #uniform(-0.1, 0.1)
        new_As[i] = new_A
        
    
    return (new_x0s, new_y0s, new_ns, new_Cs, new_As)

def radial_next(params, fixed_params):
    n, C, r0, A = params
    new_r0 = None
    while new_r0 is None or new_r0 < 0.0:
        new_r0 = r0 + numpy.random.normal(0, 1.0) #uniform
    new_n = None
    while new_n is None or new_n < 0.2 or new_n > 0.3 :
        new_n = n + numpy.random.normal(0, 0.01)
    new_C = None
    while new_C is None or new_C < 0:
        new_C = C + numpy.random.normal(0, 1.0)
    new_A = None
    while new_A is None or new_A < 100.0 or new_A > 5940.0:
        new_A = A + numpy.random.normal(0, 1.0) #uniform(-0.1, 0.1)
    return (new_n, new_C, new_r0, new_A)

def radial_model(r, n, C, r0, A):
    return numpy.piecewise(r, [r< A, r >= A], [0.0, lambda r: ((1.0/(numpy.power((1 + (r-A)/C), n))))])
    tmp = numpy.piecewise(r, [r < A, r >= A], [1.0, lambda r: numpy.exp(-1*numpy.power(((r-A)), n))])
    return tmp
    #return (1.0/(1.0 + numpy.power(r[:], n)))
#    return C*(numpy.exp((-1 * (r[:]) / n)))

def radial_likelihood(data, params):
    e_tan, rs, thetas = data
    n, C, r0, A = params
    val = radial_model(rs, n, C, r0, A)
    diff = numpy.log(numpy.power((e_tan[:] - val[:]), 2))
    log_likelihood = numpy.sum(diff)
    return log_likelihood, 0.0

def fill_ellipticities(num_skys, sky_datas, sky_pos1, sky_pos2, sky_pos3, etans, e1s, e2s):
    for jj in xrange(num_skys-1):
        if jj == 0:
            thetas = numpy.arctan2((sky_datas[:, 1] - sky_pos1[1]), (sky_datas[:, 0] - sky_pos1[0]))
            e_tans = -1*(sky_datas[:, 2]*numpy.cos(2*thetas[:])+sky_datas[:, 3]*numpy.sin(2*thetas[:]))
        elif jj == 1:
            thetas = numpy.arctan2((sky_datas[:, 1] - sky_pos2[1]), (sky_datas[:, 0] - sky_pos2[0]))
            e_tans = -1*(sky_datas[:, 2]*numpy.cos(2*thetas[:])+sky_datas[:, 3]*numpy.sin(2*thetas[:]))
        elif jj == 2:
            thetas = numpy.arctan2((sky_datas[:, 1] - sky_pos3[1]), (sky_datas[:, 0] - sky_pos3[0]))
            e_tans = -1*(sky_datas[:, 2]*numpy.cos(2*thetas[:])+sky_datas[:, 3]*numpy.sin(2*thetas[:]))
        for n,sky_data in enumerate(sky_datas):
            e1s.append(sky_data[2])
            e2s.append(sky_data[3])
            etans.append(e_tans[n])

def get_likelihood(file, range=None, prefix="testing"):
    all_dists = []
    all_thetas = []
    all_e_tans = []
    plottable_dists = []
    plottable_thetas = []
    plottable_e_tans = []
    j = 0
    pool_queue = []
    pool = Pool(processes=8)
    rs = []
    e1s = []
    e2s = []
    etans = []
    max_output= open("{0}.max_likelihoods".format(prefix), 'w')
    expectations_output = open("{0}.expectations".format(prefix), 'w')
    for sky_datas, num_skys, sky_pos1, sky_pos2, sky_pos3, sky  in get_sky_datas(file):
        j += 1
#        if j > 0:
#            break
#        if num_skys < 3 or num_skys > 3:
#            continue
        if j > range[1] or j < range[0]:
            continue
        
        data = (None, sky_datas[:, 0], sky_datas[:, 1], None, sky_datas[:, 2], sky_datas[:, 3])
#        grid_search(data, sky_pos1, sky_pos2, sky_pos3)
#        fill_ellipticities(num_skys, sky_datas, [numpy.random.uniform(0,4200),numpy.random.uniform(0,4200)] , [numpy.random.uniform(0,4200),numpy.random.uniform(0,4200)] , [numpy.random.uniform(0,4200),numpy.random.uniform(0,4200)] , etans, e1s, e2s)
#        fill_ellipticities(num_skys, sky_datas, sky_pos1, sky_pos2, sky_pos3, etans, e1s, e2s)
        params = ([2100], [2100], [0.2], [1.0], [100.0])
        for ii in xrange(num_skys-1):
            if ii == 1:
                params[0].append(2100)
                params[1].append(2100)
            else:
                params[0].append(2100)
                params[1].append(2100)
            params[2].append(0.2)
            params[3].append(1.0)
            params[4].append(100.0)
        true_values = None
        if num_skys == 1 and sky_pos1 is not None:
            true_values = [[sky_pos1[0],sky_pos1[1]]]
        elif num_skys == 2 and sky_pos2 is not None: 
            true_values = [[sky_pos1[0],sky_pos1[1]], [sky_pos2[0], sky_pos2[1]]]
        elif sky_pos3 is not None:
            true_values = [[sky_pos1[0],sky_pos1[1]], [sky_pos2[0], sky_pos2[1]], [sky_pos3[0], sky_pos3[1]]]
        
        # fill in all values for truth positions
#        for ii in xrange(num_skys):
#            if ii == 0:
#                r, theta = get_r_theta(data, sky_pos1)
#                e_tan = -1 * (sky_datas[:, 2] * numpy.cos(2 * theta[:]) + sky_datas[:, 3] * numpy.sin(2 * theta[:]))
#            elif ii == 1:
#                r, theta = get_r_theta(data, sky_pos2)
#                e_tan = -1 * (sky_datas[:, 2] * numpy.cos(2 * theta[:]) + sky_datas[:, 3] * numpy.sin(2 * theta[:]))
#            elif ii == 2:
#                r, theta = get_r_theta(data, sky_pos3)
#                e_tan = -1 * (sky_datas[:, 2] * numpy.cos(2 * theta[:]) + sky_datas[:, 3] * numpy.sin(2 * theta[:]))
#            for jj in xrange(len(r)):
#                all_dists.append(r[jj])
#                all_thetas.append(theta[jj])
#                all_e_tans.append(e_tan[jj])
#                
        # distribution of halos relative to one another, see outside this loop
#        if num_skys == 2:
#            rs.append(numpy.sqrt(numpy.power(sky_pos1[0] - sky_pos2[0], 2)+numpy.power(sky_pos1[1] - sky_pos2[1], 2)))
#        elif num_skys == 3:
#            rs.append(numpy.sqrt(numpy.power(sky_pos1[0] - sky_pos2[0], 2)+numpy.power(sky_pos1[1] - sky_pos2[1], 2))) 
#            rs.append(numpy.sqrt(numpy.power(sky_pos1[0] - sky_pos3[0], 2)+numpy.power(sky_pos1[1] - sky_pos3[1], 2)))
#            rs.append(numpy.sqrt(numpy.power(sky_pos2[0] - sky_pos3[0], 2)+numpy.power(sky_pos2[1] - sky_pos3[1], 2))) 
 
        # 
        e1 = numpy.copy(sky_datas[:, 2]) 
        e2 = numpy.copy(sky_datas[:, 3])
        data = (None, sky_datas[:, 0], sky_datas[:, 1], None, e1, e2)
        if num_skys == 3 and sky_pos1 is not None:
            prior = x0_y0_prior(data, ([sky_pos1[0], sky_pos2[0], sky_pos3[0]], [sky_pos1[1], sky_pos2[1], sky_pos3[1]], [0.2, 0.2, 0.2], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]))
            like = x0_y0_likelihood(data, ([sky_pos1[0], sky_pos2[0], sky_pos3[0]], [sky_pos1[1], sky_pos2[1], sky_pos3[1]], [0.2, 0.2, 0.2], [100.0, 001.0, 100.0], [100.0, 100.0, 100.0]))
            print "Actual likelihood of exact centers: {0}, priors: {1}".format(like, numpy.sum(prior))
        elif num_skys == 2 and sky_pos2 is not None:
            prior = x0_y0_prior(data, ([sky_pos1[0], sky_pos2[0]], [sky_pos1[1], sky_pos2[1]], [0.2, 0.2], [100.0, 100.0], [100.0, 100.0]))
            like = x0_y0_likelihood(data, ([sky_pos1[0], sky_pos2[0]], [sky_pos1[1], sky_pos2[1]], [0.2, 0.2], [100.0, 100.0], [100.0, 100.0]))
            print "Actual likelihood of exact centers: {0}, priors: {1}".format(like, numpy.sum(prior))
        elif sky_pos3 is not None:
            prior = x0_y0_prior(data, ([sky_pos1[0]], [sky_pos1[1]], [0.2], [100.0], [100.0]))
            like = x0_y0_likelihood(data, ([sky_pos1[0]], [sky_pos1[1]], [0.2], [100.0], [100.0]))
            print "Actual likelihood of exact centers: {0}, priors: {1}".format(like, numpy.sum(prior))

        
        nn = 0              # start 4 worker processes
        total_done = 0
        
#        pool_queue.append([sky_datas, num_skys, params, true_values, sky])
#        subtract_e(sky_datas, num_skys, params, true_values, sky)
        
#        e1 = numpy.copy(sky_datas[:, 2]) 
#        e2 = numpy.copy(sky_datas[:, 3])
#        data = (None, sky_datas[:, 0], sky_datas[:, 1], None, e1, e2)
#        print "initial params {0}, truth {1}".format(params, true_values)
#        print "Actual likelihood of exact centers: {0}, priors: {1}".format(like, numpy.sum(prior))
        pool_queue.append([x0_y0_likelihood_additive, x0_y0_next, data, params, int(16E6), True, true_values, "{0}.samples".format(sky)])
#        maxes, states = markov_chain(x0_y0_likelihood_additive, x0_y0_next, data, params, number=int(1E6), save_states=True, true_values=true_values, saved_states_filename="{0}.samples".format(sky))
       
       # try and fix in on starting parameters then run MCMC
#        all_maxes = []
#        e1 = numpy.copy(sky_datas[:, 2]) 
#        e2 = numpy.copy(sky_datas[:, 3])
#        data = (None, sky_datas[:, 0], sky_datas[:, 1], None, e1, e2)        
#        new_params = params[:]
#        for jj in xrange(num_skys):
#            params = ([2100], [2100], [0.2], [100.0], [100.0])
##            for ii in xrange(jj):
##                params[0].append(2100)
##                params[1].append(2100)
##                params[2].append(0.2)
##                params[3].append(1.0)
#            maxes, states = markov_chain(x0_y0_likelihood, x0_y0_next, data, params, number=int(1E5), save_states=True, true_values=true_values)
#            r, theta = get_r_theta(data, [maxes[0][0][0], maxes[0][1][0]])
#            e_tan_t = x0_y0_model(r, maxes[0][2][0], maxes[0][3][0], maxes[0][4][0])
#            e1 -= -1*e_tan_t * numpy.cos(2.0 * theta[:])
#            e2 -= -1*e_tan_t * numpy.sin(2.0 * theta[:])
#            data = (None, sky_datas[:, 0], sky_datas[:, 1], None, e1, e2)
#            all_maxes.append(str(maxes[0][0][0]))
#            all_maxes.append(str(maxes[0][1][0]))
#            new_params[0][jj] =  maxes[0][0][0]
#            new_params[1][jj] = maxes[0][1][0]
#            new_params[2][jj] = maxes[0][2][0]
#            new_params[3][jj] = maxes[0][3][0]
#            new_params[4][jj] = maxes[0][4][0]
#        maxes, states = markov_chain(x0_y0_likelihood_additive, x0_y0_next, data, new_params, number=int(1E6), save_states=True, true_values=true_values, saved_states_filename="{0}.samples".format(sky))
       

#        print "actual positions: {0}".format(true_values)
#        print "maximum likelihood: {0} {1} {2} {3} {4}".format(maxes[0][0], maxes[0][1], maxes[0][2], maxes[0][3], maxes[1])
#        if num_skys == 1:
#            likelihood = x0_y0_likelihood((e_tans, sky_datas[:, 0], sky_datas[:, 1], thetas, sky_datas[:, 2], sky_datas[:, 3]), ([sky_pos1[0]], [sky_pos1[1]], maxes[0][2], maxes[0][3]))
#            print "likelihood of exact max likelihood position: {0} {1}".format(likelihood, (-1*maxes[1])-(-1*likelihood))
#    params = [0.2, 100, 100, 100]
#    data = [numpy.asarray(all_e_tans), numpy.asarray(all_dists), numpy.asarray(all_thetas)]
#    maxes, states = markov_chain(radial_likelihood, radial_next, data, params, number=int(1E5), save_states=True, true_values=None)
#    r_model = numpy.linspace(0, 4200, 420)
#    e_tan_model = radial_model(r_model, maxes[0][0], maxes[0][1], maxes[0][2], maxes[0][3])
#    plt.plot(data[1], data[0], 'bo', r_model, e_tan_model, 'ro')
#    plt.ylabel("{0}".format("e_tan"))
#    plt.xlabel("{0}".format("Radius"))
#    plt.show()

    pool_queue.reverse()
    result = pool.map(markov_chain_proxy, pool_queue, 1)
#    for i, res in enumerate(result):
#        max = res[0]
#        expect = res[1]
#        max_output.write("{0},{1}\n".format("Sky{0}".format(i+1), ",".join(max)))
#        expectations_output.write("{0},{1}\n".format("Sky{0}".format(i+1), ",".join(expect)))
#    max_output.close()
#    expectations_output.close()

def get_r_theta(data, sky_pos):
    r = numpy.sqrt(numpy.power(data[1][:] - sky_pos[0], 2) + numpy.power(data[2][:] - sky_pos[1], 2))
    theta = numpy.arctan2( (data[2][:] - sky_pos[1]), (data[1][:] - sky_pos[0]))
    return r, theta

# This stuff figures out a rough distribution of the halos relative to each other in terms of distance between each other
# seems to be a cutoff gaussian peaking at 2100
#    counts, bins = numpy.histogram(rs, bins=10)
#    print numpy.std(rs)
#    center = (bins[:-1]+bins[1:])/2
#    import matplotlib.pyplot as plt
#    plt.plot(center, counts, 'bo')
#    plt.xlabel("{0}".format("Radius"))
#    plt.ylabel("{0}".format("counts"))
#    plt.show()
#    pool_queue.reverse()
#    result = pool.map(markov_chain_proxy, pool_queue, 1)
#    max_output.close()

#        import matplotlib.pyplot as plt
#        from scipy.interpolate import griddata
#        xi = numpy.linspace(0,4200,4200)
#        yi = numpy.linspace(0,4200,4200)
#        states_t = random.sample(states, 100000)
#        x = states_t[0][:][0]
#        y = states_t[:][1]
#        z = states_t[:][4]
#        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
#        plt.figure()
#        CS = plt.contour(xi, yi, zi,15,linewidths=0.5,colors='k')
#        plt.clabel(CS, inline=1, fontsize=10)
#        plt.show()

def grid_search(data, sky_pos1, sky_pos2, sky_pos3):
    max_like = None
    max_likelihood = None
    states = []
    C = 40.0
    A = 400.0
    n = 0.2
    x = []
    y = [] 
    z = []
    for grid_x in xrange(0, 4200, 50):
        for grid_y in xrange(0, 4200, 50):
            likelihood, priors = x0_y0_likelihood_additive(data, ([grid_x], [grid_y], [n], [C], [A]))
            if max_likelihood is None or max_likelihood < likelihood :
                max_like = ([[grid_x], [grid_y], [n], [C], [A]], likelihood)
                max_likelihood = likelihood
                print "Max like value: {0}".format(max_like)
            states.append((([grid_x], [grid_y], [n], [C], [A]), likelihood))
            x.append(grid_x)
            y.append(grid_y)
            z.append(likelihood)
        print "working on {0}".format(grid_x)
    print max_like, sky_pos1, sky_pos2, sky_pos3
    sorted_states = sorted(states, key=sort_likelihood, reverse=True)
    print sorted_states[0:10]
#    return max_like, states
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    plt.figure()
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    z = numpy.asarray(z)
    zi = griddata((x, y), z, (x[None, :], y[:, None]), method='cubic')
    CS = plt.contour(x, y, zi, colors='k')
    plt.plot(sky_pos1[0], sky_pos1[1], 'bo')
    plt.plot(sky_pos2[0], sky_pos2[1], 'ro')
    plt.plot(sky_pos3[0], sky_pos3[1], 'go')
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()

def process_samples(output_prefix):
    expectations_output = open("{0}.expectations".format(output_prefix), 'w')
    max_output= open("{0}.max_likelihoods".format(output_prefix), 'w')
    pool_queue = []
    pool = Pool(processes=8)
    for i in xrange(0, 130):
        pool_queue.append([i])
    result = pool.map(process_sample_file_proxy, pool_queue, 1)
    for sky_pos_max,sky_pos_expect in result: 
        expectations_output.write(",".join(sky_pos_expect)+"\n")
        max_output.write(",".join(sky_pos_max)+"\n")
        expectations_output.flush()
        max_output.flush()
    max_output.close()
    expectations_output.close()


def process_sample_file_proxy(args):
    i = args[0]
    return process_sample_file(i)

def process_sample_file(i):
    try:
        file = open("Sky{0}.samples".format(i), 'r')
    except Exception as e:
        pass
#    print "working on {0}".format(i)
    sum = 0.0
    states = []
    maxes = None
    max = None
    num_skys = None
    for line in file:
        data = line.split(",")
        if len(data) < 4:
            continue
#        for jj, datum in enumerate(data):
#            data[jj] = float(datum)
        data[len(data)-1] = float(data[len(data)-1])
#        print len(data), i
        if len(data) == 6:
            num_skys = 1
        elif len(data) == 11:
            num_skys = 2
        elif len(data) == 16:
            num_skys = 3
        prob = data[len(data)-1] #(numpy.exp(state[len(state)-1])/sum)
        if max is None or prob > max:
            max = prob
            maxes = data
#        states.append(data)
#    states = numpy.asarray(states)
    sum = 0.0
#    sum = numpy.sum(numpy.exp(state[:][len(states[0])-1]))
#    for state in states:
#        sum += numpy.exp(state[len(state)-1])
#    print len(states[0])
    
    
#    print "working on: {0}, num_skys{1}".format("Sky{0}.samples".format(i), num_skys)
    x_0_sum = []
    y_0_sum = []
    n_sum = []
    C_sum = []
    for ii in xrange(num_skys):
        x_0_sum.append(0.0)
        y_0_sum.append(0.0)
        n_sum.append(0.0)
        C_sum.append(0.0)
    
#    maxes = None
#    max = None
#    for state in states:
#        prob = state[len(state)-1] #(numpy.exp(state[len(state)-1])/sum)
#        if max is None or prob > max:
#            max = prob
#            maxes = state
#        jj = 0
#        for ii in xrange(num_skys):
#            x_0_sum[ii] += state[jj]*prob
#            y_0_sum[ii] += state[jj+1]*prob
#            n_sum[ii] += state[jj+2]*prob
#            C_sum[ii] += state[jj+3]*prob
#            jj += 3
#    print maxes
    if num_skys == 1:
        sky_pos_max = ["Sky{0}".format(i), str(maxes[0]), str(maxes[1]), str(0.0), str(0.0), str(0.0), str(0.0)]
        sky_pos_expect = ["Sky{0}".format(i), str(x_0_sum[0]), str(y_0_sum[0]), str(0.0), str(0.0), str(0.0), str(0.0)]
    elif num_skys == 2:
        sky_pos_max = ["Sky{0}".format(i), str(maxes[0]), str(maxes[1]), str(maxes[5]), str(maxes[6]), str(0.0), str(0.0)]
        sky_pos_expect = ["Sky{0}".format(i), str(x_0_sum[0]), str(y_0_sum[0]), str(x_0_sum[1]), str(y_0_sum[1]), str(0.0), str(0.0)]
    elif num_skys == 3:
        sky_pos_max = ["Sky{0}".format(i), str(maxes[0]), str(maxes[1]), str(maxes[5]), str(maxes[6]), str(maxes[10]), str(maxes[11])]
        sky_pos_expect = ["Sky{0}".format(i), str(x_0_sum[0]), str(y_0_sum[0]), str(x_0_sum[1]), str(y_0_sum[1]), str(x_0_sum[2]), str(y_0_sum[2])]
    print ",".join(sky_pos_max)
#    print sky_pos_expect
    return sky_pos_max, sky_pos_expect

def get_radial_likelihood():
    for sky_datas, num_skys, sky_pos1, sky_pos2, sky_pos3, sky  in get_sky_datas(file):
        if num_skys > 1:
            continue
        xs = []
        ys = []
        if (sky_pos1[0] != 0 and sky_pos1[1] != 0):
            clazz = 1
            xs.append(sky_pos1[0])
            ys.append(sky_pos1[1])
        i = xs[0]
        j = ys[0]
        dists = numpy.sqrt((sky_datas[:, 0]-i)*(sky_datas[:, 0]-i) + (sky_datas[:, 1]-j)*(sky_datas[:, 1]-j))
        thetas = numpy.arctan2((sky_datas[:, 1] - j), (sky_datas[:, 0] - i))
        e_tans = -1*(sky_datas[:, 2]*numpy.cos(2*thetas[:])+sky_datas[:, 3]*numpy.sin(2*thetas[:]))
        for i in xrange(0, len(dists)):
            if dists[i] < 10000000:
                all_dists.append(dists[i])
                all_thetas.append(thetas[i])
                all_e_tans.append(e_tans[i])
            plottable_dists.append(dists[i])
            plottable_e_tans.append(e_tans[i])
            plottable_thetas.append(thetas[i])
    data = (numpy.asarray(all_e_tans), [numpy.asarray(all_dists)])
    params = (300, [1.0], 500.0)
    maxes, states = markov_chain(radial_likelihood, radial_next, data, params, number=int(1E4), saved_states_filename="{0}.samples".format(sky), save_states=True)
    all_dists_c = numpy.linspace(0, 4200, 4200)
    predicted_etans = radial_model(numpy.asarray(all_dists_c), maxes[0][0], maxes[0][1][0], maxes[0][2])
    import matplotlib.pyplot as plt
    plt.plot(plottable_dists, plottable_e_tans, 'bo', numpy.asarray(all_dists_c), predicted_etans, 'r')
    plt.xlabel("{0}".format("Radius"))
    plt.ylabel("{0}".format("E_tan"))
    plt.show()

def subtract_e_proxy(args):
    sky_datas = args[0]
    num_skys = args[1]
    params = args[2]
    true_values = args[3]
    sky = args[4]
    return subtract_e(sky_datas, num_skys, params, true_values, sky)

def plot_e_tan(jj, data, true_values, e1, e2, maxes, i):
    e1_tmp = numpy.copy(e1)
    e2_tmp = numpy.copy(e2)
    r_model = numpy.linspace(0, 4200, 420)
    r_t, theta_t = get_r_theta(data, true_values)
    e_tans_exact = -1 * (e1 * numpy.cos(2 * theta_t[:]) + e2 * numpy.sin(2 * theta_t[:]))
    if jj == i:
        e_tan = x0_y0_model(r_t, maxes[0][2][0], maxes[0][3][0], maxes[0][4][0])
        e1_tmp -=  -1*e_tan * numpy.cos(2.0 * theta_t[:])
        e2_tmp -=  -1*e_tan * numpy.sin(2.0 * theta_t[:])
        e_tans = -1 * (e1_tmp * numpy.cos(2 * theta_t[:]) + e2_tmp * numpy.sin(2 * theta_t[:]))
        e_tan_model = x0_y0_model(r_model, maxes[0][2][0], maxes[0][3][0], maxes[0][4][0])
        plt.plot(r_t, e_tans_exact, 'bo', r_model, e_tan_model, 'ro', r_t, e_tans, 'go')
        plt.title("First halo")
        plt.ylabel("{0}".format("e_tan"))
        plt.xlabel("{0}".format("Radius"))
        plt.show()
    if jj == i:
        e_tan = x0_y0_model(r_t, maxes[0][2][0], maxes[0][3][0], maxes[0][4][0])
        e1_tmp -=  -1*e_tan * numpy.cos(2.0 * theta_t[:])
        e2_tmp -=  -1*e_tan * numpy.sin(2.0 * theta_t[:])
        e_tans = -1 * (e1_tmp * numpy.cos(2 * theta_t[:]) + e2_tmp * numpy.sin(2 * theta_t[:]))
        e_tan_model = x0_y0_model(r_model, maxes[0][2][0], maxes[0][3][0], maxes[0][4][0])
        plt.plot(r_t, e_tans_exact, 'bo', r_model, e_tan_model, 'ro', r_t, e_tans, 'go')
        plt.title("Second halo")
        plt.ylabel("{0}".format("e_tan"))
        plt.xlabel("{0}".format("Radius"))
        plt.show()
    if jj == i:
        e_tan = x0_y0_model(r_t, maxes[0][2][0], maxes[0][3][0], maxes[0][4][0])
        e1_tmp -=  -1*e_tan * numpy.cos(2.0 * theta_t[:])
        e2_tmp -=  -1*e_tan * numpy.sin(2.0 * theta_t[:])
        e_tans = -1 * (e1_tmp * numpy.cos(2 * theta_t[:]) + e2_tmp * numpy.sin(2 * theta_t[:]))
        e_tan_model = x0_y0_model(r_model, maxes[0][2][0], maxes[0][3][0], maxes[0][4][0])
        plt.plot(r_t, e_tans_exact, 'bo', r_model, e_tan_model, 'ro', r_t, e_tans, 'go')
        plt.title("Third halo")
        plt.ylabel("{0}".format("e_tan"))
        plt.xlabel("{0}".format("Radius"))
        plt.show()


def subtract_e(sky_datas, num_skys, params, true_values, sky):
    e1 = numpy.copy(sky_datas[:, 2]) 
    e2 = numpy.copy(sky_datas[:, 3])
    data = (None, sky_datas[:, 0], sky_datas[:, 1], None, e1, e2)
    all_maxes = []
    all_expectations = []
    for jj in xrange(num_skys):
        params_tmp = ([2100], [2100], [0.2], [100.0], [100.0])
#        maxes, states = markov_chain(x0_y0_likelihood_additive, x0_y0_next, data, params_tmp, number=int(5E5), save_states=True, true_values=true_values)
        maxes, states = grid_search(data, true_values[0], true_values[1], true_values[2])
        r, theta = get_r_theta(data, [maxes[0][0][0], maxes[0][1][0]])
        plot_e_tan(jj, data, true_values[0], e1, e2, maxes, 0)
        plot_e_tan(jj, data, true_values[1], e1, e2, maxes, 1)
        plot_e_tan(jj, data, true_values[2], e1, e2, maxes, 2)
        e_tan = x0_y0_model(r, maxes[0][2][0], maxes[0][3][0], maxes[0][4][0])
        e1 -= -1*e_tan * numpy.cos(2.0 * theta[:])
        e2 -= -1*e_tan * numpy.sin(2.0 * theta[:])
        data = (None, sky_datas[:, 0], sky_datas[:, 1], None, e1, e2)
        all_maxes.append(str(maxes[0][0][0]))
        all_maxes.append(str(maxes[0][1][0]))
        sum = 0.0
        for state in states:
            sum += numpy.exp(state[len(state)-1])
        x_0_sum = 0.0
        y_0_sum = 0.0
        n_sum = 0.0
        C_sum = 0.0
        for state in states:
            prob = (numpy.exp(state[len(state)-1])/sum)
            x_0_sum += state[0][0][0]*prob
            y_0_sum += state[0][1][0]*prob
            n_sum += state[0][2][0]*prob
            C_sum += state[0][3][0]*prob
        all_expectations.append(str(x_0_sum))
        all_expectations.append(str(y_0_sum))
    if len(all_maxes)  == 2:
        all_maxes.append(0.0)        
        all_maxes.append(0.0)
        all_maxes.append(0.0)        
        all_maxes.append(0.0)
    if len(all_maxes) == 4:
        all_maxes.append(0.0)        
        all_maxes.append(0.0)
    if len(all_expectations)  == 2:
        all_expectations.append(0.0)        
        all_expectations.append(0.0)
        all_expectations.append(0.0)        
        all_expectations.append(0.0)
    if len(all_maxes) == 4:
        all_expectations.append(0.0)        
        all_expectations.append(0.0)
    return all_maxes, all_expectations


#        likelihood = x0_y0_likelihood(e_tans, sky_datas[:, 0], sky_datas[:, 1], [sky_pos1[0]], [sky_pos1[1]], n_sum, [C_sum])
#        print "likelihood of exact expectation position: {0}".format(likelihood)
#        print "done"

def markov_chain_proxy(args):
    model = args[0]
    next = args[1]
    data = args[2]
    params = args[3]
    number = args[4]
    save_states = args[5]
    true_values = args[6]
    saved_states_filename = args[7]
    maxes, states = markov_chain(model, next, data, params, number, save_states, true_values, saved_states_filename)
    return maxes, states
    
def markov_chain(model, next, data, params, number=10000, save_states=False, true_values=[], saved_states_filename=None, fixed_states=[]):
    saved_states = []
    maxes = ()
    saved_states_file = None
    if saved_states_filename is not None:
        saved_states_file = open(saved_states_filename, 'w+')
    previous_likelihood, priors = model(data, params)
    i = 0
    while i < number:
        new_params = next(params, fixed_states)
        likelihood, priors = model(data, new_params)
        a = likelihood/previous_likelihood
        if a < 1.0:
            params = new_params
            previous_likelihood = likelihood
            if save_states:
                if saved_states_file is not None:
                    params_str = []
                    for jj in xrange(len(params[0])):
                        for param in params:
                            params_str.append(str(param[jj]))
                    params_str.append(str(likelihood))
                    saved_states_file.write(",".join(params_str) + "\n")
                else:
                    saved_states.append((params, likelihood))
            if i == 0 or  likelihood > maxes[1]: 
                maxes = (params, likelihood)
            prob = numpy.exp(likelihood)
            if i % (number / 1000) == 0:
                diff = []
                matches = []
                if true_values is not None:
                    for jj in xrange(len(params[0])):
                        min_diff = None
                        match = None
                        for ii in xrange(len(params[0])):
                            if ii in matches:
                                continue
                            diff_t = numpy.sqrt(numpy.power((maxes[0][0][jj] - true_values[ii][0]), 2) + numpy.power((maxes[0][1][jj] - true_values[ii][1]), 2))
                            if min_diff is None or diff_t < min_diff:
                                min_diff = diff_t
                                match = ii
                        matches.append(match)
                        diff.append(min_diff)
                print "Finished: {0}, Diff: {2} True Values: {1}, Priors {3} ".format(i, true_values, diff, numpy.sum(priors))
                print  "Maximum Likelihood values: {0} ".format(maxes)
            i += 1
        else:
            if a > numpy.random.random():
                params = new_params
                previous_likelihood = likelihood
                if save_states:
                    if saved_states_file is not None:
                        params_str = []
                        for jj in xrange(len(params[0])):
                            for param in params:
                                params_str.append(str(param[jj]))
                        params_str.append(str(likelihood))
                        saved_states_file.write(",".join(params_str) + "\n")
                    else:
                        saved_states.append((params, likelihood))
                if i == 0 or  likelihood > maxes[1]: 
                    maxes = (params, likelihood)
                if i % (number / 1000) == 0:
                    diff = []
                    matches = []
                    if true_values is not None:
                        for jj in xrange(len(params[0])):
                            min_diff = None
                            match = None
                            for ii in xrange(len(params[0])):
                                if ii in matches:
                                    continue
                                diff_t = numpy.sqrt(numpy.power((maxes[0][0][jj] - true_values[ii][0]), 2) + numpy.power((maxes[0][1][jj] - true_values[ii][1]), 2))
                                if min_diff is None or diff_t < min_diff:
                                    min_diff = diff_t
                                    match = ii
                            matches.append(match)
                            diff.append(min_diff)
                    print "Finished: {0}, Diff: {2} True Values: {1}, Priors {3} ".format(i, true_values, diff, numpy.sum(priors))
                    print  "Maximum Likelihood values: {0} ".format(maxes)

                i += 1
            else:
                pass
    if saved_states_file is not None:
        saved_states_file.close()
    return maxes, saved_states
            
        
def get_sky_datas(file):
    with open(file) as _f:
        for k, line in enumerate(_f):
                data = line.strip().split(",")
                if k == 0:
                    continue
                sky = data[0]
                num_skys = int(data[1])
                sky_pos1 = None
                sky_pos2 = None
                sky_pos3 = None
                sky_file = "/Users/jostheim/workspace/SLA/data/DarkWorlds/{0}/{1}_{2}.csv".format(dir, file_prefix, sky)
                if len(data) > 2:
                    data[4] = float(data[4])
                    data[5] = float(data[5])
                    data[6] = float(data[6])
                    data[7] = float(data[7])
                    data[8] = float(data[8])
                    data[9] = float(data[9])
    
                    sky_pos1 = data[4:6]
                    sky_pos2 = data[6:8]
                    sky_pos3 = data[8:10]
                sky_datas = []
                print "working on sky file {0}".format(sky_file)
                with open(sky_file) as _sf:
                    for j, line in enumerate(_sf):
                        if j == 0:
                            continue
                        sky_data = line.strip().split(",")
                        sky_data[1] = float(sky_data[1])
                        sky_data[2] = float(sky_data[2])
                        sky_data[3] = float(sky_data[3])
                        sky_data[4] = float(sky_data[4])
                        sky_datas.append(sky_data[1:])
                    sky_datas = numpy.asarray(sky_datas)
                    _sf.close()
                yield sky_datas, num_skys, sky_pos1, sky_pos2, sky_pos3, sky

def prob_inter_halo():
    pass
    #    if len(x0) > 1:
#        for i in xrange(len(x0)):
#            for j in xrange(i+1, len(x0)):
#                rad = numpy.sqrt(numpy.power(x0[i] - x0[j], 2) + numpy.power(y0[i] - y0[j], 2))
#                if len(x0) == 2:
#                    p_rad *= gauss(rad, 1.0, 2100.0, 863.0)
#                else:
#                    p_rad *= gauss(rad, 1.0, 2100.0, 974.0)


def plot_data(file):
    import matplotlib.pyplot as plt
    all_dists = []
    all_thetas = []
    all_e_tans = []
    for sky_datas, num_skys, sky_pos1, sky_pos2, sky_pos3, sky  in get_sky_datas(file):
        if num_skys > 2:
            continue
        xs = []
        ys = []
        if (sky_pos1[0] != 0 and sky_pos1[1] != 0):
            clazz = 1
            xs.append(sky_pos1[0])
            ys.append(sky_pos1[1])
        i = xs[0]
        j = ys[0]
        dists = numpy.sqrt((sky_datas[:, 0] - i) * (sky_datas[:, 0] - i) + (sky_datas[:, 1] - j) * (sky_datas[:, 1] - j))
        thetas = numpy.arctan2((sky_datas[:, 1] - j), (sky_datas[:, 0] - i))
        e_tans = -1 * (sky_datas[:, 2] * numpy.cos(2 * thetas[:]) + sky_datas[:, 3] * numpy.sin(2 * thetas[:]))
        for i in xrange(0, len(dists)):
            all_dists.append(dists[i])
            all_thetas.append(thetas[i])
            all_e_tans.append(e_tans[i])
    
    r = numpy.linspace(0, 4200)
    r = r + 200
    etans = radial_model(r, 700, numpy.exp(200. / 700.), 100.0)
    plt.plot(all_dists, all_e_tans, 'bo', r, etans, 'r')
    plt.xlabel("{0}".format("Radius"))
    plt.ylabel("{0}".format("E_tan"))
    plt.show()

def convert_and_filter(file, output_file, svm=False, test=False):
    all_data = []
    num_positive = 0
    num_negative = 0
    
    with open(file) as _f:
        with open(output_file + ".{0}".format(number_random) + ".positive.tab", 'w') as _of_positive:
                _of_1 = open(output_file + ".{0}".format(number_random) + ".{0}.tab".format("1"), 'w')
                _of_2 = open(output_file + ".{0}".format(number_random) + ".{0}.tab".format("2"), 'w')
                _of_3 = open(output_file + ".{0}".format(number_random) + ".{0}.tab".format("3"), 'w')
                header = []
                header.append("class:integer")
                for i in xrange(1, 2):
                    for col in cols:
                        header.append("{0}_{1}".format(i, col))
                if not svm:
                    _of_1.write("%Sparse\tdefault=-1\n")
                    _of_1.write("%" + "\t".join(header) + "\n")
                header = []
                header.append("class:integer")
                for i in xrange(1, 3):
                    for col in cols:
                        header.append("{0}_{1}".format(i, col))
                if not svm:
                    _of_2.write("%Sparse\tdefault=-1\n")
                    _of_2.write("%" + "\t".join(header) + "\n")
                header = []
                header.append("class:integer")
                for i in xrange(1, 4):
                    for col in cols:
                        header.append("{0}_{1}".format(i, col))
                if not svm:
                    _of_3.write("%Sparse\tdefault=-1\n")
                    _of_3.write("%" + "\t".join(header) + "\n")
                
                for k, line in enumerate(_f):
                    data = line.strip().split(",")
                    if k == 0:
                        continue
                    sky = data[0]
                    num_skys = int(data[1])
                    sky_file = "/Users/jostheim/workspace/SLA/data/DarkWorlds/{0}/{1}_{2}.csv".format(dir, file_prefix, sky)
                    data[4] = float(data[4])
                    data[5] = float(data[5])
                    data[6] = float(data[6])
                    data[7] = float(data[7])
                    data[8] = float(data[8])
                    data[9] = float(data[9])
    
                    sky_pos1 = data[4:6]
                    sky_pos2 = data[6:8]
                    sky_pos3 = data[8:10]
                    sky_datas = []
                    print "working on sky file {0}".format(sky_file)
                    with open(sky_file) as _sf:
                        for j, line in enumerate(_sf):
                            if j == 0:
                                continue
                            sky_data = line.strip().split(",")
                            sky_data[1] = float(sky_data[1])
                            sky_data[2] = float(sky_data[2])
                            sky_data[3] = float(sky_data[3])
                            sky_data[4] = float(sky_data[4])
                            sky_datas.append(sky_data[1:])
                        sky_datas = numpy.asarray(sky_datas)
                        _sf.close()
                    if not test:
                        xs = []
                        ys = []
                        if (sky_pos1[0] != 0 and sky_pos1[1] != 0):
                            clazz = 1
                            num_positive += 1
                            num_random = 1
                            xs.append(sky_pos1[0])
                            ys.append(sky_pos1[1])
                        if (sky_pos2[0] != 0 and sky_pos2[1] != 0):
                            clazz = 1
                            num_positive += 1
                            num_random = 2
                            xs.append(sky_pos2[0])
                            ys.append(sky_pos2[1])
                        if (sky_pos3[0] != 0 and sky_pos3[1] != 0):
                            clazz = 1
                            num_positive += 1
                            num_random = 3
                            xs.append(sky_pos3[0])
                            ys.append(sky_pos3[1])

                        out = write_line(sky_datas, xs, ys, clazz, 1, svm)
                        if num_skys == 1:
                            _of_1.write(out)
                        if num_skys == 2:
                            _of_2.write(out)
                        if num_skys == 3:
                            _of_3.write(out)
                        _of_positive.write(out)
                        
                        jj = 1
                        random_xs = []
                        random_ys = []
                        while jj < num_random + 1:
                            # half random, half around the halo center to sample close to the margin
                            if random.random() > 0.5:
                                random_x = random.uniform(0, 4200)
                                random_y = random.uniform(0, 4200)
                            else:
                                random_x = sky_pos1[0] + random.uniform(-50, 50)
                                random_y = sky_pos1[1] + random.uniform(-50, 50)
                                if sky_pos2[0] != 0 and sky_pos2[1] != 0:
                                    random_x = sky_pos2[0] + random.uniform(-50, 50)
                                    random_y = sky_pos2[1] + random.uniform(-50, 50)
                                if sky_pos3[0] != 0 and sky_pos3[1] != 0:
                                    random_x = sky_pos3[0] + random.uniform(-50, 50)
                                    random_y = sky_pos3[1] + random.uniform(-50, 50)
                            if ((sky_pos1[0] == 0 and sky_pos1[1] == 0)  or (random_x != sky_pos1[0] and random_y != sky_pos1[1])) \
                            and ((sky_pos2[0] == 0 and sky_pos2[1] == 0)  or (random_x != sky_pos2[0] and random_y != sky_pos2[1])) \
                            and ((sky_pos3[0] == 0 and sky_pos3[1] == 0)  or (random_x != sky_pos3[0] and random_y != sky_pos3[1])):
                                random_xs.append(random_x)
                                random_ys.append(random_y)
                                num_negative += 1
                                jj += 1
                        out = write_line(sky_datas, random_xs, random_ys, -1, jj, svm)
                        if num_skys == 1:
                            _of_1.write(out)
                        if num_skys == 2:
                            _of_2.write(out)
                        if num_skys == 3:
                            _of_3.write(out)
    output_prefix = output_file + ".{0}".format(number_random) + ".{0}".format("1")
    if svm:
        os.popen("/usr/local/bin/svm-scale -s {0} {1} > {2}".format(output_prefix + ".scaled", output_prefix + ".tab", output_prefix + ".scaled.tab"))
        os.popen("/usr/local/bin/svm-train -b 1 -h 0 -s 0 -t 2 -v 10 {0} > {1} ".format(output_prefix + ".scaled.tab", output_prefix + ".scaled.tab.learn.log"))
        os.popen("/usr/local/bin/svm-train -b 1 -h 0 -s 0 -t 2 {0} {1} >> {2} ".format(output_prefix + ".scaled.tab", output_prefix + ".x0_y0_likelihood", output_prefix + ".scaled.tab.learn.log"))
        os.popen("/usr/local/bin/svm-scale -r {0} {1} > {2}".format(output_prefix + ".scaled", output_prefix + ".positive.tab", output_prefix + ".positive.scaled.tab"))
    print "Number positive {0}, Number negative {1}, Ratio {2}".format(num_positive, num_negative, float(num_negative) / float(num_positive))

def produce_test_files(file):
    svm = True
    sky_data_files = []
    with open(file) as _f:
        for k, line in enumerate(_f):
            data = line.strip().split(",")
            if k == 0:
                continue
            sky = data[0]
            sky_data_files.append(sky)
    for sky in sky_data_files:
        sky_file = "/Users/jostheim/workspace/SLA/data/DarkWorlds/{0}/{1}_{2}.csv".format(dir, file_prefix, sky)
        print "working on sky file {0}".format(sky_file)
        sky_datas = []
        with open(sky_file) as _sf:
            for j, line in enumerate(_sf):
                if j == 0:
                    continue
                sky_data = line.strip().split(",")
                sky_data[1] = float(sky_data[1])
                sky_data[2] = float(sky_data[2])
                sky_data[3] = float(sky_data[3])
                sky_data[4] = float(sky_data[4])
                sky_datas.append(sky_data[1:])
            _sf.close()
            sky_datas = numpy.asarray(sky_datas)
        # write out unscaled data
        tmp_of = open("/Users/jostheim/workspace/SLA/data/DarkWorlds/Train_Skies/{0}.tab".format(sky.replace(".csv", "")), 'w')
        out = ""
        pool = Pool(processes=8)
        nn = 0              # start 4 worker processes
        pool_queue = []
        total_done = 0
#        with Timer() as t:
        for i in xrange(0, 4200, x_y_skip):
                for j in xrange(0, 4200, x_y_skip):
                    pool_queue.append([sky_datas, i, j, -1, svm, test])
                    if len(pool_queue) > 1E6:
                        #write_line(sky_datas, i, j, -1, svm, test)
                #        print "Took {0} to build the sky_data".format(t.interval)
                        print "starting: {0}".format(len(pool_queue))
                        result = pool.map(write_line_proxy, pool_queue, 1000)
                        print len(result)
                        for out in result:
                            tmp_of.write(out)
                        result = None
                        pool_queue = []
        print "starting: {0}".format(len(pool_queue))
        result = pool.map(write_line_proxy, pool_queue, 1000)
        print len(result)
        for out in result:
            tmp_of.write(out)
        pool.terminate()
        pool_queue = None
        result = None
        tmp_of.close()
                
def test(sky_file, output_file, sky_to_analyze=None):
    print "testing {0}".format(sky_to_analyze)
    halos = []
    with open(sky_file) as _f:
        for k, line in enumerate(_f):
            data = line.strip().split(",")
            if k == 0:
                continue
            halos.append(data)
    if sky_to_analyze is None:
        for data in halos:
            sky = data[0]
            with open("/Users/jostheim/workspace/SLA/data/DarkWorlds/Train_Skies/{0}.tab".format(sky.replace(".csv", "")), 'r') as tmp_if:
                predict_data = []
                for line in tmp_if:
                    splitter = line.split(" ")
                    predict_data.append(splitter[2:])
    else:
        sky = halos[sky_to_analyze][0]
        predict_data = []
        with open("/Users/jostheim/workspace/SLA/data/DarkWorlds/Train_Skies/Sky{0}.tab".format(sky_to_analyze), 'r') as tmp_if:
            with open("/Users/jostheim/workspace/SLA/data/DarkWorlds/tmp/tmp_{0}.unscaled.tab".format(sky.replace(".csv", "")), 'w') as tmp_of:
                for line in tmp_if:
                    splitter = line.strip().split(" ")
                    predict_data.append(splitter[0:2])
                    tmp_of.write(" ".join(splitter[2:]) + "\n")
            os.popen("/usr/local/bin/svm-scale -r {0} /Users/jostheim/workspace/SLA/data/DarkWorlds/tmp/tmp_{1}.unscaled.tab > /Users/jostheim/workspace/SLA/data/DarkWorlds/tmp/tmp_{1}.scaled" \
                     .format(output_file + ".{0}".format(number_random) + ".scaled", sky.replace(".csv", "")))
            os.popen("/usr/local/bin/svm-predict -b 1 /Users/jostheim/workspace/SLA/data/DarkWorlds/tmp/tmp_{0}.scaled {1} /Users/jostheim/workspace/SLA/data/DarkWorlds/tmp/tmp_{0}.predict.out" \
                     .format(sky.replace(".csv", ""), output_file + ".{0}".format(number_random) + ".x0_y0_likelihood"))
            with open("/Users/jostheim/workspace/SLA/data/DarkWorlds/tmp/tmp_{0}.predict.out".format(sky.replace(".csv", ""))) as p_if:
                predictions = []
                for i, data in enumerate(p_if):
                    data = data.split(" ")
                    if i == 0:
                        continue
#                    print data
#                    print predict_data[i-1]
#                    print halos[sky_to_analyze][2],halos[sky_to_analyze][3]
                    if int(data[0]) == 1 and len(data) > 2:
                        predictions.append((data[0], data[1], data[2], predict_data[i - 1][0], predict_data[i - 1][1], halos[sky_to_analyze][2], halos[sky_to_analyze][3]))
                predictions = sorted(predictions, key=sort_predictions, reverse=True)
                for prediction in predictions[0:100]:
                    print prediction
                
def write_line_proxy(args):
    sky_datas = args[0]
    i = args[1] 
    j = args[2]
    clazz = args[3]
    svm = args[4]
    test = args[5]
    return write_line(sky_datas, i, j, clazz, svm, test)

def write_line(sky_datas, xs, ys, clazz, halo_number, svm=False, test=False):
    global total_done
    new_sky_datas = []
#    with Timer() as t:
    for iii, x in enumerate(xs):
        i = x
        j = ys[iii]
        dists = numpy.sqrt((sky_datas[:, 0] - i) * (sky_datas[:, 0] - i) + (sky_datas[:, 1] - j) * (sky_datas[:, 1] - j))
        thetas = numpy.arctan2((sky_datas[:, 1] - j), (sky_datas[:, 0] - i))
        e_tans = -1 * (sky_datas[:, 2] * numpy.cos(2 * thetas[:]) + sky_datas[:, 3] * numpy.sin(2 * thetas[:]))
        thetas = numpy.rad2deg(thetas)
        new_sky_datas_tmp = []
        for k, dist in enumerate(dists):
            new_sky_datas_tmp.append((dist, thetas[k], e_tans[k]))
        new_sky_datas.append(new_sky_datas_tmp)
#    print "Took {0} to build the sky_data for {1}, {2}".format(t.interval, i, j)
    # sort by dist and theta
#    with Timer() as t:
#    new_sky_datas = sorted(new_sky_datas, cmp=sort_compare)
#    print "Took {0} to sort the sky_data for {1}, {2}".format(t.interval, i, j)
    # create a new data point
#    with Timer() as t:
    out = ""
#    if not svm and not test:
#        out = "{0}:{1}\t".format("class", clazz)
#    elif svm and not test:
#        out = "{0} ".format(clazz)
#    elif test:
#        out = "{0} {1} ".format(i, j)
    if svm:
        n = 1
    else:
        n = 0
        
    tmp = []
    for zz, datum in enumerate(new_sky_datas[0]):
        if not svm:
            tmp1 = ["class:{0}".format(clazz)]
        else: 
            n = 1
            tmp1 = ["{0}".format(clazz)]
        for z, new_sky_data in enumerate(new_sky_datas):
            for ii, val in enumerate(new_sky_data[zz]):
                if not svm:
                    tmp1.append("{0}_{1}:{2}".format(z + 1, col1s[ii], val))
                else:
                    tmp1.append("{0}:{1}".format(n, val))
                n += 1
        if not svm:
            tmp.append("\t".join(tmp1))
        else:
            tmp.append(" ".join(tmp1))
    if not svm:
        out += "\n".join(tmp)
    else:
        out += "\n".join(tmp)
    total_done += 1
#    print "Took {0} to build string the sky_data for {1}, {2}".format(t.interval, i, j)
#    if total_done%1000 == 0:
#        print "Done {0}/{1}".format(total_done, total)
    return out + "\n"
    

def plot(file, prefix="training"):
    all_dists = []
    all_thetas = []
    all_e_tans = []
    for sky_datas, num_skys, sky_pos1, sky_pos2, sky_pos3, sky  in get_sky_datas(file):
#        if j > 0:
#            break
        if num_skys < 2 or num_skys > 2:
            continue
        data = (None, sky_datas[:, 0], sky_datas[:, 1], None, sky_datas[:, 2], sky_datas[:, 3])
#        sky_pos1 = [numpy.random.uniform(0, 4200), numpy.random.uniform(0, 4200)]
#        sky_pos2 = [numpy.random.uniform(0, 4200), numpy.random.uniform(0, 4200)]
#        sky_pos3 = [numpy.random.uniform(0, 4200), numpy.random.uniform(0, 4200)]

        for jj in xrange(num_skys):
            if jj == 0:
                rs, thetas = get_r_theta(data, sky_pos1)
                e_tans_exact = -1 * (sky_datas[:, 2] * numpy.cos(2 * thetas[:]) + sky_datas[:, 3] * numpy.sin(2 * thetas[:]))
                for nn,r in enumerate(rs):
                    if r < 510000:
                        all_dists.append(rs[nn])
                        all_e_tans.append(e_tans_exact[nn])
            if jj == 1:
                r, theta = get_r_theta(data, sky_pos2)
                e_tans_exact = -1 * (sky_datas[:, 2] * numpy.cos(2 * thetas[:]) + sky_datas[:, 3] * numpy.sin(2 * thetas[:]))
                for nn,r in enumerate(rs):
                    if r < 100500:
                        all_dists.append(rs[nn])
                        all_e_tans.append(e_tans_exact[nn])
            if jj == 2:
                r, theta = get_r_theta(data, sky_pos2)
                e_tans_exact = -1 * (sky_datas[:, 2] * numpy.cos(2 * thetas[:]) + sky_datas[:, 3] * numpy.sin(2 * thetas[:]))
                for nn,r in enumerate(rs):
                    if r < 510000:
                        all_dists.append(rs[nn])
                        all_e_tans.append(e_tans_exact[nn])
    import matplotlib.pyplot as plt
    plt.plot(all_dists, all_e_tans, 'bo')
    plt.ylabel("{0}".format("e_tan"))
    plt.xlabel("{0}".format("Radius"))
    plt.show()
    counts, bins = numpy.histogram(all_e_tans, bins=10)
    print "mean {0}, stddev {1}".format(numpy.mean(all_e_tans), numpy.std(all_e_tans))
    center = (bins[:-1]+bins[1:])/2
    import matplotlib.pyplot as plt
    plt.plot(center, counts, 'b')
    plt.xlabel("{0}".format("Radius"))
    plt.ylabel("{0}".format("counts"))
    plt.show()




    
    
if __name__ == '__main__':
    print "Working on file {0}".format(sys.argv[2])
    if len(sys.argv) > 3:
        number_random = int(sys.argv[4])
    if len(sys.argv) > 4:
        x_y_skip = int(sys.argv[5])
    if sys.argv[1] == "write_test_files":
        produce_test_files(sys.argv[2])
    elif sys.argv[1] == "test":
        test(sys.argv[2], sys.argv[3] + ".svm", 1)
    elif sys.argv[1] == "bayesian":
        convert_and_filter(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "svm":
        convert_and_filter(sys.argv[2], sys.argv[3] + ".svm", True)
    elif sys.argv[1] == "plot":
        plot_data(sys.argv[2])
    elif sys.argv[1] == "likelihood":
#        plot(sys.argv[2])
#        fwhm = 2*numpy.sqrt(2*numpy.log(2))*1.0
#        print int_gauss(0.000000000001, 3, fwhm)
        range = [0, 62]
        get_likelihood(sys.argv[2], range)
    elif sys.argv[1] == "samples":
        process_samples(sys.argv[2])
