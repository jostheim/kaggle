
# In[3]:

import pandas as pd
import pylab as p
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
import scipy.misc
from multiprocessing import Pool
import numpy as np
from PIL import Image
import random
import time
from scipy import integrate
pool = Pool(processes=8)
p.rcParams['figure.figsize'] = (10.0, 8.0)


# In[4]:

def compute_smoothness(data):
    non_nan_mask = ~isnan(data)
    tmp_data = data[non_nan_mask]
    non_infinity_mask = ~isinf(tmp_data)
    tmp_data= tmp_data[non_infinity_mask]
    first_deriv = np.diff(tmp_data)
    tmp = np.std(first_deriv)/np.abs(np.mean(first_deriv))
    return tmp

def radial_data(data, nbins=10, theta_range=[-np.pi,np.pi], x=None, y=None, rmax=600):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)
    
    A function to reduce an image to a radial cross-section.
    
    INPUT:
    ------
    data   - whatever data you are radially averaging.  Data is
            binned into a series of annuli of width 'annulus_width'
            pixels.
    annulus_width - width of each annulus.  Default is 1.
    working_mask - array of same size as 'data', with zeros at
                      whichever 'data' points you don't want included
                      in the radial data computations.
      x,y - coordinate system in which the data exists (used to set
             the center of the data).  By default, these are set to
             integer meshgrids
      rmax -- maximum radial value over which to compute statistics
    
     OUTPUT:
     -------
      r - a data structure containing the following
                   statistics, computed across each annulus:
          .r      - the radial coordinate used (outer edge of annulus)
          .mean   - mean of the data in the annulus
          .std    - standard deviation of the data in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """
    
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
 
    class radialDat:
        """Empty object container.
        """
        def __init__(self): 
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None

    #---------------------
    # Set up input parameters
    #---------------------
    data = np.array(data)
    mirror_data = np.array(data)
        
    npix, npiy = data.shape
    if x==None or y==None:
        x1 = np.arange(-npix/2.,npix/2.)
        y1 = np.arange(-npiy/2.,npiy/2.)
        x,y = np.meshgrid(y1,x1)

    theta = np.arctan2(y, x)
    
    r = np.sqrt(x**2 + y**2) #abs(x+1j*y)
    
    if rmax==None:
        rmax = r.max()

    mirror_theta_range = [theta_range[0]+ np.pi, theta_range[1]+np.pi]
        
    #---------------------
    # Prepare the data container
    #---------------------
#    dr = np.abs([x[0,0] - x[0,1]]) * annulus_width
#     radial = np.arange(rmax/dr)*dr + dr/2.
#    radial = np.logspace(np.log10(1.0), np.log10(rmax), num=nbins, base=10)
    radial = np.linspace(0.0, rmax, num=nbins)
    nrad = nbins
    radialdata = radialDat()
    radialdata.r = radial
    radialdata.mean = np.zeros(nrad)
    radialdata.smoothness_mean = np.zeros(nrad)
    radialdata.std = np.zeros(nrad)
    radialdata.smoothness_std = np.zeros(nrad)
    radialdata.median = np.zeros(nrad)
    radialdata.smoothness_median = np.zeros(nrad)
    radialdata.max = np.zeros(nrad)
    radialdata.smoothness_max = np.zeros(nrad)
    radialdata.min = np.zeros(nrad)
    radialdata.smoothness_min = np.zeros(nrad)
    radialdata.std = np.zeros(nrad)
    radialdata.skewness = np.zeros(nrad)
    radialdata.smoothness_skewness = np.zeros(nrad)
    radialdata.kurtosis = np.zeros(nrad)
    radialdata.smoothness_kurtosis = np.zeros(nrad)
    radialdata.intensity = np.zeros(nrad)
    radialdata.smoothness_intensity = np.zeros(nrad)
    radialdata.pixel_sum = np.zeros(nrad)
    radialdata.smoothness_pixel_sum = np.zeros(nrad)
    radialdata.numel = np.zeros(nrad)
    radialdata.area = np.zeros(nrad)
    radialdata.integrated_intensity = np.zeros(nrad)
    radialdata.smoothness_integrated_intensity = np.zeros(nrad)
    
    radialdata.cummulative_mean = np.zeros(nrad)
    radialdata.cummulative_smoothness_mean = np.zeros(nrad)
    radialdata.cummulative_std = np.zeros(nrad)
    radialdata.cummulative_smoothness_std = np.zeros(nrad)
    radialdata.cummulative_median = np.zeros(nrad)
    radialdata.cummulative_smoothness_median = np.zeros(nrad)
    radialdata.cummulative_max = np.zeros(nrad)
    radialdata.cummulative_smoothness_max = np.zeros(nrad)
    radialdata.cummulative_min = np.zeros(nrad)
    radialdata.cummulative_smoothness_min = np.zeros(nrad)
    radialdata.cummulative_std = np.zeros(nrad)
    radialdata.cummulative_skewness = np.zeros(nrad)
    radialdata.cummulative_smoothness_skewness = np.zeros(nrad)
    radialdata.cummulative_kurtosis = np.zeros(nrad)
    radialdata.cummulative_smoothness_kurtosis = np.zeros(nrad)
    radialdata.cummulative_intensity = np.zeros(nrad)
    radialdata.cummulative_smoothness_intensity = np.zeros(nrad)
    radialdata.cummulative_pixel_sum = np.zeros(nrad)
    radialdata.cummulative_smoothness_pixel_sum = np.zeros(nrad)
    radialdata.cummulative_numel = np.zeros(nrad)
    radialdata.cummulative_area = np.zeros(nrad)
    
    
    #---------------------
    # Loop through the bins
    #---------------------
    for irad, minrad in enumerate(radial): #= 1:numel(radial)
      if irad+1 == len(radial):
        continue
      maxrad = radial[irad+1]
      thisindex = ((r>=minrad) * (r<maxrad)) * ((theta >= theta_range[0]) * (theta < theta_range[1])) 
      allrindex = ((r<maxrad)) * ((theta >= theta_range[0]) * (theta < theta_range[1])) 
      if not thisindex.ravel().any():
        radialdata.mean[irad] = np.nan
        radialdata.smoothness_mean[irad] = np.nan
        radialdata.std[irad]  = np.nan
        radialdata.smoothness_std[irad]  = np.nan
        radialdata.median[irad] = np.nan
        radialdata.smoothness_median[irad] = np.nan
        radialdata.max[irad] = np.nan
        radialdata.smoothness_max[irad] = np.nan
        radialdata.min[irad] = np.nan
        radialdata.smoothness_min[irad] = np.nan
        radialdata.kurtosis[irad] = np.nan
        radialdata.smoothness_kurtosis[irad] = np.nan
        radialdata.skewness[irad] = np.nan       
        radialdata.smoothness_skewness[irad] = np.nan       
        radialdata.pixel_sum[irad] = np.nan
        radialdata.smoothness_pixel_sum[irad] = np.nan
        radialdata.intensity[irad] = np.nan
        radialdata.smoothness_intensity[irad] = np.nan
        radialdata.area[irad] = np.nan
        radialdata.numel[irad] = np.nan
        radialdata.integrated_intensity[irad] = np.nan
        radialdata.smoothness_integrated_intensity[irad] = np.nan
        
        radialdata.cummulative_mean[irad] = np.nan
        radialdata.cummulative_smoothness_mean[irad] = np.nan
        radialdata.cummulative_std[irad]  = np.nan
        radialdata.cummulative_smoothness_std[irad]  = np.nan
        radialdata.cummulative_median[irad] = np.nan
        radialdata.cummulative_smoothness_median[irad] = np.nan
        radialdata.cummulative_max[irad] = np.nan
        radialdata.cummulative_smoothness_max[irad] = np.nan
        radialdata.cummulative_min[irad] = np.nan
        radialdata.cummulative_smoothness_min[irad] = np.nan
        radialdata.cummulative_kurtosis[irad] = np.nan
        radialdata.cummulative_smoothness_kurtosis[irad] = np.nan
        radialdata.cummulative_skewness[irad] = np.nan       
        radialdata.cummulative_smoothness_skewness[irad] = np.nan       
        radialdata.cummulative_pixel_sum[irad] = np.nan
        radialdata.cummulative_smoothness_pixel_sum[irad] = np.nan
        radialdata.cummulative_intensity[irad] = np.nan
        radialdata.cummulative_smoothness_intensity[irad] = np.nan
        radialdata.cummulative_area[irad] = np.nan
      else:
        size, min_max, mean, variance, skewness, kurtosis = scipy.stats.describe(data[thisindex], axis=None)
        radialdata.mean[irad] = mean
        radialdata.smoothness_mean[irad] = compute_smoothness(radialdata.mean)
        radialdata.std[irad]  = np.sqrt(variance)
        radialdata.smoothness_std[irad] = compute_smoothness(radialdata.std)
        radialdata.median[irad] = np.median(data[thisindex])
        radialdata.smoothness_median[irad] = compute_smoothness(radialdata.median)
        radialdata.max[irad] = min_max[1]
        radialdata.smoothness_max[irad] = compute_smoothness(radialdata.max)
        radialdata.min[irad] = min_max[0]
        radialdata.smoothness_min[irad] = compute_smoothness(radialdata.min)
        radialdata.kurtosis[irad] = kurtosis
        radialdata.smoothness_kurtosis[irad] = compute_smoothness(radialdata.kurtosis)
        radialdata.skewness[irad] = skewness
        radialdata.smoothness_skewness[irad] = compute_smoothness(radialdata.skewness)
        radialdata.pixel_sum[irad] = np.sum(data[thisindex])
        radialdata.smoothness_pixel_sum[irad] = compute_smoothness(radialdata.pixel_sum)
        radialdata.area[irad] = np.pi*maxrad**2 - np.pi*minrad**2
        radialdata.intensity[irad] = radialdata.pixel_sum[irad]/size
        radialdata.smoothness_intensity[irad] = compute_smoothness(radialdata.intensity)
        radialdata.numel[irad] = size
        if irad == 0:
            radialdata.integrated_intensity[irad] = radialdata.intensity[irad]
        else:
            nan_mask = ~isnan(radialdata.intensity)
            masked_intensity = radialdata.intensity[nan_mask]
            masked_r = radialdata.r[nan_mask]
            radialdata.integrated_intensity[irad] = scipy.integrate.trapz((masked_intensity[0:irad] * 2*np.pi*masked_r[0:irad]), masked_r[0:irad])
            radialdata.smoothness_integrated_intensity[irad] = compute_smoothness(radialdata.integrated_intensity)
        
        size, min_max, mean, variance, skewness, kurtosis = scipy.stats.describe(data[allrindex], axis=None)
        radialdata.cummulative_mean[irad] = mean
        radialdata.cummulative_smoothness_mean[irad] = compute_smoothness(radialdata.cummulative_mean)
        radialdata.cummulative_std[irad]  = np.sqrt(variance)
        radialdata.cummulative_smoothness_std[irad] = compute_smoothness(radialdata.cummulative_std)
        radialdata.cummulative_median[irad] = np.median(data[allrindex])
        radialdata.cummulative_smoothness_median[irad] = compute_smoothness(radialdata.cummulative_median)
        radialdata.cummulative_max[irad] = min_max[1]
        radialdata.cummulative_smoothness_max[irad] = compute_smoothness(radialdata.cummulative_max)
        radialdata.cummulative_min[irad] = min_max[0]
        radialdata.cummulative_smoothness_min[irad] = compute_smoothness(radialdata.cummulative_min)
        radialdata.cummulative_kurtosis[irad] = kurtosis
        radialdata.cummulative_smoothness_kurtosis[irad] = compute_smoothness(radialdata.cummulative_kurtosis)
        radialdata.cummulative_skewness[irad] = skewness
        radialdata.cummulative_smoothness_skewness[irad] = compute_smoothness(radialdata.cummulative_skewness)
        radialdata.cummulative_pixel_sum[irad] = np.sum(data[allrindex])
        radialdata.cummulative_smoothness_pixel_sum[irad] = compute_smoothness(radialdata.cummulative_pixel_sum)
        radialdata.cummulative_area[irad] = np.pi*maxrad**2
        radialdata.cummulative_intensity[irad] = radialdata.cummulative_pixel_sum[irad]/size
        radialdata.cummulative_smoothness_intensity[irad] = compute_smoothness(radialdata.cummulative_intensity)
        radialdata.cummulative_numel[irad] = size
    #---------------------
    # Return with data
    #---------------------
    
    return radialdata


# In[5]:

def radial_profile(image, number_radial_bins, theta_range, galaxy_id):
    profile_data = radial_data(image, number_radial_bins, theta_range=theta_range)
#     first_deriv_mean = np.diff(profile_data.mean, 1)
#     first_deriv_median = np.diff(profile_data.median, 1)    
#     first_deriv_max = np.diff(profile_data.max, 1)
#     first_deriv_min = np.diff(profile_data.min, 1)
#     first_deriv_var = np.diff(profile_data.variance, 1)
    #print 0.2*profile_data.integrated_intensity, (profile_data.pixel_sum/profile_data.area)
    petrosian_radius = profile_data.r[np.nanargmin(np.abs((profile_data.pixel_sum/profile_data.area) - (0.2*(profile_data.integrated_intensity/profile_data.cummulative_area))))]
    #print petrosian_radius
    pr2 = petrosian_radius*2.0
    integration_index = nanargmin(np.abs(profile_data.r - pr2))
    nan_mask = ~isnan(profile_data.intensity)
    masked_intensity = profile_data.intensity[nan_mask]
    masked_r = profile_data.r[nan_mask]
    petrosian_flux = scipy.integrate.trapz(masked_intensity[0:integration_index]*2*np.pi*masked_r[0:integration_index], masked_r[0:integration_index])
    index50rp = nanargmin(np.abs(0.5*petrosian_flux - profile_data.integrated_intensity))
    index90rp = nanargmin(np.abs(0.9*petrosian_flux - profile_data.integrated_intensity))
    
    radial_profile = []
    for i, r in enumerate(profile_data.r):
        tmp = {"GalaxyId":galaxy_id, "r":r,
               "petrosian_r":petrosian_radius, 
               "petrosian_flux":petrosian_flux, 
               "petrosian_50_percent_radius":profile_data.r[index50rp],
               "petrosian_90_percent_radius":profile_data.r[index90rp],
               "petrosian_concentration": profile_data.r[index90rp]/profile_data.r[index50rp],
               "theta_range_min":theta_range[0], 
               "theta_range_max":theta_range[1], 
               }
        for key, val in profile_data.__dict__.iteritems():
            if val is not None and len(val) >= i:
                tmp[key] = val[i]
#         if i < (len(profile_data.r)-1):
#             tmp['mean_first_deriv'] = first_deriv_mean[i]
#             tmp['median_first_deriv'] = first_deriv_median[i]
#             tmp['max_first_deriv'] = first_deriv_max[i]
#             tmp['min_first_deriv'] = first_deriv_min[i]
#             tmp['variance_first_deriv'] = first_deriv_var[i]
        radial_profile.append(tmp)
    return radial_profile


# In[6]:

def theta_profile(image_id, theta_bins, number_radial_bins, do_subtract=True):
    image = scipy.misc.imread('/Users/jostheim/workspace/kaggle/data/galaxy_zoo/images_training_rev1/{0}.jpg'.format(image_id), flatten=True)
    if do_subtract:
        mirror_image = scipy.misc.imrotate(image, 180)
        image_on_side = scipy.misc.imrotate(image, 90)
        flipped_image = scipy.misc.imrotate(image_on_side, 180)
    #radial_profiles_dict = []
    radial_profiles_data = []
    mirror_sub_radial_profiles_data = []
    flipped_sub_radial_profiles_data = []
    for i, theta in enumerate(theta_bins):
        if (i+1) == len(theta_bins):
            continue
        profile_data = radial_profile(image, number_radial_bins, [theta, theta_bins[i+1]], image_id)
        radial_profiles_data = radial_profiles_data + profile_data
        if do_subtract:
            profile_data = radial_profile(image-mirror_image, number_radial_bins, [theta, theta_bins[i+1]], image_id)
            mirror_sub_radial_profiles_data = mirror_sub_radial_profiles_data + profile_data
            profile_data = radial_profile(image_on_side-flipped_image, number_radial_bins, [theta, theta_bins[i+1]], image_id)
            flipped_sub_radial_profiles_data = flipped_sub_radial_profiles_data + profile_data
    return radial_profiles_data, mirror_sub_radial_profiles_data, flipped_sub_radial_profiles_data


# In[7]:

def calculate_ellipticity_and_position_angle(tmp_df):
    theta_groups = tmp_df.groupby(['theta_range_min', 'theta_range_max'])
    integrated_intensities = []
    data = []
    for i, ((theta_min, theta_max), theta_group) in enumerate(theta_groups):
        integrated_intensities.append((np.max(theta_group['cummulative_pixel_sum']), (theta_min, theta_max)))
    # the one with the most total light wins
    integrated_intensities = sorted(integrated_intensities, reverse=True)
    theta_range = integrated_intensities[0][1]
    max_intensity = integrated_intensities[0][0]
    # find the 50% cummulative light point along the major axis
    index50 = nanargmin(np.abs(tmp_df[(tmp_df['theta_range_min'] == theta_range[0]) & (tmp_df['theta_range_max'] == theta_range[1])]['cummulative_pixel_sum']-0.5*max_intensity))
    cum50major = tmp_df['r'].iloc[index50]
    mid_theta = ((theta_range[0] + ((theta_range[1] - theta_range[0])/2.0))+np.pi/2.0)
    mid_theta_bin = tmp_df['theta_range_min']+((tmp_df['theta_range_max'] - tmp_df['theta_range_min'])/2.0)
    min_idx = ( mid_theta_bin - mid_theta ).idxmin()
    theta_range = (tmp_df['theta_range_min'].ix[min_idx], tmp_df['theta_range_max'].ix[min_idx])
    # find the 50% cummulative ligth point along the minor axis
    index50 = nanargmin(np.abs(tmp_df[(tmp_df['theta_range_min'] == theta_range[0]) & (tmp_df['theta_range_max'] == theta_range[1])]['cummulative_pixel_sum']-0.5*max_intensity))
    cum50minor = tmp_df['r'].iloc[index50]
    position_angle = mid_theta - np.pi/2.0
    # 1.0/major and 1.0/minor b/c the scale is inverted, the 50% point is CLOSER for brighter parts (major axis) and farther for dimmer (minor axis)
    ellipticity = np.sqrt(1 - ((1.0/cum50minor)**2/(1.0/cum50major)**2))
    return position_angle, ellipticity, cum50major, cum50minor

def theta_profile_proxy(args):
    image_id = args[0]
    theta_bins = args[1]
    number_radial_bins = args[2]
    ret = theta_profile(image_id, theta_bins, number_radial_bins)
    return ret

def ellipticity_and_position_proxy(args):
    image_id = args[0]
    theta_bins = args[1]
    number_radial_bins = args[2]
    ret = theta_profile(image_id, theta_bins, number_radial_bins)
    tmp_df = pd.DataFrame(ret[0])
    calc_bins = np.linspace(10, 150, num=50)
    output = []
    for bin in calc_bins:
        t = tmp_df[tmp_df['r'] < bin]
        output.append(calculate_ellipticity_and_position_angle(t))
    return image_id, output, ret[0]


def get_galaxy_id_perecentile_samples(clazz, n_percentiles=10):
    ret = []
    for i in xrange(0, n_percentiles+1):
        if i == 0:
            prob = 0.0
        else:
            prob = float(i)/n_percentiles
#         print galaxy_training[galaxy_training[clazz] == prob].index[0:1].values[0]
#        print prob
        if len(galaxy_training[galaxy_training[clazz] == prob].index[3:4].values) > 0:
            ret.append((prob, galaxy_training[galaxy_training[clazz] == prob].index[3:4].values[0]))
    ret = sorted(ret)
    return ret


# In[ ]:

def plot_images_for_percentiles(percentile_galaxy_ids):
    for key, val in percentile_galaxy_ids:
        image = scipy.misc.imread('/Users/jostheim/workspace/kaggle/data/galaxy_zoo/images_training_rev1/{0}.jpg'.format(val), flatten=True)
        mirror_image = scipy.misc.imrotate(image, 180)
        plt.imshow(image, label="{0}".format(key))
        plt.title("{0} {1}".format(key, val))
        plt.legend()
        plt.show()
        plt.imshow(mirror_image, label="{0}".format(key))
        plt.title("{0}".format(key))
        plt.legend()
        plt.show()
        plt.imshow(image - mirror_image, label="{0}".format(key))
        plt.title("{0}".format(key))
        plt.legend()
        plt.show()

def plot_deriviative_profiles_for_percentiles(percentile_galaxy_ids):
    for key, val in percentile_galaxy_ids:
        profiles_data = theta_profile(val, theta_bins, 100)
        df = pd.DataFrame(profiles_data)
        integral = numpy.trapz(df['mean'].values, df['r'].values)
        df['mean'] = df['mean']/integral
        df = df[df['r'] < 30]
        first_deriv = np.diff(df['median'].values, 1)
        smoothness = np.std(first_deriv)/np.abs(np.mean(first_deriv))
        print key, smoothness
        #plt.plot(df['r'], df['mean'], label="{0}".format(key))
        plt.plot(df['r'][0:(len(df['mean'])-1)]/df['petrosian_r'][0:(len(df['mean'])-1)], first_deriv, label="{0}".format(key))
    plt.legend()
    plt.show()
        
def plot_radial_profiles_for_percentiles(percentile_galaxy_ids):
    for key, val in percentile_galaxy_ids:
        profiles_data = theta_profile(val, theta_bins, 200)
        subtract_df = pd.DataFrame(profiles_data)
#         theta_groups = df.groupby(['theta_range_min', 'theta_range_max'])
#         for i, ((theta_min, theta_max), profile) in enumerate(theta_groups):
#             #print i, theta_min, theta_max
#             if i==0 and theta_max < np.pi:
#                 subtract_df = profile
        subtract_df = subtract_df[subtract_df['r'] < 100]
#         df = df[df['r'] < 100]
        #print subtract_df['r'].values
#        plt.plot(subtract_df['r'], subtract_df['smoothness_mean'], label="{0}".format(key))
        plt.plot(subtract_df['r'], subtract_df['std'], label="{0}".format(key))
#        plt.plot(subtract_df['r'], subtract_df['smoothness_intensity'], label="{0}".format(key))
#         plt.legend()
#         plt.show()
#         plt.plot(df['r'], df['mean'], label="{0}".format(key))
#         plt.plot(mirror_profile['r'], mirror_profile['mean'])
#         plt.legend()
#         plt.show()
        #plt.plot(df['r'][0:(len(df['mean'])-1)], first_deriv, label="{0}".format(key))
    plt.legend()
    plt.show()


def calculate_ellipticity_and_position_angle(tmp_df):
    tmp_df = tmp_df[tmp_df['r'] < 100]
    theta_groups = tmp_df.groupby(['theta_range_min', 'theta_range_max'])
    integrated_intensities = []
    data = []
    for i, ((theta_min, theta_max), theta_group) in enumerate(theta_groups):
        #print theta_min, theta_max
        d = {'petrosian_r':theta_group['petrosian_r'].iloc[0], 'petrosian_90_percent_radius':theta_group['petrosian_90_percent_radius'].iloc[0], 'petrosian_50_percent_radius':theta_group['petrosian_50_percent_radius'].iloc[0], 'integrated_intensity':np.max(theta_group['integrated_intensity']), 'cummulative_area':np.max(theta_group['cummulative_area']), 'cummulative_intensity':np.max(theta_group['cummulative_intensity']), 'cummulative_pixel_sum':np.max(theta_group['cummulative_pixel_sum']), 'theta_mid':np.degrees(theta_min)+(np.degrees(theta_max) - np.degrees(theta_min))}
        data.append(d)
        integrated_intensities.append((np.max(theta_group['cummulative_pixel_sum']), (theta_min, theta_max)))
    df = pd.DataFrame(data)
    plt.plot(df['theta_mid'],df['petrosian_50_percent_radius'])
    plt.show()
    plt.plot(df['theta_mid'],df['cummulative_pixel_sum'])
    plt.show()
    plt.plot(df['theta_mid'],df['petrosian_90_percent_radius'])
    plt.show()
    integrated_intensities = sorted(integrated_intensities, reverse=True)
    #print integrated_intensities
    theta_range = integrated_intensities[0][1]
    max_intensity = integrated_intensities[0][0]
    index50 = nanargmin(np.abs(tmp_df[(tmp_df['theta_range_min'] == theta_range[0]) & (tmp_df['theta_range_max'] == theta_range[1])]['cummulative_pixel_sum']-0.5*max_intensity))
    print '50% cummulative r: ', tmp_df['r'].iloc[index50]
    cum50major = tmp_df['r'].iloc[index50]
    print np.degrees(theta_range[0]), np.degrees(theta_range[1])
    petrosian_r_major_axis = 1.0/tmp_df[(tmp_df['theta_range_min'] == theta_range[0]) & (tmp_df['theta_range_max'] == theta_range[1])]['petrosian_50_percent_radius'].values[0]
    print 'major',petrosian_r_major_axis
    mid_theta = ((theta_range[0] + ((theta_range[1] - theta_range[0])/2.0))+np.pi/2.0)
    print 'pa', np.degrees(mid_theta - np.pi/2.0) 
    mid_theta_bin = tmp_df['theta_range_min']+((tmp_df['theta_range_max'] - tmp_df['theta_range_min'])/2.0)
#     print np.abs(mid_theta_bin - mid_theta).values
    min_idx = ( mid_theta_bin - mid_theta ).idxmin()
    theta_range = (tmp_df['theta_range_min'].ix[min_idx], tmp_df['theta_range_max'].ix[min_idx])
#     print tmp_df[(tmp_df['theta_range_min'] == theta_range[0]) & (tmp_df['theta_range_max'] == theta_range[1])]['cummulative_pixel_sum']
    index50 = nanargmin(np.abs(tmp_df[(tmp_df['theta_range_min'] == theta_range[0]) & (tmp_df['theta_range_max'] == theta_range[1])]['cummulative_pixel_sum']-0.5*max_intensity))
    print '50% cummulative r: ', tmp_df['r'].iloc[index50]
    cum50minor = tmp_df['r'].iloc[index50]
    print tmp_df['theta_range_max'].ix[min_idx], tmp_df['theta_range_min'].ix[min_idx]
    petrosian_r_minor_axis = 1.0/tmp_df.ix[min_idx]['petrosian_50_percent_radius']
    print 'minor',petrosian_r_minor_axis
    position_angle = mid_theta - np.pi/2.0
    ellipticity = np.sqrt(1 - ((1.0/cum50minor)**2/(1.0/cum50major)**2))
    return ellipticity, position_angle

