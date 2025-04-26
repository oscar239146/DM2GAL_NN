from basemodule import *
from nnmodule import *
from Corrfunc.theory.wp import wp

def SHMR_mask(Mh0, Ms0, Mh3, Ms3, sig = 2, n_bins = 5000, threshold = np.log10(5e+12), bool_gate = True):
    '''
    Returns boolean mask of all values with Ms within x sigma for certain number of Mh bins
    '''
    m_bins = np.linspace(np.min(Mh0), np.max(Mh0), n_bins+1)
    ms_bins = []

    #Filter masses below 5e+12 to 2 sigma allowed in training data
    #for i in range(len(m_bins)-1):
    #    ms_bins.append([np.min(Ms0[(Mh0<m_bins[i+1])&(Mh0>=m_bins[i])]), np.max(Ms0[(Mh0<m_bins[i+1])&(Mh0>=m_bins[i])])])
    for i in range(len(m_bins) - 1):
        mask = (Mh0 >= m_bins[i]) & (Mh0 < m_bins[i + 1])
        if np.any(mask):
            ms_mean = np.mean(Ms0[mask])
            ms_std = np.std(Ms0[mask])
            ms_bins.append([ms_mean - sig * ms_std, ms_mean + sig * ms_std])
        else:
            ms_bins.append([np.inf, -np.inf])

    Mh_thresh = threshold#np.log10(8e+12)
    keep_mask = np.zeros_like(Mh3, dtype=bool)

    for i in range(len(m_bins) - 1):
        m_low, m_high = m_bins[i], m_bins[i + 1]
        m_mask = (Mh3 >= m_low) & (Mh3 < m_high)

        if m_high <= Mh_thresh:  # Apply filtering only below threshold
            ms_low, ms_high = ms_bins[i]
            s_mask = (Ms3 >= ms_low) & (Ms3 <= ms_high)
            keep_mask |= (m_mask & s_mask)
        else:
            keep_mask |= m_mask 
    if bool_gate:
        return keep_mask
    else:
        return keep_mask.astype(int)
    
def SHMR_mask_high_bias(Mh0, Ms0, Mh3, Ms3, sig = 2, n_bins = 5000, threshold = np.log10(5e+12), bias_fact = 0.5, bool_gate = True):
    '''
    Returns boolean mask of all values with Ms within x sigma for certain number of Mh bins
    '''
    m_bins = np.linspace(np.min(Mh0), np.max(Mh0), n_bins+1)
    ms_bins = []

    #Filter masses below 5e+12 to 2 sigma allowed in training data
    #for i in range(len(m_bins)-1):
    #    ms_bins.append([np.min(Ms0[(Mh0<m_bins[i+1])&(Mh0>=m_bins[i])]), np.max(Ms0[(Mh0<m_bins[i+1])&(Mh0>=m_bins[i])])])
    for i in range(len(m_bins) - 1):
        mask = (Mh0 >= m_bins[i]) & (Mh0 < m_bins[i + 1])
        if np.any(mask):
            ms_mean = np.mean(Ms0[mask])
            ms_std = np.std(Ms0[mask])
            ms_bins.append([ms_mean - sig* bias_fact * ms_std, ms_mean + sig * ms_std])
        else:
            ms_bins.append([np.inf, -np.inf])

    Mh_thresh = threshold#np.log10(8e+12)
    keep_mask = np.zeros_like(Mh3, dtype=bool)

    for i in range(len(m_bins) - 1):
        m_low, m_high = m_bins[i], m_bins[i + 1]
        m_mask = (Mh3 >= m_low) & (Mh3 < m_high)

        if m_high <= Mh_thresh:  # Apply filtering only below threshold
            ms_low, ms_high = ms_bins[i]
            s_mask = (Ms3 >= ms_low) & (Ms3 <= ms_high)
            keep_mask |= (m_mask & s_mask)
        else:
            keep_mask |= m_mask 
    if bool_gate:
        return keep_mask
    else:
        return keep_mask.astype(int)


def Vol_splits(coms, n_volbins):  #, masses):
    """
    Creates list of list indices and COMs within each volume bin for a specified number of volume bins
    """
    #COM data has (0,0,0) in corner
    #n_volbins gives number of vol bins for one side
    L_max = np.floor(np.max(coms))+1
    vol_bins = np.linspace(0, L_max, n_volbins + 1) #bin edges
    
    cube_edges = [] # have (N_cubes+1)**3 cube edges
    for i in vol_bins:
        for j in vol_bins:
            for k in vol_bins:
                cube_edges.append([i,j,k])
    N_cubes = n_volbins**3
    
    cube_coords = np.zeros(shape=(N_cubes,8,3))
    #masses_binned = []
    coms_binned = []
    index_binned = []
    
    c_x = coms[:,0]
    c_y = coms[:,1]
    c_z = coms[:,2]
    
    count =  0
    #create coordinates for edges of each cube
    for q in range(n_volbins):
        #corners = cube_edges[cube_edges[:,2] == vol_bins[q]] #iterate every row of cubes for bottom corners
        z = [vol_bins[q],vol_bins[q+1]]
        for n in range(n_volbins): #cubes per row
            #corners2 = corners[corners[:,1] == vol_bins[n]] #filter in y
            y = [vol_bins[n],vol_bins[n+1]] #possible y values
            for m in range(n_volbins):
                #corners3 = corners2[corners2[:,0] == vol_bins[m]]
                x = [vol_bins[m],vol_bins[m+1]]
                com_coords = []
                #cube_masses = []
                com_indices = []
                for i in range(len(coms)):
                    if ((x[0]<=c_x[i]<x[1])&(y[0]<=c_y[i]<y[1])&(z[0]<=c_z[i]<z[1])):
                        com_coords.append(coms[i])
                        com_indices.append(i)
                        #cube_masses.append(masses[i])
                coords = []
                for i in x:
                    for j in y:
                        for k in z:
                            coords.append([i,j,k])
                cube_coords[count,:,:] = coords
                com_coords = np.array(com_coords)
                #cube_masses = np.array(cube_masses)
                com_indices = np.array(com_indices)
                
                index_binned.append(com_indices)
                coms_binned.append(com_coords)
                #masses_binned.append(cube_masses)
                count = count+1 #cube count
       
    return index_binned, coms_binned

def JK_wp_err(box_L, pimax, nthreads, bins, coms, n_volbins):
    """
    Uses Jackknife sampling as in  doi:10.1111/j.1365-2966.2009.14389.x for error propagation of wp.
    Returns standard deviation for wp measurements.
    """
    if len(coms) < 1:
        return np.zeros(len(bins))
    N_cubes = n_volbins**3
    binned_coms = Vol_splits(coms, n_volbins)[1]
    wp_arr = []
    cube_inds = np.arange(0,N_cubes, dtype=int)
    for i in cube_inds:
        b_coms2 = []
        b_coms = np.array(binned_coms, dtype=object)[cube_inds != i]
        for i in b_coms:
            if len(i)>0:
                b_coms2.append(i)
                #Ensure at least 1 galaxy in each box
        b_coms2 = np.vstack(b_coms2) #all non-zero COMs with one box missing 
        wp_300_mass = wp(box_L, pimax, nthreads, bins, b_coms2[:,0], b_coms2[:,1], b_coms2[:,2], output_rpavg = True)
        wp_arr.append(wp_300_mass["wp"])
    #wp_mean = np.sum(wp_arr, axis=0)/N_cubes
    wp_err = np.std(wp_arr, axis = 0, ddof =0)
    return wp_err


def JK_wp_cov(box_L, pimax, nthreads, bins, coms, n_volbins):
    """
    Uses Jackknife sampling as in  doi:10.1111/j.1365-2966.2009.14389.x for error propagation of wp.
    Returns covariance matrix for wp measurements.
    """
    #if len(coms) < 1:
    #    return np.zeros(len(bins))
    N_cubes = n_volbins**3
    binned_coms = Vol_splits(coms, n_volbins)[1]
    wp_arr = []
    cube_inds = np.arange(0,N_cubes, dtype=int)
    for i in cube_inds:
        b_coms2 = []
        b_coms = np.array(binned_coms, dtype=object)[cube_inds != i]
        for i in b_coms:
            if len(i)>0:
                b_coms2.append(i)
                #Ensure at least 1 galaxy in each box
        b_coms2 = np.vstack(b_coms2) #all non-zero COMs with one box missing 
        wp_300_mass = wp(box_L, pimax, nthreads, bins, b_coms2[:,0], b_coms2[:,1], b_coms2[:,2], output_rpavg = True)
        wp_arr.append(wp_300_mass["wp"])
        
    wp_arr = np.array(wp_arr)
    wp_mean = np.sum(wp_arr, axis=0)/N_cubes
    #wp_err = np.std(wp_arr, axis = 0, ddof =1)
    
    #wp_mean = np.mean(wp_arr, axis=0)

    diff = wp_arr - wp_mean[None, :] #change shape to 2D
    factor = (N_cubes - 1) / N_cubes
    cov_matrix = factor * diff.T @ diff
    
    return cov_matrix

def chisq(obs, exp, err, normalised = False, p=0):
    if normalised:
        return np.sum((obs - exp) ** 2 / (err ** 2))/(len(obs)-p)
    else:
        return np.sum((obs - exp) ** 2 / (err ** 2))
    
def chisq_cov(obs, exp, cov, normalised = False, p=0):
    diff = obs - exp
    cov_inv = np.linalg.pinv(cov)
    if normalised:
        return diff.T @ cov_inv @ diff / (len(obs)-p)
    else:
        return diff.T @ cov_inv @ diff


