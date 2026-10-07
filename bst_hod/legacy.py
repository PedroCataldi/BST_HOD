"""Original BST functions, copied unchanged from BST_method.py.

Kept only for reproducibility and for the regression tests: the new code in
:mod:`bst_hod.core` gives the same numbers. Use :func:`bst_hod.compute_hod`
for new work.
"""

import numpy as np

########################### start the loop over groups ###########################
def background_method_new(j,mlim,z_gal_target,halo_gr_id,d_com_gr,ring_label,ring_radius,factor_rmax,factor_r,r_200,delta_gr,alfa_gr,M_200,halo_id,z_gr,mabs,magr_gal,delta_gal,alfa_gal,z_gal):
    
    index_len = np.where((halo_gr_id[j] == halo_id) & (mabs<mlim))
    N_true=len(index_len[0])    
    #############Absolute magnitude of each Galaxy in the Group frame ###################    
    mabs_HOD= magr_gal-25.0-5*np.log10(d_com_gr[j]*(1+z_gr[j]))
    ############ R proyect tested ######################

    if np.ndim(factor_r) > 0:   # array-like
        rproy_gr = factor_r[j] * r_200[j]
    else:                       # scalar
        rproy_gr = factor_r * r_200[j]
        
    if (rproy_gr<0.05):
        
        rproy_gr=0.05
        
    if (rproy_gr>factor_rmax):
        
        rproy_gr=factor_rmax
        
    if (ring_label == 'fix'):
        
        radioext = rproy_gr + 2.
        radioint = rproy_gr + 1.
    else:
        
        radioext = ring_radius[1] * rproy_gr
        radioint = ring_radius[0] * rproy_gr

    ######## I centre galaxies in each group ############### 

    alfa_tmp = alfa_gal
    delta_tmp = delta_gal
    d_phys_gr=d_com_gr[j]/(1+z_gr[j])

    ######## Radius of each galaxy centre in each group (spherical coordinates) #########
    radio = np.sin(delta_gr[j])*np.sin(delta_gal)+np.cos(delta_gr[j])*np.cos(delta_gal)*np.cos((alfa_gr[j]-alfa_gal ))         
    tang_tita = np.sqrt(1. - radio*radio) / radio 
    radio_sqr = tang_tita * d_phys_gr
        
    ##################The outside square of size la "x" eta #################################
    radioint_rad = np.arctan(radioint/d_phys_gr) # rad
    radioext_rad = np.arctan(radioext/d_phys_gr) # rad

    lamin  = alfa_gr[j] - radioext_rad
    lamax  = alfa_gr[j] + radioext_rad

    etamin = delta_gr[j] - radioext_rad / np.cos(alfa_gr[j])
    etamax = delta_gr[j] + radioext_rad / np.cos(alfa_gr[j])

    xmin = lamin
    xmax = lamax
    ymin = etamin
    ymax = etamax

    ################ resolution of the mesh ################################################
    resolucion = 56;
    nx = resolucion * 56;
    ny = resolucion * 56;
    npixels = nx * ny;

    ################ create a 2d histogram  of nx and ny bins ################################

    try:
        #print('entre')
        xi = np.linspace(np.floor(xmin),np.ceil(xmax),nx)
        yi = np.linspace(np.floor(ymin),np.ceil(ymax),ny)
        H, xedges, yedges = np.histogram2d(alfa_tmp, delta_tmp, bins=(xi, yi), density=False)

        ################ We move to the centre of the  bins###########################################

        centrox_pixel = xedges[:-1] + (xedges[1:] - xedges[:-1]) / 2
        centroy_pixel = yedges[:-1] + (yedges[1:] - yedges[:-1]) / 2
        
        ################  We create a mesh with the binned galaxy coordinates in each group, each cell we call them "pixel "########
        
        xx_pixel, yy_pixel = np.meshgrid(centrox_pixel, centroy_pixel)

        ########  Spherical coordinates for the pixeles #########
        
        radio_pixel = np.sin(delta_gr[j]) * np.sin(yy_pixel) + np.cos(delta_gr[j]) * np.cos(yy_pixel) * np.cos((alfa_gr[j] - xx_pixel))
        tang_tita_pixel = np.sqrt(1. - radio_pixel*radio_pixel) / radio_pixel
        radio_pixel_sqr = tang_tita_pixel * d_phys_gr

        ########### We count the number of galaxies and bins in a circle of virial radius ################ 
        
        index_circ = (radio_sqr < rproy_gr)
        alfa_circ, delta_circ, mabs_gr_cir, z_gal_cir = alfa_tmp[index_circ], delta_tmp[index_circ], mabs_HOD[index_circ], z_gal[index_circ]

        index_pixel_circ = ((radio_pixel_sqr < rproy_gr)) #& (H.T > 0)
        centrox_pixel_circ, centroy_pixel_circ = xx_pixel[index_pixel_circ], yy_pixel[index_pixel_circ]
        
        ########### We count the number of galaxies and bins in a ring of radioint<r<radioext ################ 
        
        index_an = ((radio_sqr > radioint) & (radio_sqr < radioext))
        alfa_an, delta_an, mabs_gr_an, z_gal_an = alfa_tmp[index_an], delta_tmp[index_an], mabs_HOD[index_an], z_gal[index_an]

        index_pixel_an = ((radio_pixel_sqr > radioint) & (radio_pixel_sqr < radioext)) #& (H.T > 0)
        centrox_pixel_an, centroy_pixel_an = xx_pixel[index_pixel_an], yy_pixel[index_pixel_an]      

        ############## We calculate the Number of galaxies according to eq.2 of Rodriguez et al. 2015 ##################
        
        if (len(centrox_pixel_an)>0):
            N= len(alfa_circ) - len(alfa_an) * len(centrox_pixel_circ) / len(centrox_pixel_an)
        else:
            N= 0

    except ValueError:
        print("Oops! ValueError. Try again in index:",j)
        N=0

    return N_true,N, M_200[j]
########################### start the loop over groups ###########################
def background_method_rmax(j,mlim,z_gal_target,halo_gr_id,d_com_gr,ring_label,ring_radius,factor_rmax,factor_r,r_200,delta_gr,alfa_gr,M_200,halo_id,z_gr,mabs,magr_gal,delta_gal,alfa_gal,z_gal):
    
    index_len = np.where((halo_gr_id[j] == halo_id) & (mabs<mlim))
    N_true=len(index_len[0])    
    #############Absolute magnitude of each Galaxy in the Group frame ###################    
    mabs_HOD= magr_gal-25.0-5*np.log10(d_com_gr[j]*(1+z_gr[j]))
    ############ R proyect tested ######################

    if np.ndim(factor_r) > 0:   # array-like
        rproy_gr = factor_r[j] * r_200[j]
    else:                       # scalar
        rproy_gr = factor_r * r_200[j]
        
    if (rproy_gr<0.05):
        
        rproy_gr=0.05
        
    if (rproy_gr>factor_rmax):
        
        rproy_gr=factor_rmax
        
    if (ring_label == 'fix'):
        
        radioext = rproy_gr + 2.
        radioint = rproy_gr + 1.
    else:
        
        radioext = ring_radius[1] * rproy_gr
        radioint = ring_radius[0] * rproy_gr

    ######## I centre galaxies in each group ############### 

    alfa_tmp = alfa_gal
    delta_tmp = delta_gal
    d_phys_gr=d_com_gr[j]/(1+z_gr[j])

    ######## Radius of each galaxy centre in each group (spherical coordinates) #########
    radio = np.sin(delta_gr[j])*np.sin(delta_gal)+np.cos(delta_gr[j])*np.cos(delta_gal)*np.cos((alfa_gr[j]-alfa_gal ))         
    tang_tita = np.sqrt(1. - radio*radio) / radio 
    radio_sqr = tang_tita * d_phys_gr
        
    ##################The outside square of size la "x" eta #################################
    radioint_rad = np.arctan(radioint/d_phys_gr) # rad
    radioext_rad = np.arctan(radioext/d_phys_gr) # rad

    lamin  = alfa_gr[j] - radioext_rad
    lamax  = alfa_gr[j] + radioext_rad

    etamin = delta_gr[j] - radioext_rad / np.cos(alfa_gr[j])
    etamax = delta_gr[j] + radioext_rad / np.cos(alfa_gr[j])

    xmin = lamin
    xmax = lamax
    ymin = etamin
    ymax = etamax

    ################ resolution of the mesh ################################################
    resolucion = 56;
    nx = resolucion * 56;
    ny = resolucion * 56;
    npixels = nx * ny;

    ################ create a 2d histogram  of nx and ny bins ################################

    try:
        #print('entre')
        xi = np.linspace(np.floor(xmin),np.ceil(xmax),nx)
        yi = np.linspace(np.floor(ymin),np.ceil(ymax),ny)
        H, xedges, yedges = np.histogram2d(alfa_tmp, delta_tmp, bins=(xi, yi), density=False)

        ################ We move to the centre of the  bins###########################################

        centrox_pixel = xedges[:-1] + (xedges[1:] - xedges[:-1]) / 2
        centroy_pixel = yedges[:-1] + (yedges[1:] - yedges[:-1]) / 2
        
        ################  We create a mesh with the binned galaxy coordinates in each group, each cell we call them "pixel "########
        
        xx_pixel, yy_pixel = np.meshgrid(centrox_pixel, centroy_pixel)

        ########  Spherical coordinates for the pixeles #########
        
        radio_pixel = np.sin(delta_gr[j]) * np.sin(yy_pixel) + np.cos(delta_gr[j]) * np.cos(yy_pixel) * np.cos((alfa_gr[j] - xx_pixel))
        tang_tita_pixel = np.sqrt(1. - radio_pixel*radio_pixel) / radio_pixel
        radio_pixel_sqr = tang_tita_pixel * d_phys_gr

        ########### We count the number of galaxies and bins in a circle of virial radius ################ 
        
        index_circ = (radio_sqr < rproy_gr)
        alfa_circ, delta_circ, mabs_gr_cir, z_gal_cir = alfa_tmp[index_circ], delta_tmp[index_circ], mabs_HOD[index_circ], z_gal[index_circ]

        index_pixel_circ = ((radio_pixel_sqr < rproy_gr))# & (H.T > 0))
        centrox_pixel_circ, centroy_pixel_circ = xx_pixel[index_pixel_circ], yy_pixel[index_pixel_circ]
        
        ########### We count the number of galaxies and bins in a ring of radioint<r<radioext ################ 
        
        index_an = ((radio_sqr > radioint) & (radio_sqr < radioext))
        alfa_an, delta_an, mabs_gr_an, z_gal_an = alfa_tmp[index_an], delta_tmp[index_an], mabs_HOD[index_an], z_gal[index_an]

        index_pixel_an = ((radio_pixel_sqr > radioint) & (radio_pixel_sqr < radioext))# & (H.T > 0))
        centrox_pixel_an, centroy_pixel_an = xx_pixel[index_pixel_an], yy_pixel[index_pixel_an]      

        ################ Only for this test we set M_lim<-18 and the corresponding redshift cut #####################

        ii = ((mabs_gr_cir < mlim))
        jj = ((mabs_gr_an < mlim)) 

        ############## We calculate the Number of galaxies according to eq.2 of Rodriguez et al. 2015 ##################
        
        if (len(centrox_pixel_an)>0):
            N= len(alfa_circ[ii]) - len(alfa_an[jj]) * len(centrox_pixel_circ) / len(centrox_pixel_an)
        else:
            N= 0

    except ValueError:
        print("Oops! ValueError. Try again in index:",j)
        N=0

    return N_true,N, M_200[j]
