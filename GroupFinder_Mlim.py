#################load package ###########################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units  as u
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic
import random
from joblib import Parallel, delayed
from datetime import datetime
startTime = datetime.now()

print("entre")
################ Chosen parameters - Corrected for COSMO-DC2 (WMAP-7)###########################
cte_VEL = 299792.458
GRAD2RAD = 0.0174532925199433
cte_VEL = 299792.458
GRAD2RAD = 0.0174532925199433
Omega0    = 0.2648      
omegal    = 0.7352     
omegak    = 0.  
omegar    = 0.    
Omega = Omega0 + omegal
h0        = 100.  
hh = 0.71
G = 4.30091e-6
H0_km_s_Mpc = 100.0 * hh  


#slope,ordenada=1868175147.2854986, -42424487.794043064 # Old
slope,ordenada= 1634917221.9876266, -43596515.20330819 # New
###################### Load galaxy catalog ########################### 

################CATALOGO Galaxias####################### 
data_gal =  pd.read_table('../../Tabs/mock_21_large_z05_cosmoDC2.mpeg', delimiter=' ') #Anto
data_gal = data_gal[(data_gal['z,'] > 0.)]

z_gal_16 = 0.0563 # ----> M_abs = -16.0021
z_gal_17 = 0.0875 #----> M_abs = -17.0068
z_gal_18 = 0.1345 #----> M_abs = -18.0078

z_gal_19 = 0.2055
z_gal_20 = 0.3038
z_gal_21 = 0.4547

z_corte= [z_gal_17,z_gal_18,z_gal_19,z_gal_20]
m_corte = [-17.0,-18.0,-19.0,-20.0]
#r_lum_arr_f=[1.5,2.0,2.5]
#r_lum_arr_f=[2.5]

M_lim_arr = [-17, -18, -19, -20]
r_lum_arr_f=[2.5]
z_bin_arr=[0.05]

for m,z_gal_lim in enumerate(z_corte):
    for factor_raddi in r_lum_arr_f:
        for z_bin in z_bin_arr:
            galaxy_id = np.array(data_gal['#galid,'])
            alfa_gal = np.array(data_gal['ra,']*np.pi/180.)
            delta_gal = np.array(data_gal['dec,']*np.pi/180.)
            z_gal = np.array(data_gal['z,'])
            magr_gal = np.array(data_gal['mr_lsst,'])
            halo_id = np.array(data_gal['halo_id,'])
            halo_mass = np.array(data_gal['loghalo_mass,'])
            stellar_mass = np.array(data_gal['logstellar_mass,'])
            is_central = np.array(data_gal['is_central,'])
            
            ii=np.where(alfa_gal>np.pi)   
            alfa_gal[ii]=alfa_gal[ii]-2.*np.pi
            
            cosmo = FlatLambdaCDM(H0=hh*100, Om0=Omega0)
            d_com = cosmo.comoving_distance(z_gal).value
            mabs= magr_gal-25.0-5*np.log10(d_com*(1+z_gal))
            
            ################## Luminosity radius #############
            Lum=10**(-0.4*(mabs))
            r_Lum=(Lum-ordenada)/slope    
            r_Lum_ang = np.arctan(r_Lum/d_com)*180/np.pi  
            
            
            #################### Rho_crit and a dot ###################################
            aexp_gr= 1/(1+z_gal)
            E = np.sqrt(Omega0*aexp_gr**(-3) + omegar*aexp_gr**(-4) + omegak*aexp_gr**(-2) + omegal)
            H_km_s_kpc = (H0_km_s_Mpc * E) * 1e-3
            rho_crit = 3.0 * (H_km_s_kpc**2) / (8.0 * np.pi * G)
            
            r_200 = 1e-3*(3.0 * (10**halo_mass) / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)            
            
            ################ Groups ######################
            mlim =M_lim_arr[m]
            
            Mlim_central=-19.5
            
            galaxy_id_cand=galaxy_id[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]
            halo_id_cand = halo_id[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]
            alfa_cand = alfa_gal[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]
            delta_cand = delta_gal[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]
            z_cand = z_gal[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]
            magr_cand = magr_gal[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]
            M_200_cand = 10**halo_mass[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]
            stellar_cand = 10**stellar_mass[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]
            is_central_cand = is_central[(mabs<Mlim_central)&(z_gal<z_gal_lim) & (mabs<mlim)]
            r_Lum_cand = r_Lum[(mabs<Mlim_central)&(z_gal<z_gal_lim) & (mabs<mlim)]  
            r_Lum_ang_cand = r_Lum_ang[(mabs<Mlim_central)&(z_gal<z_gal_lim) & (mabs<mlim)]
            mabs_cand=mabs[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]
            r_200_cand=r_200[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]
            d_com_cand = d_com[(mabs<Mlim_central)& (z_gal<z_gal_lim) & (mabs<mlim)]/(1+z_cand)
            
            ############# Magnitude order first ############
            galaxy_id_cand = galaxy_id_cand[mabs_cand.argsort()]
            halo_id_cand = halo_id_cand[mabs_cand.argsort()]
            alfa_cand = alfa_cand[mabs_cand.argsort()]
            delta_cand = delta_cand[mabs_cand.argsort()]
            z_cand = z_cand[mabs_cand.argsort()]
            magr_cand = magr_cand[mabs_cand.argsort()]
            stellar_cand = stellar_cand[mabs_cand.argsort()]
            M_200_cand = M_200_cand[mabs_cand.argsort()]
            r_Lum_cand = r_Lum_cand[mabs_cand.argsort()]
            r_Lum_ang_cand = r_Lum_ang_cand[mabs_cand.argsort()]
            r_200_cand = r_200_cand[mabs_cand.argsort()]
            is_central_cand=is_central_cand[mabs_cand.argsort()]
            d_com_cand=d_com_cand[mabs_cand.argsort()]
            mabs_cand=mabs_cand[mabs_cand.argsort()]    
            
            ############ Run in serial ############
            
            Group_Descart,central_Descart=[],[]
           
            for j,gal_target in enumerate(galaxy_id_cand):
                if (len(galaxy_id_cand)>0):
                    ############ Targe Galaxy ######################                
                    is_central_target= is_central_cand[galaxy_id_cand==gal_target]
                    z_target = z_cand[galaxy_id_cand==gal_target]
                    alfa_target = alfa_cand[galaxy_id_cand==gal_target]        
                    delta_target = delta_cand[galaxy_id_cand==gal_target]
                    r_Lum_target = r_Lum_cand[galaxy_id_cand==gal_target]
                    d_com_target = d_com_cand[galaxy_id_cand==gal_target]  
        
                    if (len(np.where(galaxy_id_cand==gal_target)[0])>0):
                        ############# Save the id ######################
                        Group_Descart.append(gal_target)    
                        central_Descart.append(is_central_target)    
                        ######## Radius of each galaxy centre in each group (spherical coordinates) #########
                        radio = np.sin(delta_target)*np.sin(delta_cand)+np.cos(delta_target)*np.cos(delta_cand)*np.cos((alfa_target-alfa_cand))  
                        tang_tita = np.sqrt(1. - radio*radio) / radio 
                        radio_sqr = tang_tita * d_com_target
                        ########### We count the number of central galaxies and bins in a circle of virial radius ################ 
                        index_circ = np.where((radio_sqr < factor_raddi*r_Lum_target) & (radio_sqr > 0) & (np.abs(z_cand-z_target)<z_bin)) # No descarto a target misma        
                        if (len(index_circ[0])):        
                            galaxy_id_cand=np.delete(galaxy_id_cand, index_circ)
                            is_central_cand=np.delete(is_central_cand, index_circ)
                            z_cand=np.delete(z_cand, index_circ)
                            alfa_cand=np.delete(alfa_cand, index_circ)
                            delta_cand=np.delete(delta_cand, index_circ)
                            r_Lum_cand=np.delete(r_Lum_cand, index_circ)
                            d_com_cand=np.delete(d_com_cand, index_circ)
            
        
            data_file_path = '../../Tabs/CGFnew_fr_' + str(factor_raddi)+ '_fz_' + str(z_bin) + 'Mlim_'+str(m_corte[m])
            np.savez(data_file_path,Gal_id=Group_Descart,central_id=central_Descart)  
        
print(datetime.now() - startTime)


