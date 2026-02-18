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
from Background_method import background_method,background_method_rmax
from datetime import datetime
startTime = datetime.now()

print("entre")
################ Chosen parameters - Corrected for COSMO-DC2 (WMAP-7)###########################
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
H0_km_s_Mpc = 100.0 * hh        # km/s/Mpc
###################### Load galaxy catalog ########################### 
data_gal =  pd.read_table('Tabs/mock_21_large_FZ_z05_cosmoDC2.mpeg', delimiter=' ') # FlexZBoost
w
#data_gal =  pd.read_table('Tabs/mock_21_large_z05_cosmoDC2.mpeg', delimiter=' ') 
data_gal = data_gal[(data_gal['z,'] > 0.) ]


galaxy_id = np.array(data_gal['#galid,']) ######## id ##########
halo_id = np.array(data_gal['halo_id,']) ########## halo id ############
is_central = np.array(data_gal['is_central,']) ############ central== 1, sat==0############

alfa_gal = np.array(data_gal['ra,']*np.pi/180.) #### sky coord ##########
delta_gal = np.array(data_gal['dec,']*np.pi/180.) ###### sky coord #########
z_gal = np.array(data_gal['z,']) ########### redshift ##########
#magr_ssd_gal = np.array(data_gal['mr_sdss,']) ########## magnitude sloan #######
magr_gal = np.array(data_gal['mr_lsst,']) ########### magnitude lsst #######

halo_mass = np.array(data_gal['loghalo_mass,']) ########## log stellar mass #########
stellar_mass = np.array(data_gal['logstellar_mass,']) ############ log halo mass #########

#################### Redifine azimutal angle #############################

ii=np.where(alfa_gal>np.pi)   
alfa_gal[ii]=alfa_gal[ii]-2.*np.pi

#########################################################################

cosmo = FlatLambdaCDM(H0=hh*100, Om0=Omega0)
d_com = cosmo.comoving_distance(z_gal).value
mabs= magr_gal-25.0-5*np.log10(d_com*(1+z_gal))

####################Cuts in redshift for each magnitude ###################
z_gal_16 = 0.0563 # ----> M_abs = -16.0021
z_gal_17 = 0.0875 #----> M_abs = -17.0068
z_gal_18 = 0.1345 #----> M_abs = -18.0078

z_gal_19 = 0.2055
z_gal_20 = 0.3038
z_gal_21 = 0.4547

z_cut_arr = [ z_gal_17, z_gal_18, z_gal_19, z_gal_20]
M_lim_arr = [-17, -18, -19, -20]

ring_radius=[1.5,3.5]

for m,z_gal_target in enumerate(z_cut_arr):
	mlim =M_lim_arr[m]    
	#for ring_radius in ring_radius_arr:
	print('entre Mlim =',mlim)    
	################ Groups ######################
	halo_gr_id = halo_id[(is_central==1) & (mabs<-19.5)& (z_gal<z_gal_target) & (mabs<mlim)]
	alfa_gr = alfa_gal[(is_central==1) & (mabs<-19.5)& (z_gal<z_gal_target) & (mabs<mlim)]
	delta_gr = delta_gal[(is_central==1) & (mabs<-19.5)& (z_gal<z_gal_target) & (mabs<mlim)]
	z_gr = z_gal[(is_central==1) & (mabs<-19.5)& (z_gal<z_gal_target) & (mabs<mlim)]
	magr_gr = magr_gal[(is_central==1) & (mabs<-19.5)& (z_gal<z_gal_target) & (mabs<mlim)]
	M_200 = 10**halo_mass[(is_central==1) & (mabs<-19.5)& (z_gal<z_gal_target) & (mabs<mlim)]
	stellar_gr = 10**stellar_mass[(is_central==1) & (mabs<-19.5)& (z_gal<z_gal_target) & (mabs<mlim)]
	mabs_gr=mabs[(is_central==1) & (mabs<-19.5)& (z_gal<z_gal_target) & (mabs<mlim)]
	
	############# Massive order first ############
	
	halo_gr_id = halo_gr_id[M_200.argsort()[::-1]]
	alfa_gr = alfa_gr[M_200.argsort()[::-1]]
	delta_gr = delta_gr[M_200.argsort()[::-1]]
	z_gr = z_gr[M_200.argsort()[::-1]]
	magr_gr = magr_gr[M_200.argsort()[::-1]]
	#magr_ssd_gr = magr_ssd_gr[M_200.argsort()[::-1]]
	stellar_gr = stellar_gr[M_200.argsort()[::-1]]
	mabs_gr=mabs_gr[M_200.argsort()[::-1]]
	M_200 = M_200[M_200.argsort()[::-1]]
	
	#################### Rho_crit and a dot ###################################
	aexp_gr= 1/(1+z_gr)
	E = np.sqrt(Omega0*aexp_gr**(-3) + omegar*aexp_gr**(-4) + omegak*aexp_gr**(-2) + omegal)
	H_km_s_kpc = (H0_km_s_Mpc * E) * 1e-3
	rho_crit = 3.0 * (H_km_s_kpc**2) / (8.0 * np.pi * G)
	
	r_200 = 1e-3*(3.0 * M_200 / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)
	d_com_gr = cosmo.comoving_distance(z_gr).value ################## co-moving distance #################33
	mabs_gr= magr_gr-25.0-5*np.log10(d_com_gr*(1+z_gr)) ################ absolute magnitude #################
	n_sample = len(z_gr)           
	
	step_size = n_sample // 4
	results_accum = []
	r_max=1.25
	for chunk_start in range(0, n_sample, step_size):
		chunk_end = min(chunk_start + step_size, n_sample)
		print(f"Processing {chunk_start} to {chunk_end} out of {n_sample} ({(chunk_end/n_sample)*100:.1f}%)")
		
		Result = Parallel(n_jobs=40, prefer="threads")(
			delayed(background_method_rmax)(
				j, mlim, z_gal_target, halo_gr_id, d_com_gr,'classic',ring_radius,r_max,1.0, r_200,
				delta_gr, alfa_gr, M_200, halo_id, z_gr,
				mabs, magr_gal, delta_gal, alfa_gal, z_gal
			) for j in range(chunk_start, chunk_end)
		)
		
		results_accum.extend(Result)

		# Save partial result
		part_file = f'Tabs/BST_DC2_{r_max}rmax_r200_Mlim_{mlim}_part_{chunk_start}_{chunk_end}.npz'
		Results_part = np.array(Result)
		np.savez(
			part_file,
			N_arr_18=Results_part[:,1],
			Mhalo_arr_18=Results_part[:,2],
			N_arr_true_18=Results_part[:,0].astype(int),
			halo_id=halo_gr_id[chunk_start:chunk_end]
		)
		print(f"Saved {part_file}")

	# Save full result at the end
	Results = np.array(results_accum)
	final_file = f'Tabs/BST_DC2_BPZ_r200_Mlim_{mlim}.npz'
	np.savez(
	final_file,
	N_arr_18=Results[:,1],
	Mhalo_arr_18=Results[:,2],
	N_arr_true_18=Results[:,0].astype(int),
	halo_id=halo_gr_id[:n_sample]
	)
	print(f"Final file saved: {final_file}")

    



