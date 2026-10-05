import numpy as np

###### CASO 1 (background_method)
#rproy_gr = factor_r * r_200[j]  ## r200 del grupo j
#rproy_gr = max(rproy_gr, 0.05)
#rproy_gr = min(rproy_gr, factor_rmax)
#
#r_int = 1.5 * rproy_gr
#r_ext = 3.5 * rproy_gr

###### CASO 2 (background_method_new)
## r_int y r_ext igual al Caso 1
# poner  apply_mlim=False


###### CASO 3 (background_method_rmax)
#rproy_gr = factor_r * r_200[j]
#rproy_gr = max(rproy_gr, 0.05)
#rproy_gr = min(rproy_gr, factor_rmax)
#
#r_int = rproy_gr + 1.
#r_ext = rproy_gr + 2.

###### CASO 4 (background_method_rgroup)
#rproy_gr = factor_r * rgroup
#rproy_gr = max(rproy_gr, 0.05)
#
#r_int = 1.5 * rproy_gr
#r_ext = 3.5 * rproy_gr


## FUNCIÓN GENERAL
def background_method(
    j, mlim, halo_gr_id, d_com_gr,
    rproy_gr, r_int, r_ext,
    delta_gr, alfa_gr, M_200, halo_id, z_gr,
    mabs, magr_gal, delta_gal, alfa_gal, z_gal,
    apply_mlim=True ):

    # Number of true galaxies in the group
    index_len = np.where((halo_gr_id[j] == halo_id) & (mabs < mlim))
    N_true = len(index_len[0])

    # Absolute magnitude of galaxies in the group frame
    mabs_HOD = magr_gal - 25.0 - 5*np.log10(d_com_gr[j] * (1 + z_gr[j]))

    # Minimum projected radius
    rproy_gr = max(rproy_gr, 0.05)  ## rproy_gr del grupo j-esimo, es un escalar

    # Physical distance to group
    alfa_tmp = alfa_gal
    delta_tmp = delta_gal

    d_phys_gr = d_com_gr[j] / (1 + z_gr[j])

    # Projected distance of galaxies from group centre
    radio = np.sin(delta_gr[j]) * np.sin(delta_gal) + np.cos(delta_gr[j]) * np.cos(delta_gal) * np.cos(alfa_gr[j] - alfa_gal)
    tang_tita = np.sqrt(1. - radio*radio) / radio
    radio_sqr = tang_tita * d_phys_gr

    # Angular size of the outer region
    radioext_rad = np.arctan(r_ext / d_phys_gr)

    lamin = alfa_gr[j] - radioext_rad
    lamax = alfa_gr[j] + radioext_rad

    etamin = delta_gr[j] - radioext_rad / np.cos(alfa_gr[j])
    etamax = delta_gr[j] + radioext_rad / np.cos(alfa_gr[j])

    xmin = lamin
    xmax = lamax
    ymin = etamin
    ymax = etamax

    # Resolution
    nx = 56**2
    ny = 56**2

    try:

        xi = np.linspace(np.floor(xmin), np.ceil(xmax), nx)
        yi = np.linspace(np.floor(ymin), np.ceil(ymax), ny)

        # Bin centres
        centrox_pixel = xi[:-1] + (xi[1:] - xi[:-1]) / 2
        centroy_pixel = yi[:-1] + (yi[1:] - yi[:-1]) / 2

        # Pixel mesh
        xx_pixel, yy_pixel = np.meshgrid(centrox_pixel,centroy_pixel)

        # Projected distance of pixels from group centre
        radio_pixel = np.sin(delta_gr[j]) * np.sin(yy_pixel) + np.cos(delta_gr[j]) * np.cos(yy_pixel) * np.cos(alfa_gr[j] - xx_pixel)
        tang_tita_pixel = np.sqrt(1. - radio_pixel*radio_pixel) / radio_pixel
        radio_pixel_sqr = tang_tita_pixel * d_phys_gr


        # Galaxies inside rproy_gr
        index_circ = (radio_sqr < rproy_gr)
        alfa_circ, delta_circ, mabs_gr_cir, z_gal_cir = alfa_tmp[index_circ], delta_tmp[index_circ], mabs_HOD[index_circ], z_gal[index_circ]

        # Pixels inside rproy_gr
        index_pixel_circ = (radio_pixel_sqr < rproy_gr)
        centrox_pixel_circ, centroy_pixel_circ = xx_pixel[index_pixel_circ], yy_pixel[index_pixel_circ]

        # Galaxies in background ring
        index_an = ((radio_sqr > r_int) & (radio_sqr < r_ext))
        alfa_an, delta_an, mabs_gr_an, z_gal_an = alfa_tmp[index_an], delta_tmp[index_an], mabs_HOD[index_an], z_gal[index_an]
    
        # Pixels in background ring
        index_pixel_an = ((radio_pixel_sqr > r_int) & (radio_pixel_sqr < r_ext))
        centrox_pixel_an, centroy_pixel_an = xx_pixel[index_pixel_an], yy_pixel[index_pixel_an]  

        # Magnitude cut
        if apply_mlim:

            ii = mabs_gr_cir < mlim
            jj = mabs_gr_an < mlim

        else:

            ii = np.ones(len(mabs_gr_cir), dtype=bool)
            jj = np.ones(len(mabs_gr_an), dtype=bool)


        # Background-corrected number of galaxies
        if len(centrox_pixel_an) > 0:

            N = len(alfa_circ[ii]) - len(alfa_an[jj]) * len(centrox_pixel_circ) / len(centrox_pixel_an)
            
        else:
            N = 0

    except ValueError:

        print("Oops! ValueError. Try again in index:", j)
        N = 0

    return N_true, N, M_200[j]


