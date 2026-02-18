# HOD Background Subtraction

This repository provides a Python implementation of a statistical background subtraction technique to estimate the Halo Occupation Distribution (HOD) from projected galaxy counts in both spectroscopic and purely photometric surveys.

The original implementation of the background subtraction technique (BST) was developed in C by Facundo Rodriguez. 
This repository extends the method to photometric catalogues and provides a modular, user-friendly Python framework developed by Pedro Cataldi.

The method enables robust HOD measurements in the absence of full 3D membership information by statistically removing foreground and background contaminants using control fields and projected annular regions.

## Features

- Estimation of HOD from projected galaxy counts
- Statistical background subtraction using configurable control annuli
- Support for both spectroscopic and photometric catalogues
- Flexible magnitude cut selections
- Configurable projected central aperture radius
- Adjustable background annulus inner and outer radii
- Customisable halo mass binning
- Bootstrap and jackknife uncertainty estimation
- Modular pipeline suitable for simulations and survey data

## Complementary Method: CGF

The repository also includes a complementary **Central Galaxy Finder (CGF)** module, designed to be used alongside the BST framework.  

The CGF method identifies candidate central galaxies within halos or projected overdensities, allowing a consistent joint analysis of:
- Central occupation statistics
- Satellite occupation statistics
- Total HOD measurements

This makes the repository suitable for cluster, group, and protocluster studies where central identification and satellite background decontamination must be treated consistently.
