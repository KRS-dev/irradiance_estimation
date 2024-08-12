from typing import OrderedDict, Tuple
import numpy as np
import pandas as pd
import torch


def get_satellite_look_angles(phi, theta, degree=False, dtype=torch.float32) -> torch.Tensor: # lat, lon
    """Calculate the azimuth and solar zenith angles of the SEVIR satellite from a given point on the Earth
    Uses spherical geometry to approximate the angles.
    Method based on the paper: "Determination of Look Angles to Geostationary Communication Satellites" by T. Soler, DOI: 10.1061/(ASCE)0733-9453(1994)120:3(115)

    Parameters
    ----------
    phi : float or int or torch.tensor like
        Latitude Spherical coordinate
    theta : float or int or torch.tensor like
        Longitude Spherical coordinate
    degree : bool, optional
        use degree input/output or radians, by default False

    Returns
    -------
    alpha : torch.tensor
        Satellite azimuth angle
    z : torch.tensor
        Satellite zenith angle
    """

    R = 6371*10**3 # m
    r = 35800*10**3  + R# m Geostationary orbit height
    theta_seviri = 0 # degree Longitude orbit Seviri
    
    if isinstance(phi, torch.Tensor) and isinstance(theta, torch.Tensor):
        if phi.shape != theta.shape:
            raise ValueError("phi and theta must have the same shape")
    elif isinstance(phi, (int, float)) and isinstance(theta, (int, float)):
        phi = torch.tensor([phi])
        theta = torch.tensor([theta])

    

    if degree:
        phi = torch.deg2rad(phi)
        theta = torch.deg2rad(theta)
    
    theta_east = torch.where(theta <0, 2*torch.pi + theta, theta) # convert to 0-2pi


    phi_abs = torch.abs(phi)
    
    gamma = torch.arccos(torch.cos(phi_abs)*torch.cos(theta_seviri -theta_east))

    d = r*(1 + (R/r)**2 - 2*R/r*torch.cos(gamma))**0.5
    z = torch.arcsin((r/d)*torch.sin(gamma))

    v = torch.pi/2 - z
    

    r = torch.abs(torch.tan(phi_abs)/torch.tan(gamma))
    beta = torch.arccos(
        torch.where(r > 1, 1, r) # torch.min(r, 1) clip to 1 to avoid nan due to numerical errors
        )


    # if (phi >=0) and (theta >= 0):
    #     alpha = beta + torch.pi
    # elif (phi < 0) and (theta >= 0):
    #     alpha = 2*torch.pi - beta
    # elif (phi < 0) and (theta < 0):
    #     alpha = beta
    # elif (phi >= 0) and (theta < 0):
    #     alpha = torch.pi - beta

    # The former but then vectorized
    alpha = torch.zeros_like(phi)
    alpha[torch.where((phi >=0) & (theta >= 0))] = beta[torch.where((phi >=0) & (theta >= 0))] + torch.pi
    alpha[torch.where((phi < 0) & (theta >= 0))] = 2*torch.pi - beta[torch.where((phi < 0) & (theta >= 0))]
    alpha[torch.where((phi < 0) & (theta < 0))] = beta[torch.where((phi < 0) & (theta < 0))]
    alpha[torch.where((phi >= 0) & (theta < 0))] = torch.pi - beta[torch.where((phi >= 0) & (theta < 0))]

    

    # if phi == 0 and theta == 0:
    #     alpha = 0 # NADIR, undefined in this case
    #     z = 0
    #     return alpha, z
    
    alpha[torch.where((phi == 0) & (theta == 0))] = 0 # NADIR, undefined in this case
    z[torch.where((phi == 0) & (theta == 0))] = 0 # NADIR, undefined in this case

    if degree:
        alpha = torch.rad2deg(alpha)
        z = torch.rad2deg(z)
        v = torch.rad2deg(v)


    return alpha.to(dtype), z.to(dtype) # azi, sza


def coscattering_angle(alpha_sat, z_sat, alpha_sun, z_sun, 
                       degree=False, dtype=torch.float32) -> torch.Tensor:
    """Calculates the scattering angle between the satellite and the sun

    Parameters
    ----------
    alpha_sat : float or torch.tensor(float)
        azimuth angle of the satellite
    z_sat : float or torch.tensor(float)
        satellite zenith angle
    alpha_sun : float or torch.tensor(float)
        azimuth angle of the sun
    z_sun : float or torch.tensor(float)
        solar zenith angle
    degree : bool, optional
        degree or radians input/output, by default False

    Returns
    -------
    torch.tensor
        coscattering angle in radians or degrees
    """

    if isinstance(alpha_sat, (int, float)):
        alpha_sat = torch.tensor([alpha_sat])
    if isinstance(z_sat, (int, float)):
        z_sat = torch.tensor([z_sat])
    if isinstance(alpha_sun, (int, float)):
        alpha_sun = torch.tensor([alpha_sun])
    if isinstance(z_sun, (int, float)):
        z_sun = torch.tensor([z_sun])
    
    if isinstance(alpha_sat, list):
        alpha_sat = torch.tensor(alpha_sat)
        z_sat = torch.tensor(z_sat)
        alpha_sun = torch.tensor(alpha_sun)
        z_sun = torch.tensor(z_sun)

    alpha_sat = alpha_sat.to(dtype)
    z_sat = z_sat.to(dtype)
    alpha_sun = alpha_sun.to(dtype)
    z_sun = z_sun.to(dtype)

    assert alpha_sat.shape == z_sat.shape and alpha_sun.shape == z_sun.shape, "Input arrays must have the same shape"
    
    if alpha_sat.shape != alpha_sun.shape:
        assert len(alpha_sat) == 1 or len(alpha_sun) == 1, "Only one of the input arrays can have multiple elements if they are not the same shape."

    if degree:
        alpha_sat = torch.deg2rad(alpha_sat)
        z_sat = torch.deg2rad(z_sat)
        alpha_sun = torch.deg2rad(alpha_sun)
        z_sun = torch.deg2rad(z_sun)


    vec_sat = torch.stack(
        [torch.sin(z_sat)*torch.sin(alpha_sat),
         torch.sin(z_sat)*torch.cos(alpha_sat),
         torch.cos(z_sat)],
         dim=0,
    ).squeeze().view(3,-1)

    vec_sun = torch.stack(
        [torch.sin(z_sun)*torch.sin(alpha_sun),
         torch.sin(z_sun)*torch.cos(alpha_sun),
         torch.cos(z_sun)], 
         dim=0,
    ).squeeze().view(3,-1)

    dot_product = torch.einsum('i...,i...->...', vec_sat, vec_sun) # elementwise dot product
    theta = torch.arccos(dot_product)

    if degree:
        theta = torch.rad2deg(theta)

    return theta