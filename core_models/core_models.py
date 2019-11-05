import numpy as np
from astropy import units as u
from astropy import constants

def plummer_density(radii, mtot=1.0*u.M_sun, a_scale=0.1*u.pc, zh2=2.8,
                    mh=u.Da):
    """
    Return the density given a Plummer profile

    Parameters
    ----------
    radii : u.m equivalent
        The radii at which to evaluate the density
    mtot : u.M_sun equivalent
        The total mass of the core
    a_scale : u.m equivalent
        The characteristic length scale of the core
    zh2 : float
        The ratio of number of H2 particles to total mass, the mean molecular
        weight.  This is generally ~2.8 for determining the mass, but 2.33-2.37
        when determining pressure or other quantities that depend on all free
        particles.
    mh : u.g equivalent
        The mass of a Hydrogen atom

    Returns
    -------
    density : u.Quantity
        The density in cm**-3

    """

    rho = 3 * mtot / (4*np.pi*a_scale**3) * (1. + radii**2/a_scale**2)**(-2.5)
    nh2 = (rho / (mh*zh2)).to(u.cm**-3)
    return nh2

def broken_powerlaw(radii, rbreak=1.0*u.pc, power=-2.0, n0=1e5*u.cm**-3, **kwargs):
    """
    Return the density of a broken power law density profile where
    the central density is flat
    """

    if np.isscalar(radii):
        if radii < rbreak:
            return n0
        else:
            # n(rbreak) = n0
            n = n0 * (radii/rbreak)**power
            return n
    else:
        narr = u.Quantity(np.zeros(radii.shape), n0.unit)
        narr[radii<rbreak] = n0
        # this was previously
        # narr[r>=rbreak] = n0 * (r/rbreak)**power
        # which is wrong on two levels - first, the math is wrong, but second,
        # the assignment is supposed to be disallowed because of shape mismatch
        # numpy has failed me.
        narr[radii>=rbreak] = n0 * (radii[radii>=rbreak]/rbreak)**power
        return narr

#def bonnorebert(radii, ncenter=1e5*u.cm**-3, ximax=6.9, viso=0.24*u.km/u.s,
#                zh2=2.8, mh=u.Da):
#    """
#    approximation to a bonnor-ebert sphere using broken power law
#    6.9 taken from Alves, Lada, Lada B68 Nature paper 2001
#    viso is for zh2=2.8, T=20K
#    """
#
#    rbreak = ximax*(viso) / np.sqrt(4*np.pi*constants.G*ncenter*mh*zh2)
#
#    return broken_powerlaw(radii, rbreak=rbreak, n0=ncenter, power=-2.0)

def bonnorebert(radii, ncenter=None, mass=None, zh2=2.8, mh=u.Da, ximax=6.9,
                viso=0.24*u.km/u.s):
    """
    Approximation from
    www.am.ub.edu/~robert/Documents/bonnor-ebert.pdf

    """

    if mass is None and ncenter is None:
        raise ValueError("Must specify one of `mass` or `ncenter`")
    elif ncenter is not None:
        rho_center = ncenter * zh2 * mh
        r_center = ximax*(viso) / np.sqrt(4*np.pi*constants.G*ncenter*mh*zh2)
    elif mass is not None:
        rho_outer = mass / (1.18 * viso**3) * constants.G**1.5
        r_out = mass / (2.42 * viso**2 / constants.G)
        external_pressure = 1.41 * viso**8 / constants.G**3 / mass**2

        rho_center = rho_outer * (1+(r_out/(2.88*r_center)))**1.47
        r_center = ximax*(viso) / np.sqrt(4*np.pi*constants.G*rho_center)
    else:
        raise ValueError('not possible')

    return rho_center * (1+(radii/(2.88*r_center))**2)**-1.47
