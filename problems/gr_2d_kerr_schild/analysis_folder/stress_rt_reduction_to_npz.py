from scipy.integrate import simpson
import numpy as np
import sys

import pplotutil as pu

import datalib_gr_ks as ks

# This function now provided by datalib_gr_ks
# # Calculate T^mu_nu for the electromagnetic field
# def calc_Tmunu_em(data):
#     Dru  = data.E1
#     Dthu = data.E2
#     Dphu = data.E3
#     Bru  = data.B1
#     Bthu = data.B2
#     Bphu = data.B3
# 
#     Erd  = data.Ed1
#     Ethd = data.Ed2
#     Ephd = data.Ed3
#     Hrd  = data.Hd1
#     Hthd = data.Hd2
#     Hphd = data.Hd3
# 
#     DdotE = Dru*Erd + Dthu*Ethd + Dphu*Ephd
#     BdotH = Bru*Hrd + Bthu*Hthd + Bphu*Hphd
# 
#     alpha = ks.alpha(data._rv, data._thetav, data.a) # np.sqrt(-1.0/data.g_upper[:,:,0,0])
#     sgam  = ks.gmsqrt(data._rv, data._thetav, data.a)
# 
#     Tmuunud = np.zeros( (Dru.shape[0], Dru.shape[1], 4, 4) )
# 
#     # Use the formulae in Komissarov 2004, MNRAS 350 427
#     # T^t_t
#     Tmuunud[:,:,0,0] = -1/(2*alpha) * (DdotE + BdotH)
# 
#     # T^i_t
#     ixnz = np.where(sgam > 0.0)
#     Tmuunud[:,:,1,0][ixnz] = -1.0/(sgam*alpha)[ixnz] * (Ethd*Hphd - Ephd*Hthd)[ixnz]
#     Tmuunud[:,:,2,0][ixnz] =  1.0/(sgam*alpha)[ixnz] * (Erd *Hphd - Ephd*Hrd )[ixnz]
#     Tmuunud[:,:,3,0][ixnz] = -1.0/(sgam*alpha)[ixnz] * (Erd *Hthd - Ethd*Hrd )[ixnz]
# 
#     # T^t_i
#     Tmuunud[:,:,0,1] =  sgam/alpha * (Dthu*Bphu - Dphu*Bthu)
#     Tmuunud[:,:,0,2] = -sgam/alpha * (Dru *Bphu - Dphu*Bru)
#     Tmuunud[:,:,0,3] =  sgam/alpha * (Dru *Bthu - Dthu*Bru)
# 
#     # T^i_j
#     # Diagonal terms
#     Tmuunud[:,:,1,1] = -1.0/alpha * (Dru *Erd  + Bru *Hrd ) - Tmuunud[:,:,0,0]
#     Tmuunud[:,:,2,2] = -1.0/alpha * (Dthu*Ethd + Bthu*Hthd) - Tmuunud[:,:,0,0]
#     Tmuunud[:,:,3,3] = -1.0/alpha * (Dphu*Ephd + Bphu*Hphd) - Tmuunud[:,:,0,0]
#     # Off-diagonal terms
#     Tmuunud[:,:,1,2] = -1.0/alpha * (Dru *Ethd + Bru *Hthd)
#     Tmuunud[:,:,1,3] = -1.0/alpha * (Dru *Ephd + Bru *Hphd)
#     Tmuunud[:,:,2,3] = -1.0/alpha * (Dthu*Ephd + Bthu*Hphd)
#     Tmuunud[:,:,2,1] = -1.0/alpha * (Dthu*Erd  + Bthu*Hrd )
#     Tmuunud[:,:,3,1] = -1.0/alpha * (Dphu*Erd  + Bphu*Hrd )
#     Tmuunud[:,:,3,2] = -1.0/alpha * (Dphu*Ethd + Bphu*Hthd)
# 
#     return Tmuunud

# scipy now has this function, but I don't have access to the latest version
def cumulative_simpson_bootleg(y, x):
    return np.asarray([0] + [simpson(y[:i], x=x[:i])
                       for i in range(2, len(y) + 1)])

def replace_inf_with_zero(a):
    return np.where(np.isinf(a), np.zeros(a.shape), a)

# Calculate T^mu_nu of the plasma
def calc_Tmunu_pl(data):
    # g_upper diverges close to the axis; replacing those values with zero
    return np.einsum(
        "ijab,ijbc->ijac", replace_inf_with_zero(data.g_upper),
        data.stress_e + data.stress_p
    )

def calc_ptcl_flux_d(data, species = "both"):
    emask = 0.0
    pmask = 0.0
    if species == "both":
        return data.flux_lower
    elif species == "electrons":
        emask = 1.0
    elif species == "positrons":
        pmask = 1.0
    else:
        msg = "arg 'species' must be one of ['both', 'electrons', 'positrons']"
        msg += ". Got %s" % species
        raise ValueError(msg)
    return np.stack(
        [
            emask * data.num_e   + pmask * data.num_p,
            emask * data.flux_e1 + pmask * data.flux_p1,
            emask * data.flux_e2 + pmask * data.flux_p2,
            emask * data.flux_e3 + pmask * data.flux_p3,
        ],
        axis=-1
    )


def calc_ptcl_flux_u(data, species = "both"):
    # g_upper diverges close to the axis; replacing those values with zero
    return np.einsum(
        "ijab,ijb->ija",
        replace_inf_with_zero(data.g_upper),
        calc_ptcl_flux_d(data, species = species)
    )

def calc_conserved_quantities_r(data, rtdata, step, i0 = 0, verbose = True):
    if verbose:
        print("Working at dump step: " + str(step))

    data.load(step)

    R     = data._rv
    Theta = data._thetav
    
    sg   = ks.gsqrt(R, Theta, data.a)
    sgam = ks.gmsqrt(R, Theta, data.a)
    
    Tmuunud_EM = ks.calc_Tmunu_em(data)
    Tmuunud_PL = calc_Tmunu_pl(data)
    Nmuu       = calc_ptcl_flux_u(data)
    Nmud       = calc_ptcl_flux_d(data)

    Nmuu_el    = calc_ptcl_flux_u(data, species = "electrons")
    
    Ed1     = data.Ed1
    Ed2     = data.Ed2
    Ed3     = data.Ed3
    Ju1     = data.J1
    Ju2     = data.J2
    Ju3     = data.J3
    rhofido = data.Rho_total
    Bu1     = data.B1
    Bu2     = data.B2

    heating = Ed1*Ju1 + Ed2*Ju2 + Ed3*Ju3
    torque  = rhofido * Ed3 + sgam*(Ju1*Bu2 - Ju2*Bu1)
    
    Umuu       = data.fluid_u_upper
    Umud       = data.fluid_u_lower
    bhatmuu    = data.fluid_b_upper
    bhatmud    = np.einsum("ijkl,ijl->ijk", data.g_lower, bhatmuu)
    Tmudnud    = data.stress_e + data.stress_p
    
    bmuu       = data.frf_B
    bmag2      = data.inner_product_4d_contravariant(bmuu, bmuu)
    bmud       = data.lower_4d_vec(bmuu)
    
    # g_upper diverges close to the axis; replacing those values with zero
    Deltamuunuu = replace_inf_with_zero(data.g_upper + np.einsum("ijm,ijn->ijmn", Umuu   , Umuu   ))
    bbhatmuunuu =                       Deltamuunuu  - np.einsum("ijm,ijn->ijmn", bhatmuu, bhatmuu)
    
    # Heat flux
    qmuu        = np.einsum("ijma,ijb,ijab->ijm", Deltamuunuu, Umuu, Tmudnud)
    qmud        = data.lower_4d_vec(qmuu)
    
    # (T_particle)^mu_nu contains the heat flux term q^mu u_nu + u^mu q_nu
    Hmuunud     = np.einsum("ija,ijb->ijab", qmuu, Umud) + \
                  np.einsum("ija,ijb->ijab", Umuu, qmud)
    
    delta4      = np.diag([1., 1., 1., 1.])
    delta4      =          np.einsum("mn,ijmn->ijmn", delta4, np.ones(data.g_upper.shape))
    Deltamuunud = delta4 + np.einsum("ijm,ijn->ijmn", Umuu  , Umud                       )
    
    edens_proper =          np.einsum("ijk,ijl,ijkl->ij",  Umuu   , Umuu   , Tmudnud)
    ppara        =          np.einsum("ijk,ijl,ijkl->ij",  bhatmuu, bhatmuu, Tmudnud)
    pperp        = (1./2) * np.einsum("ijkl,ijkl->ij", bbhatmuunuu, Tmudnud)
    deltap       = pperp - ppara
    ptot         = (1./3) * (2*pperp + ppara)
    
    enthalpy_density = edens_proper + ptot
    
    # Ballistic transport of energy-momentum for (T_particle)^mu_nu includes only the particle
    # enthalpy density; ballistic transport for (T_MHD)^mu_nu also includes the magnetic enthalpy
    Nmuunud    = np.einsum("ij,ija,ijb->ijab", enthalpy_density        , Umuu   , Umud   )
    NBmuunud   = np.einsum("ij,ija,ijb->ijab", enthalpy_density + bmag2, Umuu   , Umud   )
    
    # T^mu_nu due to pressure anisotropy is -deltap*(bhat^mu bhat_nu - 1./3 Delta^mu_nu)
    Bmuunud    = np.einsum("ij,ija,ijb->ijab", -deltap                 , bhatmuu, bhatmud)
    Bmuunud    = Bmuunud + np.einsum("ij,ijab->ijab", deltap / 3., Deltamuunud)
    
    # Maxwell stress is just -b^mu b_nu
    Mmuunud    = - np.einsum("ija,ijb->ijab", bmuu, bmud)
    
    dTmuunud_PL = Tmuunud_PL - (
        np.einsum("ij,ija,ijb->ijab", edens_proper, Umuu,  Umud) +
        np.einsum("ij,ijab->ijab"   , ptot        , Deltamuunud) + 
        Bmuunud +
        Hmuunud
    )
    dTmuunud_EM = Tmuunud_EM - (
        np.einsum("ij,ijab->ijab"   , -0.5*bmag2  , delta4     ) +
        np.einsum("ij,ijab->ijab"   , bmag2       , Deltamuunud) +
        Mmuunud
    )
    
    r     = R[0,:]
    theta = Theta[:,0]

    thbuffer = 0.0  # 0.1*np.pi
    mask = np.ones(R.shape)
    mask[np.where((Theta > np.pi - thbuffer) | (Theta < thbuffer))] = 0.0
    
    r_axis     = 1
    theta_axis = 0
    
    for key in rtdata.keys():
        #=============================================================================
        # For fluxes, use a sign convention where inflowing flux (i.e. flux in the
        # -r direction) is positive.
        # Note that T^r_t is already positive for inflowing energy flux
        # However, T^r_phi is NOT already positive.
        # (This is because energy-at-infinity is -T^t_t; the flux is -T^r_t)
        if   key == "Trutd_EM":
            sign = 1.0
            r_array = sign * 2 * np.pi * simpson(
                Tmuunud_EM[:,:,1,0] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Trutd_PL":
            sign = 1.0
            r_array = sign * 2 * np.pi * simpson(
                Tmuunud_PL[:,:,1,0] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Ttutd_EM":
            sign = -1.0
            integrand = sign * 2 * np.pi * simpson(
                Tmuunud_EM[:,:,0,0] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Ttutd_PL":
            sign = -1.0
            integrand = sign * 2 * np.pi * simpson(
                Tmuunud_PL[:,:,0,0] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Truphid_EM":
            sign = -1.0
            r_array = sign * 2 * np.pi * simpson(
                Tmuunud_EM[:,:,1,3] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Truphid_PL":
            sign = -1.0
            r_array = sign * 2 * np.pi * simpson(
                Tmuunud_PL[:,:,1,3] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Ttuphid_EM":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                Tmuunud_EM[:,:,0,3] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Ttuphid_PL":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                Tmuunud_PL[:,:,0,3] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Nru":
            sign = -1.0
            r_array = sign * 2 * np.pi * simpson(
                Nmuu[:,:,1] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Ntu":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                Nmuu[:,:,0] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Nru_el":
            sign = -1.0
            r_array = sign * 2 * np.pi * simpson(
                Nmuu_el[:,:,1] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Ntu_el":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                Nmuu_el[:,:,0] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Nrutd":
            sign = 1.0
            r_array = sign * 2 * np.pi * simpson(
                NBmuunud[:,:,1,0] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Ntutd":
            sign = -1.0
            integrand = sign * 2 * np.pi * simpson(
                NBmuunud[:,:,0,0] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)    
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Nruphid":
            sign = -1.0
            r_array = sign * 2 * np.pi * simpson(
                NBmuunud[:,:,1,3] * sg * mask, x=theta, axis=theta_axis
            )            
        elif key == "Ntuphid":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                NBmuunud[:,:,0,3] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)   
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Brutd":
            sign = 1.0
            r_array = sign * 2 * np.pi * simpson(
                Bmuunud[:,:,1,0] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Btutd":
            sign = -1.0
            integrand = sign * 2 * np.pi * simpson(
                Bmuunud[:,:,0,0] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)    
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Bruphid":
            sign = -1.0
            r_array = sign * 2 * np.pi * simpson(
                Bmuunud[:,:,1,3] * sg * mask, x=theta, axis=theta_axis
            )            
        elif key == "Btuphid":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                Bmuunud[:,:,0,3] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)   
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Mrutd":
            sign = 1.0
            r_array = sign * 2 * np.pi * simpson(
                Mmuunud[:,:,1,0] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Mtutd":
            sign = -1.0
            integrand = sign * 2 * np.pi * simpson(
                Mmuunud[:,:,0,0] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)    
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Mruphid":
            sign = -1.0
            r_array = sign * 2 * np.pi * simpson(
                Mmuunud[:,:,1,3] * sg * mask, x=theta, axis=theta_axis
            )            
        elif key == "Mtuphid":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                Mmuunud[:,:,0,3] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)   
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Hrutd":
            sign = 1.0
            r_array = sign * 2 * np.pi * simpson(
                Hmuunud[:,:,1,0] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Htutd":
            sign = -1.0
            integrand = sign * 2 * np.pi * simpson(
                Hmuunud[:,:,0,0] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)    
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Hruphid":
            sign = -1.0
            r_array = sign * 2 * np.pi * simpson(
                Hmuunud[:,:,1,3] * sg * mask, x=theta, axis=theta_axis
            )            
        elif key == "Htuphid":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                Hmuunud[:,:,0,3] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)   
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Rrutd_PL":
            sign = 1.0
            r_array = sign * 2 * np.pi * simpson(
                dTmuunud_PL[:,:,1,0] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Rtutd_PL":
            sign = -1.0
            integrand = sign * 2 * np.pi * simpson(
                dTmuunud_PL[:,:,0,0] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)    
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Rruphid_PL":
            sign = -1.0
            r_array = sign * 2 * np.pi * simpson(
                dTmuunud_PL[:,:,1,3] * sg * mask, x=theta, axis=theta_axis
            )            
        elif key == "Rtuphid_PL":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                dTmuunud_PL[:,:,0,3] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)   
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Rrutd_EM":
            sign = 1.0
            r_array = sign * 2 * np.pi * simpson(
                dTmuunud_EM[:,:,1,0] * sg * mask, x=theta, axis=theta_axis
            )
        elif key == "Rtutd_EM":
            sign = -1.0
            integrand = sign * 2 * np.pi * simpson(
                dTmuunud_EM[:,:,0,0] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)    
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Rruphid_EM":
            sign = -1.0
            r_array = sign * 2 * np.pi * simpson(
                dTmuunud_EM[:,:,1,3] * sg * mask, x=theta, axis=theta_axis
            )            
        elif key == "Rtuphid_EM":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                dTmuunud_EM[:,:,0,3] * sg * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)   
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Bflux":
            r_array = 0.5 * 2 * np.pi * np.sum(
                np.abs(data.B1) * sgam * mask * data._dtheta,
                axis=theta_axis,
            )
        elif key == "Bflux_signed":
            mask_here = np.where(Theta <= np.pi / 2., mask, np.zeros(Theta.shape))
            r_array = 2 * np.pi * np.sum(
                # data.B1[idx] * sgam[idx] * mask[idx] * data._dtheta,
                data.B1 * sgam * mask_here * data._dtheta,
                axis=theta_axis,
            )
        elif key == "Dflux":
            r_array = 2 * np.pi * np.sum(
                data.E1 * sgam * mask * data._dtheta,
                axis=theta_axis,
            )
        elif key == "Heat2Ptcl":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                heating * sgam * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)   
            # r_array = cumulative_simpson_bootleg(integrand, x=r)
            r_array = integrand
        elif key == "Stress2Ptcl":
            sign = 1.0
            integrand = sign * 2 * np.pi * simpson(
                torque * sgam * mask, x=theta, axis=theta_axis
            )
            # r_array = cumulative_simpson(integrand, x=r, axis=r_axis, initial=0.0)   
            # r_array = cumulative_simpson_bootleg(integrand, x=r)  
            r_array = integrand
        else:
            msg =  "Unrecognized quantity: " + key
            msg += " Cannot compute."
            raise ValueError(msg)
            
        if np.isnan(r_array).any():
            print("Warning: NaNs at step %d in array %s" % (step, key))
        if np.isinf(r_array).any():
            print("Warning: +/- inf at step %d in array %s" % (step, key))
            
        # print("Key = %s; step = %d; maxval = %.3g" % (key, step, np.max(r_array)))
        rtdata[key][step - i0, :] = r_array


def calc_conserved_quantities_rt(data, steps, allsteps = None, verbose = True):
    rtdata = {}

    if allsteps is None:
        allsteps = steps

    array_factory = lambda : np.zeros( (len(allsteps), data._rv.shape[1]) )

    #=============================================================================
    # At the end of the calculation, we will have arrays of the fluxes of several
    # conserved quantities through concentric spherical shells
    #
    # We want the radial fluxes of energy T^r_t, angular momentum, T^r_phi, and
    # particle number N^r
    #
    # For each flux, we will also calculate the integral of the quantity contained
    # in a spherical volume. We will calculate the total energy T^t_t, angular
    # momentum T^t_phi, and particle number (rest mass) N^0.
    #
    # Update: I have changed this routine to just calculate the integral of
    # N^t, T^t_t, and T^t_phi on spherical surfaces. You have to take the
    # radial integral yourself in postprocessing if you want to know the total
    # charge of each conserved current in a spherical volume.
    #
    # Energy and angular momentum we will decompose into plasma and field
    # components
    rtdata["Trutd_EM"]   = array_factory()
    rtdata["Trutd_PL"]   = array_factory()
    rtdata["Ttutd_EM"]   = array_factory()
    rtdata["Ttutd_PL"]   = array_factory()

    rtdata["Truphid_EM"] = array_factory()
    rtdata["Truphid_PL"] = array_factory()
    rtdata["Ttuphid_EM"] = array_factory()
    rtdata["Ttuphid_PL"] = array_factory()

    rtdata["Nru"]        = array_factory()
    rtdata["Ntu"]        = array_factory()
    # For particle number flux, split into total and electrons so as to be able
    # to reconstruct charge density (using, e.g.,
    #     positron_flux = total_flux - electron_flux)
    rtdata["Nru_el"]        = array_factory()
    rtdata["Ntu_el"]        = array_factory()

    # These define the stress carried by the bulk flow
    # N^mu_nu = (e + P + b^2) u^mu u_nu
    rtdata["Nrutd"]   = array_factory()
    rtdata["Ntutd"]   = array_factory()
    rtdata["Nruphid"] = array_factory()
    rtdata["Ntuphid"] = array_factory()

    # These define the energy and angular momentum flux carried by the pressure anisotropy.
    # B^mu_nu = - deltap (bhat^mu bhat_nu - Delta^mu_nu / 3)
    rtdata["Brutd"]   = array_factory()
    rtdata["Btutd"]   = array_factory()
    rtdata["Bruphid"] = array_factory()
    rtdata["Btuphid"] = array_factory()

    # Energy and angular momentum flux due to Maxwell stress
    # M^mu_nu = - b^mu b_nu
    rtdata["Mrutd"]   = array_factory()
    rtdata["Mtutd"]   = array_factory()
    rtdata["Mruphid"] = array_factory()
    rtdata["Mtuphid"] = array_factory()

    # Energy and angular momentum flux due to heat flux
    # H^mu_nu = q^mu u_nu + u^mu q_nu
    rtdata["Hrutd"]   = array_factory()
    rtdata["Htutd"]   = array_factory()
    rtdata["Hruphid"] = array_factory()
    rtdata["Htuphid"] = array_factory()

    # Residual fluxes for particles
    # The residual flux tensor for particles is defined as
    # (R_PL)^mu_nu = (T_PL)^mu_nu -
    #     (e u^mu u_nu + P Delta^mu_nu - deltap (bhat^mu bhat_nu - Delta^mu_nu / 3) + q^mu n_nu + u^mu q_nu)
    rtdata["Rrutd_PL"]   = array_factory()
    rtdata["Rtutd_PL"]   = array_factory()
    rtdata["Rruphid_PL"] = array_factory()
    rtdata["Rtuphid_PL"] = array_factory()

    # Residual fluxes for EM fields
    # The residual flux tensor for fields is defined as
    # (R_EM)^mu_nu = (T_EM)^mu_nu - (-b^2 g^mu_nu / 2 + b^2 Delta^mu_nu - b^nu b_nu)
    rtdata["Rrutd_EM"]   = array_factory()
    rtdata["Rtutd_EM"]   = array_factory()
    rtdata["Rruphid_EM"] = array_factory()
    rtdata["Rtuphid_EM"] = array_factory()

    # Ohmic heating and Lorentz-force stress
    # Positive values contribute to d(-T^t_t)/dt > 0 and d(T^t_phi)/dt > 0
    # for heating and Lorentz-force stress, respectively.
    rtdata["Heat2Ptcl"]   = array_factory()
    rtdata["Stress2Ptcl"] = array_factory()

    # Calculate magnetic flux penetrating radius r
    rtdata["Bflux"]        = array_factory()
    rtdata["Bflux_signed"] = array_factory()

    # And the electric flux through radius r (charge interior of r)
    rtdata["Dflux"] = array_factory()

    for step in steps:
        calc_conserved_quantities_r(data, rtdata, step, i0 = allsteps[0], verbose = verbose)

    # np.savez("conserved_quantities_tr.npz",
    #     r=data._rv, dumps=steps, **rtdata
    # )
    return rtdata

def calc_conserved_quantities_rt_mpi(data, allsteps, fname):
    # allsteps = data.fld_steps
    # if nsteps is not None:
    #     allsteps = allsteps[:nsteps]
    allsteps = np.array(allsteps)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # print("My rank is ", rank)
    
    mysteps = allsteps[rank::size]
    # print("My rank is ", rank, ". My steps are ", mysteps)

    if rank == 0:
        print(f"Starting reduction calculation on dumps [{allsteps[0]}, {allsteps[-1]}].")
        print(f"Using {size} parallel processes.")
        print("Here we go...")

    rtdata = calc_conserved_quantities_rt(data, mysteps, allsteps = allsteps)

    # # Don't want to reduce these arrays, so pop them off the dictionary
    # r = rtdata.pop("r")
    # dumps = rtdata.pop("dumps")

    # Create a master dictionary to reduce everything to
    master_rtdata = {}
    for key in rtdata:
        master_rtdata[key] = np.zeros(rtdata[key].shape)

    if size > 1:
        # Reduce!
        keys = sorted(list(rtdata.keys()))
        for key in keys:
            comm.Reduce(rtdata[key], master_rtdata[key], op=MPI.SUM, root=0)
    else:
        print("Since only on one process, skipping MPI reduce")
        keys = sorted(list(rtdata.keys()))
        for key in keys:
            master_rtdata[key] = rtdata[key]

    if rank == 0:
        np.savez(fname,
            r = data._rv[0,:], dumps = allsteps,
            t = allsteps * data.conf["fld_output_interval"] * data.conf['dt'],
            dt = data.conf['dt'],
            **master_rtdata,
        )


def load_rt_data(aperture_data, rt_file):
    rh = 1.0 + np.sqrt(1.0 - aperture_data.a**2)

    rtdata = {}
    with np.load(rt_file) as rtdfile:
        for key in rtdfile:
            rtdata[key] = rtdfile[key]
    
    r = rtdata["r"]
    ridx = np.where(r > rh)[0]
    for key in rtdata:
        if key != "dumps" and key != "r" and key != "dt" and key != "t":
            # print("Limiting to radii outside horizon for array: " + key)
            rtdata[key] = rtdata[key][:, ridx]
            # print("(max, min) at key %s are: (%.3g, %.3g)" % (key, rtdata[key].min(), rtdata[key].max()))

    rtdata["r"] = r[ridx]

    return rtdata

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("usage: mpirun -np <nproc> python stress_rt_reduction_to_npz.py <simulation data directory>")
        print("           [it0 [it1]]")
        print("  If it0 given, process dumps up through dump it0 [exclusive]")
        print("  If it0 and it1 given, process dumps it0 [inclusive] to it1 [exclusive]")
        sys.exit(0)

    args, kwargs = pu.getArgsAndKwargs(sys.argv[1:])
    datadir = args[0]

    data = ks.DataKerrSchild(datadir)

    steps = data.fld_steps

    it0 = 0
    it1 = len(steps)
    if len(args) == 2:
        it0 = 0
        it1 = int(args[1])
    elif len(args) == 3:
        it0 = int(args[1])
        it1 = int(args[2])

    fname = "conserved_quantities_tr_parallel.npz"
    if "save" in kwargs:
        fname = kwargs["save"]

    calc_conserved_quantities_rt_mpi(data, steps[it0:it1], fname)
