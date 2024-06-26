import numpy as np

def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
  """
  Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule 
  """
  
  # left and right energies
  en_L = P_L/(gamma-1)+0.5*rho_L * (vx_L**2+vy_L**2)
  en_R = P_R/(gamma-1)+0.5*rho_R * (vx_R**2+vy_R**2)

  # compute star (averaged) states
  rho_star  = 0.5*(rho_L + rho_R)
  momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
  momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
  en_star   = 0.5*(en_L + en_R)
  
  P_star = (gamma-1)*(en_star-0.5*(momx_star**2+momy_star**2)/rho_star)
  
  # compute fluxes (local Lax-Friedrichs/Rusanov)
  flux_Mass   = momx_star
  flux_Momx   = momx_star**2/rho_star + P_star
  flux_Momy   = momx_star * momy_star/rho_star
  flux_Energy = (en_star+P_star) * momx_star/rho_star
  
  # find wavespeeds
  C_L = np.sqrt(gamma*P_L/rho_L) + np.abs(vx_L)
  C_R = np.sqrt(gamma*P_R/rho_R) + np.abs(vx_R)
  C = np.maximum( C_L, C_R )
  
  # add stabilizing diffusive term
  flux_Mass   -= C * 0.5 * (rho_L - rho_R)
  flux_Momx   -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
  flux_Momy   -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
  flux_Energy -= C * 0.5 * ( en_L - en_R )

  return flux_Mass, flux_Momx, flux_Momy, flux_Energy

def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
  """
  Apply fluxes to conserved variables
  """
  # directions for np.roll() 
  R = -1   # right
  L = 1    # left
  
  # update solution
  F += - dt * dx * flux_F_X
  F +=   dt * dx * np.roll(flux_F_X,L,axis=0)
  F += - dt * dx * flux_F_Y
  F +=   dt * dx * np.roll(flux_F_Y,L,axis=1)
  
  return F