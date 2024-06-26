import numpy as np
import rusanov_flux as flux

def getConserved( rho, vx, vy, P, gamma, vol):
  """
  Calculate the conserved variable from the primitive
  """
  Mass   = rho * vol
  Momx   = rho * vx * vol
  Momy   = rho * vy * vol
  Energy = (P/(gamma-1) + 0.5*rho*(vx**2+vy**2))*vol
  
  return Mass, Momx, Momy, Energy

def getPrimitive( Mass, Momx, Momy, Energy, gamma, vol, ghostCell = False ):
  """
  Calculate the primitive variable from the conservative
  """
  rho = Mass / vol
  vx  = Momx / rho / vol
  vy  = Momy / rho / vol
  P   = (Energy/vol - 0.5*rho * (vx**2+vy**2)) * (gamma-1)
  
  if ghostCell: 
    rho, vx, vy, P = setGhostCells(rho, vx, vy, P)	
  
  return rho, vx, vy, P

def getGradient(f, dx, ghostCell = False):
  """
  Calculate the gradients of a field
  """
  # directions for np.roll() 
  R = -1   # right
  L = 1    # left
  
  f_dx = ( np.roll(f,R,axis=0) - np.roll(f,L,axis=0) ) / (2*dx)
  f_dy = ( np.roll(f,R,axis=1) - np.roll(f,L,axis=1) ) / (2*dx)

  if ghostCell:
    f_dx, f_dy = setGhostGradients(f_dx, f_dy)
  
  return f_dx, f_dy

def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
  """
  Calculate the gradients of a field
  """
  # directions for np.roll() 
  R = -1   # right
  L = 1    # left
  
  f_XL = f - f_dx * dx/2
  f_XL = np.roll(f_XL,R,axis=0)
  f_XR = f + f_dx * dx/2
  
  f_YL = f - f_dy * dx/2
  f_YL = np.roll(f_YL,R,axis=1)
  f_YR = f + f_dy * dx/2
  
  return f_XL, f_XR, f_YL, f_YR

def addGhostCells( rho, vx, vy, P ):
	"""
    Add ghost cells to the top and bottom
	"""
	rho = np.hstack((rho[:,0:1], rho, rho[:,-1:]))
	vx  = np.hstack(( vx[:,0:1],  vx,  vx[:,-1:]))
	vy  = np.hstack(( vy[:,0:1],  vy,  vy[:,-1:]))
	P   = np.hstack((  P[:,0:1],   P,   P[:,-1:]))
	
	return rho, vx, vy, P
	
def setGhostCells( rho, vx, vy, P ):
	"""
    Set ghost cells at the top and bottom
	"""
	
	rho[:,0]  = rho[:,1]
	vx[:,0]   =  vx[:,1]
	vy[:,0]   = -vy[:,1]
	P[:,0]    =   P[:,1]
	
	rho[:,-1] = rho[:,-2]
	vx[:,-1]  =  vx[:,-2]
	vy[:,-1]  = -vy[:,-2]
	P[:,-1]   =   P[:,-2]
	
	return rho, vx, vy, P
	
def setGhostGradients( f_dx, f_dy ):
	"""
    Set ghost cell y-gradients at the top and bottom to be reflections
	"""
	
	f_dy[:,0]  = -f_dy[:,1]  
	f_dy[:,-1] = -f_dy[:,-2] 
	
	return f_dx, f_dy

def addSourceTerm( Mass, Momx, Momy, Energy, g, dt ):
	"""
    Add gravitational source term to conservative variables
	"""
	
	Energy += dt * Momy * g
	Momy += dt * Mass * g
	
	return Mass, Momx, Momy, Energy