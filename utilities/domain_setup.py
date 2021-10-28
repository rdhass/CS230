import numpy as np

def setup_domain(Lx,Ly,Lz,nx,ny,nz,zPeriodic = False):
    nxf, nyf, nzf = np.float64((nx,ny,nz))
    dx, dy, dz = (Lx/nxf, Ly/nyf, Lz/nzf)
    
    x = np.arange(0., Lx, dx)
    y = np.arange(0., Ly, dy)
    if zPeriodic:
        z = np.arange(0., Lz, dz)
    else:
        z = np.arange(0.5*dz, Lz+0.5*dz, dz)

    xmesh, ymesh, zmesh = np.meshgrid(x, y, z, indexing='ij')

    return dx, dy, dz, xmesh, ymesh, zmesh
