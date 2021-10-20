import numpy as np
import ddx

def test_ddx(nx):
    # Define spatial domain
    Lx = 2.*np.pi
    nx = 64
    ny = nx
    nz = nx
    nxf, nyf, nzf = np.float64((nx,ny,nz))
    dx, dy, dz = (Lx/nxf, Ly/nyf, Lz/nzf)
    x = np.arange(0.,Lx, dx)
    y = x
    z = x

    xmesh, ymesh, zmesh = np.meshgrid(x, y, z, indexing='ij')
    xcos = np.cos(xmesh)
    xsin = np.sin(xmesh)
    ycos = np.cos(ymesh)
    zcos = np.cos(zmesh)
    yzcos = np.multiply(ycos,zcos)
    f = np.multiply(xcos,yzcos)

    dfdx = ddx(f,dx)
    dfdx_true = np.multiply(xsin,yzcos)
    
    compare = dfdx - dfdx_true
    assert np.amax(compare) < 1.e-12, "ddx test FAILED"
    print("ddx test PASSED!")
    return
