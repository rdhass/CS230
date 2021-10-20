from derivative_operators import ddx, ddy
import numpy as np
import sys

def setup_domain(A,B,C,nx,ny,nz):
    coefs = [A, B, C]
    Lx, Ly, Lz = tuple([np.pi*i for i in coefs])
    nxf, nyf, nzf = np.float64((nx,ny,nz))
    dx, dy, dz = (Lx/nxf, Ly/nyf, Lz/nzf)
    
    x = np.arange(0.,Lx, dx)
    y = np.arange(0.,Ly, dy)
    z = np.arange(0.,Lz, dz)

    xmesh, ymesh, zmesh = np.meshgrid(x, x, x, indexing='ij')

    return dx, dy, dz, xmesh, ymesh, zmesh

def test_ddx(A,B,C,nx,ny,nz):
    # Define spatial domain
    dx, dy, dz, xmesh, ymesh, zmesh = setup_domain(A,B,C,nx,ny,nz)

    xcos = np.cos(xmesh)
    xsin = np.sin(xmesh)
    ycos = np.cos(ymesh)
    ysin = np.sin(ymesh)
    zcos = np.cos(zmesh)

    xzcos = np.multiply(xcos,zcos)
    yzcos = np.multiply(ycos,zcos)

    f = np.multiply(xcos,yzcos)

    dfdx = ddx(f,dx)
    dfdx_true = np.multiply(-1.*xsin,yzcos)
    
    dfdy = ddy(f,dy)
    dfdy_true = np.multiply(-1.*ysin,xzcos)
    
    comparex = dfdx - dfdx_true
    comparey = dfdy - dfdy_true
    assert np.amax(comparex) < 1.e-12, "ddx test FAILED. max difference = {}".\
            format(np.amax(comparex))
    print("ddx test PASSED!")
    assert np.amax(comparey) < 1.e-12, "ddy test FAILED. max difference = {}".\
            format(np.amax(comparey))
    print("ddy test PASSED!")
    return

if len(sys.argv) < 7:
    print("Usage:")
    print*("  python3 test_derivatives.py <Lx_onpi> <Ly_onpi> <Lz_onpi> <nx> <ny> <nz>")
    sys.exit()

Lx_onpi, Ly_onpi, Lz_onpi, nx, ny, nz = (sys.argv[1], sys.argv[2], sys.argv[3],\
        sys.argv[4], sys.argv[5], sys.argv[6])
Lx_onpi, Ly_onpi, Lz_onpi = np.float64((Lx_onpi, Ly_onpi, Lz_onpi))
test_ddx(Lx_onpi,Ly_onpi,Lz_onpi,nx,ny,nz)
