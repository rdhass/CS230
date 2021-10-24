import sys
sys.path.insert(0,'/Users/ryanhass/Documents/MATLAB/CS_230/Final_project/utilities')
#from derivative_operators import ddx, ddy
from diff import DiffOps
import numpy as np
from domain_setup import setup_domain

def test_ddx(A,B,C,nx,ny,nz):
    # Define spatial domain
    coefs = [A, B, C]
    Lx, Ly, Lz = tuple([2.*np.pi*i for i in coefs])
    dx, dy, dz, xmesh, ymesh, zmesh = setup_domain(Lx,Ly,Lz,nx,ny,nz)

    xcos = np.cos(xmesh)
    xsin = np.sin(xmesh)
    ycos = np.cos(ymesh)
    ysin = np.sin(ymesh)
    zcos = np.cos(zmesh)

    xzcos = np.multiply(xcos,zcos)
    yzcos = np.multiply(ycos,zcos)

    f = np.multiply(xcos,yzcos)

    dops = DiffOps(nx = nx, ny = ny, nz = nz, Lx = Lx, Ly = Ly, Lz = Lz)
#    dfdx = ddx(f,dx)
    dfdx = dops.ddx(f)
    dfdx_true = np.multiply(-1.*xsin,yzcos)
    
#    dfdy = ddy(f,dy)
    dfdy = dops.ddy(f)
    dfdy_true = np.multiply(-1.*ysin,xzcos)

    comparex = dfdx - dfdx_true
    comparey = dfdy - dfdy_true
    assert np.amax(comparex) < 1.e-12, "ddx test 1 FAILED. max difference = {}".\
            format(np.amax(comparex))
    print("ddx test 1 PASSED!")
    assert np.amax(comparey) < 1.e-12, "ddy test 1 FAILED. max difference = {}".\
            format(np.amax(comparey))
    print("ddy test 1 PASSED!")
    
    xsinzcos = np.multiply(xsin,zcos)
    g = np.multiply(xsin,yzcos)
#    dgdx = ddx(g,dx)
    dgdx = dops.ddx(g)
    dgdx_true = np.multiply(xcos,yzcos)
    
#    dgdy = ddy(g,dy)
    dgdy = dops.ddy(g)
    dgdy_true = np.multiply(-1.*ysin,xsinzcos)
    
    comparex = dgdx - dgdx_true
    comparey = dgdy - dgdy_true
    assert np.amax(comparex) < 1.e-12, "ddx test 2 FAILED. max difference = {}".\
            format(np.amax(comparex))
    print("ddx test 2 PASSED!")
    assert np.amax(comparey) < 1.e-12, "ddy test 2 FAILED. max difference = {}".\
            format(np.amax(comparey))
    print("ddy test 2 PASSED!")
    return

if len(sys.argv) < 7:
    print("Usage:")
    print*("  python3 test_derivatives.py <Lx_on2pi> <Ly_on2pi> <Lz_on2pi> <nx> <ny> <nz>")
    sys.exit()

Lx_on2pi, Ly_on2pi, Lz_on2pi, nx, ny, nz = (sys.argv[1], sys.argv[2], sys.argv[3],\
        sys.argv[4], sys.argv[5], sys.argv[6])
Lx_on2pi, Ly_on2pi, Lz_on2pi = np.float64((Lx_on2pi, Ly_on2pi, Lz_on2pi))
nx, ny, nz = (int(nx),int(ny),int(nz))
test_ddx(Lx_on2pi,Ly_on2pi,Lz_on2pi,nx,ny,nz)
