def read_fortran_data(fname, array_order = 'F'):
  buf = np.empty((nx,ny,nz), dtype = np.dtype(np.float64, align = True),\
          order = array_order)
  buf = h5read...
