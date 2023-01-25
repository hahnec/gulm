# ctypes_test.py
import ctypes
from pathlib import Path
import numpy as np
import time


class EllipseIntersection(object):

    def __init__(self, *args, **kwargs):
        super(EllipseIntersection, self).__init__(*args, **kwargs)

        self.abs_pth = Path(__file__).parent.resolve() if 'abs_path' not in kwargs else Path(kwargs['abs_path'])
        self.lib_pth = self.abs_pth / "ellipse_intersect.so"
        self.c_lib = ctypes.CDLL(self.lib_pth)
        self.c_lib.ellipseIntersect.argtypes = [ctypes.c_double,]*12
        self.c_lib.ellipseIntersect.restype = ctypes.c_bool
        self.c_lib.ellipseIntersectLoopP.restype = ctypes.c_bool
        self.c_lib.ellipseIntersectLoopP.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),]*13+[ctypes.c_int]
        self.args = []

    def set_ellipses(self, center_pos_a, center_pos_b, exa, eya, exb, eyb, axisa, axisb):
        self.args = [
            center_pos_a[0], center_pos_a[1],
            center_pos_b[0], center_pos_b[1],
            exa, eya,
            exb, eyb,
            axisa[0], axisa[1],
            axisb[0], axisb[1],
        ]

    def get_intersection(self):

        x = np.zeros(8, dtype='double')
        x_ctypes = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        valid = self.c_lib.ellipseIntersect(*self.args, x_ctypes)

        return x.reshape(-1, 2), valid

    def set_all_ellipses(self, c1xs, c1ys, c2xs, c2ys, r1xs, r1ys, r2xs, r2ys, axes00, axes01, axes10, axes11):

        self.args = [c1xs, c1ys, c2xs, c2ys, r1xs, r1ys, r2xs, r2ys, axes00, axes01, axes10, axes11]

    def get_intersection_multiple(self):

        el_num = np.array(self.args).shape[-1]
        x = np.zeros(el_num*8, dtype='double')

        valid = self.c_lib.ellipseIntersectLoopP(*[np.array(vec, dtype='double') for vec in self.args], x, el_num)

        return x.reshape(el_num, -1, 2), valid


if __name__ == "__main__":

    # Load the shared library into ctypes
    libname = Path(__file__).parent.resolve() / "ellipse_intersect.so"
    c_lib = ctypes.CDLL(libname)

    args = [100.0, 100.0, 75.0, 100.0, 100.0, 100.0, 125.0, 50.0, 1, 0, 1, 0]

    c_lib.ellipseIntersect.argtypes = [ctypes.c_double,]*len(args)
    c_lib.ellipseIntersect.restype = ctypes.c_bool

    x = np.zeros(8, dtype='double')
    x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    start = time.time()
    iters = 1000
    for i in range(0, iters):
        res = c_lib.ellipseIntersect(*args, x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    
    print((time.time()-start)/iters)
    
    print(x.reshape(-1, 2))

    #   (x0,y0) = (199.99999999999997, 100.00000238418579)
    #   (x1,y1) = (199.99999999999997,  99.999997615814209)
    #   (x2,y2) = (9.5238095238095184, 142.59177099999599)
    #   (x3,y3) = (9.5238095238095184, 57.408229000004013)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(30, 15))
    ell1 = Ellipse(xy=args[0:2], width=2*args[4], height=2*args[5], angle=0, edgecolor='b', fc='None')
    ell2 = Ellipse(xy=args[2:4], width=2*args[6], height=2*args[7], angle=0, edgecolor='r', fc='None')
    ax.plot(*x[0:2], 'kx')
    ax.plot(*x[2:4], 'kx')
    ax.plot(*x[4:6], 'kx')
    ax.plot(*x[6:8], 'kx')
    ax.add_patch(ell1)
    ax.add_patch(ell2)
    plt.show()
