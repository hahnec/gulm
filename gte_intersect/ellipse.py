import ctypes
from pathlib import Path
import numpy as np


class EllipseIntersection(object):

    def __init__(self, *args, **kwargs):
        super(EllipseIntersection, self).__init__(*args, **kwargs)

        self.abs_pth = Path(__file__).parent.resolve() if 'abs_path' not in kwargs else Path(kwargs['abs_path'])
        self.lib_pth = self.abs_pth / "ellipse_intersect.so"
        self.c_lib = ctypes.CDLL(self.lib_pth)
        self.c_lib.ellipseIntersect.argtypes = [ctypes.c_double,]*12
        self.c_lib.ellipseIntersect.restype = ctypes.c_bool
        self.c_lib.ellipseIntersectLoopP.restype = ctypes.c_bool
        self.c_lib.ellipseIntersectLoopP.argtypes = [np.ctypeslib.ndpointer(dtype='double', ndim=1, flags='C_CONTIGUOUS'),]*13+[np.ctypeslib.ndpointer(dtype='bool', ndim=1, flags='C_CONTIGUOUS')]+[ctypes.c_int]
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

        ret = self.c_lib.ellipseIntersect(*self.args, x_ctypes)

        return x.reshape(-1, 2), ret

    def set_all_ellipses(self, c1xs, c1ys, c2xs, c2ys, r1xs, r1ys, r2xs, r2ys, axes00, axes01, axes10, axes11):

        self.args = [c1xs, c1ys, c2xs, c2ys, r1xs, r1ys, r2xs, r2ys, axes00, axes01, axes10, axes11]

    def get_intersection_multiple(self):

        el_num = np.array(self.args).shape[-1]
        x = np.zeros(el_num*8, dtype='double')
        v = np.zeros(el_num, dtype='bool')

        ret = self.c_lib.ellipseIntersectLoopP(*self.args, x, v, el_num)

        return x.reshape(el_num, -1, 2), v
