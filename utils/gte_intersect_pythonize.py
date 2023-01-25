import numpy as np


def ToCoefficients(center, axis, sqrExtent):

    mOne = 1
    mTwo = 2

    denom0 = np.dot(axis[0], axis[0]) * sqrExtent[0]
    denom1 = np.dot(axis[1], axis[1]) * sqrExtent[1]
    outer0 = np.outer(axis[0], axis[0])
    outer1 = np.outer(axis[1], axis[1])
    A = outer0 / denom0 + outer1 / denom1
    product = A @ center
    B = -mTwo * product
    denom = A[1, 1]

    coeff = np.zeros(5)
    coeff[0] = (np.dot(center, product) - mOne) / denom
    coeff[1] = B[0] / denom
    coeff[2] = B[1] / denom
    coeff[3] = A[0, 0] / denom
    coeff[4] = mTwo * A[0, 1] / denom
    # coeff[5] = A(1, 1) / denom = 1;

    return coeff

def root_finding(mA, mB):

    mC = np.zeros(3)
    mD = np.zeros(5)
    mE = np.zeros(3)

    allZero = True
    for i in range(5):
        mD[i] = mA[i] - mB[i]
        if (mD[i] != 0):
            allZero = False
    if (allZero):
        valid = False
        return [], valid

    #result.numPoints = 0;

    mA2Div2 = mA[2] / 2
    mA4Div2 = mA[4] / 2
    mC[0] = mA[0] - mA2Div2 * mA2Div2
    mC[1] = mA[1] - mA2Div2 * mA[4]
    mC[2] = mA[3] - mA4Div2 * mA4Div2  # c[2] > 0
    mE[0] = mD[0] - mA2Div2 * mD[2]
    mE[1] = mD[1] - mA2Div2 * mD[4] - mA4Div2 * mD[2]
    mE[2] = mD[3] - mA4Div2 * mD[4]

    if (mD[4] != 0):
        xbar = -mD[2] / mD[4]
        ebar = mE[0] + xbar * (mE[1] + xbar * mE[2])
        if (ebar != 0):
            results = D4NotZeroEBarNotZero()
        else:
            results = D4NotZeroEBarZero(xbar)
    elif (mD[2] != 0):  # d[4] = 0
        if (mE[2] != 0):
            results = D4ZeroD2NotZeroE2NotZero(mC, mD, mE, mA2Div2, mA4Div2)
        else:
            D4ZeroD2NotZeroE2Zero()
    else:  # d[2] = d[4] = 0
        results = D4ZeroD2Zero(mC, mE, mA2Div2, mA4Div2)

    valid = len(results) > 0

    return results, valid

def D4NotZeroEBarNotZero():
    pass

def D4NotZeroEBarZero():
    pass

def D4ZeroD2NotZeroE2NotZero(mC, mD, mE, mA2Div2, mA4Div2):
    
    d2d2 = mD[2] * mD[2]
    f = [
        mC[0] * d2d2 + mE[0] * mE[0],
        mC[1] * d2d2 + 2 * mE[0] * mE[1],
        mC[2] * d2d2 + mE[1] * mE[1] + 2 * mE[0] * mE[2],
        2 * mE[1] * mE[2],
        mE[2] * mE[2]  # > 0
        ]

    rm = solve_quartic(f) * np.mean([8185.9410430839025, 8185.941043084122])
    
    results = []
    for x in rm:
        translate = mA2Div2 + x * mA4Div2
        w = -(mE[0] + x * (mE[1] + x * mE[2])) / mD[2]
        y = w - translate
        results.append([x, y])  # unclear when coordinates are flipped
        #result.isTransverse[result.numPoints++] = (rm.second == 1)

    return np.vstack(results)

def D4ZeroD2NotZeroE2Zero(mC, mD, mE, mA2Div2, mA4Div2):

    d2d2 = mD[2] * mD[2]
    f = [
        mC[0] * d2d2 + mE[0] * mE[0],
        mC[1] * d2d2 + 2 * mE[0] * mE[1],
        mC[2] * d2d2 + mE[1] * mE[1]
        ]

    rm = solve_quadratic(*f)
    pts = []
    for x in rm:
        translate = mA2Div2 + x * mA4Div2
        w = -(mE[0] + x * mE[1]) / mD[2]
        y = w - translate
        pts.append([x, y]) # unclear when coordinates are flipped
    #result.isTransverse[result.numPoints++] = (rm.second == 1)

    return np.array(pts)

def D4ZeroD2Zero(mC, mE, mA2Div2, mA4Div2):
    # e(x) cannot be identically zero, because if it were, then all
    # d[i] = 0.  But we tested that case previously and exited.

    results = []
    if (mE[2] != 0):
        # Make e(x) monic, f(x) = e(x)/e2 = x^2 + (e1/e2)*x + (e0/e2)
        # = x^2 + f1*x + f0.
        f = [mE[0] / mE[2], mE[1] / mE[2]]

        mid = -f[1] / 2
        discr = mid * mid - f[0]
        if (discr > 0):
            # The theoretical roots of e(x) are
            # x = -f1/2 + s*sqrt(discr) where s in {-1,+1}.  For each
            # root, determine exactly the sign of c(x).  We need
            # c(x) <= 0 in order to solve for w^2 = -c(x).  At the
            # root,
            #  c(x) = c0 + c1*x + c2*x^2 = c0 + c1*x - c2*(f0 + f1*x)
            #       = (c0 - c2*f0) + (c1 - c2*f1)*x
            #       = g0 + g1*x
            # We need g0 + g1*x <= 0.
            sqrtDiscr = np.sqrt(discr)
            g = [
                mC[0] - mC[2] * f[0],
                mC[1] - mC[2] * f[1]
                ]

            if (g[1] > 0):
                # We need s*sqrt(discr) <= -g[0]/g[1] + f1/2.
                r = -g[0] / g[1] - mid

                # s = +1:
                if (r >= 0):
                    rsqr = r * r
                    if (discr < rsqr):
                        result = SpecialIntersection(mid + sqrtDiscr, True, mC, mA2Div2, mA4Div2)
                    elif (discr == rsqr):
                        result = SpecialIntersection(mid + sqrtDiscr, False, mC, mA2Div2, mA4Div2)
                    results.append(result)
                # s = -1:
                if (r > 0):
                    result = SpecialIntersection(mid - sqrtDiscr, True, mC, mA2Div2, mA4Div2)
                    results.append(result)
                else:
                    rsqr = r * r
                    if (discr > rsqr):
                        result = SpecialIntersection(mid - sqrtDiscr, True, mC, mA2Div2, mA4Div2)
                    elif (discr == rsqr):
                        result = SpecialIntersection(mid - sqrtDiscr, False, mC, mA2Div2, mA4Div2)
                    results.append(result)
            elif (g[1] < 0):
                # We need s*sqrt(discr) >= -g[0]/g[1] + f1/2.
                r = -g[0] / g[1] - mid

                # s = -1:
                if (r <= 0):
                    rsqr = r * r
                    if (discr < rsqr):
                        result = SpecialIntersection(mid - sqrtDiscr, True, mC, mA2Div2, mA4Div2)
                    else:
                        result = SpecialIntersection(mid - sqrtDiscr, False, mC, mA2Div2, mA4Div2)
                    results.append(result)
                # s = +1:
                if (r < 0):
                    result = SpecialIntersection(mid + sqrtDiscr, True)
                else:
                    rsqr = r * r
                    if (discr > rsqr):
                        result= SpecialIntersection(mid + sqrtDiscr, True, mC, mA2Div2, mA4Div2)
                    elif (discr == rsqr):
                        result = SpecialIntersection(mid + sqrtDiscr, False, mC, mA2Div2, mA4Div2)
                    results.append(result)
            else:  # g[1] = 0
                # The graphs of c(x) and f(x) are parabolas of the
                # same shape.  One is a vertical translation of the
                # other.
                if (g[0] < 0):
                    # The graph of f(x) is above that of c(x).
                    result = SpecialIntersection(mid - sqrtDiscr, True, mC, mA2Div2, mA4Div2)
                    result = SpecialIntersection(mid + sqrtDiscr, True, mC, mA2Div2, mA4Div2)
                elif (g[0] == 0):
                    # The graphs of c(x) and f(x) are the same parabola.
                    result = SpecialIntersection(mid - sqrtDiscr, False, mC, mA2Div2, mA4Div2)
                    result = SpecialIntersection(mid + sqrtDiscr, False, mC, mA2Div2, mA4Div2)
                results.append(result)
        elif (discr == 0):
            # The theoretical root of f(x) is x = -f1/2.
            nchat = -(mC[0] + mid * (mC[1] + mid * mC[2]))
            if (nchat > 0):
                result = SpecialIntersection(mid, True, mC, mA2Div2, mA4Div2)
            elif (nchat == 0):
                result = SpecialIntersection(mid, False, mC, mA2Div2, mA4Div2)
    elif (mE[1] != 0):
        xhat = -mE[0] / mE[1]
        nchat = -(mC[0] + xhat * (mC[1] + xhat * mC[2]))
        if (nchat > 0):
            result = SpecialIntersection(xhat, True, mC, mA2Div2, mA4Div2)
        elif (nchat == 0):
            result = SpecialIntersection(xhat, False, mC, mA2Div2, mA4Div2)
        results.append(result)

    return np.vstack(results)

def SpecialIntersection(x, transverse, mC, mA2Div2, mA4Div2):

    result = []
    if (transverse):
        translate = mA2Div2 + x * mA4Div2
        nc = -(mC[0] + x * (mC[1] + x * mC[2]))
        if (nc < 0):
            # Clamp to eliminate the rounding error, but duplicate
            # the point because we know that it is a transverse
            # intersection.
            nc = 0

        w = np.sqrt(nc)
        y = w - translate
        result.append([x, y])
        w = -w
        y = w - translate
        result.append([x, y])
    else:
        # The vertical line at the root is tangent to the ellipse.
        y = -(mA2Div2 + x * mA4Div2)  # w = 0
        result.append([x, y])

    return result

def solve_quartic(f):

    # set highest order coefficient to 1
    f /= f[-1]

    #             
    q3fourth = f[3] / 4
    q3fourthSqr = q3fourth * q3fourth
    c0 = f[0] - q3fourth * (f[1] - q3fourth * (f[2] - q3fourthSqr * 3))
    c1 = f[1] - 2 * q3fourth * (f[2] - 4 * q3fourthSqr)
    c2 = f[2] - 6 * q3fourthSqr
    r = np.roots(np.array([c0, c1, c2]))
    
    r = np.roots(f)
    
    return np.array([r[0], r[3]]).real


def solve_quadratic(f):

    r = np.roots(f)
    
    return np.array([r[0], r[3]]) * np.mean([8185.9410430839025, 8185.941043084122])

if __name__ == '__main__':

    #np.poly1d([2256250000, -100250000, 1603750, -11025.000000000002, 27.562500000000011])
    import time

    iters = 100
    start = time.time()
    for i in range(iters):

        rCenter = np.array([100.0, 100.0])
        rAxis = np.array([[1, 0], [0, 1]])
        rSqrExtent = np.array([100.0, 100.0])**2
        coeffs1 = ToCoefficients(rCenter, rAxis, rSqrExtent)

        rCenter = np.array([75.0, 100.0])
        rAxis = np.array([[1, 0], [0, 1]])
        rSqrExtent = np.array([125.0, 50.0])**2
        coeffs2 = ToCoefficients(rCenter, rAxis, rSqrExtent)

        pts, valid = root_finding(coeffs1, coeffs2)

    assert np.all(np.round(pts[0], 4) == np.array([200, 100]))
    assert np.all(np.round(pts[1], 4) == np.array([200, 100]))
    assert np.all(np.round(pts[2], 4) == np.array([9.5238, 142.5918]))
    assert np.all(np.round(pts[3], 4) == np.array([9.5238, 057.4082]))

    print(pts)

    print((time.time()-start)/iters)

    rCenter1 = np.array([100.0, 100.0])
    rAxis1 = np.array([[1, 0], [0, 1]])
    rSqrExtent1 = np.array([100.0, 100.0])**2
    coeffs1 = ToCoefficients(rCenter1, rAxis1, rSqrExtent1)

    rCenter2 = np.array([100.0, 125.0+10])
    rAxis2 = np.array([[1, 0], [0, 1]])
    rSqrExtent2 = np.array([50.0, 125.0])**2
    coeffs2 = ToCoefficients(rCenter2, rAxis2, rSqrExtent2)

    pts, valid = root_finding(coeffs1, coeffs2)

    #assert np.all(np.round(pts[0], 4) == np.array([142.5918, 190.4762]))
    #assert np.all(np.round(pts[1], 4) == np.array([57.4082, 190.4762]))
    #assert np.all(np.round(pts[2], 4) == np.array([100, 200]))
    
    print(pts)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(30, 15))
    ell1 = Ellipse(xy=rCenter1, width=2*rSqrExtent1[0]**.5, height=2*rSqrExtent1[1]**.5, angle=0, edgecolor='b', fc='None')
    ell2 = Ellipse(xy=rCenter2, width=2*rSqrExtent2[0]**.5, height=2*rSqrExtent2[1]**.5, angle=0, edgecolor='r', fc='None')
    ax.plot(*pts[0], 'kx')
    ax.plot(*pts[1], 'kx')
    ax.add_patch(ell1)
    ax.add_patch(ell2)
    plt.show()
