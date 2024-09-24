import numpy as np
import scipy.interpolate


class CubicSplineInterpolationIndependant:
    """
    [Summay] : Exact Pass Through Interpolator using piecewise natural cubic polynomial with C2 continuous.

    [Method] :

    - Interpolate for each configuration variable independantly.
    - Lack local control.

    """

    def __init__(self, s, pathArray) -> None:  # pathArray shape (numDoF, numSeq)
        self.spl = scipy.interpolate.CubicSpline(s, pathArray, axis=1)  # scipy.interpolate.CubicHermiteSpline(s, pathArray, dydx, axis=1)
        self.spld = self.spl.derivative(nu=1)
        self.spldd = self.spld.derivative(nu=1)
        self.splddd = self.spldd.derivative(nu=1)

    def eval_pose(self, snew):
        return self.spl(snew)

    def eval_velo(self, snew):
        return self.spld(snew)

    def eval_accel(self, snew):
        return self.spldd(snew)

    def eval_jerk(self, snew):
        return self.splddd(snew)


class MonotoneSplineInterpolationIndependant:
    """
    [Summay] : Exact Pass Through Interpolator with prevention of Overshooting unlike Natural Cubic Spline.
    (Runge's Phenomenon) a problem of oscillation at the edges of an interval that occurs when using polynomial interpolation
    with polynomials of high degree over a set of equispaced interpolation points.

    """

    def __init__(self, s, pathArray, mode) -> None:
        self.pathArray = pathArray
        if mode == 1:
            self.spl = scipy.interpolate.Akima1DInterpolator(s, pathArray, axis=1)
        elif mode == 2:
            self.spl = scipy.interpolate.PchipInterpolator(s, pathArray, axis=1)
        self.spld = self.spl.derivative(nu=1)
        self.spldd = self.spld.derivative(nu=1)
        self.splddd = self.spldd.derivative(nu=1)

    def eval_pose(self, snew):
        return self.spl(snew)

    def eval_velo(self, snew):
        return self.spld(snew)

    def eval_accel(self, snew):
        return self.spldd(snew)

    def eval_jerk(self, snew):
        return self.splddd(snew)


class BSplineInterpolationIndependant:
    """
    [Summay] : B-Spline (Basis Spline) does not necessary pass through all

    """

    def __init__(self, s, pathArray, degree) -> None:
        self.pathArray = pathArray
        self.spl = scipy.interpolate.make_interp_spline(s, pathArray, k=degree, axis=1)  # degree 3 by default, cubic
        self.spld = self.spl.derivative(nu=1)
        self.spldd = self.spld.derivative(nu=1)
        self.splddd = self.spldd.derivative(nu=1)

    def eval_pose(self, snew):
        return self.spl(snew)

    def eval_velo(self, snew):
        return self.spld(snew)

    def eval_accel(self, snew):
        return self.spldd(snew)

    def eval_jerk(self, snew):
        return self.splddd(snew)


class SmoothSpline:
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html#scipy.interpolate.make_smoothing_spline

    """

    def __init__(self, s, pathArray, lam) -> None:
        self.pathArray = pathArray
        w = np.ones(pathArray.shape[1]) # force first and last point to always the same
        w[0] = 100
        w[-1] = 100
        self.spl = [scipy.interpolate.make_smoothing_spline(s, pathArray[i], w=w, lam=lam) for i in range(pathArray.shape[0])]
        self.spld = [bspl.derivative(1) for bspl in self.spl]
        self.spldd = [bspld.derivative(1) for bspld in self.spld]
        self.splddd = [bspldd.derivative(1) for bspldd in self.spldd]

    def eval_pose(self, snew):
        return [spl(snew) for spl in self.spl]

    def eval_velo(self, snew):
        return [spld(snew) for spld in self.spld]

    def eval_accel(self, snew):
        return [spldd(snew) for spldd in self.spldd]

    def eval_jerk(self, snew):
        return [splddd(snew) for splddd in self.splddd]


class BSplineSmoothingUnivariant:
    """
    Find a smooth approximating spline curve

    """

    def __init__(self, s, pathArray, smoothc, degree=None) -> None:
        self.pathArray = pathArray
        w = np.ones(pathArray.shape[1]) # force first and last point to always the same
        w[0] = 100
        w[-1] = 100
        self.bspl = [scipy.interpolate.UnivariateSpline(s, pathArray[i], w=w, s=smoothc, k=degree if degree is not None else 3) for i in range(pathArray.shape[0])]  # default degree k = 3
        self.bspld = [bspl.derivative(1) for bspl in self.bspl]
        self.bspldd = [bspld.derivative(1) for bspld in self.bspld]
        self.bsplddd = [bspldd.derivative(1) for bspldd in self.bspldd]

    def eval_pose(self, snew):
        return [bspl(snew) for bspl in self.bspl]

    def eval_velo(self, snew):
        return [bspld(snew) for bspld in self.bspld]

    def eval_accel(self, snew):
        return [bspldd(snew) for bspldd in self.bspldd]

    def eval_jerk(self, snew):
        return [bsplddd(snew) for bsplddd in self.bsplddd]


if __name__ == "__main__":
    import os
    import sys
    wd = os.path.abspath(os.getcwd())
    sys.path.append(str(wd))

    import matplotlib.pyplot as plt
    from datasave.joint_value.pre_record_value import PreRecordedPathMobileRobot, PreRecordedPath

    pathArray = np.array(PreRecordedPathMobileRobot.warehouse_path).T
    print(f"==>> pathArray.shape: {pathArray.shape}")

    tf = 60
    s = np.linspace(0, tf, pathArray.shape[1])
    snew = np.linspace(0, tf, 1001)
    cb = CubicSplineInterpolationIndependant(s, pathArray)
    # cb = MonotoneSplineInterpolationIndependant(s, pathArray, 2)
    # cb = BSplineInterpolationIndependant(s, pathArray, degree=3)
    # cb = BSplineSmoothingUnivariant(s, pathArray, smoothc=0.0001)

    p = cb.eval_pose(5)
    print(p[0])
    print(f"> p.shape: {p.shape}")
    print(f"> type(p): {type(p)}")
    print(f"> p: {p}")

    p = cb.eval_pose(snew)
    v = cb.eval_velo(snew)
    a = cb.eval_accel(snew)

    plt.plot(pathArray[0], pathArray[1], "*")
    plt.plot(p[0], p[1], "--")
    plt.show()

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(s, pathArray[0], "*")
    axs[0, 0].plot(snew, p[0], "--")
    axs[1, 0].plot(snew, v[0])
    axs[2, 0].plot(snew, a[0])

    axs[0, 1].plot(s, pathArray[1], "*")
    axs[0, 1].plot(snew, p[1], "--")
    axs[1, 1].plot(snew, v[1])
    axs[2, 1].plot(snew, a[1])

    plt.show()

    # cubic spline 1D interplate
    x = np.array([0, 6, -1, 6, 0, 6, 0, 6, 0, 6, 0, 6, 0])
    time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    timenew = np.arange(0, 12, 0.01)
    cv = CubicSplineInterpolationIndependant(time, x)
    pv = cv.eval_pose(timenew)
    vv = cv.eval_velo(timenew)
    av = cv.eval_accel(timenew)
    plt.plot(time, x, "*")
    plt.plot(timenew, pv)
    plt.plot(timenew, vv)
    plt.plot(timenew, av)
    plt.show()
