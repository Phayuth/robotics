import numpy as np


class IKJacobianInverseSolver: # THIS METHOD OF IK IS NOT WORKING WHEN THE ARM IS AT SINGULARITY

    def __init__(self, baseClass, clampMagConstant=None) -> None:
        self.baseClass = baseClass
        self.clampMagConstant = clampMagConstant

    def update_term(self):
        e = self.baseClass.get_error()
        if self.clampMagConstant:
            e = clampMag(e, self.clampMagConstant)
        Jac = self.baseClass.get_jacobian()
        return np.linalg.inv(Jac).dot(e)


class IKJacobianPseudoInverseSolver:

    def __init__(self, baseClass) -> None:
        self.baseClass = baseClass

    def update_term(self):
        e = self.baseClass.get_error()
        Jac = self.baseClass.get_jacobian()
        return np.linalg.pinv(Jac).dot(e)


class IKJacobianTransposeSolver:

    def __init__(self, baseClass) -> None:
        self.baseClass = baseClass

    def cal_alpha(self, e, Jac):
        JacT = np.transpose(Jac)
        JacJacTe = Jac @ JacT @ e
        return (np.dot(np.transpose(e), JacJacTe)) / (np.dot(np.transpose(JacJacTe), JacJacTe))

    def update_term(self):
        e = self.baseClass.get_error()
        Jac = self.baseClass.get_jacobian()
        alpha = self.cal_alpha(e, Jac)
        return alpha * np.transpose(Jac).dot(e)


class IKDampedLeastSquareSolver:

    def __init__(self, baseClass, dampConstant, clampMagConstant=None) -> None:
        self.baseClass = baseClass
        self.dampConstant = dampConstant
        self.clampMagConstant = clampMagConstant

    def update_term(self):
        e = self.baseClass.get_error()
        if self.clampMagConstant:
            e = clampMag(e, self.clampMagConstant)
        Jac = self.baseClass.get_jacobian()
        JacT = np.transpose(Jac)
        return JacT @ np.linalg.inv(Jac @ JacT + np.identity(2) * (self.dampConstant**2)) @ e


class IKSelectivelyDampedLeastSquareSolver:

    def __init__(self, baseClass, dampConstant, clampMagConstant=None) -> None:
        self.baseClass = baseClass
        self.dampConstant = dampConstant
        self.clampMagConstant = clampMagConstant
        self.gammaMax = np.pi/4 #  the maximum permissible change in any joint angle in a single step

    def update_term(self): # NOT CORRECT YET
        e = self.baseClass.get_error()
        Jac = self.baseClass.get_jacobian()
        U,Sigma,VT = np.linalg.svd(Jac)
        V = VT.T
        Alpha = U.T @ e
        Gamma = np.minimum(1, N/M) * self.gammaMax
        Phi = clampMagAbs()
        return clampMagAbs(np.sum(Phi), self.gammaMax)


def clampMag(w, d):
    if euclideanNormW := np.linalg.norm(w) <= d:
        return w
    else:
        return d * (w / euclideanNormW)


def clampMagAbs(w, d):
    if oneNormW := np.max(abs(w)) <= d:
        return w
    else:
        return d * (w / oneNormW)
