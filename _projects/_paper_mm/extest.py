import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np

np.set_printoptions(precision=3, suppress=True)

frankie = rtb.models.Frankie()


frankie.q = frankie.qr
print(f"Frankie joint angles: {frankie.q}")
print(f"Frankie joint r: {frankie.qr}")

numjoints = frankie.n
print(f"Frankie number of joints: {numjoints}")

wTe = frankie.fkine(frankie.q)
print(f"Frankie end-effector pose:\n{wTe}")
print(f"Frankie end-effector pose:\n{wTe.A}")

jac_in_ee = frankie.jacobe(frankie.q)
print(f"Frankie end-effector Jacobian:\n{jac_in_ee}")

jac_in_base = frankie.jacob0(frankie.q)
print(f"Frankie base Jacobian:\n{jac_in_base}")


jac_manip = frankie.jacobm(start=frankie.links[4])
print(f"Frankie manipulability Jacobian:\n{jac_manip}")