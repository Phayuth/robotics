import pinocchio

urdf_filename = "./datasave/urdf/ur5e_extract_calibrated.urdf"

model = pinocchio.buildModelFromUrdf(urdf_filename)
print("model name: " + model.name)

data = model.createData()

q = pinocchio.randomConfiguration(model)
print(f"q: {q.T}")

pinocchio.forwardKinematics(model, data, q)

# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
