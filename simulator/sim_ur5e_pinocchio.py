import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import pinocchio
from pinocchio.visualize import MeshcatVisualizer

# unset PYTHONPATH


class UR5ePinocchio:
    def __init__(self):
        self.urdf_filename = "./datasave/urdf/ur5e_extract_calibrated.urdf"
        self.model = pinocchio.buildModelFromUrdf(self.urdf_filename)
        self.modelname = self.model.name
        self.tip = self.model.getFrameId("gripper")
        self.data = self.model.createData()

        print("model name: " + self.modelname)
        print("tip name: " + self.model.frames[self.tip].name)
        print("tip id: " + str(self.tip))

    def random_configuration(self):
        q = pinocchio.randomConfiguration(self.model)
        return q

    def forward_kinematics(self, q):
        pinocchio.forwardKinematics(self.model, self.data, q)
        return (
            self.data.oMf[self.tip].translation,
            self.data.oMf[self.tip].rotation,
        )

    def inverse_kinematics(self, q_init, target_position, target_rotation):
        pinocchio.forwardKinematics(self.model, self.data, q_init)
        pinocchio.updateFramePlacements(self.model, self.data)
        target_frame = pinocchio.SE3(target_rotation, target_position)
        q_sol = pinocchio.ik(self.model, self.data, q_init, target_frame)
        return q_sol

    def visualize(self):
        # Create geometry model
        model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
            self.urdf_filename
        )
        print(model)
        print(collision_model)
        print("-----------------------------")
        print(visual_model)

        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)

        viz.loadViewerModel(rootNodeName="ur5e")
        print("-----------------------------")
        print(viz.visual_model)

        q0 = pinocchio.neutral(model)
        viz.display(q0)
        viz.displayVisuals(True)


if __name__ == "__main__":
    ur5e = UR5ePinocchio()

    q = ur5e.random_configuration()
    print("Random configuration: ", q)

    q = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
    tran, rot = ur5e.forward_kinematics(q)
    print("Forward kinematics: ", tran, rot)

    ur5e.visualize()
