import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import time
from joint_process import JointProcess


class PlannerManipulator:

    def __init__(self, xStart, xApp, xGoal, config):
        # process joint
        xStart = JointProcess.wrap_to_pi(xStart)
        if isinstance(xApp, list):
            xApp = [JointProcess.wrap_to_pi(x) for x in xApp]
            xGoal = [JointProcess.wrap_to_pi(x) for x in xGoal]
        else:
            xApp = JointProcess.wrap_to_pi(xApp)
            xGoal = JointProcess.wrap_to_pi(xGoal)

        self.planner = config["planner"](xStart, xApp, xGoal, config)

    def planning(self):
        timePlanningStart = time.perf_counter_ns()
        self.planner.start()
        path = self.planner.get_path()
        timePlanningEnd = time.perf_counter_ns()
        self.planner.update_perf(timePlanningStart, timePlanningEnd)
        return path