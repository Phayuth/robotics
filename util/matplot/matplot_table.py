import matplotlib.pyplot as plt
import numpy as np

columnDataName = ['RRT-Connect', 'RRT*', 'Informed-RRT*', 'RRT*-Connect', 'RRT-Cont ast Ifm-RRT* (me)', 'RRT-Cont LocOpt (me)', 'RRT* LocOpt (me)']

rowDataName = ['# Node',
               'Initial Path Cost',
               'Initial Path Found on Iteration',
               'Final Path Cost',
               'Full Planning Time (sec)',
               'Planning Time Only (sec)',
               'Collision Check Time (sec)',
               '# Collision Check',
               'Avg Col. Check Time (sec)']

ExpResPerPlanner = [[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]],
                    [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]]


data = [[f"{np.mean(plannerTrialData):.4f} $\pm$ {np.std(plannerTrialData):.4f}" for plannerTrialData in perfMatric] for perfMatric in ExpResPerPlanner]
print(f"==>> data: {len(data)}")
print(f"==>> data: \n{data}")

fig, ax = plt.subplots()
table = ax.table(rowLabels=rowDataName, colLabels=columnDataName, cellText=data, loc='center')

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)  # Adjust the table size
table.auto_set_column_width(range(len(columnDataName)))  # Automatically set column widths based on content
ax.axis('off')  # Turn off the axis
ax.set_title('Performance of Sampling Based in 2D')

plt.show()
