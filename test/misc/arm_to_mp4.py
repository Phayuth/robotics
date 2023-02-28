import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


l1 = 1
l2 = 1
theta1 = 0
theta2 = 0

def plot_arm(theta1, theta2, *args, **kwargs):
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    wrist = elbow + np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])

    plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
    plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

    plt.plot(shoulder[0], shoulder[1], 'ro')
    plt.plot(elbow[0], elbow[1], 'ro')
    plt.plot(wrist[0], wrist[1], 'ro')
    
    
    title = kwargs.get('title', None)
    plt.annotate("X pos = "+str(wrist[0]), xy=(0, 1.8+2), xycoords="data",va="center", ha="center")
    plt.annotate("Y pos = "+str(wrist[1]), xy=(0, 1.5+2), xycoords="data",va="center", ha="center")
    
    circle1 = plt.Circle((0, 0), l1+l2,alpha=0.5, edgecolor='none')
    plt.gca().add_patch(circle1)
    plt.gca().set_aspect('equal')

    plt.title(title)
    
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

# Settings
video_file = "myvid.mp4"
clear_frames = True
fps = 24

# Output video writer
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
writer = FFMpegWriter(fps=fps, metadata=metadata)

fig = plt.figure()
with writer.saving(fig, video_file, 100):
	for i in range(300):
		print(i)
		theta1 = np.sin(i*0.01)
		theta2 += 0.01 + np.sin(i*0.01)
		if clear_frames:
			fig.clear()
		plot_arm(theta1,theta2,title='The edge of the circle is singularity')
		writer.grab_frame()