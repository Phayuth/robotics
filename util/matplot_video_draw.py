import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import time


def draw_results(input_video_path, event_csv_path, output_video_path, configs, n_frames_limit=np.inf, index_col="frame"):

    # load data
    df = pd.read_csv(event_csv_path, index_col=index_col)
    cap = cv2.VideoCapture(input_video_path)


    # names
    gait_events_list = df.columns
    col_names = configs["column_names"] or list(df.columns)


    # setup frame and writer
    N_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    org_frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, int(org_frame_rate // configs["slow_down"]), (v_width, v_height))


    # plot configs
    plt.style.use(configs["theme"])
    plt.rcParams.update({"font.size": configs["font_size"]})
    plt.rcParams.update({"font.weight": configs["font_weight"]})
    plt.rcParams.update({"axes.linewidth": configs["axes_linewidth"]})

    # create fig
    fig = Figure(figsize=(v_width / 100, v_height / 100 / configs["aspect_ratio"]), dpi=50)
    canvas = FigureCanvas(fig)

    # fig axis
    axs = [fig.add_axes([0.1, pos, configs["subplot_width"], configs["subplot_height"]]) for pos in configs["subplot_positions"]]
    for i, axi in enumerate(axs):
        axi.set_ylim(0, 1.0)
        axi.set_title(col_names[i])
        axi.xaxis.set_major_locator(plt.MaxNLocator(4))

    first_frame_idx = min(df.index)
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(1)) - 1  # Get current frame index

        if frame_idx in df.index:
            start_idx = max(first_frame_idx, frame_idx - configs["x_range"])
            # da = df[start_idx:frame_idx]
            da = df[frame_idx:frame_idx+5]

            xlimstart = 0
            xlimend = configs["x_range"]
            
            for i, gait_event in enumerate(gait_events_list):
                # draw graph
                # axs[i].clear()
                axs[i].plot(da.index, da[gait_event], configs["event_colors"][i], linewidth=4.0)
                # axs[i].set_xlim(start_idx, max(configs["x_range"], frame_idx))
                axs[i].set_xlim(xlimstart, xlimend)


            # put graph into buffer
            canvas.draw()
            buf = canvas.buffer_rgba()
            buf = np.asarray(buf)
            buf = buf[:, :, :-1]  # Remove alpha channel
            buf = buf[:, :, ::-1]  # rgb to bgr

            buf_gray = buf.mean(axis=-1, keepdims=True).repeat(3, axis=-1)
            mask = 1 - (buf_gray / 255)**configs["alpha"]

            frame[configs["margin_top"]:configs["margin_top"] + buf.shape[0], configs["margin_left"]:configs["margin_left"] + buf.shape[1], :,] = \
            (frame[configs["margin_top"]:configs["margin_top"] + buf.shape[0], configs["margin_left"]:configs["margin_left"] + buf.shape[1], :,] * mask + (1-mask) * buf)

        # write frame to output file
        out.write(frame)

    # Release video capture and writer
    cap.release()
    out.release()


if __name__ == "__main__":
    tic = time.perf_counter()
    configs = {
        "event_colors": ["white", "red", "cyan", "yellow"],  # color for each subgraph
        "column_names": [
            "[LEFT] Heel Strike",
            "[LEFT] Toe Off",
            "[RIGHT] Heel Strike",
            "[RIGHT] Toe Off",
        ],  # column names for better visualization, put None to use default column names from the input csv file
        "alpha": 0.7,
        "font_size": 25,
        "font_weight": "bold",
        "axes_linewidth": 3,
        "theme": "dark_background",
        "threshold": 5,
        "x_range": 150,  # range of the horizontal axis, default to 150 frames
        "subplot_width": 0.75,  # width of each subplot, relative to the total width of the graph
        "subplot_height": 0.15,  # similar to subplot_width
        "subplot_positions": [
            0.05,
            0.3,
            0.55,
            0.8,
        ],  # y-positions of each subgraph in the bigger graph
        "aspect_ratio": 0.5,  # controls the aspect ratio of the graph to be more rectangular or square
        "margin_left": 10,  # margin of the graph to the left corner of the input video
        "margin_top": 0,  # margin of the graph from the top corner of the input video
        "slow_down": 2,  # how slow the output compares to the original video, e.g. value of 2 will slow it down by 2x times, thus making it 2x times longer in duration
    }

    draw_results(
        input_video_path="/home/yuth/Downloads/PJB_MO.mp4",
        event_csv_path="/home/yuth/Downloads/GAIT_output_2023_10_12_13_54_20.csv",
        output_video_path="/home/yuth/Downloads/rrrresult.mp4",
        configs=configs,
        # n_frames_limit=200,  # comment out to process the whole video
    )
    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.2f} seconds")
