import numpy as np


def update_y_limits(ax, data_range, padding_factor=0.1):
    """Update axis limits with padding"""
    data_min, data_max = np.min(data_range), np.max(data_range)
    if data_min != data_max:
        data_pad = (data_max - data_min) * padding_factor
        ax.set_ylim(data_min - data_pad, data_max + data_pad)
    else:
        ax.set_ylim(0, max(1, data_max * 1.1))
