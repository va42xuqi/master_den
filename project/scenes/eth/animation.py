"""
This module is responsible for providing the dataloader for the soc environment.
"""


def func_color(axs, idx, input_plot, target_plot):
    """
    Function to color the plot based on the object type
    Parameters
    ----------
    axs
    idx
    input_plot
    target_plot

    Returns
    -------

    """
    if idx == 1:
        axs.plot(
            input_plot[idx, 0, :],
            input_plot[idx, 1, :],
            label="Target Team",
            color="red",
            alpha=0.8,
        )
        axs.plot(
            target_plot[idx, 0, :], target_plot[idx, 1, :], color="salmon", alpha=0.5
        )
    elif idx < 5:
        axs.plot(input_plot[idx, 0, :], input_plot[idx, 1, :], color="red", alpha=0.8)
        axs.plot(
            target_plot[idx, 0, :], target_plot[idx, 1, :], color="salmon", alpha=0.5
        )
    elif idx == 5:
        axs.plot(
            input_plot[idx, 0, :],
            input_plot[idx, 1, :],
            label="Other Team",
            color="blue",
            alpha=0.8,
        )
        axs.plot(
            target_plot[idx, 0, :], target_plot[idx, 1, :], color="lightblue", alpha=0.5
        )
    elif idx < 10:
        axs.plot(input_plot[idx, 0, :], input_plot[idx, 1, :], color="blue", alpha=0.8)
        axs.plot(
            target_plot[idx, 0, :], target_plot[idx, 1, :], color="lightblue", alpha=0.5
        )
    else:
        axs.plot(
            input_plot[idx, 0, :],
            input_plot[idx, 1, :],
            label="Ball",
            color="gold",
            alpha=0.8,
        )
        axs.plot(
            target_plot[idx, 0, :], target_plot[idx, 1, :], color="yellow", alpha=0.5
        )
