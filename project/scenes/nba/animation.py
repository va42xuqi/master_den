def func_color(axs, i, input_plot, target_plot):
    if i == 1:
        axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], label='Target Team', color='red', alpha=0.8)
        axs.plot(target_plot[i, 0, :], target_plot[i, 1, :], color='salmon', alpha=0.5)
    elif i < 5:
        axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], color='red', alpha=0.8)
        axs.plot(target_plot[i, 0, :], target_plot[i, 1, :], color='salmon', alpha=0.5)
    elif i == 5:
        axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], label='Other Team', color='blue', alpha=0.8)
        axs.plot(target_plot[i, 0, :], target_plot[i, 1, :], color='lightblue', alpha=0.5)
    elif i < 10:
        axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], color='blue', alpha=0.8)
        axs.plot(target_plot[i, 0, :], target_plot[i, 1, :], color='lightblue', alpha=0.5)
    else:
        axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], label='Ball', color='gold', alpha=0.8)
        axs.plot(target_plot[i, 0, :], target_plot[i, 1, :], color='yellow', alpha=0.5)
