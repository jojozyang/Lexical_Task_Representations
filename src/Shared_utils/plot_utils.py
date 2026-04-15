import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import pickle
import numpy as np

def plot_curve(data_dictionaries=None, labels=None, color_list=None,
    linestyle_list=None, y_lim=(0, 1.02), figsize=(10, 6),
    axis_label_size=14, axis_tick_size=12,
    y_label="Accuracy", bbox_to_anchor=(1.19, 1), x_ticks=None,
    save_path=None, title=None,
): 

    plt.figure(figsize=figsize)
    for i, dictionary in enumerate(data_dictionaries):
        x_values = list(dictionary.keys())
        y_values = list(dictionary.values())
        plt.plot(x_values, y_values,  
            #marker=marker_list[i], markersize=10,
            color=color_list[i], 
            label=labels[i], 
            linestyle=linestyle_list[i],
            )

    plt.xlabel('Num of top heads', fontsize=axis_label_size)
    if x_ticks is None:
        plt.xticks(sorted(data_dictionaries[0].keys()), fontsize=axis_tick_size)
    else:
        plt.xticks(x_ticks, fontsize=axis_tick_size)
    plt.ylabel(y_label, fontsize=axis_label_size)
    plt.title(title, fontsize=axis_label_size)
    plt.ylim(y_lim)
    plt.legend()
    plt.legend(loc='upper right',  bbox_to_anchor=bbox_to_anchor, fontsize=axis_tick_size)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_three_subplots(
    data_dictionaries_list,
    labels_list,
    color_list,
    linestyle_list,
    y_labels,
    titles,
    figsize=(15, 5),
    y_lim=(0, 1.02),
    axis_label_size=14,
    axis_tick_size=12,
    bbox_to_anchor=(1.05, 1.0),
    x_label="Num of top heads",
    x_ticks=None, title=None,
    save_path=None
):
    """
    Creates a figure with three subplots, each similar to the plot_curve function.

    Args:
        data_dictionaries_list (list of lists of dicts): A list containing three lists of dictionaries.
                                                        Each inner list corresponds to a subplot.
        labels_list (list of lists of str): A list of three lists of labels.
        color_list (list of lists of str): A list of three lists of colors.
        linestyle_list (list of lists of str): A list of three lists of line styles.
        y_labels (list of str): A list of three y-axis labels.
        titles (list of str): A list of three titles for the subplots.
        figsize (tuple): Figure size.
        y_lim (tuple): Y-axis limits.
        axis_label_size (int): Font size for axis labels.
        axis_tick_size (int): Font size for axis ticks.
        bbox_to_anchor (tuple): Bounding box for the legend.
        x_ticks (list): Optional list of x-axis tick values.
        save_path (str): Optional path to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i in range(3):
        ax = axes[i]
        for j, dictionary in enumerate(data_dictionaries_list[i]):
            x_values = list(dictionary.keys())
            y_values = list(dictionary.values())
            ax.plot(
                x_values,
                y_values,
                color=color_list[i][j],
                label=labels_list[i][j],
                linestyle=linestyle_list[i][j]
            )

        ax.set_xlabel(x_label, fontsize=axis_label_size)
        if x_ticks is None:
            ax.set_xticks(sorted(data_dictionaries_list[i][0].keys()))
        else:
            ax.set_xticks(x_ticks)
        ax.set_ylabel(y_labels[i], fontsize=axis_label_size)
        ax.set_title(titles[i], fontsize=axis_label_size)
        ax.set_ylim(y_lim)
        if i == 2:
            ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=axis_tick_size)
        ax.tick_params(axis='both', which='major', labelsize=axis_tick_size)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_two_curves_subplots(
    data_dictionaries_list,
    labels_list,
    color_list,
    linestyle_list,
    y_labels,
    titles,
    figsize=(15, 5),
    y_lim=(0, 1.02),
    axis_label_size=14,
    axis_tick_size=12,
    bbox_to_anchor=(1.05, 1.0),
    x_label="Num of top heads",
    x_ticks=None, title=None,
    save_path=None
):
    """
    Creates a figure with three subplots, each similar to the plot_curve function.

    Args:
        data_dictionaries_list (list of lists of dicts): A list containing three lists of dictionaries.
                                                        Each inner list corresponds to a subplot.
        labels_list (list of lists of str): A list of three lists of labels.
        color_list (list of lists of str): A list of three lists of colors.
        linestyle_list (list of lists of str): A list of three lists of line styles.
        y_labels (list of str): A list of three y-axis labels.
        titles (list of str): A list of three titles for the subplots.
        figsize (tuple): Figure size.
        y_lim (tuple): Y-axis limits.
        axis_label_size (int): Font size for axis labels.
        axis_tick_size (int): Font size for axis ticks.
        bbox_to_anchor (tuple): Bounding box for the legend.
        x_ticks (list): Optional list of x-axis tick values.
        save_path (str): Optional path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for i in range(2):
        ax = axes[i]
        for j, dictionary in enumerate(data_dictionaries_list[i]):
            x_values = list(dictionary.keys())
            y_values = list(dictionary.values())
            ax.plot(
                x_values,
                y_values,
                color=color_list[i][j],
                label=labels_list[i][j],
                linestyle=linestyle_list[i][j]
            )

        ax.set_xlabel(x_label, fontsize=axis_label_size)
        if x_ticks is None:
            ax.set_xticks(sorted(data_dictionaries_list[i][0].keys()))
        else:
            ax.set_xticks(x_ticks)
        ax.set_ylabel(y_labels[i], fontsize=axis_label_size)
        ax.set_title(titles[i], fontsize=axis_label_size)
        ax.set_ylim(y_lim)
        if i == 1:
            ax.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=axis_tick_size)
        ax.tick_params(axis='both', which='major', labelsize=axis_tick_size)
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
    
def get_save_2_curves_plot(model_name:str=None, 
    d_name:str=None, 
    prompt_type_2_be_fixed:str=None, prompt_index_2_be_fixed:int=None, 
    fix_component:str=None, scaling_factors:list=None,
    n_shot_list:list=None, n_inst_list:list=None, 
    save_root:str=None, save:bool=True, y_lim=(-0.4, 1.02),
    figsize=(17, 5), bbox_to_anchor=(1.59, 1.0),
    plot_control:bool=True, 
    save_path_EP:str=None, save_path_IP:str=None,
    result_dict_IP:dict=None, result_dict_EP:dict=None,
): 
    if result_dict_IP is None and result_dict_EP is None:
        if save_path_EP is None:
            save_path_EP = os.path.join(save_root, model_name, d_name, 
                "Fix_incorrect_prompts", 
                f"fix_{prompt_type_2_be_fixed}_{prompt_index_2_be_fixed}_with_EP_{fix_component}_scales_vary_prompt_instances_average.pkl")
        with open(save_path_EP,"rb") as f:
            result_dict_EP = pickle.load(f)
        if save_path_IP is None:
            save_path_IP = os.path.join(save_root, model_name, d_name, 
                "Fix_incorrect_prompts", 
                f"fix_{prompt_type_2_be_fixed}_{prompt_index_2_be_fixed}_with_IP_{fix_component}_scales_vary_prompt_instances_average.pkl")
        with open(save_path_IP, "rb") as f:
            result_dict_IP = pickle.load(f)

    if plot_control:
        data_rr = [result_dict_EP[f"rr_ranks_dict_{index}"] for index in n_shot_list
            ] + [result_dict_IP[f"rr_ranks_dict_{index}"] for index in n_inst_list
            ] + [result_dict_EP["rr_ranks_dict_control"]]
        data_acc = [result_dict_EP[f"interv_acc_dict_{index}"] for index in n_shot_list
            ] + [result_dict_IP[f"interv_acc_dict_{index}"] for index in n_inst_list
            ] + [result_dict_EP["interv_acc_dict_control"]]
    else: 
        data_rr = [result_dict_EP[f"rr_ranks_dict_{index}"] for index in n_shot_list
            ] + [result_dict_IP[f"rr_ranks_dict_{index}"] for index in n_inst_list]
        data_acc = [result_dict_EP[f"interv_acc_dict_{index}"] for index in n_shot_list
            ] + [result_dict_IP[f"interv_acc_dict_{index}"] for index in n_inst_list]
    
    # Styles for 2 curve plots 
    line_colors_all = np.zeros((len(n_shot_list)+len(n_inst_list)+1, 4))
    line_colors_EP = np.zeros((len(n_shot_list), 4))
    N_LINES_colors = len(n_shot_list)
    color_indices_EP = np.linspace(0.5, 1, N_LINES_colors)
    cmap_EP = cm.get_cmap("Greens")
    line_colors_EP = cmap_EP(color_indices_EP)

    line_colors_IP = np.zeros((len(n_inst_list), 4))
    N_LINES_colors = len(n_inst_list)
    color_indices_IP = np.linspace(0.4, 0.8, N_LINES_colors)
    cmap_IP = cm.get_cmap("Oranges")
    line_colors_IP = cmap_IP(color_indices_IP)

    line_colors_all[:len(n_shot_list), :] = line_colors_EP
    line_colors_all[len(n_shot_list):len(n_shot_list)+len(n_inst_list), :] = line_colors_IP
    line_colors_all[-1, :] = np.array(mcolors.to_rgba("gray"))

    labels_EP = [f"Example_{n_shot}_shot" for n_shot in n_shot_list] 
    labels_IP = [f"Instruction_{n_inst}" for n_inst in n_inst_list] 
    labels_all = labels_EP + labels_IP + ["Control"]

    styles_all = ['-'] * len(n_shot_list) + ['-'] * len(n_inst_list) + ['--'] 

    if save is True:
        save_path = os.path.join(save_root, model_name, d_name, "plots", "Fix")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, 
            f"fix_{prompt_type_2_be_fixed}_{prompt_index_2_be_fixed}_with_EP_{fix_component}_average.pdf")
    else:
        save_path = None

    plot_two_curves_subplots(
        data_dictionaries_list=[data_rr, data_acc],
        labels_list=[labels_all, labels_all],
        color_list=[line_colors_all, line_colors_all],
        linestyle_list=[styles_all, styles_all],
        y_labels=['Average Reciprocal Rank', 'Accuracy'],
        titles=['', ''],
        bbox_to_anchor=bbox_to_anchor,
        y_lim=y_lim,
        figsize=figsize, title=f"{d_name}, fix {prompt_type_2_be_fixed}_{prompt_index_2_be_fixed}",
        x_ticks=scaling_factors, x_label="Scaling factor",
        save_path=save_path
    )
