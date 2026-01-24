import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
from pogema_toolbox.views.view_utils import View
from typing import Union, List,Literal,Tuple,Optional

def custom_palette():
    q = list(sns.color_palette("deep"))
    q[1], q[9] = q[9], q[1]
    q[5], q[9] = q[9], q[5]
    q[6], q[9] = q[9], q[6]
    return q

class PlotView(View):
    type: Literal['plot'] = 'plot'
    name: str = None
    x: str = None
    y: str = None
    by: str = 'algorithm'
    width: float = 2.6
    height: float = 2.8
    remove_title: bool = False
    line_width: float = 2.0

    error_bar: Tuple[str, int] = ('ci', 95)

    plt_style: str = 'seaborn-v0_8-colorblind'
    figure_dpi: int = 300
    font_size: int = 8
    legend_font_size: int = 7
    legend_loc: Literal[
        'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right',
        'center left', 'center right', 'lower center', 'upper center', 'center'] = 'best'
    figure_face_color: str = '#FFFFFF'
    use_log_scale_x: bool = False
    use_log_scale_y: bool = False
    markers: bool = True
    line_types: bool = True
    extension: Literal['svg', 'png', 'pdf'] = 'pdf'
    palette: list = custom_palette()
    hue_order: list = None

    tight_layout: bool = True
    ticks: Optional[list] = None
    remove_legend_title: bool = True

def prepare_plt(view):
    plt.style.use(view.plt_style)
    plt.rcParams['figure.figsize'] = (view.width, view.height)
    plt.rcParams['figure.dpi'] = view.figure_dpi
    plt.rcParams['font.size'] = view.font_size
    plt.rcParams['legend.fontsize'] = view.legend_font_size
    plt.rcParams['legend.loc'] = view.legend_loc
    plt.rcParams['figure.facecolor'] = view.figure_face_color

    if view.name:
        plt.title(view.name)

def eval_logs_to_pandas(eval_configs):
    data = {}
    for idx, config in enumerate(eval_configs):
        data[idx] = {**config['env_grid_search'], 'algorithm': config['algorithm']}

        # Adding metrics separately to skip possible lists of metrics (e.g. every step throughput)
        for key, value in config['metrics'].items():
            if isinstance(value, list):
                continue
            data[idx][key] = value
    return pd.DataFrame.from_dict(data, orient='index')

def process_multi_plot_view(results,view,save_path=None):
    df = eval_logs_to_pandas(results)

    over_keys = sorted(df[view.over].unique())
    num_cols = view.num_cols
    num_rows = len(over_keys) // num_cols + (1 if len(over_keys) % num_cols else 0)

    prepare_plt(view)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(view.width * num_cols, view.height * num_rows),
                            sharex=view.share_x, sharey=view.share_y)

    # Adjust for when axs is not a 2D array
    if num_rows == 1 or num_cols == 1:
        axs = np.array(axs).reshape(num_rows, num_cols)

    x, y, hue = prepare_plot_fields(view)
    if view.ticks:
        plt.setp(axs, xticks=view.ticks)

    for idx, over in enumerate(over_keys):
        ax = axs[idx // num_cols, idx % num_cols]
        g = sns.lineplot(x=x, y=y, data=df[df[view.over] == over], errorbar=view.error_bar, hue=hue, ax=ax,
                         style=hue if view.line_types else None, markers=view.markers, palette=view.palette,
                         linewidth=view.line_width, hue_order=view.hue_order, style_order=view.hue_order)
        ax.set_title(over if not view.remove_individual_titles else '')

        if view.remove_individual_titles:
            legend = g.get_legend()
            if legend is not None:  # Check if the legend exists before removing
                legend.remove()

        if view.use_log_scale_x:
            ax.set_xscale('log', base=2)
            from matplotlib.ticker import ScalarFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter())

        g.grid()

    # Remove unused axes
    for idx in range(len(over_keys), num_rows * num_cols):
        fig.delaxes(axs.flatten()[idx])

    if view.tight_layout:
        plt.tight_layout()

    # Handle legend outside the loop to prevent duplication
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=view.legend_bbox_to_anchor, loc=view.legend_loc,
                   ncol=view.legend_columns, fancybox=True, shadow=False)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

import re

def parse_value(value_str):
    # 使用正则表达式提取中等值和误差
    match = re.match(r"([0-9.-]+) ± ([0-9.-]+)", value_str)
    
    if match:
        mean = float(match.group(1))  # 中等值
        error = float(match.group(2))  # 误差
        
        # 计算上下限
        upper = mean + error
        lower = mean - error
        
        return [upper,mean,lower]
    else:
        raise ValueError("输入格式不正确")

def read_lacam_metric(map_index):
    path = f"/home/mapf-gpt/exp/compare_other_method/{map_index}_metric.json"
    with open(path,"r") as f:
        data = json.load(f)
    lacam_data = {}
    for i in data:
        if i["Algorithm"] == "LaCAM":
            lacam_data[int(i["Number of Agents"])] = i
    return lacam_data


def SoC_ratio(json_path,agent_num,map_index,alg):
    path = json_path+f"/{map_index}_metric.json"
    with open(path,"r") as f:
        data = json.load(f)
    result = {i:{} for i in agent_num}    # "±"
    expert_data = read_lacam_metric(map_index)

    for i in agent_num:
        for j in data:
            if j["Number of Agents"] == i:
                parse_data = parse_value(j["SoC"])
                temp_expert_data = parse_value(expert_data[i]["SoC"])[1]
                isr = parse_value(j["ISR"])[1]
                csr = parse_value(j["CSR"])[1]
                soc = parse_value(j["SoC"])[1]
                makespan = parse_value(j["makespan"])[1]
                temp = [k/temp_expert_data for k in parse_data]
                if j["Algorithm"] == alg:
                    # result[i].update({j["Algorithm"]:{"ratio":temp,"isr":isr,"csr":csr}})
                    result[i].update({j["Algorithm"]:{"soc":soc,"makespan":makespan,"csr":csr}})

    return result

def draw_plot(x,bar_data,line_data,labels):
    bar_labels = labels
    line_labels = labels

    fig, ax1 = plt.subplots()

    # ------ 左 y 轴：bar ------
    width = 0.1  # 柱子宽度
    for i, bar_vals in enumerate(bar_data):
        ax1.bar(x + i * width, bar_vals, width, alpha=0.6, label=bar_labels[i])

    ax1.set_ylabel('Bar Values')
    ax1.set_xlabel('X Axis')

    # ------ 右 y 轴：line ------
    ax2 = ax1.twinx()
    for i, line_vals in enumerate(line_data):
        ax2.plot(x, line_vals, marker='o', linewidth=2, label=line_labels[i])

    ax2.set_ylabel('Line Values')
    fig.legend(labels,loc='lower center',ncol=len(labels),fontsize=6)
    # ------ 合并 legend ------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title("Multiple Bars + Multiple Lines with Dual Y Axes")
    plt.show()

if __name__ == "__main__":
    json_path = "/home/mapf-gpt/exp/actions"
    # agent_num = np.array([8,16,32,48,64])
    # index = [1,2]

    # agent_num = np.array([64,128,192,256])
    # index = [4]

    # agent_num = np.array([2,3,4])
    # index = [5]

    agent_num = np.array([[8,16,32,48,64],[32,64,96,128,160,192],[64,128,192,256]])
    index = [1,3,4]
    real_data = []
    alg = "MA-MAPF-GPT-4actions"
    p_1 = []
    p_2 = []
    p_3 = []
    for index_,k in enumerate(agent_num):
        # for i in index:
        data = SoC_ratio(json_path,k,index[index_],alg)
        real_data.append(data)
        for i in k:
            p_1.append(str(real_data[0][i][alg]["makespan"]))
            p_2.append(str(real_data[0][i][alg]["soc"]))
            p_3.append(str(real_data[0][i][alg]["csr"]))
        real_data = []
    print("csr"+"&"+"& ".join(p_3))
    print("makespan:"+"&"+"& ".join(p_1))
    print("soc:"+"&"+"& ".join(p_2))

    
    # isr = []
    # csr = []
    # soc_rat = []
    # for i in data:
    #     temp_isr = []
    #     temp_csr = []
    #     temp_rat = []
    #     for name in data[i]:
    #         temp_isr.append(data[i][name]["isr"])
    #         temp_csr.append(data[i][name]["csr"])
    #         temp_rat.append(data[i][name]["ratio"][1])
    #     isr.append(temp_isr)
    #     csr.append(temp_csr)
    #     soc_rat.append(temp_rat)
    # isr = np.array(isr).T
    # csr = np.array(csr).T
    # soc_rat = np.array(soc_rat).T
    # name = [i for i in data[agent_num[0]].keys()]
    # bar_data = soc_rat
    # line_data = isr
    # x = np.array([1,2,3,4,5])
    # draw_plot(x,bar_data,line_data,name)


