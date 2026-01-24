import json
import seaborn as sns

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from typing import Optional

from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.views.view_utils import View, eval_logs_to_pandas, drop_na
from typing import Tuple

from typing_extensions import Literal
from pathlib import Path

from pogema_toolbox.evaluator import run_views
from pogema_toolbox.views.view_utils import load_from_folder, check_seeds
from matplotlib.ticker import MultipleLocator


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
    width: float = 3
    height: float = 2
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
    locator_set: int = None

class MultiPlotView(PlotView):
    type: Literal['multi-plot'] = 'multi-plot'
    over: str = None
    num_cols: int = 5
    num_rows: int = 1
    share_x: bool = False
    share_y: bool = False
    remove_individual_titles: bool = True
    legend_bbox_to_anchor: Tuple[float, float] = (0.5, -0.05)
    legend_loc: str = 'lower center'
    legend_columns: int = 5
    width: float = 3
    height: float = 3

def prepare_plt(view: PlotView):
    plt.style.use(view.plt_style)
    plt.rcParams['figure.figsize'] = (view.width, view.height)
    plt.rcParams['figure.dpi'] = view.figure_dpi
    plt.rcParams['font.size'] = view.font_size
    plt.rcParams['legend.fontsize'] = view.legend_font_size
    plt.rcParams['legend.loc'] = view.legend_loc
    plt.rcParams['figure.facecolor'] = view.figure_face_color

    if view.name:
        plt.title(view.name)

def prepare_plot_fields(view):
    x = view.x if view.x not in view.rename_fields else view.rename_fields[view.x]
    y = view.y if view.y not in view.rename_fields else view.rename_fields[view.y]
    hue = view.by if view.by not in view.rename_fields else view.rename_fields[view.by]
    return x, y, hue


def process_plot_view(results,view, ax,save_path=None,save_json=False):
    df = eval_logs_to_pandas(results)
    df = drop_na(df)
    if view.hue_order is None:
        view.hue_order = sorted(df['algorithm'].unique())
    if view.sort_by:
        df.sort_values(by=['map_name', 'algorithm'], inplace=True)
    if view.rename_fields:
        df = df.rename(columns=view.rename_fields)
    if save_json:
        record_metric = df.groupby(['Algorithm','Number of Agents'])[['CSR','ISR','SoC']].mean().reset_index().to_dict(orient='records')
        json_path =  '/'.join(str(save_path).split("/")[:-1])+'/metric.json'
        with open(json_path,'w',encoding='utf-8') as f:
            json.dump(record_metric,f,ensure_ascii=False,indent=4)
    prepare_plt(view)
    x, y, hue = prepare_plot_fields(view)

    # fig, ax = plt.subplots()
    if x not in df.keys():
        ToolboxRegistry.warning(f"Could not interpret value {x} for parameter 'x'. Skipping this plot.")
        return
    if y not in df.keys():
        ToolboxRegistry.warning(f"Could not interpret value {y} for parameter 'y'. Skipping this plot.")
        return

    sns.lineplot(x=x, y=y, data=df, errorbar=view.error_bar, hue=hue, hue_order=view.hue_order,
                 style_order=view.hue_order, linewidth=view.line_width,
                 style=hue if view.line_types else None, markers=view.markers,
                 palette=view.palette[:len(view.hue_order)], ax=ax,
                 )
    if not view.remove_title:
        ax.set_title(view.name)

    if view.remove_legend_title:
        ax.legend().set_title('')

    if view.use_log_scale_x:
        ax.set_xscale('log', base=2)
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())

    if view.use_log_scale_y:
        ax.set_yscale('log', base=2)
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
    
    if view.locator_set:
        ax.xaxis.set_major_locator(MultipleLocator(view.locator_set))
    ax.grid(True)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Success Rate")
    ax.get_legend().remove()
    
    return ax

def process_multi_plot_view(json_path,view: MultiPlotView, type,save_path=None):
    num_cols = view.num_cols
    num_rows = view.num_rows
    figsize=(view.width * num_cols, view.height * num_rows)
    fig = plt.figure(figsize=figsize)

    # gs = GridSpec(1,2,figure=fig,wspace=0.3,hspace=0.3)
    # ax1 = fig.add_subplot(gs[0,0])
    # ax2 = fig.add_subplot(gs[0,1])
    # axs = [ax1,ax2]

    gs = GridSpec(1,5,figure=fig,wspace=0.3,hspace=0.3)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    gs2 = GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1,:3],wspace=0.3)
    ax4 = fig.add_subplot(gs2[0,0])
    ax5 = fig.add_subplot(gs2[0,2])
    ax6 = fig.add_subplot(gs2[0,1])

    axs = [ax1,ax2,ax3,ax4,ax5]
    fig.delaxes(ax6)
    prepare_plt(view)

    for index,path in enumerate(json_path):
        results, evaluation_config = load_from_folder(Path(path))
        for key, view_ in evaluation_config['results_views'].items():
            if key[-3:] == "CSR":
                ax = process_plot_view(results,PlotView(**view_),axs[index])

    # # Remove unused axes
    # for idx in range(len(over_keys), num_rows * num_cols):
    #     fig.delaxes(axs.flatten()[idx])
    # Handle legend outside the loop to prevent duplication

        
    if view.tight_layout:
        plt.tight_layout()
        
    handles, labels = ax.get_legend_handles_labels()
    # actions
    labels[0],labels[1],labels[2],labels[3] = labels[2],labels[3],labels[1],labels[0]
    handles[0],handles[1],handles[2],handles[3] = handles[2],handles[3],handles[1],handles[0]
    # noise
    # labels[0],labels[1],labels[4] = labels[1],labels[4],labels[0]
    # handles[0],handles[1],handles[4] = handles[1],handles[4],handles[0]
    # mian
    # labels[0],labels[1],labels[-1] = labels[1],labels[-1],labels[0]
    # handles[0],handles[1],handles[-1] = handles[1],handles[-1],handles[0]
    # noise
    # labels[0],labels[-1] = labels[-1],labels[0]
    # handles[0],handles[-1] = handles[-1],handles[0]
    labels[-1] = "Forcast-MAPF(Ours)"

    if handles:
        fig.legend(handles, labels, bbox_to_anchor=view.legend_bbox_to_anchor, loc=view.legend_loc,
                   ncol=view.legend_columns, fancybox=True, shadow=False)
        
    pos4 = ax4.get_position()
    pos5 = ax5.get_position()

    new4_x = pos4.x0+0.12
    new5_x = pos5.x0-0.12
    ax4.set_position([new4_x,pos4.y0,pos4.width,pos4.height])
    ax5.set_position([new5_x,pos5.y0,pos5.width,pos5.height])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def process_multi_plot_view_(json_path,view: MultiPlotView, type,save_path=None):
    num_cols = view.num_cols
    num_rows = view.num_rows
    figsize=(view.width * num_cols, view.height * num_rows)
    fig = plt.figure(figsize=figsize)

    # gs = GridSpec(1,2,figure=fig,wspace=0.3,hspace=0.3)
    # ax1 = fig.add_subplot(gs[0,0])
    # ax2 = fig.add_subplot(gs[0,1])
    # axs = [ax1,ax2]

    gs = GridSpec(1,5,figure=fig,wspace=0.3,hspace=0.3)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    # gs2 = GridSpecFromSubplotSpec(1,3,subplot_spec=gs[1,:3],wspace=0.3)
    ax4 = fig.add_subplot(gs[0,3])
    ax5 = fig.add_subplot(gs[0,4])
    # ax6 = fig.add_subplot(gs2[0,1])

    axs = [ax1,ax2,ax3,ax4,ax5]
    # fig.delaxes(ax6)
    prepare_plt(view)

    for index,path in enumerate(json_path):
        results, evaluation_config = load_from_folder(Path(path))
        for key, view_ in evaluation_config['results_views'].items():
            if key[-3:] == "CSR":
                ax = process_plot_view(results,PlotView(**view_),axs[index])

    # # Remove unused axes
    # for idx in range(len(over_keys), num_rows * num_cols):
    #     fig.delaxes(axs.flatten()[idx])
    # Handle legend outside the loop to prevent duplication

        

        
    handles, labels = ax.get_legend_handles_labels()
    # actions
    labels[0],labels[1],labels[2],labels[3] = labels[2],labels[3],labels[1],labels[0]
    handles[0],handles[1],handles[2],handles[3] = handles[2],handles[3],handles[1],handles[0]
    # noise
    # labels[0],labels[1],labels[4] = labels[1],labels[4],labels[0]
    # handles[0],handles[1],handles[4] = handles[1],handles[4],handles[0]
    # mian
    # labels[0],labels[1],labels[-1] = labels[1],labels[-1],labels[0]
    # handles[0],handles[1],handles[-1] = handles[1],handles[-1],handles[0]
    # noise
    # labels[1],labels[-1] = labels[-1],labels[1]
    # handles[1],handles[-1] = handles[-1],handles[1]
    labels[-1] = "Forecast-MAPF(Ours)"
    view.legend_bbox_to_anchor = (0.5,-0.2)
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=view.legend_bbox_to_anchor, loc=view.legend_loc,
                   ncol=view.legend_columns, fancybox=True, shadow=False)
    if view.tight_layout:
        plt.tight_layout()
    # pos4 = ax4.get_position()
    # pos5 = ax5.get_position()

    # new4_x = pos4.x0+0.12
    # new5_x = pos5.x0-0.12
    # ax4.set_position([new4_x,pos4.y0,pos4.width,pos4.height])
    # ax5.set_position([new5_x,pos5.y0,pos5.width,pos5.height])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def main():
    current_dir_name = Path(__file__).parent

    results, evaluation_config = load_from_folder(current_dir_name)
    # filtered_results = [result for result in results if result['env_grid_search']['map_name'] != 'puzzle-06']
    # results = filtered_results
    check_seeds(results)
    run_views(results, evaluation_config, eval_dir=current_dir_name)

if __name__ == "__main__":
    paths = ["/home/mapf-gpt/greedy_action_plot/1","/home/mapf-gpt/greedy_action_plot/2","/home/mapf-gpt/greedy_action_plot/3",
             "/home/mapf-gpt/greedy_action_plot/4","/home/mapf-gpt/greedy_action_plot/5"]
    # paths = ["/home/mapf-gpt/greedy_action_plot/2","/home/mapf-gpt/greedy_action_plot/5"]
    view = MultiPlotView()
    save_path = "/home/mapf-gpt/greedy_action_plot/test.pdf"
    process_multi_plot_view_(paths,view,type="Success Rate",save_path=save_path)