from typing import List, Callable, Tuple, Dict
from pprint import pprint
import matplotlib.pyplot as plt


def check_assumptions(
        assumption_funcs: List[Callable],
        **assuption_func_kwargs
) -> Tuple[bool, Dict[str, bool]]:
    """
    Checks linear regression assumptions and displays each plot separately in Quarto.
    """

    assumptions_dict = {
        assumption_func.__name__: assumption_func(
            **assuption_func_kwargs
        )
        for assumption_func in assumption_funcs
    }

    # Check if all assumptions hold
    assumptions_hold = all(assumptions_dict.values())

    print(f"All {assumptions_hold=}")
    pprint(assumptions_dict)

    return assumptions_hold, assumptions_dict



def show_plot(
        formal_test_holds: bool,
        plot_func: callable,
        plot_func_args: tuple = tuple(),
        plot_func_kwargs: dict = None,
        message: str = "",
        title: str = "",
        xlabel: str = None,
        ylabel: str = None,
        y0line: bool = False,
        pass_axis: bool = False,
):
    if formal_test_holds and not isinstance(formal_test_holds, str):
        return
    
    if plot_func_kwargs is None:
        plot_func_kwargs = {}
    
    fig = plt.figure(figsize=(4, 2))

    if pass_axis:
        ax = fig.add_subplot(111)
        plot_func_kwargs['ax'] = ax

    plot_func(*plot_func_args, **plot_func_kwargs)

    plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if y0line:
        plt.axhline(y=0, color='r', linestyle='--')

    plt.subplots_adjust(bottom=0.18)
    plt.figtext(
        0.5, 0.01, message, wrap=True,
        horizontalalignment='center', fontsize=9,
        bbox=dict(facecolor='white', alpha=0.8, pad=6)
    )
    plt.show()
