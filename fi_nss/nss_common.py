import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def ns(beta0, beta1, beta2, lambda0, df_maturity):
    result = (
        (beta0) +
        (beta1 * ((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) +
        (beta2 * ((((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) - (np.exp(-df_maturity / lambda0))))
    )
    return result


def nss(beta0, beta1, beta2, beta3, lambda0, lambda1, df_maturity):
    result = (
        (beta0) +
        (beta1 * ((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) +
        (beta2 * ((((1 - np.exp(-df_maturity / lambda0)) / (df_maturity / lambda0))) - (np.exp(-df_maturity / lambda0)))) +
        (beta3 * ((((1 - np.exp(-df_maturity / lambda1)) / (df_maturity / lambda1))) - (np.exp(-df_maturity / lambda1))))
    )
    return result


def plot_all(
    x_maturity: pd.Series,
    y_yield: pd.Series,
    y_marker="o",
    y_color="blue",
    y_label="Yield",
    y_ns=None,
    ns_marker="o",
    ns_color="green",
    ns_label="NS",
    y_nss=None,
    nss_marker="o",
    nss_color="orange",
    nss_label="NSS",
    y_qs=None,
    qs_marker="o",
    qs_color="red",
    qs_label="QS",
    title="",
    grid=True,
    xticks=None,
    yticks=None,
    fontsize=15,
    logx=False,
):
    fig = plt.figure(figsize=(13, 7))
    plt.title(title, fontsize=fontsize)
    plt.gca().set_facecolor("black")
    fig.patch.set_facecolor("white")
    plt.scatter(x_maturity, y_yield, marker=y_marker, c=y_color)
    plt.gca().plot(x_maturity, y_yield, color=y_color, label=y_label)
    if y_ns is not None:
        if ns_marker is not None:
            plt.scatter(x_maturity, y_ns, marker=ns_marker, c=ns_color)
        plt.gca().plot(x_maturity, y_ns, color=ns_color, label=ns_label)
    if y_nss is not None:
        if nss_marker is not None:
            plt.scatter(x_maturity, y_nss, marker=nss_marker, c=nss_color)
        plt.gca().plot(x_maturity, y_nss, color=nss_color, label=nss_label)
    if y_qs is not None:
        if qs_marker is not None:
            plt.scatter(x_maturity, y_qs, marker=qs_marker, c=qs_color)
        plt.gca().plot(x_maturity, y_qs, color=qs_color, label=qs_label)

    plt.xlabel("Period", fontsize=fontsize)
    plt.ylabel("Interest", fontsize=fontsize)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    if yticks is None:
        plt.gca().yaxis.set_ticks(
            np.arange(0, np.ceil(y_yield.max()), 0.2)
        )
    else:
        plt.gca().yaxis.set_ticks(yticks)
    if xticks is not None:  # FIXME
        x_ticks = x_maturity.to_list()
        plt.gca().xaxis.set_ticks(x_ticks)
    if logx:  # true if we need a logaritmic x axis
        plt.gca().set_xscale("log")
    plt.gca().legend(loc="lower right", title="Yield")
    if grid:
        plt.grid()
    plt.show()
