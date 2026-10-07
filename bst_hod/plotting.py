"""Quick-look plot of the HOD."""

import numpy as np


def plot_hod(curves, ax=None, show_true=True, labels=None, logy=True):
    """Plot <N|M> from :func:`bst_hod.hod_curve`.

    Parameters
    ----------
    curves : DataFrame or dict {label: DataFrame}
        One HOD curve, or several (e.g. one per M_lim).
    show_true : bool
        Also draw N_true (mock truth) as a dashed line when available.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    if not isinstance(curves, dict):
        curves = {labels or "BST": curves}
    for i, (lab, c) in enumerate(curves.items()):
        color = f"C{i}"
        ax.errorbar(c["logM_mean"], c["N_bst"], yerr=c["N_bst_err"], fmt="o", color=color,
                    ms=4, capsize=2, label=f"{lab}")
        if show_true and "N_true" in c:
            ax.plot(c["logM_mean"], c["N_true"], "--", color=color, lw=1.2,
                    label=f"{lab} (true)")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M_{200}\,/\,M_\odot)$")
    ax.set_ylabel(r"$\langle N \,|\, M \rangle$")
    ax.legend(fontsize=9)
    return ax
