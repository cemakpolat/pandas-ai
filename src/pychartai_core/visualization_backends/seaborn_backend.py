from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .catalog import BUILTIN_CHART_TYPES
from .base import ChartBackend, register_backend
from .utils import ensure_dir, setup_mpl, setup_sns


class SeabornBackend(ChartBackend):
    name = "seaborn"
    supported_charts = BUILTIN_CHART_TYPES

    def render(self, chart_type: str, df: pd.DataFrame, output_file: str, **kwargs) -> str:
        ensure_dir(output_file)
        handler = getattr(self, f"_{chart_type}", None)
        if handler is None:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        with warnings.catch_warnings():
            # Seaborn's legacy internals currently trigger pandas FutureWarnings.
            warnings.filterwarnings(
                "ignore",
                message="use_inf_as_na option is deprecated.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="When grouping with a length-1 list-like.*",
                category=FutureWarning,
            )
            return handler(df=df, output_file=output_file, **kwargs)

    def _bar(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(11, 6))
        order = df.groupby(x)[y].mean().sort_values(ascending=False).index.tolist()
        import seaborn as _sns_ver
        _sns_new = tuple(int(x) for x in _sns_ver.__version__.split(".")[:2]) >= (0, 13)
        effective_hue = hue if hue is not None else (x if _sns_new else None)
        barplot_kwargs = dict(
            data=df, x=x, y=y, hue=effective_hue,
            order=order, ax=ax, palette="Set2", errorbar=None,
        )
        if _sns_new:
            barplot_kwargs["legend"] = hue is not None
        sns.barplot(**barplot_kwargs)
        for patch in ax.patches:
            height = patch.get_height()
            if height > 0 and not np.isnan(height):
                ax.annotate(
                    f"{height:,.0f}",
                    xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.xticks(rotation=35, ha="right")
        if hue:
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _line(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(13, 5))
        lineplot_kwargs = {
            "data": df,
            "x": x,
            "y": y,
            "hue": hue,
            "ax": ax,
            "linewidth": 2.0,
            "markers": True,
            "dashes": False,
            "errorbar": None,
        }
        if hue:
            lineplot_kwargs["palette"] = "tab10"
        else:
            lineplot_kwargs["color"] = sns.color_palette("tab10", n_colors=1)[0]
        sns.lineplot(**lineplot_kwargs)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.xticks(rotation=45, ha="right")
        if hue:
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _scatter(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, size: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(9, 7))
        scatter_kwargs = {
            "data": df,
            "x": x,
            "y": y,
            "hue": hue,
            "size": size,
            "ax": ax,
            "alpha": 0.65,
            "edgecolor": "none",
        }
        if hue:
            scatter_kwargs["palette"] = "Set1"
        else:
            scatter_kwargs["color"] = sns.color_palette("Set1", n_colors=1)[0]
        sns.scatterplot(**scatter_kwargs)
        try:
            sns.regplot(
                data=df,
                x=x,
                y=y,
                ax=ax,
                scatter=False,
                ci=95,
                line_kws={"color": "#e74c3c", "linewidth": 2, "linestyle": "--"},
            )
        except Exception:
            pass
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        if hue:
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8, markerscale=1.4)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _histogram(self, df: pd.DataFrame, output_file: str, column: str, title: str = "", bins: int = 20, hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        is_numeric = pd.api.types.is_numeric_dtype(df[column])
        
        sns.histplot(
            data=df,
            x=column,
            hue=hue,
            bins=bins if is_numeric else "auto",
            kde=is_numeric,
            ax=ax,
            alpha=0.55 if is_numeric else 0.8,
            element="bars",
            palette="Set2" if hue else None,
            stat="density" if is_numeric else "count",
            common_norm=False,
            shrink=1.0 if is_numeric else 0.8,
        )
        if is_numeric:
            sns.rugplot(
                data=df,
                x=column,
                hue=hue,
                ax=ax,
                height=0.04,
                palette="Set2" if hue else None,
                alpha=0.4,
            )
            
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(column.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Density" if is_numeric else "Count", fontsize=11)
        
        if not is_numeric:
            plt.xticks(rotation=45, ha="right")
            
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        if hue and ax.legend_ is not None:
            ax.legend_.set_title(hue.replace("_", " ").title())
            ax.legend_.set_frame_on(True)
            ax.legend_.get_frame().set_alpha(0.8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _heatmap(self, df: pd.DataFrame, output_file: str, title: str = "Correlation Heatmap", **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        corr = df.select_dtypes(include="number").corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            linewidths=0.5,
            ax=ax,
            square=True,
            vmin=-1,
            vmax=1,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
            annot_kws={"size": 9},
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        plt.xticks(rotation=40, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _box(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.boxplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            palette="Set2",
            linewidth=1.4,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
            boxprops={"alpha": 0.85},
        )
        if len(df) <= 800:
            stripplot_kwargs = {
                "data": df,
                "x": x,
                "y": y,
                "hue": hue,
                "ax": ax,
                "alpha": 0.25,
                "size": 3,
                "dodge": bool(hue),
                "jitter": True,
                "legend": False,
            }
            if hue:
                stripplot_kwargs["palette"] = "dark:black"
            else:
                stripplot_kwargs["color"] = "black"
            sns.stripplot(**stripplot_kwargs)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.xticks(rotation=35, ha="right")
        if hue:
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _violin(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.violinplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            inner="quart",
            palette="pastel",
            linewidth=1.5,
            cut=0,
            density_norm="width",
        )
        sample = df.sample(min(len(df), 600), random_state=0)
        stripplot_kwargs = {
            "data": sample,
            "x": x,
            "y": y,
            "hue": hue,
            "ax": ax,
            "alpha": 0.30,
            "size": 2.5,
            "dodge": bool(hue),
            "jitter": True,
            "legend": False,
        }
        if hue:
            stripplot_kwargs["palette"] = "deep"
        else:
            stripplot_kwargs["color"] = sns.color_palette("deep", n_colors=1)[0]
        sns.stripplot(**stripplot_kwargs)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.xticks(rotation=35, ha="right")
        if hue:
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _pie(self, df: pd.DataFrame, output_file: str, labels: str, values: str, title: str = "", **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(9, 8))
        categories = df[labels].astype(str)
        palette = sns.color_palette("Set2", n_colors=max(len(df), 3))
        wedges, _, autotexts = ax.pie(
            df[values],
            labels=categories,
            colors=palette[: len(df)],
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.82,
            wedgeprops={"edgecolor": "white", "linewidth": 1.4},
        )
        centre_circle = plt.Circle((0, 0), 0.55, fc="white")
        ax.add_artist(centre_circle)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color("#333333")
        ax.legend(
            wedges,
            categories,
            title=labels.replace("_", " ").title(),
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            framealpha=0.85,
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _area(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(12, 5))
        if hue:
            grouped = df.sort_values([hue, x]).groupby(hue, sort=True)
            palette = sns.color_palette("tab10", n_colors=df[hue].nunique())
            for (label, group), color in zip(grouped, palette):
                ordered = group.sort_values(x)
                sns.lineplot(data=ordered, x=x, y=y, ax=ax, color=color, linewidth=1.9, legend=False)
                ax.fill_between(ordered[x], ordered[y], alpha=0.24, color=color, label=str(label))
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.85)
        else:
            ordered = df.sort_values(x)
            color = sns.color_palette("deep", n_colors=1)[0]
            sns.lineplot(data=ordered, x=x, y=y, ax=ax, color=color, linewidth=2.0, legend=False)
            ax.fill_between(ordered[x], ordered[y], alpha=0.24, color=color)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(x.replace("_", " ").title())
        ax.set_ylabel(y.replace("_", " ").title())
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _count(self, df: pd.DataFrame, output_file: str, x: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        order = df[x].value_counts().index.tolist()
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.countplot(
            data=df,
            x=x,
            hue=hue,
            order=order,
            ax=ax,
            palette="Set2",
            saturation=0.8,
        )
        for patch in ax.patches:
            height = patch.get_height()
            if height > 0 and not np.isnan(height):
                ax.annotate(
                    f"{int(height):,}",
                    xy=(patch.get_x() + patch.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#333333",
                )
        ax.set_title(title or f"Count of {x.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.xticks(rotation=35, ha="right")
        if hue:
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _kde(self, df: pd.DataFrame, output_file: str, column: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(10, 6))
        kde_kwargs: dict = {
            "data": df,
            "x": column,
            "ax": ax,
            "fill": True,
            "linewidth": 2.0,
            "alpha": 0.45,
        }
        if hue:
            kde_kwargs["hue"] = hue
            kde_kwargs["palette"] = "Set1"
            kde_kwargs["common_norm"] = False
        else:
            kde_kwargs["color"] = sns.color_palette("deep", n_colors=1)[0]
        sns.kdeplot(**kde_kwargs)
        mean_val = df[column].mean()
        ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"mean={mean_val:.2f}")
        ax.legend(framealpha=0.8)
        ax.set_title(title or f"KDE — {column.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(column.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _strip(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(11, 6))
        strip_kwargs: dict = {
            "data": df,
            "x": x,
            "y": y,
            "ax": ax,
            "alpha": 0.55,
            "size": 4,
            "jitter": True,
            "dodge": bool(hue),
        }
        if hue:
            strip_kwargs["hue"] = hue
            strip_kwargs["palette"] = "Set1"
        else:
            strip_kwargs["color"] = sns.color_palette("Set1", n_colors=1)[0]
        sns.stripplot(**strip_kwargs)
        # Overlay mean markers
        means = df.groupby(x)[y].mean()
        for i, (cat, mean_val) in enumerate(means.items()):
            ax.hlines(mean_val, i - 0.35, i + 0.35, colors="#333333", linewidth=2.5, zorder=5)
        ax.set_title(title or f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.xticks(rotation=35, ha="right")
        if hue:
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _regression(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        if hue:
            groups = df[hue].unique()
            palette = sns.color_palette("Set1", n_colors=len(groups))
            fig, ax = plt.subplots(figsize=(10, 7))
            for color, group_val in zip(palette, groups):
                subset = df[df[hue] == group_val]
                sns.regplot(
                    data=subset,
                    x=x,
                    y=y,
                    ax=ax,
                    scatter_kws={"alpha": 0.4, "s": 25, "color": color, "edgecolors": "none"},
                    line_kws={"linewidth": 2.0, "color": color},
                    ci=95,
                    label=str(group_val),
                )
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8)
        else:
            fig, ax = plt.subplots(figsize=(10, 7))
            color = sns.color_palette("deep", n_colors=1)[0]
            sns.regplot(
                data=df,
                x=x,
                y=y,
                ax=ax,
                scatter_kws={"alpha": 0.45, "s": 30, "color": color, "edgecolors": "none"},
                line_kws={"linewidth": 2.0, "color": "#e74c3c"},
                ci=95,
            )
        ax.set_title(title or f"{y.replace('_', ' ').title()} ~ {x.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _pairplot(self, df: pd.DataFrame, output_file: str, title: str = "", hue: str | None = None, columns: list | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if columns:
            plot_cols = [c for c in columns if c in df.columns]
        else:
            # Cap at 6 columns to keep the grid readable
            plot_cols = numeric_cols[:6]
        plot_df = df[plot_cols + ([hue] if hue and hue not in plot_cols else [])].copy()
        pairplot_kwargs: dict = {
            "data": plot_df,
            "vars": plot_cols,
            "diag_kind": "kde",
            "plot_kws": {"alpha": 0.4, "s": 15, "edgecolors": "none"},
            "diag_kws": {"fill": True, "alpha": 0.5},
            "corner": True,
        }
        if hue:
            pairplot_kwargs["hue"] = hue
            pairplot_kwargs["palette"] = "Set1"
        else:
            pairplot_kwargs["plot_kws"]["color"] = sns.color_palette("deep", n_colors=1)[0]
        grid = sns.pairplot(**pairplot_kwargs)
        if title:
            grid.figure.suptitle(title, y=1.01, fontsize=14, fontweight="bold")
        grid.figure.tight_layout()
        grid.figure.savefig(output_file, dpi=130, bbox_inches="tight")
        plt.close("all")
        return output_file

    def _stacked_bar(self, df: pd.DataFrame, output_file: str, x: str, y: str, stack: str, title: str = "", normalize: bool = False, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        pivot = df.groupby([x, stack])[y].sum().unstack(fill_value=0)
        if normalize:
            pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
        palette = sns.color_palette("Set2", n_colors=len(pivot.columns))
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(pivot))
        for col, color in zip(pivot.columns, palette):
            ax.bar(range(len(pivot)), pivot[col].values, bottom=bottom,
                   color=color, label=str(col), alpha=0.87, width=0.7)
            bottom += pivot[col].values
        ax.set_xticks(range(len(pivot)))
        ax.set_xticklabels(pivot.index, rotation=35, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Share (%)" if normalize else y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(title=stack.replace("_", " ").title(), bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _bubble(self, df: pd.DataFrame, output_file: str, x: str, y: str, size: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        from matplotlib.lines import Line2D

        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(10, 7))

        # Keep bubble areas in a readable range and build legends manually.
        # Seaborn's automatic (hue + size) legend can add confusing extra items.
        size_min_plot, size_max_plot = 80, 520
        scatter_kwargs: dict = {
            "data": df,
            "x": x,
            "y": y,
            "ax": ax,
            "size": size,
            "sizes": (size_min_plot, size_max_plot),
            "alpha": 0.62,
            "edgecolor": "white",
            "linewidth": 0.6,
            "legend": False,
        }
        if hue:
            palette = sns.color_palette("Set2", n_colors=max(1, df[hue].nunique()))
            scatter_kwargs["hue"] = hue
            scatter_kwargs["palette"] = palette
        else:
            palette = [sns.color_palette("deep", n_colors=1)[0]]
            scatter_kwargs["color"] = palette[0]

        sns.scatterplot(**scatter_kwargs)

        def _to_plot_size(v: float, vmin: float, vmax: float) -> float:
            if vmax <= vmin:
                return float((size_min_plot + size_max_plot) / 2)
            return float(size_min_plot + ((v - vmin) / (vmax - vmin)) * (size_max_plot - size_min_plot))

        size_vals = pd.to_numeric(df[size], errors="coerce").dropna().values
        if len(size_vals) > 0:
            vmin = float(np.min(size_vals))
            vmax = float(np.max(size_vals))
            reps = np.unique(np.round(np.linspace(vmin, vmax, 3), 0))
            if len(reps) == 1:
                reps = np.array([reps[0]])
            size_handles = [
                Line2D(
                    [0], [0], marker="o", linestyle="",
                    markersize=max(4.0, np.sqrt(_to_plot_size(float(val), vmin, vmax)) * 0.55),
                    markerfacecolor="#555555", markeredgecolor="white", markeredgewidth=0.6,
                    alpha=0.7, label=f"{int(val):,}" if float(val).is_integer() else f"{val:,.1f}",
                )
                for val in reps
            ]
            size_legend = ax.legend(
                handles=size_handles,
                title=size.replace("_", " ").title(),
                loc="upper right",
                framealpha=0.85,
            )
            ax.add_artist(size_legend)

        if hue:
            levels = list(pd.unique(df[hue]))
            hue_handles = [
                Line2D(
                    [0], [0], marker="o", linestyle="", markersize=8,
                    markerfacecolor=palette[idx % len(palette)], markeredgecolor="white",
                    markeredgewidth=0.6, alpha=0.9, label=str(level),
                )
                for idx, level in enumerate(levels)
            ]
            ax.legend(
                handles=hue_handles,
                title=hue.replace("_", " ").title(),
                loc="upper left",
                framealpha=0.85,
            )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _step(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(13, 5))
        palette = sns.color_palette("tab10")
        if hue:
            agg_df = df.groupby([x, hue], as_index=False)[y].mean()
            for idx, (label, group) in enumerate(agg_df.groupby(hue)):
                ordered = group.sort_values(x)
                ax.step(ordered[x], ordered[y], where="post",
                        color=palette[idx % len(palette)], linewidth=2.2,
                        label=str(label), alpha=0.9)
                ax.fill_between(ordered[x], ordered[y], step="post",
                                alpha=0.12, color=palette[idx % len(palette)])
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8)
        else:
            ordered = df.groupby(x, as_index=False)[y].mean().sort_values(x)
            color = palette[0]
            ax.step(ordered[x], ordered[y], where="post", color=color, linewidth=2.2)
            ax.fill_between(ordered[x], ordered[y], step="post", alpha=0.15, color=color)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _swarm(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        sample = df.sample(min(len(df), 500), random_state=0) if len(df) > 500 else df
        fig, ax = plt.subplots(figsize=(11, 6))
        swarm_kwargs: dict = {
            "data": sample,
            "x": x,
            "y": y,
            "ax": ax,
            "size": 4,
            "alpha": 0.75,
            "warn_thresh": 1.0,
        }
        if hue:
            swarm_kwargs["hue"] = hue
            swarm_kwargs["palette"] = "Set1"
            swarm_kwargs["dodge"] = True
        else:
            swarm_kwargs["color"] = sns.color_palette("Set1", n_colors=1)[0]
        sns.swarmplot(**swarm_kwargs)
        sns.boxplot(
            data=sample, x=x, y=y, ax=ax,
            boxprops={"alpha": 0.25}, flierprops={"alpha": 0},
            whiskerprops={"alpha": 0.5}, capprops={"alpha": 0.5},
            medianprops={"color": "#333333", "linewidth": 1.8},
            palette=["#ffffff"] * df[x].nunique(),
            hue=hue,
        )
        # Remove duplicate legend entries from the overlay boxplot
        handles, labels_list = ax.get_legend_handles_labels()
        if hue and handles:
            seen = {}
            for h, lb in zip(handles, labels_list):
                if lb not in seen:
                    seen[lb] = h
            ax.legend(list(seen.values()), list(seen.keys()),
                      title=hue.replace("_", " ").title(), framealpha=0.8)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.xticks(rotation=35, ha="right")
        if hue:
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _ecdf(self, df: pd.DataFrame, output_file: str, column: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        fig, ax = plt.subplots(figsize=(10, 6))
        ecdf_kwargs: dict = {
            "data": df,
            "x": column,
            "ax": ax,
            "linewidth": 2.2,
            "alpha": 0.9,
        }
        if hue:
            ecdf_kwargs["hue"] = hue
            ecdf_kwargs["palette"] = "Set1"
        else:
            ecdf_kwargs["color"] = sns.color_palette("deep", n_colors=1)[0]
        sns.ecdfplot(**ecdf_kwargs)
        ax.set_title(title or f"ECDF — {column.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(column.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Cumulative Probability", fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        if hue:
            ax.legend(title=hue.replace("_", " ").title(), framealpha=0.8)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _funnel(self, df: pd.DataFrame, output_file: str, labels: str, values: str, title: str = "", **kwargs) -> str:
        plt = setup_mpl()
        sns = setup_sns()
        sorted_df = df.sort_values(values, ascending=False).reset_index(drop=True)
        max_val = float(sorted_df[values].max())
        palette = sns.color_palette("Blues_d", n_colors=len(sorted_df))
        fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_df) * 0.85)))
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            width = float(row[values])
            ax.barh(i, width, color=palette[i], alpha=0.9, height=0.65)
            ax.barh(i, -width, color=palette[i], alpha=0.9, height=0.65)
            ax.text(0, i, f"  {row[labels]}  {row[values]:,.0f}",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(-max_val * 1.1, max_val * 1.1)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file


register_backend(SeabornBackend())