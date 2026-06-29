from __future__ import annotations

import pandas as pd

from .catalog import BUILTIN_CHART_TYPES
from .base import ChartBackend, register_backend
from .utils import ensure_dir, setup_mpl


class MatplotlibBackend(ChartBackend):
    name = "matplotlib"
    supported_charts = BUILTIN_CHART_TYPES

    def render(self, chart_type: str, df: pd.DataFrame, output_file: str, **kwargs) -> str:
        ensure_dir(output_file)
        handler = getattr(self, f"_{chart_type}", None)
        if handler is None:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        return handler(df=df, output_file=output_file, **kwargs)

    def _bar(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(11, 6))
        if hue:
            hue_values = df[hue].unique()
            categories = df[x].unique()
            width = 0.8 / len(hue_values)
            cmap = plt.cm.get_cmap("tab10", len(hue_values))
            for index, hue_value in enumerate(hue_values):
                subset = df[df[hue] == hue_value]
                values = [subset[subset[x] == category][y].mean() for category in categories]
                offset = (index - len(hue_values) / 2 + 0.5) * width
                ax.bar(
                    [position + offset for position in range(len(categories))],
                    values,
                    width,
                    label=str(hue_value),
                    color=cmap(index),
                    alpha=0.85,
                )
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=35, ha="right")
            ax.legend(title=hue.replace("_", " ").title())
        else:
            aggregate = df.groupby(x)[y].mean().sort_values(ascending=False)
            colors = plt.cm.get_cmap("tab10", len(aggregate))(range(len(aggregate)))
            ax.bar(range(len(aggregate)), aggregate.values, color=colors, alpha=0.85)
            ax.set_xticks(range(len(aggregate)))
            ax.set_xticklabels(aggregate.index, rotation=35, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _line(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(13, 5))
        cmap = plt.cm.get_cmap("tab10")
        
        # Aggregate y to mean in case there are multiple observations per x
        if hue:
            agg_df = df.groupby([x, hue], as_index=False)[y].mean()
            for index, (label, group) in enumerate(agg_df.groupby(hue)):
                # Also sort to avoid lines jumping back and forth
                group = group.sort_values(x)
                ax.plot(group[x], group[y], marker="o", linewidth=2, color=cmap(index), label=str(label), alpha=0.9)
            ax.legend(title=hue.replace("_", " ").title())
        else:
            agg_df = df.groupby(x, as_index=False)[y].mean().sort_values(x)
            ax.plot(agg_df[x], agg_df[y], marker="o", linewidth=2, color=cmap(0))
            
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _scatter(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, size: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(9, 7))
        cmap = plt.cm.get_cmap("tab10")
        if hue:
            for index, (label, group) in enumerate(df.groupby(hue)):
                scatter_size = group[size] * 20 if size else 40
                ax.scatter(group[x], group[y], s=scatter_size, color=cmap(index), label=str(label), alpha=0.65)
            ax.legend(title=hue.replace("_", " ").title())
        else:
            scatter_size = df[size] * 20 if size else 40
            ax.scatter(df[x], df[y], s=scatter_size, color=cmap(0), alpha=0.65)
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

    def _histogram(self, df: pd.DataFrame, output_file: str, column: str, title: str = "", bins: int = 20, hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.get_cmap("tab10")
        if hue:
            for index, (label, group) in enumerate(df.groupby(hue)):
                ax.hist(group[column], bins=bins, color=cmap(index), alpha=0.55, density=True, label=str(label))
            ax.legend(title=hue.replace("_", " ").title())
        else:
            ax.hist(df[column], bins=bins, color=cmap(0), alpha=0.7, density=True)
            mean_value = df[column].mean()
            ax.axvline(mean_value, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"mean={mean_value:.1f}")
            ax.legend()
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(column.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _heatmap(self, df: pd.DataFrame, output_file: str, title: str = "Correlation Heatmap", **kwargs) -> str:
        plt = setup_mpl()
        corr = df.select_dtypes(include="number").corr()
        fig, ax = plt.subplots(figsize=(11, 9))
        image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(image, ax=ax, label="Pearson r", shrink=0.8)
        columns = corr.columns.tolist()
        ax.set_xticks(range(len(columns)))
        ax.set_yticks(range(len(columns)))
        ax.set_xticklabels(columns, rotation=40, ha="right")
        ax.set_yticklabels(columns)
        for row_index in range(len(columns)):
            for column_index in range(len(columns)):
                value = corr.values[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) > 0.6 else "black",
                )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _box(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(11, 6))
        categories = df[x].unique()
        data_by_category = [df[df[x] == category][y].dropna().values for category in categories]
        boxplot = ax.boxplot(
            data_by_category,
            patch_artist=True,
            notch=False,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
        )
        cmap = plt.cm.get_cmap("Set2", len(categories))
        for patch, color in zip(boxplot["boxes"], [cmap(index) for index in range(len(categories))]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels(categories, rotation=35, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _violin(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(11, 6))
        categories = df[x].unique()
        data_by_category = [df[df[x] == category][y].dropna().values for category in categories]
        parts = ax.violinplot(data_by_category, positions=range(1, len(categories) + 1), showmedians=True, showextrema=True)
        cmap = plt.cm.get_cmap("Pastel1", len(categories))
        for index, body in enumerate(parts["bodies"]):
            body.set_facecolor(cmap(index))
            body.set_alpha(0.8)
        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels(categories, rotation=35, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _pie(self, df: pd.DataFrame, output_file: str, labels: str, values: str, title: str = "", **kwargs) -> str:
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = plt.cm.get_cmap("tab20", len(df))(range(len(df)))
        _, _, autotexts = ax.pie(
            df[values],
            labels=df[labels],
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _area(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(12, 5))
        
        if hue:
            agg_df = df.groupby([x, hue], as_index=False)[y].mean()
            for label, group in agg_df.groupby(hue):
                ordered = group.sort_values(x)
                ax.fill_between(ordered[x], ordered[y], alpha=0.35, label=str(label))
                ax.plot(ordered[x], ordered[y], linewidth=1.5)
            ax.legend(title=hue.replace("_", " ").title())
        else:
            agg_df = df.groupby(x, as_index=False)[y].mean()
            ordered = agg_df.sort_values(x)
            ax.fill_between(ordered[x], ordered[y], alpha=0.35)
            ax.plot(ordered[x], ordered[y], linewidth=1.5)
            
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(x.replace("_", " ").title())
        ax.set_ylabel(y.replace("_", " ").title())
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _count(self, df: pd.DataFrame, output_file: str, x: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(11, 6))
        cmap = plt.cm.get_cmap("Set2")
        order = df[x].value_counts().index.tolist()
        if hue:
            hue_values = df[hue].unique()
            width = 0.8 / len(hue_values)
            for idx, hue_val in enumerate(hue_values):
                counts = [df[(df[x] == cat) & (df[hue] == hue_val)].shape[0] for cat in order]
                offset = (idx - len(hue_values) / 2 + 0.5) * width
                bars = ax.bar(
                    [pos + offset for pos in range(len(order))],
                    counts,
                    width,
                    label=str(hue_val),
                    color=cmap(idx),
                    alpha=0.85,
                )
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=35, ha="right")
            ax.legend(title=hue.replace("_", " ").title())
        else:
            counts = [df[df[x] == cat].shape[0] for cat in order]
            colors = cmap(range(len(order)))
            bars = ax.bar(range(len(order)), counts, color=colors, alpha=0.85)
            for bar, count in zip(bars, counts):
                ax.annotate(
                    f"{count:,}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_title(title or f"Count of {x.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _kde(self, df: pd.DataFrame, output_file: str, column: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        from scipy.stats import gaussian_kde  # type: ignore[import]
        import numpy as np
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.get_cmap("tab10")
        x_range = np.linspace(df[column].min(), df[column].max(), 300)
        if hue:
            for idx, (label, group) in enumerate(df.groupby(hue)):
                data = group[column].dropna().values
                if len(data) > 1:
                    kde = gaussian_kde(data)
                    ax.plot(x_range, kde(x_range), color=cmap(idx), linewidth=2.0, label=str(label))
                    ax.fill_between(x_range, kde(x_range), alpha=0.25, color=cmap(idx))
            ax.legend(title=hue.replace("_", " ").title())
        else:
            data = df[column].dropna().values
            kde = gaussian_kde(data)
            color = cmap(0)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2.0)
            ax.fill_between(x_range, kde(x_range), alpha=0.3, color=color)
            mean_val = df[column].mean()
            ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"mean={mean_val:.2f}")
            ax.legend()
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
        import numpy as np
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(11, 6))
        cmap = plt.cm.get_cmap("tab10")
        categories = df[x].unique()
        cat_to_pos = {cat: i for i, cat in enumerate(categories)}
        if hue:
            hue_values = df[hue].unique()
            for hue_idx, hue_val in enumerate(hue_values):
                subset = df[df[hue] == hue_val]
                xs = [cat_to_pos[c] + np.random.uniform(-0.2, 0.2) for c in subset[x]]
                ax.scatter(xs, subset[y], color=cmap(hue_idx), alpha=0.5, s=20, label=str(hue_val), edgecolors="none")
            ax.legend(title=hue.replace("_", " ").title())
        else:
            color = cmap(0)
            xs = [cat_to_pos[c] + np.random.uniform(-0.25, 0.25) for c in df[x]]
            ax.scatter(xs, df[y], color=color, alpha=0.45, s=18, edgecolors="none")
        # Draw mean lines
        for cat, pos in cat_to_pos.items():
            mean_val = df[df[x] == cat][y].mean()
            ax.hlines(mean_val, pos - 0.35, pos + 0.35, colors="#333333", linewidth=2.5, zorder=5)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=35, ha="right")
        ax.set_title(title or f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _regression(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import numpy as np
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(10, 7))
        cmap = plt.cm.get_cmap("tab10")
        if hue:
            for idx, (label, group) in enumerate(df.groupby(hue)):
                xvals = group[x].values.astype(float)
                yvals = group[y].values.astype(float)
                mask = np.isfinite(xvals) & np.isfinite(yvals)
                ax.scatter(xvals[mask], yvals[mask], color=cmap(idx), alpha=0.4, s=25, edgecolors="none", label=str(label))
                if mask.sum() > 1:
                    coeffs = np.polyfit(xvals[mask], yvals[mask], 1)
                    x_line = np.linspace(xvals[mask].min(), xvals[mask].max(), 100)
                    ax.plot(x_line, np.polyval(coeffs, x_line), color=cmap(idx), linewidth=2.0)
            ax.legend(title=hue.replace("_", " ").title())
        else:
            xvals = df[x].values.astype(float)
            yvals = df[y].values.astype(float)
            mask = np.isfinite(xvals) & np.isfinite(yvals)
            color = cmap(0)
            ax.scatter(xvals[mask], yvals[mask], color=color, alpha=0.45, s=30, edgecolors="none")
            if mask.sum() > 1:
                coeffs = np.polyfit(xvals[mask], yvals[mask], 1)
                x_line = np.linspace(xvals[mask].min(), xvals[mask].max(), 100)
                ax.plot(x_line, np.polyval(coeffs, x_line), color="#e74c3c", linewidth=2.0)
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
        import numpy as np
        plt = setup_mpl()
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if columns:
            plot_cols = [c for c in columns if c in df.columns]
        else:
            plot_cols = numeric_cols[:5]
        n = len(plot_cols)
        cmap = plt.cm.get_cmap("tab10")
        fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
        if n == 1:
            axes = [[axes]]
        hue_vals = df[hue].unique() if hue else [None]
        for row_idx, col_y in enumerate(plot_cols):
            for col_idx, col_x in enumerate(plot_cols):
                ax = axes[row_idx][col_idx]
                if row_idx == col_idx:
                    # Diagonal: histogram
                    for h_idx, h_val in enumerate(hue_vals):
                        subset = df[df[hue] == h_val] if hue else df
                        ax.hist(subset[col_x].dropna(), bins=15, color=cmap(h_idx), alpha=0.5, density=True)
                else:
                    for h_idx, h_val in enumerate(hue_vals):
                        subset = df[df[hue] == h_val] if hue else df
                        ax.scatter(subset[col_x], subset[col_y], color=cmap(h_idx), alpha=0.35, s=8, edgecolors="none")
                if row_idx == n - 1:
                    ax.set_xlabel(col_x.replace("_", " ").title(), fontsize=8)
                else:
                    ax.set_xticklabels([])
                if col_idx == 0:
                    ax.set_ylabel(col_y.replace("_", " ").title(), fontsize=8)
                else:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=6)
        if title:
            fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()
        plt.savefig(output_file, dpi=120, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _stacked_bar(self, df: pd.DataFrame, output_file: str, x: str, y: str, stack: str, title: str = "", normalize: bool = False, **kwargs) -> str:
        import numpy as np
        plt = setup_mpl()
        pivot = df.groupby([x, stack])[y].sum().unstack(fill_value=0)
        if normalize:
            pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
        cmap = plt.cm.get_cmap("Set2", len(pivot.columns))
        colors = [cmap(i) for i in range(len(pivot.columns))]
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(pivot))
        for col, color in zip(pivot.columns, colors):
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
        import numpy as np
        from matplotlib.lines import Line2D

        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(10, 7))
        cmap = plt.cm.get_cmap("Set2")

        size_vals = pd.to_numeric(df[size], errors="coerce").values.astype(float)
        size_min, size_max = float(np.min(size_vals)), float(np.max(size_vals))
        size_min_plot, size_max_plot = 80.0, 520.0

        def _to_plot_size(values: np.ndarray) -> np.ndarray:
            if size_max <= size_min:
                return np.full_like(values, (size_min_plot + size_max_plot) / 2.0)
            return size_min_plot + ((values - size_min) / (size_max - size_min)) * (size_max_plot - size_min_plot)

        if hue:
            for idx, (label, group) in enumerate(df.groupby(hue)):
                sv = pd.to_numeric(group[size], errors="coerce").values.astype(float)
                sz = _to_plot_size(sv)
                ax.scatter(
                    group[x], group[y], s=sz,
                    color=cmap(idx % 8), alpha=0.62,
                    edgecolors="white", linewidths=0.6,
                )

            hue_handles = [
                Line2D(
                    [0], [0], marker="o", linestyle="", markersize=8,
                    markerfacecolor=cmap(idx % 8), markeredgecolor="white",
                    markeredgewidth=0.6, label=str(label),
                )
                for idx, label in enumerate(df[hue].dropna().unique().tolist())
            ]
            hue_legend = ax.legend(
                handles=hue_handles,
                title=hue.replace("_", " ").title(),
                loc="upper left",
                framealpha=0.85,
            )
            ax.add_artist(hue_legend)
        else:
            ax.scatter(df[x], df[y], s=_to_plot_size(size_vals), color=cmap(0), alpha=0.62,
                       edgecolors="white", linewidths=0.6)

        reps = np.unique(np.round(np.linspace(size_min, size_max, 3), 0))
        size_handles = [
            Line2D(
                [0], [0], marker="o", linestyle="",
                markersize=max(4.0, np.sqrt(float(_to_plot_size(np.array([val]))[0])) * 0.55),
                markerfacecolor="#555555", markeredgecolor="white", markeredgewidth=0.6,
                alpha=0.75, label=f"{int(val):,}" if float(val).is_integer() else f"{val:,.1f}",
            )
            for val in reps
        ]
        ax.legend(
            handles=size_handles,
            title=size.replace("_", " ").title(),
            loc="upper right",
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
        fig, ax = plt.subplots(figsize=(13, 5))
        cmap = plt.cm.get_cmap("tab10")
        if hue:
            agg_df = df.groupby([x, hue], as_index=False)[y].mean()
            for idx, (label, group) in enumerate(agg_df.groupby(hue)):
                ordered = group.sort_values(x)
                ax.step(ordered[x], ordered[y], where="post",
                        color=cmap(idx), linewidth=2.2, label=str(label), alpha=0.9)
                ax.fill_between(ordered[x], ordered[y], step="post", alpha=0.12, color=cmap(idx))
            ax.legend(title=hue.replace("_", " ").title())
        else:
            ordered = df.groupby(x, as_index=False)[y].mean().sort_values(x)
            ax.step(ordered[x], ordered[y], where="post", color=cmap(0), linewidth=2.2)
            ax.fill_between(ordered[x], ordered[y], step="post", alpha=0.15, color=cmap(0))
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
        """Beeswarm-style plot using jittered scatter with separation."""
        import numpy as np
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(11, 6))
        cmap = plt.cm.get_cmap("tab10")
        categories = df[x].unique()
        cat_to_pos = {cat: i for i, cat in enumerate(categories)}

        rng = np.random.default_rng(42)
        if hue:
            hue_values = df[hue].unique()
            offsets = np.linspace(-0.25, 0.25, len(hue_values))
            for hue_idx, hue_val in enumerate(hue_values):
                subset = df[df[hue] == hue_val]
                base_x = np.array([cat_to_pos[c] + offsets[hue_idx] for c in subset[x]])
                xs = base_x + rng.uniform(-0.08, 0.08, size=len(subset))
                ax.scatter(xs, subset[y], color=cmap(hue_idx), alpha=0.6, s=22,
                           edgecolors="none", label=str(hue_val))
            ax.legend(title=hue.replace("_", " ").title())
        else:
            xs = np.array([cat_to_pos[c] + rng.uniform(-0.25, 0.25) for c in df[x]])
            ax.scatter(xs, df[y], color=cmap(0), alpha=0.5, s=20, edgecolors="none")

        for cat, pos in cat_to_pos.items():
            med = df[df[x] == cat][y].median()
            ax.hlines(med, pos - 0.38, pos + 0.38, colors="#222222", linewidth=2.2, zorder=6)

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=35, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(y.replace("_", " ").title(), fontsize=11)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _ecdf(self, df: pd.DataFrame, output_file: str, column: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import numpy as np
        plt = setup_mpl()
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.cm.get_cmap("tab10")
        if hue:
            for idx, (label, group) in enumerate(df.groupby(hue)):
                vals = np.sort(group[column].dropna().values)
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ax.step(vals, cdf, color=cmap(idx), linewidth=2.2, label=str(label), alpha=0.9, where="post")
            ax.legend(title=hue.replace("_", " ").title())
        else:
            vals = np.sort(df[column].dropna().values)
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, cdf, color=cmap(0), linewidth=2.2, where="post")
            ax.fill_between(vals, cdf, step="post", alpha=0.12, color=cmap(0))
        ax.set_title(title or f"ECDF — {column.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel(column.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Cumulative Probability", fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")
        return output_file

    def _funnel(self, df: pd.DataFrame, output_file: str, labels: str, values: str, title: str = "", **kwargs) -> str:
        import numpy as np
        plt = setup_mpl()
        sorted_df = df.sort_values(values, ascending=False).reset_index(drop=True)
        max_val = float(sorted_df[values].max())
        n = len(sorted_df)
        cmap = plt.cm.get_cmap("Blues", n + 2)
        colors = [cmap(n - i + 1) for i in range(n)]
        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.85)))
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            width = float(row[values])
            ax.barh(i, width, color=colors[i], alpha=0.9, height=0.65)
            ax.barh(i, -width, color=colors[i], alpha=0.9, height=0.65)
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


register_backend(MatplotlibBackend())