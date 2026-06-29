from __future__ import annotations

import pandas as pd

from .catalog import BUILTIN_CHART_TYPES
from .base import ChartBackend, register_backend
from .utils import plotly_available, save_plotly


class PlotlyBackend(ChartBackend):
    name = "plotly"
    supported_charts = BUILTIN_CHART_TYPES

    def render(self, chart_type: str, df: pd.DataFrame, output_file: str, **kwargs) -> str:
        if not plotly_available():
            raise RuntimeError("Plotly backend requested but plotly is not installed")

        handler = getattr(self, f"_{chart_type}", None)
        if handler is None:
            raise ValueError(f"Unsupported chart type: {chart_type}")
        return handler(df=df, output_file=output_file, **kwargs)

    def _bar(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        order = df.groupby(x)[y].mean().sort_values(ascending=False).index.tolist()
        fig = px.bar(
            df,
            x=x,
            y=y,
            color=hue,
            title=title,
            barmode="group",
            template="plotly_white",
            category_orders={x: order},
            text_auto=".2s",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_tickangle=-35,
            uniformtext_minsize=8,
            uniformtext_mode="hide",
            plot_bgcolor="rgba(245,245,250,0.6)",
            bargap=0.25,
        )
        return save_plotly(fig, output_file)

    def _line(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        # Aggregate y to mean in case there are multiple observations per x
        if hue:
            plot_df = df.groupby([x, hue], as_index=False)[y].mean().sort_values(x)
        else:
            plot_df = df.groupby(x, as_index=False)[y].mean().sort_values(x)

        fig = px.line(
            plot_df,
            x=x,
            y=y,
            color=hue,
            title=title,
            template="plotly_white",
            markers=True,
            line_shape="spline",
        )
        fig.update_traces(line=dict(width=2.5))
        fig.update_layout(plot_bgcolor="rgba(245,245,250,0.5)")
        return save_plotly(fig, output_file)

    def _scatter(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, size: str | None = None, **kwargs) -> str:
        import plotly.express as px

        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=hue,
            size=size,
            title=title,
            template="plotly_white",
            opacity=0.65,
            marginal_x="rug",
            marginal_y="violin",
        )
        fig.update_layout(plot_bgcolor="rgba(245,245,250,0.5)")
        return save_plotly(fig, output_file)

    def _histogram(self, df: pd.DataFrame, output_file: str, column: str, title: str = "", bins: int = 20, hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        fig = px.histogram(
            df,
            x=column,
            color=hue,
            nbins=bins,
            title=title,
            template="plotly_white",
            marginal="box",
            barmode="overlay",
            opacity=0.6,
        )
        fig.update_layout(plot_bgcolor="rgba(245,245,250,0.5)")
        return save_plotly(fig, output_file)

    def _heatmap(self, df: pd.DataFrame, output_file: str, title: str = "Correlation Heatmap", **kwargs) -> str:
        import plotly.figure_factory as ff

        corr = df.select_dtypes(include="number").corr()
        fig = ff.create_annotated_heatmap(
            z=corr.values.round(2),
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            showscale=True,
        )
        fig.update_layout(title=title, template="plotly_white")
        return save_plotly(fig, output_file)

    def _box(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        fig = px.box(
            df,
            x=x,
            y=y,
            color=hue,
            title=title,
            template="plotly_white",
            points="outliers",
            notched=True,
        )
        fig.update_layout(plot_bgcolor="rgba(245,245,250,0.5)")
        return save_plotly(fig, output_file)

    def _violin(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        fig = px.violin(
            df,
            x=x,
            y=y,
            color=hue,
            title=title,
            template="plotly_white",
            box=True,
            points="outliers",
        )
        fig.update_layout(plot_bgcolor="rgba(245,245,250,0.5)")
        return save_plotly(fig, output_file)

    def _pie(self, df: pd.DataFrame, output_file: str, labels: str, values: str, title: str = "", **kwargs) -> str:
        import plotly.express as px

        fig = px.pie(
            df,
            names=labels,
            values=values,
            title=title,
            template="plotly_white",
            hole=0.3,
        )
        return save_plotly(fig, output_file)

    def _area(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        # Aggregate y to mean in case there are multiple observations per x
        if hue:
            plot_df = df.groupby([x, hue], as_index=False)[y].mean().sort_values(x)
        else:
            plot_df = df.groupby(x, as_index=False)[y].mean().sort_values(x)

        fig = px.area(plot_df, x=x, y=y, color=hue, title=title, template="plotly_white")
        return save_plotly(fig, output_file)

    def _count(self, df: pd.DataFrame, output_file: str, x: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        order = df[x].value_counts().index.tolist()
        if hue:
            count_df = df.groupby([x, hue]).size().reset_index(name="count")
        else:
            count_series = df[x].value_counts().rename_axis(x).reset_index(name="count")
            count_df = count_series
        fig = px.bar(
            count_df,
            x=x,
            y="count",
            color=hue,
            title=title or f"Count of {x.replace('_', ' ').title()}",
            template="plotly_white",
            barmode="group",
            category_orders={x: order},
            text="count",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_tickangle=-35,
            plot_bgcolor="rgba(245,245,250,0.6)",
            bargap=0.25,
        )
        return save_plotly(fig, output_file)

    def _kde(self, df: pd.DataFrame, output_file: str, column: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.figure_factory as ff
        import plotly.graph_objects as go

        if hue:
            groups = df[hue].unique()
            hist_data = [df[df[hue] == g][column].dropna().values for g in groups]
            group_labels = [str(g) for g in groups]
        else:
            hist_data = [df[column].dropna().values]
            group_labels = [column.replace("_", " ").title()]

        try:
            fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=True)
            fig.update_layout(
                title=title or f"KDE — {column.replace('_', ' ').title()}",
                template="plotly_white",
                xaxis_title=column.replace("_", " ").title(),
                yaxis_title="Density",
            )
        except Exception:
            # Fallback: histogram with kde-like appearance
            fig = go.Figure()
            for label, data in zip(group_labels, hist_data):
                fig.add_trace(go.Histogram(x=data, name=label, histnorm="probability density", opacity=0.6))
            fig.update_layout(
                title=title or f"KDE — {column.replace('_', ' ').title()}",
                template="plotly_white",
                barmode="overlay",
            )
        return save_plotly(fig, output_file)

    def _strip(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        fig = px.strip(
            df,
            x=x,
            y=y,
            color=hue,
            title=title or f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}",
            template="plotly_white",
        )
        fig.update_traces(jitter=0.4, opacity=0.6)
        fig.update_layout(
            plot_bgcolor="rgba(245,245,250,0.5)",
            xaxis_tickangle=-35,
        )
        return save_plotly(fig, output_file)

    def _regression(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import numpy as np
        import plotly.graph_objects as go
        import plotly.express as px

        full_title = title or f"{y.replace('_', ' ').title()} ~ {x.replace('_', ' ').title()}"

        if hue:
            fig = px.scatter(
                df, x=x, y=y, color=hue, opacity=0.5,
                title=full_title, template="plotly_white",
            )
            # Add a numpy regression line per hue group
            palette = px.colors.qualitative.Set1
            for idx, (label, group) in enumerate(df.groupby(hue)):
                xv = group[x].values.astype(float)
                yv = group[y].values.astype(float)
                mask = np.isfinite(xv) & np.isfinite(yv)
                if mask.sum() > 1:
                    coeffs = np.polyfit(xv[mask], yv[mask], 1)
                    x_line = np.linspace(xv[mask].min(), xv[mask].max(), 100)
                    fig.add_trace(go.Scatter(
                        x=x_line, y=np.polyval(coeffs, x_line),
                        mode="lines", line=dict(width=2, color=palette[idx % len(palette)]),
                        showlegend=False,
                    ))
        else:
            xv = df[x].values.astype(float)
            yv = df[y].values.astype(float)
            mask = np.isfinite(xv) & np.isfinite(yv)
            fig = px.scatter(
                df, x=x, y=y, opacity=0.45,
                title=full_title, template="plotly_white",
            )
            if mask.sum() > 1:
                coeffs = np.polyfit(xv[mask], yv[mask], 1)
                x_line = np.linspace(xv[mask].min(), xv[mask].max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_line, y=np.polyval(coeffs, x_line),
                    mode="lines", line=dict(width=2.5, color="#e74c3c"),
                    name="trend",
                ))
        fig.update_layout(plot_bgcolor="rgba(245,245,250,0.5)")
        return save_plotly(fig, output_file)

    def _pairplot(self, df: pd.DataFrame, output_file: str, title: str = "", hue: str | None = None, columns: list | None = None, **kwargs) -> str:
        import plotly.express as px

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if columns:
            plot_cols = [c for c in columns if c in df.columns]
        else:
            plot_cols = numeric_cols[:6]

        dimensions = [{"label": c.replace("_", " ").title(), "values": df[c]} for c in plot_cols]
        fig = px.scatter_matrix(
            df,
            dimensions=plot_cols,
            color=hue,
            title=title or "Scatter Matrix",
            template="plotly_white",
            opacity=0.5,
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False, marker_size=4)
        fig.update_layout(height=800)
        return save_plotly(fig, output_file)

    def _stacked_bar(self, df: pd.DataFrame, output_file: str, x: str, y: str, stack: str, title: str = "", normalize: bool = False, **kwargs) -> str:
        import plotly.graph_objects as go

        pivot = df.groupby([x, stack])[y].sum().unstack(fill_value=0)
        if normalize:
            pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for col in pivot.columns:
            fig.add_trace(go.Bar(name=str(col), x=pivot.index.astype(str), y=pivot[col].values))
        fig.update_layout(
            barmode="stack",
            title=title,
            template="plotly_white",
            xaxis_tickangle=-35,
            yaxis_title="Share (%)" if normalize else y.replace("_", " ").title(),
            legend_title=stack.replace("_", " ").title(),
            plot_bgcolor="rgba(245,245,250,0.6)",
        )
        return save_plotly(fig, output_file)

    def _bubble(self, df: pd.DataFrame, output_file: str, x: str, y: str, size: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        size_vals = pd.to_numeric(df[size], errors="coerce")
        max_size = float(size_vals.max()) if len(size_vals) else 1.0
        desired_max_px = 42.0
        sizeref = (2.0 * max_size) / (desired_max_px ** 2) if max_size > 0 else 1.0

        fig = px.scatter(
            df, x=x, y=y, size=size, color=hue,
            title=title,
            template="plotly_white",
            opacity=0.65,
            size_max=desired_max_px,
        )
        fig.update_traces(marker=dict(sizemode="area", sizeref=sizeref, sizemin=5))
        fig.update_layout(plot_bgcolor="rgba(245,245,250,0.5)")
        return save_plotly(fig, output_file)

    def _step(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.graph_objects as go
        import plotly.express as px

        if hue:
            plot_df = df.groupby([x, hue], as_index=False)[y].mean().sort_values(x)
            palette = px.colors.qualitative.Plotly
            fig = go.Figure()
            for idx, (label, group) in enumerate(plot_df.groupby(hue)):
                ordered = group.sort_values(x)
                color = palette[idx % len(palette)]
                fig.add_trace(go.Scatter(
                    x=ordered[x], y=ordered[y], mode="lines",
                    line=dict(shape="hv", width=2.2, color=color),
                    fill="tozeroy", fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba"),
                    name=str(label),
                ))
        else:
            plot_df = df.groupby(x, as_index=False)[y].mean().sort_values(x)
            fig = go.Figure(go.Scatter(
                x=plot_df[x], y=plot_df[y], mode="lines",
                line=dict(shape="hv", width=2.2), fill="tozeroy",
            ))
        fig.update_layout(
            title=title, template="plotly_white",
            xaxis_title=x.replace("_", " ").title(),
            yaxis_title=y.replace("_", " ").title(),
        )
        return save_plotly(fig, output_file)

    def _swarm(self, df: pd.DataFrame, output_file: str, x: str, y: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        fig = px.strip(
            df, x=x, y=y, color=hue,
            title=title or f"{y.replace('_', ' ').title()} by {x.replace('_', ' ').title()}",
            template="plotly_white",
        )
        fig.update_traces(jitter=0.45, opacity=0.65, marker_size=4)
        fig.update_layout(plot_bgcolor="rgba(245,245,250,0.5)", xaxis_tickangle=-35)
        return save_plotly(fig, output_file)

    def _ecdf(self, df: pd.DataFrame, output_file: str, column: str, title: str = "", hue: str | None = None, **kwargs) -> str:
        import plotly.express as px

        fig = px.ecdf(
            df, x=column, color=hue,
            title=title or f"ECDF — {column.replace('_', ' ').title()}",
            template="plotly_white",
            marginal="rug",
        )
        fig.update_layout(
            xaxis_title=column.replace("_", " ").title(),
            yaxis_title="Cumulative Probability",
            plot_bgcolor="rgba(245,245,250,0.5)",
        )
        return save_plotly(fig, output_file)

    def _funnel(self, df: pd.DataFrame, output_file: str, labels: str, values: str, title: str = "", **kwargs) -> str:
        import plotly.express as px

        sorted_df = df.sort_values(values, ascending=False)
        fig = px.funnel(
            sorted_df, x=values, y=labels,
            title=title,
            template="plotly_white",
        )
        fig.update_layout(
            yaxis_title=labels.replace("_", " ").title(),
            xaxis_title=values.replace("_", " ").title(),
        )
        return save_plotly(fig, output_file)


register_backend(PlotlyBackend())