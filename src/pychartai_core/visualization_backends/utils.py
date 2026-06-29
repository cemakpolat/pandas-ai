from __future__ import annotations

import os


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def setup_mpl():
    """Import matplotlib with a non-interactive backend and clean state."""
    import matplotlib

    if matplotlib.get_backend() != "Agg":
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.close("all")
    return plt


def setup_sns():
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    return sns


def plotly_available() -> bool:
    try:
        import plotly  # noqa: F401

        return True
    except ImportError:
        return False


def save_plotly(fig, output_file: str) -> str:
    """Save a plotly figure as HTML or PNG, falling back to HTML.

    HTML files are always written as fully self-contained pages
    (``include_plotlyjs=True``) so they open correctly without internet access
    even when plotly.js cannot be loaded from a CDN.
    """
    ensure_dir(output_file)
    if output_file.endswith(".html"):
        fig.write_html(output_file, include_plotlyjs=True, full_html=True)
        return output_file

    try:
        fig.write_image(output_file)
        return output_file
    except Exception:
        html_path = os.path.splitext(output_file)[0] + ".html"
        fig.write_html(html_path, include_plotlyjs=True, full_html=True)
        return html_path


def export_figure(fig, output_file: str, *, dpi: int = 150) -> str:
    """Export a matplotlib figure to PNG, SVG, or PDF.

    The format is inferred from *output_file* extension.  Falls back to PNG.
    """
    ensure_dir(output_file)
    ext = os.path.splitext(output_file)[1].lower()
    if ext not in ('.png', '.svg', '.pdf'):
        ext = '.png'
        output_file = os.path.splitext(output_file)[0] + ext
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight', format=ext.lstrip('.'))
    return output_file


def export_plotly_data(fig) -> dict:
    """Extract JSON-serializable data from a Plotly figure."""
    return fig.to_dict()