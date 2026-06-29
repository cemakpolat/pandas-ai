"""
client_agent.py — Autonomous multi-phase pychartai chart agent.

Phase 1 (NL):     Simple charts via LLM — bar, histogram (6+5 questions)
Phase 2 (NL):     Medium charts via LLM — scatter, line, trend (5+4 questions)
Phase 3 (Direct): Rich direct chart suite called directly — guaranteed output
                  heatmap, box, violin, stacked_bar, bubble, area, kde, ecdf,
                  step, strip, count, funnel, pie, regression, scatter, swarm,
                  pairplot, line, bar — one per helper function
Phase 4 (NL):     Text analysis via LLM — rankings, ratios, summaries

Direct charts (Phase 3) bypass the LLM entirely and call the visualization
helpers with pre-aggregated DataFrames. This guarantees all chart types run
regardless of LLM code-gen quality.

Usage:
  python examples/client_agent.py                           # defaults
  python examples/client_agent.py --backend plotly          # plotly charts
    python examples/client_agent.py --dataset all             # sales + employee + operations
  python examples/client_agent.py --phases 1,2,3,4         # all four phases
  python examples/client_agent.py --workers 4

Make targets:
  make demo-agent
  make demo-agent DATASET=both PHASES=1,2,3,4
  make demo-agent BACKEND=plotly WORKERS=4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import re
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import pandas as pd
import pychartai_core as pai
from pychartai_core.visualization import (
    area_chart,
    bar_chart,
    box_chart,
    bubble_chart,
    count_chart,
    ecdf_chart,
    funnel_chart,
    heatmap,
    histogram,
    kde_chart,
    line_chart,
    pairplot_chart,
    pie_chart,
    regression_chart,
    scatter_chart,
    stacked_bar_chart,
    step_chart,
    strip_chart,
    swarm_chart,
    violin_chart,
)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _make_sales_df() -> pd.DataFrame:
    """120-row sales dataset (12 months × 10 rows) with richer dimensions."""
    products   = ['Widget-A', 'Widget-B', 'Widget-C', 'Gadget-X', 'Gadget-Y', 'Gadget-Z']
    categories = ['Widgets',  'Widgets',  'Widgets',  'Gadgets',  'Gadgets',  'Gadgets']
    channels   = ['Online', 'Retail', 'Partner']
    reps       = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
    regions    = ['North', 'South', 'East', 'West']
    segments   = ['Enterprise', 'SMB', 'Consumer']
    cities     = ['Istanbul', 'Ankara', 'Izmir', 'Bursa', 'Antalya', 'Konya']
    campaigns  = ['Brand', 'Performance', 'Retention', 'Partner']

    base_rev  = [12000, 8500, 9200, 17000, 7500, 11000]
    cost_pct  = [0.58,  0.60, 0.55, 0.56,  0.62, 0.59]
    upx       = [100,   110,  95,   112,   103,  102]   # revenue per unit divisor

    rows = []
    for month in range(1, 13):
        quarter  = (month - 1) // 3 + 1
        seasonal = 1.0 + 0.18 * math.sin((month - 3) * math.pi / 6)
        for i in range(10):
            pi        = i % 6
            product   = products[pi]
            category  = categories[pi]
            region    = regions[(i + month) % 4]
            channel   = channels[(i * 3 + month) % 3]
            rep       = reps[(i + month * 2) % 5]
            segment   = segments[(i + month) % 3]
            city      = cities[(i * 2 + month) % len(cities)]
            campaign  = campaigns[(i + month) % len(campaigns)]
            reg_mult  = [1.10, 0.95, 1.05, 1.00][(i + month) % 4]
            variation = 0.80 + 0.35 * ((i * 7 + month * 3) % 10) / 10
            revenue   = int(base_rev[pi] * seasonal * reg_mult * variation)
            cost      = int(revenue * cost_pct[pi])
            profit    = revenue - cost
            units     = max(1, revenue // upx[pi])
            unit_price = round(revenue / units, 2)
            unit_cost = round(cost / units, 2)
            shipping_cost = int(0.035 * revenue + (i * 17 + month * 13) % 120)
            ad_spend = int(0.06 * revenue + (i * 23 + month * 19) % 180)
            lead_time_days = 1 + ((i * 5 + month * 7) % 12)
            inventory_days = 7 + ((i * 9 + month * 4) % 40)
            on_time_delivery = ((i + month) % 5) != 0
            competitor_price = round(unit_price * (0.92 + ((i + month) % 8) * 0.02), 2)
            margin_pct = round((profit / revenue) * 100, 2) if revenue else 0.0
            # Use a deterministic date to create richer temporal analysis.
            day = 1 + ((i * 3 + month * 2) % 28)
            order_date = pd.Timestamp(year=2024, month=month, day=day)
            disc      = [5, 8, 12][channel_idx := channels.index(channel)]  # noqa: F841
            disc      += (i * 3 + month) % 6
            ret_rate  = round(2.0 + 5.0 * ((i * 5 + month * 7) % 10) / 10, 1)
            sat       = 60 + (i * 11 + month * 13) % 40
            rows.append(dict(
                month=month, quarter=quarter,
                order_date=order_date,
                week=order_date.isocalendar().week,
                day_of_week=order_date.day_name(),
                product=product, category=category,
                region=region, channel=channel,
                city=city, campaign=campaign,
                sales_rep=rep, customer_segment=segment,
                revenue=revenue, cost=cost, profit=profit, units=units,
                unit_price=unit_price, unit_cost=unit_cost,
                margin_pct=margin_pct,
                shipping_cost=shipping_cost, ad_spend=ad_spend,
                lead_time_days=lead_time_days, inventory_days=inventory_days,
                on_time_delivery=on_time_delivery,
                competitor_price=competitor_price,
                discount_pct=disc, return_rate=ret_rate,
                satisfaction_score=sat,
            ))
    return pd.DataFrame(rows)


def _make_employee_df() -> pd.DataFrame:
    """50-employee dataset with richer HR + delivery + risk dimensions."""
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
    roles       = ['Junior', 'Mid', 'Senior', 'Lead', 'Principal']
    regions     = ['North', 'South', 'East', 'West']
    cities      = ['Istanbul', 'Ankara', 'Izmir', 'Bursa']
    names = [
        'Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank', 'Grace', 'Hank',
        'Iris', 'Jack', 'Karen', 'Leo', 'Mia', 'Nate', 'Olivia', 'Pete',
        'Quinn', 'Rose', 'Sam', 'Tara', 'Uma', 'Victor', 'Wendy', 'Xander',
        'Yara', 'Zoe', 'Aaron', 'Bella', 'Chris', 'Diana', 'Ethan', 'Fiona',
        'George', 'Helen', 'Ivan', 'Julia', 'Kevin', 'Laura', 'Mike', 'Nina',
        'Oscar', 'Paula', 'Ray', 'Sandra', 'Tom', 'Ursula', 'Vince', 'Wilma',
        'Xavier', 'Yvonne',
    ]
    base_sal  = {'Engineering': 95000, 'Marketing': 72000, 'Sales': 68000,
                 'HR': 61000, 'Finance': 78000}
    role_mult = {'Junior': 0.72, 'Mid': 0.88, 'Senior': 1.00,
                 'Lead': 1.22, 'Principal': 1.48}
    role_exp  = {'Junior': 1, 'Mid': 3, 'Senior': 6, 'Lead': 9, 'Principal': 14}

    rows = []
    for i in range(50):
        dept   = departments[i % 5]
        role   = roles[(i // 5) % 5]
        region = regions[(i * 3 + i // 5) % 4]
        city = cities[(i + (i // 5)) % len(cities)]
        remote = (i % 3) != 0
        sal_v  = 0.82 + 0.36 * ((i * 7) % 10) / 10
        salary = int(base_sal[dept] * role_mult[role] * sal_v)
        yoe    = role_exp[role] + (i * 3) % 4
        perf   = 68 + (i * 11 + 17) % 32
        train  = 20 + (i * 7 + 5) % 61
        promo  = yoe // 3
        hire   = 2024 - yoe - (i % 3)
        bonus_pct = round(4.0 + ((i * 5 + yoe) % 16) * 0.8, 1)
        overtime_hours = 2 + ((i * 9 + yoe * 4) % 26)
        project_count = 1 + ((i + yoe) % 7)
        absenteeism_days = (i * 3 + yoe) % 12
        certification_count = (i + yoe) % 6
        engagement_score = 55 + (i * 13 + yoe * 5) % 45
        attrition_risk = round(
            0.15 * (100 - engagement_score) +
            0.35 * overtime_hours +
            0.25 * absenteeism_days,
            2,
        )
        team = f'{dept[:3].upper()}-{(i % 4) + 1}'
        manager = names[((i // 5) * 5 + 2) % len(names)]
        tenure_band = (
            '0-2y' if yoe <= 2 else
            '3-5y' if yoe <= 5 else
            '6-9y' if yoe <= 9 else
            '10y+'
        )
        rows.append(dict(
            name=names[i], department=dept, role_level=role,
            salary=salary, years_experience=yoe,
            performance_score=perf, hire_year=hire,
            region=region, city=city, remote=remote,
            team=team, manager=manager, tenure_band=tenure_band,
            training_hours=train, promotions=promo,
            bonus_pct=bonus_pct, overtime_hours=overtime_hours,
            project_count=project_count,
            absenteeism_days=absenteeism_days,
            certification_count=certification_count,
            engagement_score=engagement_score,
            attrition_risk=attrition_risk,
        ))
    return pd.DataFrame(rows)


def _make_operations_df() -> pd.DataFrame:
    """Complex operations dataset with supply-chain, quality, and risk signals."""
    plants = ['Berlin', 'Ankara', 'Austin', 'Osaka']
    suppliers = ['S-Apex', 'S-Nova', 'S-Prime', 'S-Delta', 'S-Core']
    supplier_tier = {'S-Apex': 'Tier-1', 'S-Nova': 'Tier-1', 'S-Prime': 'Tier-2',
                     'S-Delta': 'Tier-2', 'S-Core': 'Tier-3'}
    routes = ['EU-Central', 'US-South', 'APAC-East', 'MEA-Hub']
    transport_modes = ['Air', 'Sea', 'Road', 'Rail']
    product_families = ['Compute', 'Storage', 'Sensors', 'Power']

    rows = []
    for day in range(1, 181):
        week = (day - 1) // 7 + 1
        month = (day - 1) // 30 + 1
        seasonal = 1.0 + 0.16 * math.sin((day - 10) * math.pi / 45)
        for i in range(6):
            plant = plants[(day + i) % len(plants)]
            supplier = suppliers[(day * 3 + i) % len(suppliers)]
            tier = supplier_tier[supplier]
            route = routes[(day + i * 2) % len(routes)]
            mode = transport_modes[(day * 2 + i) % len(transport_modes)]
            family = product_families[(day + i * 5) % len(product_families)]

            demand = int(900 + 240 * seasonal + ((day * 11 + i * 17) % 180))
            produced = int(demand * (0.90 + 0.12 * ((day + i) % 10) / 10))
            shipped = int(produced * (0.92 + 0.06 * ((day * 2 + i) % 10) / 10))

            lead_time_days = round(
                4.0 + (2.0 if mode == 'Sea' else 0.5 if mode == 'Road' else 0.2)
                + (1.8 if tier == 'Tier-3' else 0.7 if tier == 'Tier-2' else 0.2)
                + ((day * 7 + i * 3) % 16) / 10,
                2,
            )
            on_time_pct = round(78 + 20 * ((day * 5 + i * 7) % 22) / 21, 2)
            defect_rate = round(0.8 + 3.7 * ((day * 13 + i * 11) % 25) / 24, 2)
            return_rate = round(0.6 + defect_rate * 0.72 + ((day + i) % 5) * 0.08, 2)
            stockout_hours = round(((day * 4 + i * 9) % 20) * (1.0 if tier == 'Tier-3' else 0.6), 2)
            inventory_turnover = round(3.0 + 7.5 * ((day * 3 + i) % 18) / 17, 2)
            fill_rate = round(84 + 14 * ((day * 5 + i * 4) % 20) / 19, 2)
            transport_cost_usd = int(
                shipped * (1.9 if mode == 'Air' else 0.8 if mode == 'Sea' else 1.1 if mode == 'Road' else 0.9)
            )
            procurement_cost_usd = int(produced * (4.4 + ((day + i * 2) % 11) / 10))
            co2_kg = round(shipped * (0.24 if mode == 'Air' else 0.09 if mode == 'Sea' else 0.12 if mode == 'Road' else 0.08), 2)
            forecast_error_pct = round(1.8 + 11.0 * ((day * 6 + i * 5) % 30) / 29, 2)
            incident_count = (day + i) % 5

            rows.append(dict(
                day=day,
                week=week,
                month=month,
                plant=plant,
                supplier=supplier,
                supplier_tier=tier,
                route=route,
                transport_mode=mode,
                product_family=family,
                demand_units=demand,
                produced_units=produced,
                shipped_units=shipped,
                lead_time_days=lead_time_days,
                on_time_pct=on_time_pct,
                defect_rate_pct=defect_rate,
                return_rate_pct=return_rate,
                stockout_hours=stockout_hours,
                inventory_turnover=inventory_turnover,
                fill_rate_pct=fill_rate,
                transport_cost_usd=transport_cost_usd,
                procurement_cost_usd=procurement_cost_usd,
                co2_kg=co2_kg,
                demand_forecast_error_pct=forecast_error_pct,
                incident_count=incident_count,
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NL Question bank — Phases 1, 2, 4
# ---------------------------------------------------------------------------
# Each Question = (text, kind, fallback_on_error)
# Phrasing is explicit so llama3.2 produces clean, working pandas code.

Question = Tuple[str, str, Optional[str]]

# ── Sales Phase 1 — Simple (NL) ─────────────────────────────────────────────

SALES_P1: List[Question] = [
    (
        'Group the dataframe by the region column and compute the sum of the '
        'revenue column for each region. Create a bar chart with region on '
        'the x-axis and total revenue on the y-axis.',
        'chart',
        'Bar chart of total revenue by region.',
    ),
    (
        'Group the dataframe by the product column and compute the sum of the '
        'units column for each product. Create a bar chart with product on '
        'the x-axis and total units sold on the y-axis, sorted from highest '
        'to lowest.',
        'chart',
        'Bar chart of total units by product.',
    ),
    (
        'Create a histogram of the revenue column. Use 10 equally spaced bins '
        'to show the distribution of individual sale revenue values.',
        'chart',
        'Histogram of revenue.',
    ),
    (
        'Group the dataframe by the sales_rep column and compute the sum of '
        'the profit column for each sales_rep. Create a bar chart with '
        'sales_rep on the x-axis and total profit on the y-axis.',
        'chart',
        'Bar chart of total profit by sales_rep.',
    ),
    (
        'Group the dataframe by the channel column and compute the mean of '
        'the discount_pct column for each channel. Create a bar chart with '
        'channel on the x-axis and average discount_pct on the y-axis.',
        'chart',
        'Bar chart of average discount_pct by channel.',
    ),
    (
        'Group the dataframe by the customer_segment column and compute the '
        'sum of the units column for each segment. Create a bar chart with '
        'customer_segment on the x-axis and total units on the y-axis.',
        'chart',
        'Bar chart of total units by customer_segment.',
    ),
]

# ── Sales Phase 2 — Medium (NL) ─────────────────────────────────────────────

SALES_P2: List[Question] = [
    (
        'Create a scatter plot where the x-axis is the cost column and the '
        'y-axis is the revenue column. Each point represents one row in the '
        'dataset.',
        'chart',
        'Scatter plot of cost vs revenue.',
    ),
    (
        'Group the dataframe by the month column and compute the sum of '
        'revenue for each month (values 1 through 12). Sort the result by '
        'month ascending. Create a line chart with month on the x-axis and '
        'total revenue on the y-axis. Add circle markers at each data point.',
        'chart',
        'Line chart of total revenue by month.',
    ),
    (
        'Create a scatter plot where the x-axis is discount_pct and the '
        'y-axis is profit. Each point is one row in the dataset.',
        'chart',
        'Scatter plot of discount_pct vs profit.',
    ),
    (
        'Group the dataframe by the quarter column (values 1, 2, 3, 4). '
        'Compute the sum of revenue for each quarter. Create a bar chart '
        'with quarter on the x-axis (in order 1, 2, 3, 4) and total '
        'revenue on the y-axis.',
        'chart',
        'Bar chart of total revenue by quarter.',
    ),
    (
        'Group the dataframe by the month column and compute the sum of '
        'revenue for each month. Sort by month ascending. Compute the '
        'running cumulative sum of revenue across months. Create a line '
        'chart with month on the x-axis and cumulative revenue on the '
        'y-axis.',
        'chart',
        'Line chart of cumulative revenue by month.',
    ),
]

# ── Sales Phase 4 — Text Analysis (NL) ──────────────────────────────────────

SALES_P4: List[Question] = [
    (
        'Group the dataframe by the product column. For each product compute '
        'total_profit as the sum of profit and total_revenue as the sum of '
        'revenue. Return a table with product, total_profit, and total_revenue '
        'sorted by total_profit descending.',
        'analysis',
        'What is total profit and total revenue for each product?',
    ),
    (
        'Which sales_rep has the highest total revenue? Group by sales_rep, '
        'sum the revenue column, and return the top result.',
        'analysis',
        'What is total revenue by sales_rep?',
    ),
    (
        'Which quarter had the highest total profit? Group by quarter, sum '
        'the profit column, and return the quarter number and its total profit.',
        'analysis',
        'What is total profit grouped by quarter?',
    ),
    (
        'What is the average satisfaction_score grouped by region? Return a '
        'table with region and average satisfaction_score.',
        'analysis',
        'Average satisfaction_score by region.',
    ),
    (
        'What is the total revenue and total cost grouped by channel? Return '
        'a table with channel, total_revenue, and total_cost.',
        'analysis',
        'Total revenue and cost by channel.',
    ),
]

# ── Employee Phase 1 — Simple (NL) ──────────────────────────────────────────

EMP_P1: List[Question] = [
    (
        'Group the dataframe by the department column and compute the mean of '
        'the salary column for each department. Create a bar chart with '
        'department on the x-axis and average salary on the y-axis.',
        'chart',
        'Bar chart of average salary by department.',
    ),
    (
        'Create a histogram of the salary column using 10 bins to show the '
        'distribution of salary values across all employees.',
        'chart',
        'Histogram of salary.',
    ),
    (
        'Group the dataframe by the role_level column and compute the mean of '
        'the salary column. Create a bar chart with role_level on the x-axis '
        'and average salary on the y-axis.',
        'chart',
        'Bar chart of average salary by role_level.',
    ),
    (
        'Group the dataframe by the hire_year column and count the number of '
        'rows (employees) hired each year. Create a bar chart with hire_year '
        'on the x-axis and employee count on the y-axis.',
        'chart',
        'Bar chart of headcount by hire_year.',
    ),
    (
        'Group the dataframe by the department column and compute the mean of '
        'the performance_score column. Create a bar chart with department on '
        'the x-axis and average performance_score on the y-axis.',
        'chart',
        'Bar chart of average performance_score by department.',
    ),
]

# ── Employee Phase 2 — Medium (NL) ──────────────────────────────────────────

EMP_P2: List[Question] = [
    (
        'Create a scatter plot where the x-axis is years_experience and the '
        'y-axis is salary. Each point represents one employee row.',
        'chart',
        'Scatter plot of years_experience vs salary.',
    ),
    (
        'Create a scatter plot where the x-axis is performance_score and the '
        'y-axis is salary. Each point represents one employee.',
        'chart',
        'Scatter plot of performance_score vs salary.',
    ),
    (
        'Group the dataframe by region and compute the mean of the salary '
        'column. Create a bar chart with region on the x-axis and average '
        'salary on the y-axis.',
        'chart',
        'Bar chart of average salary by region.',
    ),
    (
        'Group the dataframe by department and compute the sum of the '
        'training_hours column. Create a bar chart with department on the '
        'x-axis and total training_hours on the y-axis.',
        'chart',
        'Bar chart of total training_hours by department.',
    ),
]

# ── Employee Phase 4 — Text Analysis (NL) ───────────────────────────────────

EMP_P4: List[Question] = [
    (
        'What is the average salary grouped by department? Return a table '
        'with department and average_salary sorted from highest to lowest.',
        'analysis',
        'Average salary by department.',
    ),
    (
        'What is the average performance_score for employees where remote is '
        'True versus where remote is False? Return both averages.',
        'analysis',
        'Average performance_score grouped by remote.',
    ),
    (
        'Which department has the highest average training_hours? Group by '
        'department and compute the mean of training_hours.',
        'analysis',
        'Average training_hours by department.',
    ),
]

# ── Operations Phase 1/2/4 — richer, multi-signal dataset ─────────────────

OPS_P1: List[Question] = [
    (
        'Group by plant and compute average lead_time_days. Create a bar chart '
        'with plant on x-axis and avg lead_time_days on y-axis.',
        'chart',
        'Bar chart of average lead_time_days by plant.',
    ),
    (
        'Create a histogram of defect_rate_pct using 12 bins to show quality '
        'distribution across operations records.',
        'chart',
        'Histogram of defect_rate_pct.',
    ),
    (
        'Group by transport_mode and compute total transport_cost_usd. Create '
        'a bar chart with transport_mode on x-axis and total transport cost on y-axis.',
        'chart',
        'Bar chart of total transport_cost_usd by transport_mode.',
    ),
]

OPS_P2: List[Question] = [
    (
        'Create a scatter plot with lead_time_days on x-axis and on_time_pct '
        'on y-axis. Each point is one record.',
        'chart',
        'Scatter plot of lead_time_days vs on_time_pct.',
    ),
    (
        'Group by week and compute average fill_rate_pct. Sort by week and '
        'create a line chart of weekly fill_rate_pct.',
        'chart',
        'Line chart of weekly fill_rate_pct.',
    ),
    (
        'Create a scatter plot with defect_rate_pct on x-axis and return_rate_pct '
        'on y-axis. Each point is one record.',
        'chart',
        'Scatter plot of defect_rate_pct vs return_rate_pct.',
    ),
]

OPS_P4: List[Question] = [
    (
        'Group by supplier_tier and compute average lead_time_days, average '
        'defect_rate_pct, and average on_time_pct. Return a sorted table.',
        'analysis',
        'Average lead_time_days, defect_rate_pct, and on_time_pct by supplier_tier.',
    ),
    (
        'Group by transport_mode and compute total co2_kg and total '
        'transport_cost_usd. Return a table sorted by total co2_kg descending.',
        'analysis',
        'Total co2_kg and transport_cost_usd by transport_mode.',
    ),
]

# NL question bank — dataset → phase → questions
_QUESTION_BANK: Dict[str, Dict[int, List[Question]]] = {
    'sales':    {1: SALES_P1, 2: SALES_P2, 4: SALES_P4},
    'employee': {1: EMP_P1,   2: EMP_P2,   4: EMP_P4},
    'operations': {1: OPS_P1, 2: OPS_P2,   4: OPS_P4},
}


# ---------------------------------------------------------------------------
# Phase 3 — Direct chart helpers (all 20 chart types, no LLM)
# ---------------------------------------------------------------------------

class DirectChart:
    """A chart produced by calling a pychartai helper directly — no LLM."""

    __slots__ = ('label', 'chart_type', 'render_fn', 'backend')

    def __init__(self, label: str, chart_type: str,
                 render_fn: Callable, backend: str = 'seaborn') -> None:
        self.label = label
        self.chart_type = chart_type
        # render_fn(df: pd.DataFrame, output_file: str, backend: str) -> str
        self.render_fn = render_fn
        self.backend = backend


def _build_sales_phase3(output_dir: str, backend: str) -> List[DirectChart]:
    """Build all 15 Phase-3 charts for the sales dataset."""

    def out(name: str) -> str:
        return os.path.join(output_dir, f'sales_{name}.png')

    charts = [
        # 1. Correlation heatmap — 7 numeric metrics
        DirectChart(
            'Correlation heatmap of 7 sales metrics '
            '(revenue, cost, profit, units, discount, return_rate, satisfaction)',
            'heatmap',
            lambda df, of, bk: heatmap(
                df[['revenue', 'cost', 'profit', 'units',
                    'discount_pct', 'return_rate', 'satisfaction_score']],
                title='Sales Metrics — Pearson Correlation Matrix',
                output_file=out('01_heatmap_correlation'),
                backend=bk,
            ),
        ),

        # 2. Box plot — revenue distribution per region, coloured by category
        DirectChart(
            'Box plot: revenue distribution per region (hue=category)',
            'box',
            lambda df, of, bk: box_chart(
                df, x='region', y='revenue',
                title='Revenue Distribution by Region',
                output_file=out('02_box_revenue_region'),
                hue='category', backend=bk,
            ),
        ),

        # 3. Violin plot — revenue shape per category
        DirectChart(
            'Violin plot: revenue density shape per category',
            'violin',
            lambda df, of, bk: violin_chart(
                df, x='category', y='revenue',
                title='Revenue Distribution Shape by Category',
                output_file=out('03_violin_revenue_category'),
                backend=bk,
            ),
        ),

        # 4. Stacked bar — revenue by quarter stacked by category
        DirectChart(
            'Stacked bar: revenue by quarter, stacked by product category',
            'stacked_bar',
            lambda df, of, bk: stacked_bar_chart(
                df, x='quarter', y='revenue', stack='category',
                title='Revenue by Quarter (Widgets vs Gadgets)',
                output_file=out('04_stacked_bar_quarter_category'),
                backend=bk,
            ),
        ),

        # 5. Stacked bar normalised — 100% share by region stacked by channel
        DirectChart(
            '100% stacked bar: revenue share by region, stacked by channel',
            'stacked_bar_normalized',
            lambda df, of, bk: stacked_bar_chart(
                df, x='region', y='revenue', stack='channel',
                normalize=True,
                title='Revenue Channel Mix by Region (100% Stacked)',
                output_file=out('05_stacked_bar_region_channel_norm'),
                backend=bk,
            ),
        ),

        # 6. Bubble chart — pre-aggregated per product: units vs revenue, size=profit
        DirectChart(
            'Bubble chart: total units vs total revenue per product '
            '(bubble size = total profit)',
            'bubble',
            lambda df, of, bk: bubble_chart(
                df.groupby('product').agg(
                    total_units=('units', 'sum'),
                    total_revenue=('revenue', 'sum'),
                    total_profit=('profit', 'sum'),
                ).reset_index(),
                x='total_units', y='total_revenue', size='total_profit',
                title='Product Portfolio Bubble Chart '
                      '(units × revenue, size=profit)',
                output_file=out('06_bubble_product_portfolio'),
                hue='product', backend=bk,
            ),
        ),

        # 7. Area chart — monthly revenue by category (two coloured bands)
        DirectChart(
            'Area chart: monthly total revenue by category over 12 months',
            'area',
            lambda df, of, bk: area_chart(
                df.groupby(['month', 'category'])['revenue']
                  .sum().reset_index(),
                x='month', y='revenue',
                title='Monthly Revenue by Category (Area)',
                output_file=out('07_area_monthly_category'),
                hue='category', backend=bk,
            ),
        ),

        # 8. KDE — revenue density by category
        DirectChart(
            'KDE plot: revenue probability density per category',
            'kde',
            lambda df, of, bk: kde_chart(
                df, column='revenue',
                title='Revenue Density Estimate by Category',
                output_file=out('08_kde_revenue_category'),
                hue='category', backend=bk,
            ),
        ),

        # 9. ECDF — profit cumulative distribution by channel
        DirectChart(
            'ECDF: cumulative profit distribution per sales channel',
            'ecdf',
            lambda df, of, bk: ecdf_chart(
                df, column='profit',
                title='Profit Cumulative Distribution by Channel',
                output_file=out('09_ecdf_profit_channel'),
                hue='channel', backend=bk,
            ),
        ),

        # 10. Step chart — cumulative units sold across 12 months
        DirectChart(
            'Step chart: cumulative units sold month-by-month',
            'step',
            lambda df, of, bk: step_chart(
                df.groupby('month')['units'].sum()
                  .cumsum().reset_index(name='cumulative_units'),
                x='month', y='cumulative_units',
                title='Cumulative Units Sold (Step)',
                output_file=out('10_step_cumulative_units'),
                backend=bk,
            ),
        ),

        # 11. Strip chart — individual revenue points by region, hue=channel
        DirectChart(
            'Strip chart: every sale\'s revenue value plotted by region '
            '(colour=channel)',
            'strip',
            lambda df, of, bk: strip_chart(
                df, x='region', y='revenue',
                title='Individual Revenue by Region (Strip)',
                output_file=out('11_strip_revenue_region'),
                hue='channel', backend=bk,
            ),
        ),

        # 12. Count chart — transaction count by channel, hue=customer_segment
        DirectChart(
            'Count chart: number of transactions per channel '
            '(colour=customer segment)',
            'count',
            lambda df, of, bk: count_chart(
                df, x='channel',
                title='Transaction Count by Channel',
                output_file=out('12_count_channel_segment'),
                hue='customer_segment', backend=bk,
            ),
        ),

        # 13. Funnel chart — total revenue by customer segment (descending)
        DirectChart(
            'Funnel chart: total revenue by customer segment '
            '(Enterprise → SMB → Consumer)',
            'funnel',
            lambda df, of, bk: funnel_chart(
                df.groupby('customer_segment')['revenue'].sum()
                  .sort_values(ascending=False).reset_index()
                  .rename(columns={'customer_segment': 'segment',
                                   'revenue': 'total_revenue'}),
                labels='segment', values='total_revenue',
                title='Revenue Funnel by Customer Segment',
                output_file=out('13_funnel_customer_segment'),
                backend=bk,
            ),
        ),

        # 14. Pie chart — revenue share by customer segment
        DirectChart(
            'Pie chart: revenue share per customer segment',
            'pie',
            lambda df, of, bk: pie_chart(
                df.groupby('customer_segment')['revenue'].sum()
                  .reset_index()
                  .rename(columns={'customer_segment': 'segment',
                                   'revenue': 'total_revenue'}),
                labels='segment', values='total_revenue',
                title='Revenue Share by Customer Segment',
                output_file=out('14_pie_customer_segment'),
                backend=bk,
            ),
        ),

        # 15. Regression chart — discount_pct vs profit with trend line
        DirectChart(
            'Regression plot: discount_pct vs profit with OLS trend line '
            '(hue=category)',
            'regression',
            lambda df, of, bk: regression_chart(
                df, x='discount_pct', y='profit',
                title='Discount % vs Profit with Regression Line',
                output_file=out('15_regression_discount_profit'),
                hue='category', backend=bk,
            ),
        ),
    ]
    for dc in charts:
        dc.backend = backend
    return charts


def _build_employee_phase3(output_dir: str, backend: str) -> List[DirectChart]:
    """Build all 5 Phase-3 charts for the employee dataset."""

    def out(name: str) -> str:
        return os.path.join(output_dir, f'emp_{name}.png')

    charts = [
        # 16. Violin — salary by department, hue=remote
        DirectChart(
            'Violin plot: salary distribution per department '
            '(split by remote / on-site)',
            'violin',
            lambda df, of, bk: violin_chart(
                df, x='department', y='salary',
                title='Salary Distribution by Department '
                      '(Remote vs On-Site)',
                output_file=out('16_violin_salary_dept'),
                hue='remote', backend=bk,
            ),
        ),

        # 17. Swarm chart — performance_score per department
        DirectChart(
            'Swarm chart: every employee\'s performance score per department '
            '(colour=remote)',
            'swarm',
            lambda df, of, bk: swarm_chart(
                df, x='department', y='performance_score',
                title='Individual Performance Scores by Department '
                      '(Swarm)',
                output_file=out('17_swarm_perf_dept'),
                hue='remote', backend=bk,
            ),
        ),

        # 18. Pairplot — 4 numeric columns, hue=department
        DirectChart(
            'Pairplot: salary × years_experience × performance_score × '
            'training_hours (colour=department)',
            'pairplot',
            lambda df, of, bk: pairplot_chart(
                df,
                columns=['salary', 'years_experience',
                         'performance_score', 'training_hours'],
                title='Employee Metrics Pairplot',
                output_file=out('18_pairplot_employee'),
                hue='department', backend=bk,
            ),
        ),

        # 19. KDE — salary density by department
        DirectChart(
            'KDE plot: salary density estimate per department',
            'kde',
            lambda df, of, bk: kde_chart(
                df, column='salary',
                title='Salary Density by Department',
                output_file=out('19_kde_salary_dept'),
                hue='department', backend=bk,
            ),
        ),

        # 20. Strip chart — salary by role_level, hue=department
        DirectChart(
            'Strip chart: individual salaries per role level '
            '(colour=department)',
            'strip',
            lambda df, of, bk: strip_chart(
                df, x='role_level', y='salary',
                title='Salary by Role Level (Strip)',
                output_file=out('20_strip_salary_role'),
                hue='department', backend=bk,
            ),
        ),
    ]
    for dc in charts:
        dc.backend = backend
    return charts


def _build_operations_phase3(output_dir: str, backend: str) -> List[DirectChart]:
    """Build complex operations charts for supply-chain analytics."""

    def out(name: str) -> str:
        return os.path.join(output_dir, f'ops_{name}.png')

    charts = [
        DirectChart(
            'Correlation heatmap: lead-time, quality, service, inventory, carbon, forecast error',
            'heatmap',
            lambda df, of, bk: heatmap(
                df[['lead_time_days', 'on_time_pct', 'defect_rate_pct', 'return_rate_pct',
                    'inventory_turnover', 'fill_rate_pct', 'co2_kg', 'demand_forecast_error_pct']],
                title='Operations KPI Correlation Matrix',
                output_file=out('21_heatmap_ops_kpis'),
                backend=bk,
            ),
        ),
        DirectChart(
            'Bubble chart: avg lead time vs avg on-time by supplier tier (size=total cost)',
            'bubble',
            lambda df, of, bk: bubble_chart(
                df.groupby('supplier_tier').agg(
                    avg_lead_time=('lead_time_days', 'mean'),
                    avg_on_time=('on_time_pct', 'mean'),
                    total_cost=('transport_cost_usd', 'sum'),
                ).reset_index(),
                x='avg_lead_time', y='avg_on_time', size='total_cost',
                hue='supplier_tier',
                title='Supplier Tier Service-Cost Bubble',
                output_file=out('22_bubble_supplier_tier'),
                backend=bk,
            ),
        ),
        DirectChart(
            'Line chart: weekly average fill-rate by plant',
            'line',
            lambda df, of, bk: line_chart(
                df.groupby(['week', 'plant'])['fill_rate_pct'].mean().reset_index(),
                x='week', y='fill_rate_pct', hue='plant',
                title='Weekly Fill Rate by Plant',
                output_file=out('23_line_weekly_fillrate_plant'),
                backend=bk,
            ),
        ),
        DirectChart(
            'Area chart: monthly CO2 emissions by transport mode',
            'area',
            lambda df, of, bk: area_chart(
                df.groupby(['month', 'transport_mode'])['co2_kg'].sum().reset_index(),
                x='month', y='co2_kg', hue='transport_mode',
                title='Monthly CO2 Emissions by Transport Mode',
                output_file=out('24_area_monthly_co2_mode'),
                backend=bk,
            ),
        ),
        DirectChart(
            '100% stacked bar: shipment mix by plant and transport mode',
            'stacked_bar_normalized',
            lambda df, of, bk: stacked_bar_chart(
                df, x='plant', y='shipped_units', stack='transport_mode',
                normalize=True,
                title='Shipment Mix by Plant (100% Stacked)',
                output_file=out('25_stacked_plant_transport_mix'),
                backend=bk,
            ),
        ),
        DirectChart(
            'Regression: defect rate vs return rate (hue=supplier_tier)',
            'regression',
            lambda df, of, bk: regression_chart(
                df, x='defect_rate_pct', y='return_rate_pct', hue='supplier_tier',
                title='Quality to Returns Relationship',
                output_file=out('26_regression_defect_return'),
                backend=bk,
            ),
        ),
    ]
    for dc in charts:
        dc.backend = backend
    return charts


# Pre-build phase-3 registries keyed by dataset name
_PHASE3_REGISTRY: Dict[str, Callable[[str, str], List[DirectChart]]] = {
    'sales':    _build_sales_phase3,
    'employee': _build_employee_phase3,
    'operations': _build_operations_phase3,
}


# ---------------------------------------------------------------------------
# Dynamic question generator (LLM — optional extras for Phase 1)
# ---------------------------------------------------------------------------

_GEN_PROMPT = """\
You are a data analyst. Given this dataset schema:
{schema}

Generate exactly {n} bar-chart or histogram questions. One per line.
Prefix each line with [CHART]. Be specific about which column to group by
and which numeric column to aggregate. Do not ask for pie charts or
multi-axis plots.

Example:
[CHART] Group by region and compute average revenue. Plot a bar chart with region on x-axis and avg_revenue on y-axis.
"""


class DynamicQuestionGenerator:
    def __init__(self, llm: pai.PyChartLLM) -> None:
        self._llm = llm

    def generate(self, df: pd.DataFrame, n: int) -> List[Question]:
        cols = ', '.join(f'{c}({df[c].dtype})' for c in df.columns)
        schema = f'Shape: {df.shape[0]}×{df.shape[1]}\nColumns: {cols}'
        try:
            raw = self._llm.generate(_GEN_PROMPT.format(schema=schema, n=n))
            return self._parse(raw)[:n]
        except Exception as exc:
            print(f'  [QuestionGen] {exc}')
            return []

    @staticmethod
    def _parse(raw: str) -> List[Question]:
        out: List[Question] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if line.upper().startswith('[CHART]'):
                q = line[7:].strip().lstrip('-').strip()
                if q:
                    out.append((q, 'chart', None))
        return out


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class QueryResult:
    __slots__ = ('idx', 'dataset', 'phase', 'question', 'kind',
                 'result', 'elapsed', 'error', 'retried')

    def __init__(self, idx, dataset, phase, question, kind,
                 result, elapsed, error, retried=False):
        self.idx = idx
        self.dataset = dataset
        self.phase = phase
        self.question = question
        self.kind = kind
        self.result = result
        self.elapsed = elapsed
        self.error = error
        self.retried = retried


# ---------------------------------------------------------------------------
# ClientAgent
# ---------------------------------------------------------------------------

_ERR_RE = re.compile(
    r"'?nonetype'?\s+object\s+is\s+not\s+callable|"
    r"object\s+is\s+not\s+callable",
    re.IGNORECASE,
)


class ClientAgent:
    """Multi-phase autonomous pychartai client powered by Ollama.

    Phases 1 & 2: NL queries via LLM (ThreadPoolExecutor).
    Phase 3:      Direct chart helpers — guaranteed output, no LLM.
    Phase 4:      NL text analysis, sequential (memory builds up).
    """

    def __init__(
        self,
        datasets: Dict[str, pai.SmartDataFrame],
        llm: pai.PyChartLLM,
        *,
        phases: List[int],
        max_workers: int = 3,
        backends: Optional[List[str]] = None,
        backend: str = 'seaborn',
        output_dir: str = 'agent_output',
        extra_questions: int = 0,
        keep_nl_charts: bool = False,
    ) -> None:
        self._datasets    = datasets
        self._llm         = llm
        self._phases      = phases
        self._workers     = max_workers
        # backends: list for Phase-3 multi-library showcase; backend: primary
        # backend for NL phases (1, 2, 4).
        self._backends    = backends if backends else [backend]
        self._backend     = self._backends[0]
        self._output_dir  = output_dir
        self._extra       = extra_questions
        self._keep_nl_charts = keep_nl_charts
        self._lock        = threading.Lock()
        self._results: List[QueryResult] = []
        self._counter     = 0
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def run(self) -> List[QueryResult]:
        self._results.clear()
        self._counter = 0

        for phase_num in self._phases:
            self._section_header(phase_num)
            jobs: List[Tuple[str, object]] = []

            if phase_num == 3:
                # Run all 20 direct chart helpers once per backend.
                # When multiple backends are requested each backend gets its
                # own subdirectory so filenames never collide.
                multi = len(self._backends) > 1
                for bk in self._backends:
                    bkdir = (os.path.join(self._output_dir, bk)
                             if multi else self._output_dir)
                    os.makedirs(bkdir, exist_ok=True)
                    if multi:
                        self._log(f'\n  [Phase 3] backend: {bk}  →  {bkdir}/')
                    for ds_name in self._datasets:
                        builder = _PHASE3_REGISTRY.get(ds_name)
                        if builder:
                            for dc in builder(bkdir, bk):
                                jobs.append((ds_name, dc))
            else:
                for ds_name, sdf in self._datasets.items():
                    bank = _QUESTION_BANK.get(ds_name, {})
                    qs = list(bank.get(phase_num, []))
                    if phase_num == 1 and self._extra > 0:
                        gen = DynamicQuestionGenerator(self._llm)
                        extras = gen.generate(
                            object.__getattribute__(sdf, '_df'),
                            self._extra,
                        )
                        if extras:
                            self._log(f'  + {len(extras)} LLM-generated '
                                      f'extras for [{ds_name}]')
                        qs.extend(extras)
                    for q in qs:
                        jobs.append((ds_name, q))

            # Phase 3 (direct matplotlib calls) and Phase 4 (NL memory) run
            # sequentially. matplotlib is NOT thread-safe: plt.close("all") in
            # one thread closes figures belonging to concurrent threads, which
            # produces blank PNGs. Sequential execution avoids this entirely.
            sequential = phase_num >= 3
            if sequential:
                for ds, job in jobs:
                    r = self._execute(ds, job, phase_num)
                    if r:
                        self._record(r)
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._workers
                ) as pool:
                    futs = {
                        pool.submit(self._execute, ds, job, phase_num): (ds, job)
                        for ds, job in jobs
                    }
                    for fut in concurrent.futures.as_completed(futs):
                        r = fut.result()
                        if r:
                            self._record(r)

        self._print_summary()
        return self._results

    # ------------------------------------------------------------------

    def _execute(self, ds_name: str, job: object,
                 phase: int) -> Optional[QueryResult]:
        with self._lock:
            idx = self._counter
            self._counter += 1

        sdf = self._datasets[ds_name]
        start = time.time()

        if isinstance(job, DirectChart):
            return self._run_direct(idx, ds_name, phase, job, sdf, start)
        return self._run_nl(idx, ds_name, phase, job, sdf, start)

    def _run_direct(self, idx, ds_name, phase, dc: DirectChart,
                    sdf, start) -> QueryResult:
        # output_file is a fallback path; the render_fn typically hardcodes
        # its own path via the out() closure capturing the backend output dir.
        output_file = os.path.join(
            self._output_dir,
            f'{dc.chart_type}_{ds_name}_{idx:03d}.png',
        )
        try:
            df = object.__getattribute__(sdf, '_df')
            path = dc.render_fn(df, output_file, dc.backend)
            return QueryResult(idx, ds_name, phase, dc.label, 'chart',
                               path, time.time() - start, None)
        except Exception as exc:
            return QueryResult(idx, ds_name, phase, dc.label, 'chart',
                               None, time.time() - start, str(exc))

    def _run_nl(self, idx, ds_name, phase, question: Question,
                sdf, start) -> QueryResult:
        text, kind, fallback = question
        kwargs = {'chart_library': self._backend} if kind == 'chart' else {}
        try:
            result = sdf.chat(text, **kwargs)
            # Phase-1/2 NL charts are typically exploratory and redundant with
            # Phase-3 direct chart artifacts. Keep them only when requested.
            if (
                kind == 'chart'
                and phase in (1, 2)
                and not self._keep_nl_charts
                and isinstance(result, str)
                and os.path.isfile(result)
            ):
                try:
                    os.remove(result)
                    result = '[nl-chart-discarded]'
                except OSError:
                    pass
            return QueryResult(idx, ds_name, phase, text, kind,
                               str(result), time.time() - start, None)
        except Exception as exc:
            err = str(exc)
            if fallback and _ERR_RE.search(err):
                try:
                    result = sdf.chat(fallback, **kwargs)
                    return QueryResult(idx, ds_name, phase, text, kind,
                                       str(result), time.time() - start,
                                       None, retried=True)
                except Exception as exc2:
                    err = str(exc2)
            return QueryResult(idx, ds_name, phase, text, kind,
                               None, time.time() - start, err)

    def _record(self, r: QueryResult) -> None:
        with self._lock:
            self._results.append(r)
        self._print_result(r)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    _PHASE_LABELS = {
        1: 'Phase 1 — Simple charts    (NL → LLM parallel  → bar, histogram)',
        2: 'Phase 2 — Medium charts    (NL → LLM parallel  → scatter, line)',
        3: 'Phase 3 — Complex charts   (Direct sequential  → rich chart suite)',
        4: 'Phase 4 — Text analysis    (NL → LLM sequential → rankings)',
    }

    def _section_header(self, phase: int) -> None:
        label = self._PHASE_LABELS.get(phase, f'Phase {phase}')
        with self._lock:
            print(f'\n{"=" * 68}')
            print(f'  {label}')
            print('=' * 68)

    def _print_result(self, r: QueryResult) -> None:
        tag   = 'C' if r.kind == 'chart' else 'A'
        ds    = r.dataset[:3].upper()
        state = 'ERR' if r.error else ('OK*' if r.retried else 'OK ')
        note  = ' [retried]' if r.retried else ''
        if r.phase == 3:
            # Show which backend produced this direct chart
            bk = getattr(r, '_backend', '')
            mode = f'direct/{bk}' if bk else 'direct'
        else:
            mode = 'NL'
        with self._lock:
            print(f'\n  [{r.idx + 1:>2}] P{r.phase} [{ds}][{tag}]'
                  f'[{mode}][{state}] ({r.elapsed:.1f}s){note}')
            q_short = r.question[:90] + ('…' if len(r.question) > 90 else '')
            print(f'       {q_short}')
            if r.error:
                msg = re.sub(r'^\[Pipeline/\w+\]\s*', '', r.error)
                print(f'  !    {msg.split(chr(10))[0][:120]}')
            elif r.result:
                if os.path.isfile(r.result):
                    # Show containing directory (important when multiple backends)
                    rel = os.path.relpath(r.result, self._output_dir)
                    print(f'  →    chart: {rel}')
                else:
                    preview = r.result[:180].replace('\n', ' | ')
                    print(f'  →    {preview}{"…" if len(r.result) > 180 else ""}')

    def _log(self, msg: str) -> None:
        with self._lock:
            print(msg)

    def _print_summary(self) -> None:
        ok      = sum(1 for r in self._results if not r.error)
        err     = sum(1 for r in self._results if r.error)
        retried = sum(1 for r in self._results if r.retried)
        charts  = [r for r in self._results
                   if r.result and os.path.isfile(r.result)]
        total_t = sum(r.elapsed for r in self._results)

        phase_stats: Dict[int, Dict[str, int]] = {}
        for r in self._results:
            s = phase_stats.setdefault(r.phase, {'ok': 0, 'err': 0, 'ch': 0})
            if r.error:
                s['err'] += 1
            else:
                s['ok'] += 1
                if r.result and os.path.isfile(r.result):
                    s['ch'] += 1

        # Per-backend chart count for Phase 3 multi-library showcase
        backend_chart_counts: Dict[str, int] = {}
        if 3 in phase_stats and len(self._backends) > 1:
            for r in charts:
                if r.phase != 3:
                    continue
                # Derive backend from path: output_dir/<backend>/file.png
                parts = os.path.relpath(r.result, self._output_dir).split(os.sep)
                bk = parts[0] if len(parts) > 1 else 'unknown'
                backend_chart_counts[bk] = backend_chart_counts.get(bk, 0) + 1

        with self._lock:
            print(f'\n{"=" * 68}')
            print(f'  SUMMARY')
            print(f'{"=" * 68}')
            print(f'  Queries : {len(self._results)}'
                  f'  ({ok} OK, {err} ERR, {retried} retried)')
            print(f'  Charts  : {len(charts)}  →  {self._output_dir}/')
            print(f'  Time    : {total_t:.1f}s')
            for ph, s in sorted(phase_stats.items()):
                lbl = self._PHASE_LABELS.get(ph, f'Phase {ph}')
                print(f'  {lbl}: {s["ok"]} OK  {s["err"]} ERR  {s["ch"]} charts')

            if backend_chart_counts:
                print(f'\n  Phase 3 — charts per backend:')
                for bk in self._backends:
                    n = backend_chart_counts.get(bk, 0)
                    print(f'    {bk:<12} {n} charts')

            if charts:
                print(f'\n  Chart gallery ({len(charts)} files):')
                for r in sorted(charts, key=lambda x: x.idx):
                    q_short = r.question[:55] + ('…' if len(r.question) > 55 else '')
                    rel = os.path.relpath(r.result, self._output_dir)
                    print(f'    [{r.idx + 1:>2}] P{r.phase} {rel:<50} {q_short}')

            if err:
                print(f'\n  Failed queries ({err}):')
                for r in self._results:
                    if r.error:
                        msg = re.sub(r'^\[Pipeline/\w+\]\s*', '', r.error)
                        print(f'    [{r.idx + 1:>2}] P{r.phase} {r.question[:60]}…')
                        print(f'         {msg.split(chr(10))[0][:110]}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_VALID_BACKENDS = ('seaborn', 'matplotlib', 'plotly')


def _cleanup_legacy_nl_outputs(output_dir: str) -> None:
    """Remove legacy NL chart files previously written in output root.

    Older runs placed NL chart artifacts directly in ``output_dir`` with names
    such as ``seaborn_*.png``. New runs use ``output_dir/nl`` for NL outputs,
    so we proactively clean old root-level files to avoid confusion.
    """
    try:
        if not os.path.isdir(output_dir):
            return
        prefixes = ('seaborn_', 'matplotlib_', 'plotly_')
        for name in os.listdir(output_dir):
            p = os.path.join(output_dir, name)
            if os.path.isfile(p) and name.startswith(prefixes):
                os.remove(p)
    except OSError:
        # Best effort only; a cleanup miss should not block chart generation.
        pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Autonomous multi-phase pychartai chart agent (rich chart suite)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--model',    default='llama3.2')
    p.add_argument('--backend',  default='seaborn',
                   choices=list(_VALID_BACKENDS),
                   help='Primary backend for NL phases (default: seaborn)')
    p.add_argument('--backends', default=None,
                   help='Comma-separated backends for Phase-3 multi-library '
                        'showcase, e.g. seaborn,matplotlib,plotly. '
                        'Overrides --backend for Phase 3.')
    p.add_argument('--dataset',  default='all',
                   choices=['sales', 'employee', 'operations', 'both', 'all'])
    p.add_argument('--phases',   default='1,2,3',
                   help='Comma-separated phases to run (default: 1,2,3)')
    p.add_argument('--workers',  type=int, default=3)
    p.add_argument('--extra',    type=int, default=0,
                   help='Extra LLM-generated Phase-1 questions per dataset')
    p.add_argument('--output',   default='agent_output')
    p.add_argument('--keep-nl-charts', action='store_true',
                   help='Keep NL-generated Phase-1/2 chart files (default: discard)')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        phases = [int(x.strip()) for x in args.phases.split(',') if x.strip()]
    except ValueError:
        print('ERROR: --phases must be integers, e.g. 1,2,3')
        sys.exit(1)

    # Parse backends: --backends overrides --backend for Phase 3.
    # Validate each entry against the allowed set.
    if args.backends:
        backends = [b.strip() for b in args.backends.split(',') if b.strip()]
        invalid = [b for b in backends if b not in _VALID_BACKENDS]
        if invalid:
            print(f'ERROR: unknown backends: {invalid}. '
                  f'Choose from {list(_VALID_BACKENDS)}')
            sys.exit(1)
    else:
        backends = [args.backend]

    print('=' * 68)
    print('  pychartai — Autonomous Chart Agent (rich chart suite)')
    print(f'  model    : ollama/{args.model}')
    print(f'  backends : {", ".join(backends)}'
          + (' ← multi-library Phase-3 showcase' if len(backends) > 1 else ''))
    print(f'  dataset  : {args.dataset}')
    print(f'  phases   : {phases}')
    print(f'  workers  : {args.workers}')
    print('=' * 68)

    if not args.keep_nl_charts:
        _cleanup_legacy_nl_outputs(args.output)

    try:
        llm = pai.OllamaLLM(model=args.model)
        nl_output = os.path.join(args.output, 'nl')
        pai.config.set({
            'llm': llm,
            'chart_backend': backends[0],
            'verbose': False,
            'charts_output_dir': nl_output,
        })
        print(f'\n  LLM  : ollama/{args.model} — connected')
    except Exception as exc:
        print(f'\n  ERROR: Cannot connect to Ollama: {exc}')
        print('  Run: ollama serve')
        sys.exit(1)

    datasets: Dict[str, pai.SmartDataFrame] = {}
    if args.dataset in ('sales', 'both', 'all'):
        df = _make_sales_df()
        sdf = pai.SmartDataFrame(df)
        sdf.enable_memory(window_size=10)
        datasets['sales'] = sdf
        print(f'  Data : sales_data    '
              f'({df.shape[0]}r × {df.shape[1]}c, memory=on)')
    if args.dataset in ('employee', 'both', 'all'):
        df = _make_employee_df()
        sdf = pai.SmartDataFrame(df)
        sdf.enable_memory(window_size=10)
        datasets['employee'] = sdf
        print(f'  Data : employee_data '
              f'({df.shape[0]}r × {df.shape[1]}c, memory=on)')
    if args.dataset in ('operations', 'all'):
        df = _make_operations_df()
        sdf = pai.SmartDataFrame(df)
        sdf.enable_memory(window_size=12)
        datasets['operations'] = sdf
        print(f'  Data : operations_data '
              f'({df.shape[0]}r × {df.shape[1]}c, memory=on)')

    agent = ClientAgent(
        datasets=datasets,
        llm=llm,
        phases=phases,
        max_workers=args.workers,
        backends=backends,
        backend=backends[0],
        output_dir=args.output,
        extra_questions=args.extra,
        keep_nl_charts=args.keep_nl_charts,
    )
    agent.run()

    print('\n' + '=' * 68)
    print('  Done.')
    print('=' * 68)


if __name__ == '__main__':
    main()
