"""Report figures from the training-window EDA tables."""

from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
import pandas as pd


BLUE = "#245e85"
TEAL = "#137563"
INK = "#1f3544"


def _number(value, position=None):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:,.6f}".rstrip("0").rstrip(".")


def _heading(figure, title, subtitle):
    figure.text(0.075, 0.97, title, va="top", fontsize=17, fontweight="bold", color=INK)
    figure.text(0.075, 0.915, subtitle, va="top", fontsize=10, color=INK)


def _style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#899aa3")
    axis.tick_params(labelsize=9, labelcolor=INK, color="#899aa3")
    axis.grid(axis="y", which="major", color="#dbe1e4", linewidth=0.7)
    axis.set_axisbelow(True)


def _monthly(monthly, window):
    figure = Figure(figsize=(11, 7))
    axes = figure.subplots(2, 2, sharex=True).flat
    figure.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.13, hspace=0.35, wspace=0.21)
    _heading(figure, "Monthly purchasing activity", f"{window}  ·  Month of basket completion")
    panels = [
        ("baskets", "Customer invoices (baskets)"),
        ("basket_item_events", "Distinct items summed across baskets (events)"),
        ("units", "Quantity purchased (units)"),
        ("amount_gbp", "Recorded purchase amount (GBP)"),
    ]
    month_labels = pd.PeriodIndex(monthly["month"], freq="M").strftime("%b\n%Y")
    for axis, (column, title) in zip(axes, panels):
        _style(axis)
        axis.bar(range(len(monthly)), monthly[column], color=BLUE, width=0.70)
        axis.set_title(title, fontsize=11, loc="left", color=INK)
        axis.set_ylim(bottom=0)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=column != "amount_gbp"))
        axis.yaxis.set_major_formatter(FuncFormatter(_number))
        axis.set_xticks(range(len(monthly)), month_labels)
    return figure


def _distributions(distributions, window):
    figure = Figure(figsize=(12, 7.4))
    grid = figure.add_gridspec(2, 6, left=0.075, right=0.975, top=0.785,
                               bottom=0.09, wspace=0.78, hspace=0.52)
    _heading(figure, "Distribution of purchase lines, baskets and customer histories",
             f"{window}  ·  Share at or above each observed value  ·  Both axes logarithmic  ·  Full range")
    panels = [
        ("line_unit_price_gbp", "Unit price per purchase line", "GBP per unit", "Purchase lines (%)", grid[0, :3]),
        ("line_quantity", "Quantity per purchase line", "Units", "Purchase lines (%)", grid[0, 3:]),
        ("basket_distinct_skus", "Size of a basket", "Distinct SKUs", "Baskets (%)", grid[1, :2]),
        ("customer_history_events", "Events per customer", "Basket-item events", "Customers (%)", grid[1, 2:4]),
        ("customer_baskets", "Baskets per customer", "Completed baskets", "Customers (%)", grid[1, 4:]),
    ]
    for measure, title, unit, population, slot in panels:
        axis = figure.add_subplot(slot)
        _style(axis)
        axis.set_title(title, fontsize=11, loc="left", color=INK)
        axis.set_xlabel(unit, fontsize=10)
        axis.set_ylabel(population, fontsize=10)
        rows = distributions.loc[distributions["measure"] == measure].sort_values("value")
        if rows.empty:
            axis.text(0.5, 0.5, "No observations", transform=axis.transAxes, ha="center")
            continue
        # For an observed x, sum exact frequencies from x through the maximum.
        share = rows["frequency"].iloc[::-1].cumsum().iloc[::-1] * 100 / rows["frequency"].sum()
        axis.step(rows["value"], share, where="pre", color=TEAL, linewidth=1.8)
        axis.scatter(rows["value"].iloc[[0, -1]], share.iloc[[0, -1]], s=16, color=TEAL, zorder=3)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.minorticks_off()
        axis.yaxis.set_major_formatter(FuncFormatter(_number))
        axis.xaxis.set_major_formatter(FuncFormatter(_number))
    return figure


def _popularity(popularity, window):
    figure = Figure(figsize=(10.5, 6))
    axis = figure.subplots()
    figure.subplots_adjust(left=0.10, right=0.975, top=0.80, bottom=0.14)
    _heading(figure, "Item popularity by customer reach",
             f"{window}  ·  All {len(popularity):,} training SKUs  ·  Logarithmic customer count")
    _style(axis)
    axis.set_xlabel("Item rank by number of distinct purchasing customers")
    axis.set_ylabel("Distinct customers purchasing the item")
    if not popularity.empty:
        ordered = popularity.sort_values("rank")
        head = ordered.loc[ordered["is_head"], "rank"]
        if not head.empty:
            axis.axvspan(0.5, head.max() + 0.5, color=TEAL, alpha=0.13,
                         label=f"Top 10% of items (ranks 1–{head.max():,})")
            axis.legend(loc="upper right", frameon=False, fontsize=10)
        axis.plot(ordered["rank"], ordered["purchasing_customers"], color=BLUE, linewidth=2)
        axis.set_yscale("log")
        axis.set_xlim(0.5, ordered["rank"].max() + 0.5)
        axis.minorticks_off()
    axis.yaxis.set_major_formatter(FuncFormatter(_number))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    return figure


def build_training_figures(tables: dict[str, pd.DataFrame]) -> dict[str, Figure]:
    """Build displayable, unsaved figures. The caller chooses file format and DPI."""
    monthly = tables["monthly_activity"].sort_values("month")
    periods = pd.PeriodIndex(monthly["month"], freq="M")
    window = "Training window"
    if len(periods):
        window += f" · {periods.min().strftime('%b %Y')}–{periods.max().strftime('%b %Y')}"
    return {
        "monthly_volume": _monthly(monthly, window),
        "size_distributions": _distributions(tables["distributions"], window),
        "item_popularity": _popularity(tables["item_popularity"], window),
    }
