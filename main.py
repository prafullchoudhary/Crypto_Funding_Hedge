
import requests
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta

FUNDING_BASE_URL = "https://api.india.delta.exchange"
IST = timezone(timedelta(hours=5, minutes=30))
CORR_THRESHOLD_STRONG = 0.85
CORR_THRESHOLD_MODERATE = 0.70
ROLLING_WINDOW_DAYS = 30

def fetch_historical_funding_rates(symbol: str, from_time: int, to_time: int, resolution: str = "60"):
    """
    Fetch historical funding rates for a perpetual contract.

    Args:
        symbol     : Perpetual contract symbol, e.g. "BTCUSD", "ETHUSD"
        from_time  : Start time as Unix timestamp in seconds
        to_time    : End time as Unix timestamp in seconds
        resolution : Candle resolution - supported values:
                     "5S","1","120","15","1W","240","2W","3","30","360","5","60","D","W"
                     Default is "60" (1 hour). For funding rates, "60" is recommended.

    Returns:
        Pandas DataFrame with columns:
        - datetime : Candle timestamp in IST (UTC+5:30)
        - close    : Realized funding rate at that candle
    """
    url = f"{FUNDING_BASE_URL}/v2/chart/history"

    funding_symbol = f"FUNDING:{symbol}"

    params = {
        "symbol": funding_symbol,
        "resolution": resolution,
        "from": from_time,
        "to": to_time
    }

    try:
        response = requests.get(url, params=params, timeout=(3, 27))
        response.raise_for_status()
        data = response.json()
        # print(f"API response: {data}")  # Debugging line to inspect the API response
        if not data.get("success"):
            print(f"API returned failure response: {data}")
            return pd.DataFrame(columns=["datetime", "close"])

        records = data.get("result", {})
        timestamps = records.get("t", [])
        closes     = records.get("c", [])

        if not timestamps:
            print("No data returned for the given time range.")
            return pd.DataFrame(columns=["datetime", "close"])

        df = pd.DataFrame({
            "datetime": [
                datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M:%S")
                for ts in timestamps
            ],
            "close": closes
        })

        print(f"Fetched {len(df)} funding rate records for {symbol}")
        return df

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e} | Response: {response.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
    except requests.exceptions.Timeout:
        print("Request timed out.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return pd.DataFrame(columns=["datetime", "close"])

def filter_df(df: pd.DataFrame, rate_exchange_interval: int) -> pd.DataFrame:
    """
    Filters the funding rate DataFrame to return only rows at actual
    funding realization timestamps, calculated dynamically from the
    crypto day start time of 05:30 IST.

    The realization times are computed by adding multiples of the
    funding interval to the day start (05:30 IST), wrapping around
    midnight if necessary.

    Args:
        df                     : DataFrame returned by fetch_historical_funding_rates
                                 (must have 'datetime' and 'close' columns)
        rate_exchange_interval : Funding interval in seconds from product_specs
                                 e.g. 28800 for 8-hour funding (BTCUSD)
                                      14400 for 4-hour funding
                                      3600  for 1-hour funding

    Returns:
        Pandas DataFrame with only the realized funding rate rows.

    Examples:
        8h interval → realization times: 13:30, 21:30, 05:30 IST
        4h interval → realization times: 09:30, 13:30, 17:30, 21:30, 01:30, 05:30 IST
    """
    if df.empty:
        print("Input DataFrame is empty.")
        return df

    interval_minutes = rate_exchange_interval // 60

    # Crypto day starts at 05:30 IST = 330 minutes from midnight
    DAY_START_MINUTES = 4 * 60 + 30  # 330

    # Total minutes in a day
    TOTAL_DAY_MINUTES = 24 * 60  # 1440

    # Generate all realization times (in minutes from midnight)
    realization_times = set()
    current = DAY_START_MINUTES
    while True:
        current = (DAY_START_MINUTES + interval_minutes + 
                   (((current - DAY_START_MINUTES) // interval_minutes) * interval_minutes))
        next_time = (DAY_START_MINUTES + interval_minutes * 
                     (len(realization_times) + 1)) % TOTAL_DAY_MINUTES
        realization_times.add(next_time)
        if len(realization_times) == TOTAL_DAY_MINUTES // interval_minutes:
            break

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Total minutes from midnight for each row
    total_minutes = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute

    df = df[total_minutes.isin(realization_times)].reset_index(drop=True)

    return df

HISTORY_BASE_URL = "https://api.india.delta.exchange"

# Resolution to seconds mapping
RESOLUTION_SECONDS = {
    "1m":  60,
    "3m":  180,
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1h":  3600,
    "2h":  7200,
    "4h":  14400,
    "6h":  21600,
    "1d":  86400,
    "1w":  604800
}

MAX_CANDLES_PER_REQUEST = 4000
ENTRY_THRESHOLD = 0.01


def fetch_ohlc_page(symbol: str, resolution: str, start: int, end: int) -> list:
    """
    Fetch a single page of OHLC candles (up to 4000) from Delta Exchange.

    Args:
        symbol     (str): Trading symbol, e.g. "BTCUSD", "MARK:BTCUSD"
        resolution (str): Candle timeframe, e.g. "1m", "1h", "1d"
        start      (int): Start time as Unix timestamp in seconds
        end        (int): End time as Unix timestamp in seconds

    Returns:
        list: List of OHLC candle dicts
    """
    url = f"{HISTORY_BASE_URL}/v2/history/candles"

    params = {
        "symbol": symbol,
        "resolution": resolution,
        "start": start,
        "end": end
    }

    try:
        response = requests.get(url, params=params, timeout=(3, 27))
        response.raise_for_status()
        data = response.json()

        if not data.get("success"):
            raise ValueError(f"API returned failure: {data}")

        return data.get("result", [])

    except requests.exceptions.Timeout:
        raise RuntimeError("Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP request failed: {e}")


def fetch_ohlc(symbol: str, resolution: str, start: int, end: int) -> "pd.DataFrame":
    """
    Fetch all historical OHLC candles with automatic pagination.

    Splits the time range into chunks of MAX_CANDLES_PER_REQUEST candles
    and merges the results into a single DataFrame.

    Args:
        symbol     (str): Trading symbol, e.g. "BTCUSD", "MARK:BTCUSD", "FUNDING:BTCUSD"
        resolution (str): Candle timeframe. Supported values:
                          "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "1d", "1w"
        start      (int): Start time as Unix timestamp in seconds
        end        (int): End time as Unix timestamp in seconds

    Returns:
        pd.DataFrame: DataFrame with columns: time, open, high, low, close, volume
                      'time' column is converted to UTC datetime and set as the index.

    Raises:
        ValueError: If an unsupported resolution is provided or API returns failure
        RuntimeError: If the HTTP request fails or times out
    """

    if resolution not in RESOLUTION_SECONDS:
        raise ValueError(
            f"Unsupported resolution '{resolution}'. "
            f"Supported values: {list(RESOLUTION_SECONDS.keys())}"
        )

    candle_duration = RESOLUTION_SECONDS[resolution]
    chunk_duration  = MAX_CANDLES_PER_REQUEST * candle_duration

    all_candles = []
    chunk_start = start

    while chunk_start < end:
        chunk_end = min(chunk_start + chunk_duration, end)

        print(f"Fetching candles from {chunk_start} to {chunk_end} ...")
        page = fetch_ohlc_page(symbol, resolution, chunk_start, chunk_end)

        if page:
            all_candles.extend(page)

        chunk_start = chunk_end

        # Avoid hitting rate limits between paginated requests
        if chunk_start < end:
            time.sleep(0.2)

    if not all_candles:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_candles)

    # Deduplicate by timestamp
    df = df.drop_duplicates(subset="time")

    # Sort by time ascending
    df = df.sort_values("time").reset_index(drop=True)

    # Convert Unix timestamp to UTC datetime and set as index
    df["datetime"] = df["time"].apply(lambda ts: datetime.fromtimestamp(ts, tz=IST).strftime("%Y-%m-%d %H:%M:%S"))
    # df = df.set_index("time")

    return df


def build_trade_results(
    merged_df: pd.DataFrame,
    sym1: str,
    sym2: str,
    entry_threshold: float = ENTRY_THRESHOLD,
) -> pd.DataFrame:
    """
    Build one result row for each completed hedge trade.

    Entry rules:
    - enter long sym1 / short sym2 when net_funding > +entry_threshold
    - enter short sym1 / long sym2 when net_funding < -entry_threshold

    Exit rule:
    - if funding direction flips, exit on the previous row
    """
    result_rows = []
    trade_open = False
    trade_direction = None
    entry_index = None

    def get_entry_direction(net_funding: float):
        if net_funding > entry_threshold:
            return "long_sym1_short_sym2"
        if net_funding < -entry_threshold:
            return "short_sym1_long_sym2"
        return None

    def has_direction_flipped(direction: str, net_funding: float) -> bool:
        if direction == "long_sym1_short_sym2":
            return net_funding < 0
        return net_funding > 0

    def calculate_percent_change(entry_price: float, exit_price: float) -> float:
        return ((exit_price / entry_price) - 1) * 100

    for current_index in range(len(merged_df)):
        current_row = merged_df.iloc[current_index]
        current_net_funding = current_row["net_funding"]

        if not trade_open:
            new_direction = get_entry_direction(current_net_funding)
            if new_direction is None:
                continue

            trade_open = True
            trade_direction = new_direction
            entry_index = current_index
            continue

        if not has_direction_flipped(trade_direction, current_net_funding):
            continue

        exit_index = current_index - 1
        if exit_index < entry_index:
            trade_open = False
            trade_direction = None
            entry_index = None
            continue

        entry_row = merged_df.iloc[entry_index]
        exit_row = merged_df.iloc[exit_index]
        trade_slice = merged_df.iloc[entry_index: exit_index + 1]

        sym1_entry_close = entry_row[f"{sym1}_ohlc_close"]
        sym1_exit_close = exit_row[f"{sym1}_ohlc_close"]
        sym2_entry_close = entry_row[f"{sym2}_ohlc_close"]
        sym2_exit_close = exit_row[f"{sym2}_ohlc_close"]

        sym1_percent_change = calculate_percent_change(sym1_entry_close, sym1_exit_close)
        sym2_percent_change = calculate_percent_change(sym2_entry_close, sym2_exit_close)

        if trade_direction == "long_sym1_short_sym2":
            sym1_trade_percent_change = sym1_percent_change
            sym2_trade_percent_change = -sym2_percent_change
        else:
            sym1_trade_percent_change = -sym1_percent_change
            sym2_trade_percent_change = sym2_percent_change

        hedge_net_percentage_change = (
            sym1_trade_percent_change + sym2_trade_percent_change
        )
        sum_abs_net_funding = trade_slice["net_funding"].abs().sum()
        total_trade_return = hedge_net_percentage_change + sum_abs_net_funding

        result_rows.append({
            "direction": trade_direction,
            "start_datetime": entry_row["datetime"],
            "end_datetime": exit_row["datetime"],
            f"{sym1}_entry_close": sym1_entry_close,
            f"{sym2}_entry_close": sym2_entry_close,
            f"{sym1}_exit_close": sym1_exit_close,
            f"{sym2}_exit_close": sym2_exit_close,
            "sum_abs_net_funding": sum_abs_net_funding,
            f"{sym1}_trade_percent_change": sym1_trade_percent_change,
            f"{sym2}_trade_percent_change": sym2_trade_percent_change,
            "hedge_net_percentage_change": hedge_net_percentage_change,
            "total_trade_return": total_trade_return,
            "holding_rows": len(trade_slice),
        })

        trade_open = False
        trade_direction = None
        entry_index = None

    return pd.DataFrame(result_rows)


def plot_return_comparison(merged_df: pd.DataFrame, sym1: str, sym2: str):
    """
    Plot both 8-hour return series on the same chart for visual correlation comparison.
    """
    plt.figure(figsize=(15, 8))
    plt.plot(
        merged_df["datetime"],
        merged_df[f"{sym1}_cumm_return"],
        color="tab:blue",
        linewidth=1.4,
        label=f"{sym1} 8h Return",
    )
    plt.plot(
        merged_df["datetime"],
        merged_df[f"{sym2}_cumm_return"],
        color="tab:orange",
        linewidth=1.4,
        label=f"{sym2} 8h Return",
    )
    plt.title(f"8-Hour Return Comparison: {sym1} vs {sym2}")
    plt.xlabel("Datetime")
    plt.ylabel("Return (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"return_comparison_{sym1}_{sym2}.png", dpi=300, bbox_inches="tight")
    plt.show()


def prepare_trade_visualization_data(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add helper columns used across the strategy charts.
    """
    chart_df = result_df.copy()
    chart_df["start_datetime"] = pd.to_datetime(chart_df["start_datetime"])
    chart_df["end_datetime"] = pd.to_datetime(chart_df["end_datetime"])
    chart_df = chart_df.sort_values("start_datetime").reset_index(drop=True)

    chart_df["trade_number"] = np.arange(1, len(chart_df) + 1)
    chart_df["holding_hours"] = chart_df["holding_rows"] * 8
    chart_df["holding_days"] = chart_df["holding_hours"] / 24

    chart_df["cumulative_funding_return"] = chart_df["sum_abs_net_funding"].cumsum()
    chart_df["cumulative_hedge_return"] = chart_df["hedge_net_percentage_change"].cumsum()
    chart_df["cumulative_total_return"] = chart_df["total_trade_return"].cumsum()

    chart_df["running_peak"] = chart_df["cumulative_total_return"].cummax()
    chart_df["drawdown"] = (
        chart_df["cumulative_total_return"] - chart_df["running_peak"]
    )

    chart_df["trade_month"] = chart_df["start_datetime"].dt.to_period("M").astype(str)
    chart_df["trade_year"] = chart_df["start_datetime"].dt.year
    chart_df["trade_month_name"] = chart_df["start_datetime"].dt.strftime("%b")

    return chart_df


def plot_cumulative_strategy_return(chart_df: pd.DataFrame, sym1: str, sym2: str):
    plt.figure(figsize=(14, 7))
    plt.plot(
        chart_df["end_datetime"],
        chart_df["cumulative_total_return"],
        color="navy",
        linewidth=2,
        label="Cumulative Strategy Return",
    )
    plt.title(f"Cumulative Strategy Return: {sym1} / {sym2}")
    plt.xlabel("Trade Exit Time")
    plt.ylabel("Return (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_return_breakdown(chart_df: pd.DataFrame, sym1: str, sym2: str):
    plt.figure(figsize=(14, 7))
    plt.plot(
        chart_df["end_datetime"],
        chart_df["cumulative_funding_return"],
        color="teal",
        linewidth=2,
        label="Cumulative Funding Return",
    )
    plt.plot(
        chart_df["end_datetime"],
        chart_df["cumulative_hedge_return"],
        color="darkorange",
        linewidth=2,
        label="Cumulative Hedge Return",
    )
    plt.plot(
        chart_df["end_datetime"],
        chart_df["cumulative_total_return"],
        color="purple",
        linewidth=2.2,
        label="Cumulative Total Return",
    )
    plt.title(f"Cumulative Return Breakdown: {sym1} / {sym2}")
    plt.xlabel("Trade Exit Time")
    plt.ylabel("Return (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_per_trade_total_return(chart_df: pd.DataFrame, sym1: str, sym2: str):
    colors = ["forestgreen" if value >= 0 else "crimson" for value in chart_df["total_trade_return"]]

    plt.figure(figsize=(14, 7))
    plt.bar(chart_df["trade_number"], chart_df["total_trade_return"], color=colors, alpha=0.85)
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Per-Trade Total Return: {sym1} / {sym2}")
    plt.xlabel("Trade Number")
    plt.ylabel("Return (%)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_funding_vs_hedge_per_trade(chart_df: pd.DataFrame, sym1: str, sym2: str):
    x = np.arange(len(chart_df))
    width = 0.38

    plt.figure(figsize=(15, 7))
    plt.bar(
        x - width / 2,
        chart_df["sum_abs_net_funding"],
        width=width,
        color="steelblue",
        label="Funding Return",
    )
    plt.bar(
        x + width / 2,
        chart_df["hedge_net_percentage_change"],
        width=width,
        color="darkorange",
        label="Hedge Return",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Funding Return vs Hedge Return Per Trade: {sym1} / {sym2}")
    plt.xlabel("Trade Number")
    plt.ylabel("Return (%)")
    plt.xticks(x, chart_df["trade_number"])
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_trade_duration(chart_df: pd.DataFrame, sym1: str, sym2: str):
    plt.figure(figsize=(14, 7))
    plt.bar(chart_df["trade_number"], chart_df["holding_days"], color="slateblue", alpha=0.85)
    plt.title(f"Trade Duration in Days: {sym1} / {sym2}")
    plt.xlabel("Trade Number")
    plt.ylabel("Holding Time (days)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_holding_time_vs_total_return(chart_df: pd.DataFrame, sym1: str, sym2: str):
    plt.figure(figsize=(12, 7))
    plt.scatter(
        chart_df["holding_days"],
        chart_df["total_trade_return"],
        s=70,
        color="mediumseagreen",
        alpha=0.8,
        edgecolors="black",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Holding Time vs Total Return: {sym1} / {sym2}")
    plt.xlabel("Holding Time (days)")
    plt.ylabel("Total Trade Return (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_funding_vs_hedge_scatter(chart_df: pd.DataFrame, sym1: str, sym2: str):
    plt.figure(figsize=(12, 7))
    plt.scatter(
        chart_df["sum_abs_net_funding"],
        chart_df["hedge_net_percentage_change"],
        s=70,
        color="darkviolet",
        alpha=0.8,
        edgecolors="black",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.title(f"Funding Return vs Hedge Return: {sym1} / {sym2}")
    plt.xlabel("Funding Return (%)")
    plt.ylabel("Hedge Return (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_trade_return_distribution(chart_df: pd.DataFrame, sym1: str, sym2: str):
    plt.figure(figsize=(12, 7))
    plt.hist(
        chart_df["total_trade_return"],
        bins=min(10, max(5, len(chart_df))),
        color="cornflowerblue",
        edgecolor="black",
        alpha=0.85,
    )
    plt.axvline(0, color="red", linewidth=1.2)
    plt.title(f"Distribution of Trade Returns: {sym1} / {sym2}")
    plt.xlabel("Total Trade Return (%)")
    plt.ylabel("Number of Trades")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_drawdown_chart(chart_df: pd.DataFrame, sym1: str, sym2: str):
    plt.figure(figsize=(14, 7))
    plt.fill_between(
        chart_df["end_datetime"],
        chart_df["drawdown"],
        0,
        color="crimson",
        alpha=0.35,
    )
    plt.plot(chart_df["end_datetime"], chart_df["drawdown"], color="crimson", linewidth=1.8)
    plt.title(f"Strategy Drawdown: {sym1} / {sym2}")
    plt.xlabel("Trade Exit Time")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_direction_split(chart_df: pd.DataFrame, sym1: str, sym2: str):
    direction_summary = (
        chart_df.groupby("direction")[["sum_abs_net_funding", "hedge_net_percentage_change", "total_trade_return"]]
        .mean()
    )

    x = np.arange(len(direction_summary.index))
    width = 0.25

    plt.figure(figsize=(14, 7))
    plt.bar(x - width, direction_summary["sum_abs_net_funding"], width=width, color="steelblue", label="Funding")
    plt.bar(x, direction_summary["hedge_net_percentage_change"], width=width, color="darkorange", label="Hedge")
    plt.bar(x + width, direction_summary["total_trade_return"], width=width, color="seagreen", label="Total")
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Average Return by Trade Direction: {sym1} / {sym2}")
    plt.xlabel("Trade Direction")
    plt.ylabel("Average Return (%)")
    plt.xticks(x, direction_summary.index, rotation=10)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_monthly_return_heatmap(chart_df: pd.DataFrame, sym1: str, sym2: str):
    monthly_returns = (
        chart_df.groupby(["trade_year", "trade_month_name"])["total_trade_return"]
        .sum()
        .reset_index()
    )

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_returns["trade_month_name"] = pd.Categorical(
        monthly_returns["trade_month_name"],
        categories=month_order,
        ordered=True,
    )
    monthly_returns = monthly_returns.sort_values(["trade_year", "trade_month_name"])

    heatmap_df = monthly_returns.pivot(
        index="trade_year",
        columns="trade_month_name",
        values="total_trade_return",
    ).reindex(columns=month_order)

    fig, ax = plt.subplots(figsize=(14, 5))
    image = ax.imshow(heatmap_df.fillna(0).values, cmap="RdYlGn", aspect="auto")
    ax.set_title(f"Monthly Return Heatmap: {sym1} / {sym2}")
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns)
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)

    for row_index in range(len(heatmap_df.index)):
        for col_index in range(len(heatmap_df.columns)):
            value = heatmap_df.iloc[row_index, col_index]
            if pd.notna(value):
                ax.text(col_index, row_index, f"{value:.1f}", ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, label="Return (%)")
    plt.tight_layout()
    plt.show()


def plot_net_funding_with_trade_overlay(
    merged_df: pd.DataFrame,
    result_df: pd.DataFrame,
    sym1: str,
    sym2: str,
):
    merged_plot_df = merged_df.copy()
    merged_plot_df["datetime"] = pd.to_datetime(merged_plot_df["datetime"])

    trade_plot_df = result_df.copy()
    trade_plot_df["start_datetime"] = pd.to_datetime(trade_plot_df["start_datetime"])
    trade_plot_df["end_datetime"] = pd.to_datetime(trade_plot_df["end_datetime"])

    plt.figure(figsize=(15, 7))
    plt.plot(
        merged_plot_df["datetime"],
        merged_plot_df["net_funding"],
        color="black",
        linewidth=1.5,
        label="Net Funding",
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.axhline(ENTRY_THRESHOLD, color="green", linestyle="--", linewidth=1, alpha=0.7)
    plt.axhline(-ENTRY_THRESHOLD, color="green", linestyle="--", linewidth=1, alpha=0.7)

    for _, trade in trade_plot_df.iterrows():
        if trade["direction"] == "long_sym1_short_sym2":
            trade_color = "mediumseagreen"
        else:
            trade_color = "tomato"

        plt.axvspan(
            trade["start_datetime"],
            trade["end_datetime"],
            color=trade_color,
            alpha=0.18,
        )

    plt.title(f"Net Funding with Trade Overlay: {sym1} / {sym2}")
    plt.xlabel("Datetime")
    plt.ylabel("Net Funding")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_strategy_visual_report(
    merged_df: pd.DataFrame,
    result_df: pd.DataFrame,
    sym1: str,
    sym2: str,
):
    """
    Draw a full visualization pack for presenting the strategy.
    """
    if result_df.empty:
        print("No trades available for visualization.")
        return

    chart_df = prepare_trade_visualization_data(result_df)

    plot_return_comparison(merged_df, sym1, sym2)
    plot_cumulative_strategy_return(chart_df, sym1, sym2)
    plot_cumulative_return_breakdown(chart_df, sym1, sym2)
    plot_per_trade_total_return(chart_df, sym1, sym2)
    plot_funding_vs_hedge_per_trade(chart_df, sym1, sym2)
    plot_trade_duration(chart_df, sym1, sym2)
    plot_holding_time_vs_total_return(chart_df, sym1, sym2)
    plot_funding_vs_hedge_scatter(chart_df, sym1, sym2)
    plot_trade_return_distribution(chart_df, sym1, sym2)
    plot_drawdown_chart(chart_df, sym1, sym2)
    plot_direction_split(chart_df, sym1, sym2)
    plot_monthly_return_heatmap(chart_df, sym1, sym2)
    plot_net_funding_with_trade_overlay(merged_df, result_df, sym1, sym2)


def plot_strategy_dashboard(
    merged_df: pd.DataFrame,
    result_df: pd.DataFrame,
    sym1: str,
    sym2: str,
    save_path: str | None = None,
):
    """
    Draw a compact dashboard for presenting the strategy in one figure.
    """
    if result_df.empty:
        print("No trades available for dashboard.")
        return

    merged_plot_df = merged_df.copy()
    merged_plot_df["datetime"] = pd.to_datetime(merged_plot_df["datetime"])

    chart_df = prepare_trade_visualization_data(result_df)

    fig, axes = plt.subplots(3, 2, figsize=(20, 14))
    fig.suptitle(
        f"Funding Hedge Strategy Dashboard: {sym1} / {sym2}",
        fontsize=18,
        fontweight="bold",
    )

    ax1 = axes[0, 0]
    ax1.plot(
        merged_plot_df["datetime"],
        merged_plot_df[f"{sym1}_cumm_return"],
        color="tab:blue",
        linewidth=1.8,
        label=f"{sym1} Cumulative Return",
    )
    ax1.plot(
        merged_plot_df["datetime"],
        merged_plot_df[f"{sym2}_cumm_return"],
        color="tab:orange",
        linewidth=1.8,
        label=f"{sym2} Cumulative Return",
    )
    ax1.set_title("Instrument Return Comparison")
    ax1.set_ylabel("Return (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = axes[0, 1]
    ax2.plot(
        chart_df["end_datetime"],
        chart_df["cumulative_funding_return"],
        color="teal",
        linewidth=2,
        label="Funding",
    )
    ax2.plot(
        chart_df["end_datetime"],
        chart_df["cumulative_hedge_return"],
        color="darkorange",
        linewidth=2,
        label="Hedge",
    )
    ax2.plot(
        chart_df["end_datetime"],
        chart_df["cumulative_total_return"],
        color="purple",
        linewidth=2.2,
        label="Total",
    )
    ax2.set_title("Cumulative Return Breakdown")
    ax2.set_ylabel("Return (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = axes[1, 0]
    per_trade_colors = [
        "forestgreen" if value >= 0 else "crimson"
        for value in chart_df["total_trade_return"]
    ]
    ax3.bar(
        chart_df["trade_number"],
        chart_df["total_trade_return"],
        color=per_trade_colors,
        alpha=0.85,
    )
    ax3.axhline(0, color="black", linewidth=1)
    ax3.set_title("Per-Trade Total Return")
    ax3.set_xlabel("Trade Number")
    ax3.set_ylabel("Return (%)")
    ax3.grid(True, axis="y", alpha=0.3)

    ax4 = axes[1, 1]
    x = np.arange(len(chart_df))
    width = 0.38
    ax4.bar(
        x - width / 2,
        chart_df["sum_abs_net_funding"],
        width=width,
        color="steelblue",
        label="Funding",
    )
    ax4.bar(
        x + width / 2,
        chart_df["hedge_net_percentage_change"],
        width=width,
        color="darkorange",
        label="Hedge",
    )
    ax4.axhline(0, color="black", linewidth=1)
    ax4.set_title("Funding vs Hedge Per Trade")
    ax4.set_xlabel("Trade Number")
    ax4.set_ylabel("Return (%)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(chart_df["trade_number"])
    ax4.grid(True, axis="y", alpha=0.3)
    ax4.legend()

    ax5 = axes[2, 0]
    ax5.scatter(
        chart_df["holding_days"],
        chart_df["total_trade_return"],
        s=70,
        color="mediumseagreen",
        alpha=0.8,
        edgecolors="black",
    )
    ax5.axhline(0, color="black", linewidth=1)
    ax5.set_title("Holding Time vs Total Return")
    ax5.set_xlabel("Holding Time (days)")
    ax5.set_ylabel("Total Trade Return (%)")
    ax5.grid(True, alpha=0.3)

    ax6 = axes[2, 1]
    ax6.plot(
        merged_plot_df["datetime"],
        merged_plot_df["net_funding"],
        color="black",
        linewidth=1.3,
        label="Net Funding",
    )
    ax6.axhline(0, color="red", linestyle="--", linewidth=1)
    ax6.axhline(ENTRY_THRESHOLD, color="green", linestyle="--", linewidth=1, alpha=0.7)
    ax6.axhline(-ENTRY_THRESHOLD, color="green", linestyle="--", linewidth=1, alpha=0.7)

    for _, trade in chart_df.iterrows():
        trade_color = "mediumseagreen" if trade["direction"] == "long_sym1_short_sym2" else "tomato"
        ax6.axvspan(
            trade["start_datetime"],
            trade["end_datetime"],
            color=trade_color,
            alpha=0.18,
        )

    ax6.set_title("Net Funding with Trade Overlay")
    ax6.set_xlabel("Datetime")
    ax6.set_ylabel("Net Funding")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def print_trade_summary_report(csv_path: str, sym1: str, sym2: str):
    """
    Read the saved trade-results CSV and print a simple strategy report.
    """
    result_df = pd.read_csv(csv_path)

    if result_df.empty:
        print("\nTrade Summary Report")
        print("No trades found in the CSV.")
        return

    result_df["start_datetime"] = pd.to_datetime(result_df["start_datetime"])
    result_df["end_datetime"] = pd.to_datetime(result_df["end_datetime"])

    backtest_start = result_df["start_datetime"].min()
    backtest_end = result_df["end_datetime"].max()
    backtest_days = (backtest_end - backtest_start).total_seconds() / 86400
    annual_factor = 365 / backtest_days if backtest_days > 0 else 0

    total_trades = len(result_df)
    total_funding_return = result_df["sum_abs_net_funding"].sum()
    total_hedge_return = result_df["hedge_net_percentage_change"].sum()
    total_strategy_return = result_df["total_trade_return"].sum()

    yearly_funding_return = total_funding_return * annual_factor
    yearly_hedge_return = total_hedge_return * annual_factor
    yearly_strategy_return = total_strategy_return * annual_factor

    average_trade_return = result_df["total_trade_return"].mean()
    average_holding_rows = result_df["holding_rows"].mean()
    average_holding_hours = average_holding_rows * 8
    average_holding_days = average_holding_hours / 24

    print("\n" + "=" * 72)
    print(f"Trade Summary Report: {sym1} / {sym2} Funding-Hedge Strategy")
    print("=" * 72)
    print(f"Backtest period           : {backtest_start} to {backtest_end}")
    print(f"Backtest length (days)    : {backtest_days:.1f}")
    print(f"Total trades              : {total_trades}")
    print(
        f"Average holding time      : {average_holding_hours:.2f} hours "
        f"({average_holding_days:.2f} days)"
    )

    print("\nReturn Breakdown")
    print(f"Funding return total      : {total_funding_return:.4f}%")
    print(f"Hedge return total        : {total_hedge_return:.4f}%")
    print(f"Strategy return total     : {total_strategy_return:.4f}%")

    print("\nAnnualized Return Estimate")
    print(f"Funding return yearly     : {yearly_funding_return:.4f}%")
    print(f"Hedge return yearly       : {yearly_hedge_return:.4f}%")
    print(f"Strategy return yearly    : {yearly_strategy_return:.4f}%")
    print("=" * 72)
    

# -------------------------------------------------------
# Example Usage
# -------------------------------------------------------
if __name__ == "__main__":
    symbol1 = "BTCUSD"
    symbol2 = "BCHUSD"

    from_time = int(datetime(2025, 1, 1).timestamp())
    # to_time   = int(datetime(2026, 1, 1).timestamp())
    to_time   = int(datetime.now().timestamp())

    funding_df1 = fetch_historical_funding_rates(
        symbol=symbol1,
        from_time=from_time,
        to_time=to_time,
        resolution="60"
    )

    funding_df2 = fetch_historical_funding_rates(
        symbol=symbol2,
        from_time=from_time,
        to_time=to_time,
        resolution="60"
    )
    
    ohlc_df1 = fetch_ohlc(
        symbol     = symbol1,
        resolution = "1h",
        start      = from_time,
        end        = to_time
    )

    filtered_ohlc_df1 = filter_df(ohlc_df1[['datetime', 'close']], rate_exchange_interval=28800)  # 8-hour funding interval for BTCUSD

    ohlc_df2 = fetch_ohlc(
        symbol     = symbol2,
        resolution = "1h",
        start      = from_time,
        end        = to_time
    )

    filtered_ohlc_df2 = filter_df(ohlc_df2[['datetime', 'close']], rate_exchange_interval=28800)  # 8-hour funding interval for BCHUSD

    # Filter only funding realization candles (non-zero close)
    funding_df1 = funding_df1[funding_df1["close"].notna()].reset_index(drop=True)
    funding_df2 = funding_df2[funding_df2["close"].notna()].reset_index(drop=True)

    # Filter to only realized funding timestamps
    # BTCUSD rate_exchange_interval = 28800 seconds (8 hours)
    # Filter to realized funding timestamps (8h interval for BTCUSD)
    filtered_funding_df1 = filter_df(funding_df1, rate_exchange_interval=28800)
    filtered_funding_df2 = filter_df(funding_df2, rate_exchange_interval=28800)

    merged_df = (
        filtered_ohlc_df1.rename(columns={"close": f"{symbol1}_ohlc_close"})
        .merge(
            filtered_ohlc_df2.rename(columns={"close": f"{symbol2}_ohlc_close"}),
            on="datetime",
            how="outer",
        )
        .merge(
            filtered_funding_df1.rename(columns={"close": f"{symbol1}_funding_close"}),
            on="datetime",
            how="outer",
        )
        .merge(
            filtered_funding_df2.rename(columns={"close": f"{symbol2}_funding_close"}),
            on="datetime",
            how="outer",
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    merged_df['net_funding'] = merged_df[f"{symbol2}_funding_close"] - merged_df[f"{symbol1}_funding_close"]

    merged_df[f"{symbol1}_ohlc_pct_change"] = (
        merged_df[f"{symbol1}_ohlc_close"].pct_change() * 100
    ).round(2)
    merged_df[f"{symbol2}_ohlc_pct_change"] = (
        merged_df[f"{symbol2}_ohlc_close"].pct_change() * 100
    ).round(2)

    merged_df["abs_net_funding"] = merged_df["net_funding"].abs()

    merged_df["BTCUSD_cumm_return"] = merged_df["BTCUSD_ohlc_pct_change"].cumsum()
    merged_df["BCHUSD_cumm_return"] = merged_df["BCHUSD_ohlc_pct_change"].cumsum()

    merged_df.dropna(inplace=True)
    merged_df = merged_df.reset_index(drop=True)

    merged_df.to_csv(f"merged_data_{symbol1}_{symbol2}.csv", index=False)    

    result_df = build_trade_results(merged_df, symbol1, symbol2)

    trade_results_path = f"trade_results_{symbol1}_{symbol2}.csv"
    result_df.to_csv(trade_results_path, index=False)

    print_trade_summary_report(trade_results_path, symbol1, symbol2)
    
    plot_strategy_dashboard(
        merged_df,
        result_df,
        symbol1,
        symbol2,
        save_path=f"strategy_dashboard_{symbol1}_{symbol2}.png",
    )


