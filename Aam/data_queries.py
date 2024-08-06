import sys
import logging
import pandas as pd
import numpy as np
from timeit import default_timer as timer


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)],
# )
# logger = logging.getLogger("__main__."+__name__)
# logger.setLevel(logging.DEBUG)

MAX_PRICE = 5000

import os
import pyodbc
import configparser
DB_Config = configparser.ConfigParser()
DB_Config.read(os.path.join("C:\Config", "DatabaseConfig.ini"))

from GQDatabaseSQL.MDSDB import DATAPOOL

# class odbc_conn():
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def dw_price_guideline_connection():
#         server_name = DB_Config["GDDatabase_UAT"]["IP"]
#         database_name = DB_Config["GDDatabase_UAT"]["Dbname"]
#         # table_name =
#         UserBD = DB_Config["GDDatabase_UAT"]["UserDB"]
#         PassBD = DB_Config["GDDatabase_UAT"]["PassDB"]
#         """
#         Trusted_Connection -> connects using user's account if yes
#         """
#
#         conn = pyodbc.connect(
#             "Driver={SQL Server};"
#             f"Server={server_name};"
#             f"Database={database_name};"
#             "Trusted_Connection=No;"
#             f"UID={UserBD};"
#             f"PWD={PassBD};"
#         )
#         return conn
#
#
#     @staticmethod
#     def tick_data_connection():
#         server_name = DB_Config["SET_EMAPIR1"]["IP"]
#         database_name = DB_Config["SET_EMAPIR1"]["Dbname"]
#         # table_name =
#         UserBD = DB_Config["SET_EMAPIR1"]["UserDB"]
#         PassBD = DB_Config["SET_EMAPIR1"]["PassDB"]
#         """
#         Trusted_Connection -> connects using user's account if yes
#         """
#
#         conn = pyodbc.connect(
#             "Driver={SQL Server};"
#             f"Server={server_name};"
#             f"Database={database_name};"
#             "Trusted_Connection=No;"
#             f"UID={UserBD};"
#             f"PWD={PassBD};"
#         )
#         return conn
#
#     @staticmethod
#     def setsmart_connection():
#         server_name = DB_Config["SetSmartDB"]["IP"]
#         database_name = DB_Config["SetSmartDB"]["Dbname"]
#         # table_name =
#         UserBD = DB_Config["SetSmartDB"]["UserDB"]
#         PassBD = DB_Config["SetSmartDB"]["PassDB"]
#         """
#         Trusted_Connection -> connects using user's account if yes
#         """
#
#         conn = pyodbc.connect(
#             "Driver={SQL Server};"
#             f"Server={server_name};"
#             f"Database={database_name};"
#             "Trusted_Connection=No;"
#             f"UID={UserBD};"
#             f"PWD={PassBD};"
#         )
#         return conn


# odbc_conn = DB_conn()

def get_dw_price_guideline(dw_symbol: str, dt: pd.Timestamp):
    # dt = dt.strftime("%Y-%m-%d")
    # conn = odbc_conn.dw_price_guideline_connection()
    # query = f"""select *
    #             from
    #                 DATAPOOL.dbo.pgMatrix
    #             where
    #                 asofdate = '{dt}' and symbol like '%{dw_symbol}%' """
    # df = pd.read_sql(query, conn)
    # dt_plus_one = dt + pd.Timedelta(1,'D')
    df = DATAPOOL.get_pricing_guideline(starting_date=dt,ending_date=dt,symbols= f"{dw_symbol}%")

    df["asofdate"] = pd.to_datetime(df["asofdate"])
    return df


def get_open_close(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, freq_in_days: int = 1
):
    conn = odbc_conn.tick_data_connection()
    str_date1 = start_date.strftime("%Y-%m-%d %H:%M:%S")
    str_date2 = end_date.strftime("%Y-%m-%d %H:%M:%S")
    if symbol is None:
        # Grab All
        AND_CLAUSE = ""
    else:

        AND_CLAUSE = f"AND (SecName LIKE '{symbol}')"

    query = f"""SELECT 
                    *
                FROM 
                    EMAPIDB.dbo.DATAPREP_EquityOCStat
                WHERE 
                    UpdateTime BETWEEN '{str_date1}' AND '{str_date2}' 
                    {AND_CLAUSE}
                ORDER BY 
                    UpdateTime ASC"""
    df = pd.read_sql(query, conn)
    df["UpdateTime"] = (
        pd.to_datetime(df["UpdateTime"])
        .dt.tz_localize("UTC")
        .dt.tz_convert("Asia/Bangkok")
        .dt.tz_localize(None)
    )
    if freq_in_days == 0.5:
        offset_hours = 3
        df["UpdateTime"] = (
            df["UpdateTime"] - pd.Timedelta(hours=offset_hours)
        ).dt.ceil("1H")
    elif freq_in_days == 1.0:
        df["UpdateTime"] = df["UpdateTime"].apply(
            lambda x: pd.Timestamp(x.date()) + pd.Timedelta(hours=10)
        )
    df = df.pivot(index="UpdateTime", columns="OpenClose", values=["Mid"])
    df = df.droplevel(0, axis=1)
    df["IntradayOpenToClose"] = df["C"] - df["O"]
    df["ONCloseToOpen"] = df["O"] - df["C"].shift()
    return df[["IntradayOpenToClose", "ONCloseToOpen"]].reset_index()
    # return df


def get_equity_trades(
    date1, date2, symbol=None, bkk_time: bool = False
) -> pd.DataFrame:
    """
    Simple query to grab raw tick data (trades)
    :param date1:
    :param date2:
    :param symbol:
    :param bkk_time: boolean, data come in utc time. If we want bkk_time this must be set to True
    :return:
    """
    conn = odbc_conn.tick_data_connection()

    if bkk_time:
        date1 = date1.tz_localize("Asia/Bangkok").tz_convert("UTC")
        date2 = date2.tz_localize("Asia/Bangkok").tz_convert("UTC")

    str_date1 = date1.strftime("%Y-%m-%d %H:%M:%S")
    str_date2 = date2.strftime("%Y-%m-%d %H:%M:%S")

    if symbol is None:
        # Grab All
        AND_CLAUSE = ""
    else:

        AND_CLAUSE = f"AND (SecName LIKE '{symbol}')"

    query = f"""SELECT 
                    TradeTime, SendingTime, ReceivingTime, 
                    SeqNo, SecCode, SecName, LastPrice, Volume, 
                    BidAggressor, AskAggressor, IsTradeReport, MatchType
                FROM 
                    EMAPIDB.dbo.MDSEMAPI_EquityTicker 
                WHERE 
                    TradeTime BETWEEN '{str_date1}' AND '{str_date2}' {AND_CLAUSE}
                ORDER BY 
                    TradeTime ASC"""
    df = pd.read_sql(query, conn)
    if bkk_time:
        df["TradeTime"] = pd.to_datetime(df["TradeTime"])
        df["TradeTime"] = (
            df["TradeTime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Bangkok")
        ).dt.tz_localize(None)

    return df


def get_equity_trades_one_side_by_freq(
    symbol: str,
    include_auction: bool,
    direction: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    freq_in_hours: int,
):
    """
    Use SQL groupby to grab data at 6 hour frequency. This allows for quicker grab and less memory usage
    :param symbol:
    :param include_auction:
    :param direction:
    :param start_date:
    :param end_date:
    :param freq_in_hours:
    :return:
    """
    if symbol is None:
        secname_clause = ""
    else:
        secname_clause = f"AND (SecName LIKE '{symbol}')"

    if include_auction is False:
        exclude_auction_clause = f"AND MatchType != 7"
    else:
        ""

    if direction == 1:
        direction_clause = f"AND BidAggressor = 1"
    else:
        direction_clause = f"AND AskAggressor = 1"

    str_date1 = start_date.strftime("%Y-%m-%d")
    str_date2 = end_date.strftime("%Y-%m-%d")

    query = f"""SELECT 
                    MAX(TradeTime) as TradeTime, 
                    SecName, 
                    AVG(LastPrice) as LastPrice, 
                    SUM(Volume) as Volume,
                    COUNT(SecName) as TradeCount
                FROM 
                    EMAPIDB.dbo.MDSEMAPI_EquityTicker 
                WHERE 
                    TradeTime BETWEEN '{str_date1}' AND '{str_date2}'
                    {secname_clause}
                    {exclude_auction_clause}
                    {direction_clause}
                 GROUP BY 
                    SecName,
                    DATEPART(YEAR, TradeTime),
                    DATEPART(MONTH, TradeTime),
                    DATEPART(DAY, TradeTime),
                    (DATEPART(HOUR, TradeTime) / {freq_in_hours})
                ORDER BY 
                    TradeTime ASC"""

    s = timer()
    conn = odbc_conn.tick_data_connection()
    df = pd.read_sql(query, conn)
    df["Direction"] = direction
    df["TradeTime"] = pd.to_datetime(df["TradeTime"])
    df["TradeTime"] = (
        df["TradeTime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Bangkok")
    ).dt.tz_localize(None)

    e = timer()
    logging.info(
        f"{symbol}: Got direction={direction} trades, freq={freq_in_hours} hours. Time taken= {e-s} seconds"
    )
    return df


def get_equity_trades_by_freq(
    symbol: str,
    include_auction: bool,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    freq_in_hours: int,
):
    """
    This combines volumes from buy and sell sides but uses avg prices from BidAgressor side (ask price).
    For price change prediction, it doesn't really matter whether one looks at mid, bid or ask change.
    :param symbol:
    :param include_auction:
    :param start_date:
    :param end_date:
    :param freq_in_hours:
    :return:
    """

    buy = get_equity_trades_one_side_by_freq(
        symbol, include_auction, 1, start_date, end_date, freq_in_hours
    )
    sell = get_equity_trades_one_side_by_freq(
        symbol, include_auction, -1, start_date, end_date, freq_in_hours
    )
    buy["TradeTime"] = buy["TradeTime"].dt.ceil("1H")
    sell["TradeTime"] = sell["TradeTime"].dt.ceil("1H")
    df = (
        buy.drop(columns=["Direction"])
        .set_index("TradeTime")
        .join(
            sell.drop(columns=["SecName", "LastPrice", "Direction"]).set_index(
                "TradeTime"
            ),
            rsuffix="_s",
        )
        .reset_index()
    )
    df["SignedVolume"] = df["Volume"] - df["Volume_s"]
    df["Volume"] = df["Volume"] + df["Volume_s"]
    df["TradeCount"] = df["TradeCount"] + df["TradeCount_s"]
    df["HourOfDay"] = df["TradeTime"].dt.hour
    df["DayOfWeek"] = df["TradeTime"].dt.dayofweek

    return df.drop(columns=["Volume_s", "TradeCount_s"])


def get_level1_lob_by_freq(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, freq_in_hours: int
):
    """
    Grab Level 1 LOB data using SQL groupby.
    :param symbol:
    :param start_date:
    :param end_date:
    :param freq_in_hours:
    :return:
    """

    if symbol is None:
        secname_clause = ""
    else:
        secname_clause = f"AND (SecName LIKE '{symbol}')"

    str_date1 = start_date.strftime("%Y-%m-%d")
    str_date2 = end_date.strftime("%Y-%m-%d")

    query = f"""SELECT 
                MAX(UpdateTime) as UpdateTime, 
                SecName, 
                AVG(MDBid1Price) as Bid1Price, 
                AVG(MDAsk1Price) as Ask1Price, 
                SUM(MDBid1Size) as Bid1Size, 
                SUM(MDAsk1Size) as Ask1Size,
                AVG((MDBid1Price + MDAsk1Price)/2) as Mid, 
                AVG((MDAsk1Price - MDBid1Price)/2) as Sprd1, 
                COUNT(SecName) as TickCount
            from 
                EMAPIDB.dbo.MDSEMAPI_EquityOrderBook 
            WHERE 
                UpdateTime BETWEEN '{str_date1}' AND '{str_date2}'
                AND MDAsk1Price between 0 AND {MAX_PRICE}
                AND MDBid1Price between 0 AND {MAX_PRICE} 
                AND MDAsk1Price > MDBid1Price
                {secname_clause}
            GROUP BY 
                SecName,
                DATEPART(YEAR, UpdateTime),
                DATEPART(MONTH, UpdateTime),
                DATEPART(DAY, UpdateTime),
                (DATEPART(HOUR, UpdateTime) / {freq_in_hours})
            ORDER BY 
                UpdateTime ASC"""

    conn = odbc_conn.tick_data_connection()
    s = timer()
    df = pd.read_sql(query, conn)
    df["Imbalance1"] = df["Bid1Size"] - df["Ask1Size"]
    df["UpdateTime"] = pd.to_datetime(df["UpdateTime"])
    df["UpdateTime"] = (
        df["UpdateTime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Bangkok")
    ).dt.tz_localize(None)

    e = timer()
    logging.info(
        f"{symbol}: Got level 1 LOB freq={freq_in_hours} hours, from {str_date1} to {str_date2}. Time taken= {e-s} seconds"
    )
    return df


def get_level1_lob_by_freq_fast(
    symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, freq_in_hours: int
):
    """
    Grab Level 1 LOB half-day frequency from provided SQL table
    :param symbol:
    :param start_date:
    :param end_date:
    :param freq_in_hours:
    :return:
    """

    if symbol is None:
        secname_clause = ""
    else:
        secname_clause = f"AND (SecName LIKE '{symbol}')"

    if freq_in_hours == 6:
        table = "EMAPIDB.dbo.DATAPREP_EquityLOBStat"
    elif freq_in_hours == 2:
        table = "EMAPIDB.dbo.DATAPREP_EquityLOBStat_2Hr"
    elif freq_in_hours == 12:
        table = "EMAPIDB.dbo.DATAPREP_EquityLOBStat_12Hr"

    str_date1 = start_date.strftime("%Y-%m-%d")
    str_date2 = end_date.strftime("%Y-%m-%d")

    query = f"""SELECT 
                UpdateTime, 
                SecName, 
                Bid1Price, 
                Ask1Price, 
                Bid1Size, 
                Ask1Size,
                Mid, 
                Sprd1, 
                TickCount
            from 
                {table} 
            WHERE 
                UpdateTime BETWEEN '{str_date1}' AND '{str_date2}'
                {secname_clause}
            ORDER BY 
                UpdateTime ASC"""

    conn = odbc_conn.tick_data_connection()
    s = timer()
    df = pd.read_sql(query, conn)
    df["Imbalance1"] = df["Bid1Size"] - df["Ask1Size"]
    df["UpdateTime"] = pd.to_datetime(df["UpdateTime"])
    df["UpdateTime"] = (
        df["UpdateTime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Bangkok")
    ).dt.tz_localize(None)

    e = timer()
    logging.info(
        f"{symbol}: Got level 1 LOB, freq={freq_in_hours} hours, from {str_date1} to {str_date2}. Time taken= {e-s} seconds"
    )
    return df


def get_trades_and_quotes_daily(
    symbol: str,
    include_auction: bool,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    freq_in_days: float,
):
    if freq_in_days == 0.5:  # half daily, for DW
        freq_in_hours = 6
        offset_hours = 3
    elif freq_in_days == 1:  # daily, for block trading
        freq_in_hours = 12
        offset_hours = 7

    lob = get_level1_lob_by_freq_fast(
        symbol, start_date, end_date, freq_in_hours=freq_in_hours
    )
    trades = get_equity_trades_by_freq(
        symbol, include_auction, start_date, end_date, freq_in_hours=freq_in_hours
    )
    if freq_in_days == 0.5:
        lob["UpdateTime"] = (
            lob["UpdateTime"] - pd.Timedelta(hours=offset_hours)
        ).dt.ceil("1H")
    elif freq_in_days == 1.0:
        lob["UpdateTime"] = lob["UpdateTime"].apply(
            lambda x: pd.Timestamp(x.date()) + pd.Timedelta(hours=10)
        )

    trades["TradeTime"] = trades["TradeTime"] - pd.Timedelta(hours=offset_hours)
    lob_last2h = get_level1_lob_by_freq_fast(
        symbol, start_date, end_date, freq_in_hours=2
    ).drop(columns=["SecName"])
    cols_mapping = {
        col: col + "_last2h" for col in lob_last2h.columns if col != "UpdateTime"
    }
    lob_last2h = lob_last2h.rename(columns=cols_mapping)

    trades_last2h = get_equity_trades_by_freq(
        symbol, include_auction, start_date, end_date, freq_in_hours=2
    ).drop(columns=["SecName", "HourOfDay", "DayOfWeek"])
    cols_mapping = {
        col: col + "_last2h" for col in trades_last2h.columns if col != "TradeTime"
    }
    trades_last2h = trades_last2h.rename(columns=cols_mapping)

    df = pd.merge_asof(
        trades,
        lob.drop(columns=["SecName"]),
        left_on="TradeTime",
        right_on="UpdateTime",
        allow_exact_matches=True,
        direction="backward",
    )
    df = pd.merge_asof(
        df,
        trades_last2h,
        on="TradeTime",
        allow_exact_matches=True,
        direction="backward",
    )

    df = pd.merge_asof(
        df, lob_last2h, on="UpdateTime", allow_exact_matches=True, direction="backward",
    )

    oc = get_open_close(symbol, start_date, end_date, freq_in_days=1)
    df = pd.merge_asof(
        df, oc, on="UpdateTime", allow_exact_matches=True, direction="backward"
    )
    return df


def get_regular_interval_traded_volume(
    symbol: str,
    freq: str,
    include_auction: bool,
    split_buy_sell: bool,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    """

    :param symbol:          ticker for stock
    :param freq:            "15T" for 15-min
    :param include_auction: True or False
    :param split_buy_sell:  True if we want buy volume in one row and sell volume in another for the same timestamp
    :param start_date:
    :param end_date:
    :return:
    """
    df = get_equity_trades(start_date, end_date, symbol)
    if not include_auction:
        df = df.loc[df["MatchType"] != 7].reset_index(drop=True)

    df["Direction"] = df["BidAggressor"].replace({True: 1, False: -1})
    df["SignedVolume"] = df["Direction"] * df["Volume"]
    df["TradeTime"] = pd.to_datetime(df["TradeTime"])
    df["TradeTime"] = (
        df["TradeTime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Bangkok")
    ).dt.tz_localize(None)
    if split_buy_sell:
        df1 = (
            df[["LastPrice", "Volume", "SignedVolume", "TradeTime", "Direction"]]
            .groupby([pd.Grouper(key="TradeTime", freq=freq), "Direction"])
            .agg({"LastPrice": "mean", "Volume": sum, "SignedVolume": "count"})
            .reset_index(level=1)
        ).rename(columns={"SignedVolume": "TradeCount"})
        df2 = (
            df[["SignedVolume", "TradeTime"]]
            .groupby([pd.Grouper(key="TradeTime", freq=freq)])
            .agg({"SignedVolume": sum})
        )
        df1 = df2.join(df1).reset_index()
        df1.loc[pd.isnull(df1["Direction"]), "Volume"] = 0
    else:
        df1 = (
            (
                df[["LastPrice", "Volume", "SignedVolume", "TradeTime", "Direction"]]
                .groupby([pd.Grouper(key="TradeTime", freq=freq)])
                .agg(
                    {
                        "LastPrice": "mean",
                        "Volume": sum,
                        "SignedVolume": sum,
                        "Direction": "count",
                    }
                )
            )
            .rename(columns={"Direction": "TradeCount"})
            .reset_index()
        )

    df1["HourOfDay"] = (
        (df1["TradeTime"] - pd.to_datetime(df1["TradeTime"].dt.date))
        / np.timedelta64(1, "h")
    ).astype(float)
    df1["DayOfWeek"] = df1["TradeTime"].dt.dayofweek
    return df1


def get_equity_lob(
    date1, date2, symbol=None, bkk_time: bool = False, level1_only: bool = False
) -> pd.DataFrame:
    conn = odbc_conn.tick_data_connection()
    if bkk_time:
        date1 = date1.tz_localize("Asia/Bangkok").tz_convert("UTC")
        date2 = date2.tz_localize("Asia/Bangkok").tz_convert("UTC")

    str_date1 = date1.strftime("%Y-%m-%d %H:%M:%S")
    str_date2 = date2.strftime("%Y-%m-%d %H:%M:%S")

    if symbol is None:
        # Grab All
        symbol_clause = ""
    else:
        symbol_clause = f"AND (SecName LIKE '{symbol}')"

    if level1_only:
        field_clause = f""" UpdateTime, SeqNo, SecName, 
                            MDBid1Price, MDAsk1Price, MDBid1Size, MDAsk1Size """
    else:
        field_clause = f""" UpdateTime, SourceTime, ReceivingTime,
                SeqNo, SecCode, SecName, 
                MDBid1Price, MDBid1Size, MDBid2Price, MDBid2Size,
                MDBid3Price, MDBid3Size, MDBid4Price, MDBid4Size,
                MDBid5Price, MDBid5Size, MDAsk1Price, MDAsk1Size,
                MDAsk2Price, MDAsk2Size, MDAsk3Price, MDAsk3Size,
                MDAsk4Price, MDAsk4Size, MDAsk5Price, MDAsk5Size """

    query = f"""SELECT
                {field_clause}
            FROM 
                EMAPIDB.dbo.MDSEMAPI_EquityOrderBook
            WHERE 
                UpdateTime BETWEEN '{str_date1}' AND '{str_date2}' {symbol_clause}
                AND MDAsk1Price between 0 AND {MAX_PRICE}
                AND MDBid1Price between 0 AND {MAX_PRICE} 
            Order BY 
            UpdateTime ASC"""
    df = pd.read_sql(query, conn)
    df = df.dropna(subset=["MDBid1Price", "MDAsk1Price"])

    if bkk_time:
        df["UpdateTime"] = pd.to_datetime(df["UpdateTime"])
        df["UpdateTime"] = (
            df["UpdateTime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Bangkok")
        ).dt.tz_localize(None)

    return df


def get_open_close_from_lob(symbol: str, dt: pd.Timestamp, open_or_close: str):
    """
    Grab Level 1 Open/Close order book data
    :param symbol:
    :param dt:
    :param open_or_close: "Open" or "Close"
    :return:
    """

    if open_or_close == "Open":
        start_dt = (
            (pd.Timestamp(dt.date()) + pd.Timedelta(hours=10, minutes=5))
            .tz_localize("Asia/Bangkok")
            .tz_convert("UTC")
        )
        end_dt = (
            (pd.Timestamp(dt.date()) + pd.Timedelta(hours=10, minutes=35))
            .tz_localize("Asia/Bangkok")
            .tz_convert("UTC")
        )
    else:
        start_dt = (
            (pd.Timestamp(dt.date()) + pd.Timedelta(hours=16))
            .tz_localize("Asia/Bangkok")
            .tz_convert("UTC")
        )
        end_dt = (
            (pd.Timestamp(dt.date()) + pd.Timedelta(hours=16, minutes=30))
            .tz_localize("Asia/Bangkok")
            .tz_convert("UTC")
        )

    str_date1 = start_dt.strftime("%Y-%m-%d %H:%M")
    str_date2 = end_dt.strftime("%Y-%m-%d %H:%M")

    query = f"""SELECT 
                MAX(UpdateTime) as UpdateTime, 
                SecName, 
                AVG(MDBid1Price) as {open_or_close}Bid, 
                AVG(MDAsk1Price) as {open_or_close}Ask, 
                AVG((MDBid1Price + MDAsk1Price)/2) as {open_or_close}Mid 
            from 
                EMAPIDB.dbo.MDSEMAPI_EquityOrderBook 
            WHERE 
                UpdateTime BETWEEN '{str_date1}' AND '{str_date2}'
                AND MDAsk1Price between 0 AND {MAX_PRICE}
                AND MDBid1Price between 0 AND {MAX_PRICE} 
                AND MDAsk1Price > MDBid1Price
                AND SecName LIKE '{symbol}'
            GROUP BY 
                SecName,
                DATEPART(YEAR, UpdateTime),
                DATEPART(MONTH, UpdateTime),
                DATEPART(DAY, UpdateTime)
            ORDER BY 
                UpdateTime ASC"""

    conn = odbc_conn.tick_data_connection()
    s = timer()
    df = pd.read_sql(query, conn)
    df["UpdateTime"] = pd.to_datetime(df["UpdateTime"])
    df["UpdateTime"] = (
        df["UpdateTime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Bangkok")
    ).dt.tz_localize(None)

    e = timer()
    logging.info(
        f"{symbol}: Got {open_or_close} from UTC {str_date1} to {str_date2}. Time taken= {e-s} seconds"
    )
    return df


def get_regular_interval_equity_level1_lob(
    symbol: str, freq: str, date1: pd.Timestamp, date2: pd.Timestamp
):

    conn = odbc_conn.tick_data_connection()
    str_date1 = date1.strftime("%Y-%m-%d")
    str_date2 = date2.strftime("%Y-%m-%d")

    if symbol is None:
        # Grab All
        symbol_clause = ""
    else:
        symbol_clause = f"AND (SecName LIKE '{symbol}')"

    query = f"""SELECT UpdateTime, SecName, Mid = (MDAsk1Price + MDBid1Price)/2,
                MDBid1Price, MDAsk1Price, MDBid1Size, MDAsk1Size
            FROM 
                EMAPIDB.dbo.MDSEMAPI_EquityOrderBook
            WHERE 
                UpdateTime BETWEEN '{str_date1}' AND '{str_date2}' {symbol_clause}
                AND MDAsk1Price between 0 AND {MAX_PRICE}
                AND MDBid1Price between 0 AND {MAX_PRICE} 
            Order BY 
            UpdateTime ASC"""

    df = pd.read_sql(query, conn)
    df = df.drop_duplicates(subset=["UpdateTime"]).dropna(subset=["MDAsk1Price"])
    df = df.dropna(subset=["MDBid1Price"]).reset_index(drop=True)
    df["UpdateTime"] = pd.to_datetime(df["UpdateTime"])
    df["UpdateTime"] = (
        df["UpdateTime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Bangkok")
    ).dt.tz_localize(None)
    df1 = (
        df[
            [
                "UpdateTime",
                "Mid",
                "MDBid1Price",
                "MDAsk1Price",
                "MDBid1Size",
                "MDAsk1Size",
            ]
        ]
        .groupby([pd.Grouper(key="UpdateTime", freq=freq)])
        .mean()
    )
    df2 = (
        df[["UpdateTime", "Mid"]]
        .groupby([pd.Grouper(key="UpdateTime", freq=freq)])
        .agg(TickCount=("Mid", "count"))
    )

    return df1.dropna(subset=["Mid"]).join(df2).reset_index()


def get_regular_interval_equity_lob(
    symbol: str, freq: str, date1: pd.Timestamp, date2: pd.Timestamp
):

    conn = odbc_conn.tick_data_connection()
    str_date1 = date1.strftime("%Y-%m-%d %H:%M:%S")
    str_date2 = date2.strftime("%Y-%m-%d %H:%M:%S")

    if symbol is None:
        # Grab All
        symbol_clause = ""
    else:
        symbol_clause = f"AND (SecName LIKE '{symbol}')"

    query = f"""SELECT UpdateTime, SecName, 
                MDBid1Size, MDAsk1Size,
                Mid = (MDAsk1Price + MDBid1Price)/2, 
                Sprd1 = (MDAsk1Price - MDBid1Price)/2, 
                Sprd2 = (MDAsk2Price - MDBid2Price)/2, 
                Sprd3 = (MDAsk3Price - MDBid3Price)/2, 
                Sprd4 = (MDAsk4Price - MDBid4Price)/2,
                Sprd5 = (MDAsk5Price - MDBid5Price)/2,
                Imbalance = (MDBid1Size + MDBid2Size + MDBid3Size + MDBid4Size + MDBid5Size 
                            - MDAsk1Size - MDAsk2Size - MDAsk3Size - MDAsk4Size - MDAsk5Size),
                Imbalance1 = (MDBid1Size - MDAsk1Size), 
                Imbalance2 = (MDBid2Size - MDAsk2Size),
                Imbalance3 = (MDBid3Size - MDAsk3Size), 
                Imbalance4 = (MDBid4Size - MDAsk4Size),
                Imbalance5 = (MDBid5Size - MDAsk5Size)
            FROM 
                EMAPIDB.dbo.MDSEMAPI_EquityOrderBook
            WHERE 
                UpdateTime BETWEEN '{str_date1}' AND '{str_date2}' {symbol_clause}
                AND MDAsk1Price between 0 AND {MAX_PRICE}
                AND MDAsk2Price between 0 AND {MAX_PRICE}
                AND MDAsk3Price between 0 AND {MAX_PRICE}
                AND MDAsk4Price between 0 AND {MAX_PRICE}
                AND MDAsk5Price between 0 AND {MAX_PRICE}
                AND MDBid1Price between 0 AND {MAX_PRICE} 
                AND MDBid2Price between 0 AND {MAX_PRICE} 
                AND MDBid3Price between 0 AND {MAX_PRICE} 
                AND MDBid4Price between 0 AND {MAX_PRICE} 
                AND MDBid5Price between 0 AND {MAX_PRICE}
            Order BY 
            UpdateTime ASC"""

    df = pd.read_sql(query, conn)
    df = df.loc[df["Sprd1"] > 0]
    df = df.drop_duplicates(subset=["UpdateTime"]).dropna(subset=["Mid"])
    df = df.loc[
        (df["Sprd1"] > 0)
        & (df["Sprd2"] > 0)
        & (df["Sprd3"] > 0)
        & (df["Sprd4"] > 0)
        & (df["Sprd5"] > 0)
    ].reset_index()
    df["UpdateTime"] = pd.to_datetime(df["UpdateTime"])
    df["UpdateTime"] = (
        df["UpdateTime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Bangkok")
    ).dt.tz_localize(None)
    df1 = (
        df[
            [
                "UpdateTime",
                "Mid",
                "MDBid1Size",
                "MDAsk1Size",
                "Sprd1",
                "Sprd2",
                "Sprd3",
                "Sprd4",
                "Sprd5",
                "Imbalance",
                "Imbalance1",
                "Imbalance2",
                "Imbalance3",
                "Imbalance4",
                "Imbalance5",
            ]
        ]
        .groupby([pd.Grouper(key="UpdateTime", freq=freq)])
        .mean()
    )
    return df1.dropna(subset=["Mid"])


def get_regular_interval_trades_and_quotes(
    symbol: str,
    trade_freq: str,
    lob_freq: str,
    include_auction: bool,
    split_buy_sell: bool,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
):
    lob = get_regular_interval_equity_lob(symbol, lob_freq, start_date, end_date)
    trades = get_regular_interval_traded_volume(
        symbol, trade_freq, include_auction, split_buy_sell, start_date, end_date
    )
    return pd.merge_asof(
        trades,
        lob,
        left_on="TradeTime",
        right_on="UpdateTime",
        allow_exact_matches=True,
        direction="backward",
    )


def get_dw_descriptions():
    conn = odbc_conn.setsmart_connection()
    query = f""" SELECT B.N_SECURITY As Instrument,
        CASE WHEN C.N_SECURITY is null THEN E.N_UNDERLYING
        WHEN E.N_UNDERLYING is null THEN C.N_SECURITY
        ELSE null END As UL,
        LEFT(Right(RTrim(B.N_SECURITY),8),2) As IssueBroker,
        D.Z_EXERCISE As Strike,
        D.D_FIRST_TRADE As FirstTrade,
        D.D_LAST_TRADE As LastTrade,
        D.I_TYPE_OPTION As PutOrCall,
        D.Q_FIRST_RATIO As WarrantRatio,
        D.Q_LAST_RATIO As UlRatio,
        --D.Q_FIRST_RATIO/Q_LAST_RATIO As ExRatio,
        --D.Z_MULTIPLIER As S50ExRatio,
        CASE WHEN D.Q_FIRST_RATIO is null THEN D.Z_MULTIPLIER
        WHEN D.Z_MULTIPLIER is null THEN D.Q_FIRST_RATIO/Q_LAST_RATIO
        END As Ratio,
        Q_SHARE_ISSUED As ISSUED,
        DATENAME(W,D.D_FIRST_TRADE) AS DAYNAME,
        D.N_MKT_MAKER_E
        --,D.N_MKT_MAKER_T
        --,*
          FROM [SSETDI].[dbo].[UNDERLYING] A
            JOIN [dbo].[SECURITY] B on A.I_SECURITY=B.I_SECURITY
           LEFT JOIN [dbo].[SECURITY] C on A.I_UNDERLYING=
           ( CASE WHEN A.I_TYPE_UNDERLYING='S' THEN C.I_SECURITY
           ELSE Null END )
          JOIN [dbo].[SECURITY_DETAIL] D on  A.I_SECURITY = D.I_SECURITY
          LEFT JOIN [dbo].[UNDERLYING_ASSET] E on A.I_UNDERLYING=
           ( CASE WHEN A.I_TYPE_UNDERLYING='I' THEN E.I_UNDERLYING
            WHEN A.I_TYPE_UNDERLYING='O' THEN E.I_UNDERLYING
             ELSE Null END )
           Where B.I_SEC_TYPE = 'V' and D.D_LAST_TRADE>= GETDATE()
           --and B.N_SECURITY like '%BABA%'
           Order By B.N_SECURITY """

    df = pd.read_sql(query, conn)
    df["UL"] = df["UL"].apply(lambda x: x.split(" ")[0])
    df["Instrument"] = df["Instrument"].apply(lambda x: x.split(" ")[0])
    return df


def get_ULs_for_DWs():
    df = get_dw_descriptions()
    return df["UL"].unique()


# "BABA28C2209A" dw
# df = get_equity_trades(pd.Timestamp('2021-01-04'), pd.Timestamp('2022-07-10'), _symbol="CPALL")
# lob = get_equity_lob(pd.Timestamp('2022-01-04 9:00'), pd.Timestamp('2022-01-10'), _symbol="CPALL")
