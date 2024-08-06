import os.path
import sys
import numpy as np
import pandas as pd
from typing import List
from .data_queries import get_equity_lob
from .data_queries import get_dw_price_guideline
# from ml_trading_strategies.common.data_utils import get_tick_sizes
import logging
from collections import defaultdict


logger = logging.getLogger("__main__."+__name__)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

from GQDatabaseSQL import REDIS_DB,SETSMART,BDSupport
from TFEX_Utils import TFEX_Utils

class TickRule(object):
    def __init__(self, tick_rule):
        self.tick_rule = tick_rule

    def get_up_tick(self, x):
        if self.tick_rule == 0:
            if x >= 400:
                return 2
            elif 200 <= x and x < 400:
                return 1
            elif 100 <= x and x < 200:
                return 0.5
            elif 25 <= x and x < 100:
                return 0.25
            elif 10 <= x and x < 25:
                return 0.1
            elif 5 <= x and x < 10:
                return 0.05
            elif 2 <= x and x < 5:
                return 0.02
            elif 0 <= x and x < 2:
                return 0.01
            else:
                return np.nan
        elif self.tick_rule == 1:
            return 0.1
        elif self.tick_rule == 2:
            return 10
        elif self.tick_rule == 3:
            return 0.01

    def get_dw_tick(self, x):
        if self.tick_rule == 0:
            if x > 400:
                return 2
            elif 200 < x and x <= 400:
                return 1
            elif 100 < x and x <= 200:
                return 0.5
            elif 25 < x and x <= 100:
                return 0.25
            elif 10 < x and x <= 25:
                return 0.1
            elif 5 < x and x <= 10:
                return 0.05
            elif 2 < x and x <= 5:
                return 0.02
            elif 0 <= x and x <= 2:
                return 0.01
            else:
                return np.nan
        elif self.tick_rule == 1:
            return 0.1
        elif self.tick_rule == 2:
            return 10
        elif self.tick_rule == 3:
            return 0.01

    def get_n_ticks(self,target,current):
        diff = target - current

        if diff >0:
            tick = self.get_up_tick(current)
        else:
            tick = self.get_dw_tick(current)

        return diff/tick

SETeq_TickRule = TickRule(tick_rule=0)


# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[logging.FileHandler("../debug.log"), logging.StreamHandler(sys.stdout)],
# )


# base_folder = "C:\\Users\\sachapon.t\\Documents"

# base_folder = "X:\\Users\\Aam\\production0"
# prediction_path = base_folder+"\\daily_price_prediction\\predictions_2ticks"
prediction_path = "W:\\Users\\Aam\\dw_hitter\\production\\predictions"
signal_cache_path = "C:\\dw_signal_local_cache"

holiday_class = TFEX_Utils.SETSMART_Holidays()

def get_prev_trade_date(dt):
    today = dt - pd.Timedelta(1,'D')
    while not TFEX_Utils.isDateTradingDay(today,holiday_class.get_holidays()):
        # print(f"date:{today}")
        today = today - pd.Timedelta(1,'D')
    # pd.tseries.offsets.BusinessDay(n=1)
    # print(f"Result:{today}")
    return today

def get_next_trading_date(dt):
    today = dt + pd.Timedelta(1,'D')
    while not TFEX_Utils.isDateTradingDay(today,holiday_class.get_holidays()):
        # print(f"date:{today}")
        today = today + pd.Timedelta(1,'D')
    # pd.tseries.offsets.BusinessDay(n=1)
    # print(f"Result:{today}")
    return today



class DwBackTest:
    """
    Simulate trading of DW for a given UL and date
    :param symbol:          UL symbol
    :param trade_date:      trade date
    :param put_call:        'P' for put 'C' for call
    :param sampling_freq    frequency of DW order book sampling
    :return:
    """

    def __init__(
            self,
            symbol: str,
            trade_date: pd.Timestamp,
            RTD_Class,
            SETSMART_DW_SPEC,
            put_call: str = "C",
            tick_size_multiple: int = 2,
            start_minute: int = 2,
            stop_loss_ticks: int = 2,
            use_prev_day= False,
            outlook_email = None

    ):
        logger.info(f"Init DWBacktest: symbol:{symbol} trade_date:{trade_date} PC:{put_call} tick_size_multiple:{tick_size_multiple} "
                    f"start_minute:{start_minute}")
        self._symbol = symbol
        self._trade_date = trade_date
        # self._signals_date_assumption = get_next_trading_date(trade_date)
        self._signals_date_assumption = trade_date
        self._put_call = put_call
        self._tick_size_multiple = tick_size_multiple
        self._start_minute = start_minute
        self._signals = None
        self._RTD_class = RTD_Class
        self.RTD_Instrument = None
        self._SETSMART_DW_SPEC = SETSMART_DW_SPEC
        self._stop_loss_ticks = stop_loss_ticks
        self._stop_profit_ticks = stop_loss_ticks

        self._valid_issuer_list = ["01","19","13","11"]
        # self._valid_issuer_list = ["19","01"]
        self._price_limit = 0.05
        #Too lazy to redefine this variable dict
        self.n_ticks_limit_dict = {}
        self.n_ticks_limit_dict["KCE"] = 3
        self.n_ticks_limit_dict["SCB"] = 3
        self.n_ticks_limit_dict["SCC"] = 3
        self.n_ticks_limit_dict["SAWAD"] = 3
        self.n_ticks_limit_dict["BBL"] = 3
        self.n_ticks_limit_dict["KBANK"] = 3
        self.n_ticks_limit_dict["PTTEP"] = 3
        self.n_ticks_limit_dict["TOP"] = 3
        self.n_ticks_limit_dict["JMART"] = 3
        self.n_ticks_limit_dict["JMT"] = 3
        self.n_ticks_limit_dict["TRUE"] = 3
        self.n_ticks_limit_dict["THANI"] = 3
        self.n_ticks_limit_dict["BAM"] = 3
        self.n_ticks_limit_dict["CPALL"] = 3
        self.n_ticks_limit_dict["CPN"] = 3

        # self.outlook_email = outlook_email

        self.signal_columns =['signal_datetime', 'signal', 'action', 'ul_mid_time', 'ul_mid_price',
                                  'predicted_ul_mid_price', 'current_dw_time', 'current_dw_ask_price',
                                  'dw_price_guideline_given_ul_mid_price',
                                  'dw_price_guideline_given_predicted_ul_price', 'ul_stoploss_price',
                                  'dw_stoploss_price', 'probability', 'dw_symbol']

        self.prev_trading_date = get_prev_trade_date(self._trade_date)
        if use_prev_day:
            self.price_guideline_date = get_prev_trade_date(self._trade_date)
            self.col = "t1"
        else:
            self.price_guideline_date = self._trade_date
            self.col = "t0"
            self.col_tp1 = "t1"
        if not self._RTD_class is None:

            try:
                self.init_RTD_feed(product_type='eq',symbol=self._symbol,sub_type="orderbook",init_from_beginning=False)
            except Exception as e:
                # print(e)
                if str(e).startswith("LOB Stream Class Exist"):
                    self.RTD_Instrument: REDIS_DB.instrument_class = self._RTD_class.get_symbol(self._symbol)
                    return
                else:
                    raise e

    def check_price_guideline(self):

        pg_df = BDSupport.get_pricing_guideline(starting_date=self.price_guideline_date,ending_date=self.price_guideline_date+pd.Timedelta(hours=20))

        if pg_df.shape[0] > 0:
            pass
        # else:
        #     self.outlook_email.send_email(subject=f"DW Price Guideling for {self.price_guideline_date} missing",
        #                              message=f"DW Price Guideling for {self.price_guideline_date} missing",
        #                              cc=''
        #                                 '')


    def get_dw_symbols(self) -> List:
        # df = get_dw_price_guideline(self._symbol, self._trade_date - pd.tseries.offsets.BusinessDay(n=1) )

        df = get_dw_price_guideline(self._symbol, self.price_guideline_date )
        '''we trading at the end of day, no need to get previous price guideline '''
        # df = get_dw_price_guideline(self._symbol, self._trade_date )
        df = df.loc[~df['symbol'].str.contains(self._symbol + '24')]
        put_call_index = len(self._symbol) + 2
        df = df.loc[df.symbol.str[put_call_index] == self._put_call]
        return df["symbol"].unique()

    # def get_signals(self,trade_date):
    #     # fn = f"{base_folder}\\production0\\daily_price_prediction\\predictions_1tick\\{self._trade_date.date()}\\{self._symbol}.csv"
    #     fn = f"{base_folder}\\production0\\daily_price_prediction\\predictions_1tick\\{trade_date.date()}\\{self._symbol}.csv"
    #     df = pd.read_csv(fn, parse_dates=["datetime"])
    #     if self._put_call == "C":
    #         df = df.loc[df["predicted_class"] > 0].reset_index(drop=True)
    #     else:
    #         df = df.loc[df["predicted_class"] < 0].reset_index(drop=True)
    #
    #     dw_syms = self.get_dw_symbols()
    #     results = []
    #
    #     for dw_sym in dw_syms:
    #         date_times = df["datetime"]
    #         signals = df["predicted_class"]
    #         actions = np.zeros(len(signals))
    #         ul_mid_prices = np.zeros(len(signals))
    #         predicted_ul_mid_prices = np.zeros(len(signals))
    #         current_dw_ask_prices = np.zeros(len(signals))
    #         dw_prices_from_guideline_corresp_to_ul_mid_prices = np.zeros(len(signals))
    #         dw_prices_from_guideline_corresp_to_predicted_ul_prices = np.zeros(
    #             len(signals)
    #         )
    #         ul_stoploss_price = np.zeros(len(signals))
    #         ul_mid_price_times = [pd.Timestamp(0) for i in range(len(signals))]
    #         current_dw_ask_price_times = [pd.Timestamp(0) for i in range(len(signals))]
    #
    #         for ii, signal_dt in zip(range(len(date_times)), date_times):
    #
    #             # df_dw_price_guideline = get_dw_price_guideline(dw_sym, signal_dt - pd.tseries.offsets.BusinessDay(n=1))
    #             prev_date = get_prev_trade_date(signal_dt)
    #             df_dw_price_guideline = get_dw_price_guideline(dw_sym,prev_date)
    #
    #             if not df_dw_price_guideline.empty:
    #                 ul_dt = pd.Timestamp(signal_dt.date()) + pd.Timedelta(hours=10)
    #                 ul_dt_prev = pd.Timestamp(prev_date.date()) + pd.Timedelta(hours=10)
    #                 #Get LOB from 10am on previous trading day (previous from signal date)
    #                 #to 10+ start minute(2 in this case)
    #                 df_ul = get_equity_lob(
    #                     date1=(ul_dt - pd.tseries.offsets.BusinessDay(n=1)),
    #                     date2=(ul_dt + pd.Timedelta(minutes=self._start_minute)),
    #                     symbol=self._symbol,
    #                     bkk_time=True,
    #                     level1_only=True,
    #                 )
    #                 df_ul["Mid"] = (df_ul["MDBid1Price"] + df_ul["MDAsk1Price"]) / 2.0
    #                 # ul_mid_price = df_ul["Mid"].values[-1]
    #                 ul_mid_price = df_ul["MDAsk1Price"].values[-1]
    #                 ul_mid_price_time = df_ul["UpdateTime"].values[-1]
    #
    #                 if self._put_call == "C":
    #                     predicted_ul_price = (
    #                             ul_mid_price
    #                             + self._tick_size_multiple
    #                             * SETeq_TickRule.get_up_tick(ul_mid_price) #.values[-1]
    #                     )
    #                 else:
    #                     predicted_ul_price = (
    #                             ul_mid_price
    #                             - self._tick_size_multiple
    #                             * SETeq_TickRule.get_dw_tick(ul_mid_price) #.values[-1]
    #                     )
    #                 diffs = df_dw_price_guideline["spot"] - ul_mid_price
    #                 idx = np.argmin(abs(diffs))
    #                 col = "t0"
    #                 col = "t1"
    #                 dw_price_from_guideline_corresp_to_ul_mid_price = df_dw_price_guideline.loc[
    #                     idx, col
    #                 ]
    #
    #                 diffs = df_dw_price_guideline["spot"] - predicted_ul_price
    #                 idx = np.argmin(abs(diffs))
    #                 dw_price_from_guideline_corresp_to_predicted_ul_price = df_dw_price_guideline.loc[
    #                     idx, col
    #                 ]
    #
    #                 df_dw = get_equity_lob(
    #                     date1=(ul_dt - pd.tseries.offsets.BusinessDay(n=1)),
    #                     date2=(ul_dt + pd.Timedelta(minutes=self._start_minute)),
    #                     symbol=dw_sym,
    #                     bkk_time=True,
    #                     level1_only=True,
    #                 )
    #                 if not df_dw.empty:
    #                     # print(df_dw[['UpdateTime', 'MDBid1Price', 'MDAsk1Price']])
    #                     current_dw_price = float(
    #                         df_dw.loc[df_dw.shape[0] - 1, "MDAsk1Price"]
    #                     )
    #
    #                     stop_loss_price = current_dw_price - 2 * SETeq_TickRule.get_dw_tick(current_dw_price)
    #                     where_same = df_dw_price_guideline[col] == stop_loss_price
    #                     if self._put_call == "C":
    #                         stop_loss_ul_price = df_dw_price_guideline['spot'][where_same].max()
    #                     else:
    #                         stop_loss_ul_price = df_dw_price_guideline['spot'][where_same].min()
    #
    #
    #                     current_dw_time = df_dw.loc[df_dw.shape[0] - 1, "UpdateTime"]
    #                     logging.info(
    #                         f"UL price at {ul_mid_price_time} = {ul_mid_price}"
    #                     )
    #                     logging.info(
    #                         f"curent {dw_sym} ask price {current_dw_price} at {current_dw_time}"
    #                     )
    #                     logging.info(
    #                         f"{dw_sym} dw_price_from_guideline_corresp_to_ul_mid_price {dw_price_from_guideline_corresp_to_ul_mid_price}"
    #                     )
    #                     logging.info(
    #                         f"{dw_sym} dw_price_from_guideline_corresp_to_predicted_ul_price {dw_price_from_guideline_corresp_to_predicted_ul_price}"
    #                     )
    #
    #                     if (
    #                             np.sign(
    #                                 dw_price_from_guideline_corresp_to_ul_mid_price
    #                                 - dw_price_from_guideline_corresp_to_predicted_ul_price
    #                             )
    #                             == -1
    #                     ) & (
    #                             np.sign(
    #                                 current_dw_price
    #                                 - dw_price_from_guideline_corresp_to_predicted_ul_price
    #                             )
    #                             == -1
    #                     ):
    #                         actions[ii] = 1
    #                         ul_mid_price_times[ii] = ul_mid_price_time
    #                         ul_mid_prices[ii] = ul_mid_price
    #                         predicted_ul_mid_prices[ii] = predicted_ul_price
    #                         current_dw_ask_price_times[ii] = current_dw_time
    #                         current_dw_ask_prices[ii] = current_dw_price
    #                         dw_prices_from_guideline_corresp_to_ul_mid_prices[
    #                             ii
    #                         ] = dw_price_from_guideline_corresp_to_ul_mid_price
    #                         dw_prices_from_guideline_corresp_to_predicted_ul_prices[
    #                             ii
    #                         ] = dw_price_from_guideline_corresp_to_predicted_ul_price
    #                         ul_stoploss_price[ii] = stop_loss_ul_price
    #                 else:
    #                     logging.info(
    #                         f"{dw_sym}: NO LOB {ul_dt - pd.tseries.offsets.BusinessDay(n=3)} - {ul_dt}"
    #                     )
    #             res = pd.DataFrame(
    #                 {
    #                     "signal_datetime": date_times,
    #                     "signal": signals,
    #                     "action": actions,
    #                     "ul_mid_time": ul_mid_price_times,
    #                     "ul_mid_price": ul_mid_prices,
    #                     "predicted_ul_mid_price": predicted_ul_mid_prices,
    #                     "current_dw_time": current_dw_ask_price_times,
    #                     "current_dw_ask_price": current_dw_ask_prices, #this is limit buy price
    #                     "dw_price_guideline_given_ul_mid_price": dw_prices_from_guideline_corresp_to_ul_mid_prices,
    #                     "dw_price_guideline_given_predicted_ul_price": dw_prices_from_guideline_corresp_to_predicted_ul_prices, #thiis take profit price
    #                 }
    #             )
    #
    #             res["dw_symbol"] = dw_sym
    #             res = res.loc[res["action"] == 1].reset_index(drop=True)
    #             # print(res)
    #             results.append(res)
    #     self._signals = pd.concat(results).drop_duplicates().reset_index(drop=True)
    #     return self._signals


    # def get_signals_intraday(self):
    #     fn = f"{base_folder}\\production0\\daily_price_prediction\\predictions_1tick\\{self._trade_date.date()}\\{self._symbol}.csv"
    #     # fn = f"{base_folder}\\production0\\daily_price_prediction\\predictions_1tick\\{trade_date.date()}\\{self._symbol}.csv"
    #     df = pd.read_csv(fn, parse_dates=["datetime"])
    #     if self._put_call == "C":
    #         df = df.loc[df["predicted_class"] > 0].reset_index(drop=True)
    #     else:
    #         df = df.loc[df["predicted_class"] < 0].reset_index(drop=True)
    #
    #
    #     logger.debug(f"for {self._put_call} predicted class shape {df.shape[0]}")
    #
    #     dw_syms = self.get_dw_symbols()
    #     results = []
    #
    #     logger.debug(f"list of dw for {self._symbol} => {dw_syms} isEmpty:{ df.empty}")
    #
    #     if not df.empty:
    #         for dw_sym in dw_syms:
    #
    #             logger.debug(f"DW sym: {dw_sym}")
    #
    #             if not dw_sym in self._SETSMART_DW_SPEC.index:
    #                 logger.debug(f"dw:{dw_sym} not in DW SPEC in SETSMART skipping")
    #                 continue
    #             # else:
    #
    #             if self._SETSMART_DW_SPEC.loc[dw_sym, "IssueBroker"] in self._valid_issuer_list:
    #                     print(f"{dw_sym} valid issuer")
    #                     logger.debug(f"{dw_sym} valid issuer")
    #             else:
    #                 logger.debug(f"{dw_sym} not a valid issuer")
    #                 continue
    #
    #
    #             logger.info(f"start:{dw_sym}")
    #             date_times = df["datetime"]
    #             signals = df["predicted_class"]
    #             probability = df.at[0,str(signals.values[0])]
    #             actions = np.zeros(len(signals))
    #             ul_mid_prices = np.zeros(len(signals))
    #             predicted_ul_mid_prices = np.zeros(len(signals))
    #             current_dw_ask_prices = np.zeros(len(signals))
    #             dw_prices_from_guideline_corresp_to_ul_mid_prices = np.zeros(len(signals))
    #             dw_prices_from_guideline_corresp_to_predicted_ul_prices = np.zeros(
    #                 len(signals)
    #             )
    #             ul_stoploss_price = np.zeros(len(signals))
    #             dw_stoploss_price = np.zeros(len(signals))
    #
    #             ul_mid_price_times = [pd.Timestamp(0) for i in range(len(signals))]
    #             current_dw_ask_price_times = [pd.Timestamp(0) for i in range(len(signals))]
    #
    #             logger.debug(f"list of dt {date_times}")
    #             for ii, signal_dt in zip(range(len(date_times)), date_times):
    #
    #                 # get_prev_trade_date(signal_dt)
    #                 # prev_date = get_prev_trade_date(signal_dt)
    #                 df_dw_price_guideline = get_dw_price_guideline(dw_sym, self.price_guideline_date)
    #                 # df_dw_price_guideline = get_dw_price_guideline(dw_sym, signal_dt)
    #                 df_dw_price_guideline = df_dw_price_guideline.sort_values('spot')
    #                 if not df_dw_price_guideline.empty:
    #                     # ul_dt = pd.Timestamp(signal_dt.date()) + pd.Timedelta(hours=10)
    #
    #                     try:
    #                         df_ul = self.get_ul_LOB()
    #
    #                     except IndexError as e:
    #                         logger.debug(f"{dw_sym} get UL LOB is empty")
    #                         continue
    #                     except Exception as e:
    #                         print(e)
    #                         raise e
    #
    #                     if len(df_ul) == 0:
    #                         logger.debug(f"{dw_sym} get UL LOB is empty")
    #                         continue
    #
    #                     ohlcv =SETSMART.get_date_OHLCV(datetime_input=self.prev_trading_date,symbols=self._symbol)
    #
    #                     prev_close_bid = ohlcv.at[0, 'Z_LAST_BID']
    #                     prev_close_ask = ohlcv.at[0, 'Z_LAST_OFFER']
    #                     logger.info(f"UL LOB: {df_ul} b1:{df_ul[REDIS_DB.LOB_flat_coldict['b1']] } a1:{ df_ul[REDIS_DB.LOB_flat_coldict['a1']]}")
    #                     # current_mid_price = (df_ul[REDIS_DB.LOB_flat_coldict['b1']] + df_ul[REDIS_DB.LOB_flat_coldict['a1']]) / 2.0
    #
    #                     if self._put_call == "C":
    #                         # pass
    #                         ul_mid_price = prev_close_ask
    #                         current_p = df_ul[REDIS_DB.LOB_flat_coldict['a1']]
    #
    #                         n_tick_from_prev = SETeq_TickRule.get_n_ticks(current_p,ul_mid_price)
    #
    #                     else:
    #                         # pass
    #                         ul_mid_price = prev_close_bid
    #                         current_p = df_ul[REDIS_DB.LOB_flat_coldict['b1']]
    #                         n_tick_from_prev = SETeq_TickRule.get_n_ticks(current_p,ul_mid_price)
    #
    #                     # ul_mid_price = (df_ul[REDIS_DB.LOB_flat_coldict['a1']])
    #                     # ul_mid_price = df_ul["Mid"].values[-1]
    #                     # ul_mid_price = df_ul["MDAsk1Price"].values[-1]
    #                     logger.info(f"UL Price Previous close {ul_mid_price} current_p:{current_p}")
    #
    #                     ul_mid_price_time = pd.Timestamp(df_ul[REDIS_DB.LOB_flat_coldict['tss']],unit='ms')
    #                     ul_mid_price_time = ul_mid_price_time.tz_localize("UTC").tz_convert("ASIA/BANGKOK").tz_localize(None)
    #
    #                     logger.info(f"ul_mid_price_time {ul_mid_price_time}")
    #                     if self._put_call == "C":
    #                         predicted_ul_price = (
    #                                 current_p
    #                                 + self._tick_size_multiple
    #                                 * SETeq_TickRule.get_up_tick(current_p) #.values[-1]
    #                         )
    #                     else:
    #                         predicted_ul_price = (
    #                                 current_p
    #                                 - self._tick_size_multiple
    #                                 * SETeq_TickRule.get_dw_tick(current_p) #.values[-1]
    #                         )
    #
    #                     logger.info(f"predicted_ul_price {predicted_ul_price}")
    #
    #                     diffs = df_dw_price_guideline["spot"] - current_p
    #                     idx_mid = np.argmin(abs(diffs))
    #                     dw_price_from_guideline_corresp_to_ul_mid_price = df_dw_price_guideline.loc[
    #                         idx_mid, self.col
    #                     ]
    #
    #
    #                     logger.info(f"dw_price_from_guideline_corresp_to_ul_mid_price: {dw_price_from_guideline_corresp_to_ul_mid_price}")
    #
    #                     diffs = df_dw_price_guideline["spot"] - predicted_ul_price
    #                     idx = np.argmin(abs(diffs))
    #                     dw_price_from_guideline_corresp_to_predicted_ul_price = df_dw_price_guideline.loc[
    #                         idx, self.col
    #                     ]
    #                     logger.info(f"dw_price_from_guideline_corresp_to_predicted_ul_price: {dw_price_from_guideline_corresp_to_predicted_ul_price}")
    #
    #                     print(f"{dw_sym}")
    #                     dw_LOB = self.get_last_dw_orderbook(product_type='eq',symbol=dw_sym,sub_type='orderbook',
    #                                                         init_from_beginning=False)
    #
    #                     logger.debug(f"{dw_sym} dw_LOB:{dw_LOB}")
    #                     # df_dw = get_equity_lob(
    #                     #     date1=(ul_dt - pd.tseries.offsets.BusinessDay(n=1)),
    #                     #     date2=(ul_dt + pd.Timedelta(minutes=self._start_minute)),
    #                     #     symbol=dw_sym,
    #                     #     bkk_time=True,
    #                     #     level1_only=True,
    #                     # )
    #                     # print()
    #
    #                     if len(dw_LOB)> 0:
    #                         # print(df_dw[['UpdateTime', 'MDBid1Price', 'MDAsk1Price']])
    #                         current_dw_price = float(dw_LOB[REDIS_DB.LOB_flat_coldict['a1']]
    #                         )
    #                         DW_LOB_price = dw_price_from_guideline_corresp_to_ul_mid_price
    #
    #                         if current_dw_price > self._price_limit :
    #                             if np.isnan(self._stop_loss_ticks):
    #                                 logger.debug("Stop loss tick isnan")
    #                                 stop_loss_ul_price = np.nan
    #                                 stop_loss_price= np.nan
    #                             else:
    #                                 stop_loss_price = np.round(current_dw_price - self._stop_loss_ticks * SETeq_TickRule.get_dw_tick(current_dw_price),2)
    #                                 where_same = df_dw_price_guideline[self.col] == stop_loss_price
    #                                 # if dw_sym == "KBANK01C2303X":
    #                                 #     print()
    #                                 if not where_same.any(): #If no exact price on the PG, need to get first negative
    #                                     diff_2 = (df_dw_price_guideline[self.col] - stop_loss_price)
    #
    #                                     if  diff_2[diff_2 < 0].shape[0] == 0:
    #                                         '''Negative 1 indicates price not found on price guideline... should skip'''
    #                                         max_of_negative_idx = -1
    #                                         stop_loss_ul_price = -1
    #                                         stop_loss_price = -1
    #                                     else:
    #                                         max_of_negative_idx = diff_2[diff_2 < 0].idxmax() #First non_positive index
    #
    #                                         stop_loss_ul_price = df_dw_price_guideline.at[max_of_negative_idx,"spot"]
    #                                         stop_loss_price = df_dw_price_guideline.at[max_of_negative_idx,self.col]
    #
    #                                 else:
    #                                     if self._put_call == "C":
    #                                         stop_loss_ul_price = df_dw_price_guideline['spot'][where_same].max()
    #                                     else:
    #                                         stop_loss_ul_price = df_dw_price_guideline['spot'][where_same].min()
    #
    #                             current_dw_time = pd.Timestamp(dw_LOB[REDIS_DB.LOB_flat_coldict['tss']], unit='ms')
    #                             current_dw_time = current_dw_time.tz_localize("UTC").tz_convert("ASIA/BANGKOK").tz_localize(
    #                                 None)
    #
    #                             logger.info(
    #                                 f"UL price at {ul_mid_price_time} = {ul_mid_price}"
    #                             )
    #                             logger.info(
    #                                 f"curent {dw_sym} ask price {current_dw_price} at {current_dw_time}"
    #                             )
    #                             logger.info(
    #                                 f"{dw_sym} dw_price_from_guideline_corresp_to_ul_mid_price {dw_price_from_guideline_corresp_to_ul_mid_price}"
    #                             )
    #                             logger.info(
    #                                 f"{dw_sym} dw_price_from_guideline_corresp_to_predicted_ul_price {dw_price_from_guideline_corresp_to_predicted_ul_price}"
    #                             )
    #
    #                             logger.info(f"Condition 1 {np.sign(dw_price_from_guideline_corresp_to_ul_mid_price- dw_price_from_guideline_corresp_to_predicted_ul_price)} == -1"
    #                                          f" Condition 2 {np.sign(current_dw_price- dw_price_from_guideline_corresp_to_predicted_ul_price)} == -1"
    #                                         f" StopLoss Price: {stop_loss_ul_price}  n_tick_from_prev: {n_tick_from_prev} ")
    #
    #                             if  (
    #                                 (
    #                                     np.sign(
    #                                         dw_price_from_guideline_corresp_to_ul_mid_price
    #                                         - dw_price_from_guideline_corresp_to_predicted_ul_price
    #                                     )
    #                                     == -1
    #                                 )
    #                                 # &
    #                                 # (
    #                                 #     np.sign(
    #                                 #         ul_mid_price
    #                                 #         - dw_price_from_guideline_corresp_to_predicted_ul_price
    #                                 #     )
    #                                 #     == -1
    #                                 # )
    #                                 ) \
    #                                     and not (stop_loss_ul_price <= 0) and np.abs(n_tick_from_prev) <= self.n_ticks_limit_dict.get(self._symbol,1):
    #                                 actions[ii] = 1
    #                                 ul_mid_price_times[ii] = ul_mid_price_time
    #                                 ul_mid_prices[ii] = ul_mid_price
    #                                 predicted_ul_mid_prices[ii] = predicted_ul_price
    #                                 current_dw_ask_price_times[ii] = current_dw_time
    #                                 current_dw_ask_prices[ii] = DW_LOB_price
    #                                 dw_prices_from_guideline_corresp_to_ul_mid_prices[
    #                                     ii
    #                                 ] = dw_price_from_guideline_corresp_to_ul_mid_price
    #                                 dw_prices_from_guideline_corresp_to_predicted_ul_prices[
    #                                     ii
    #                                 ] = dw_price_from_guideline_corresp_to_predicted_ul_price
    #                                 ul_stoploss_price[ii] = stop_loss_ul_price
    #                                 dw_stoploss_price[ii] = stop_loss_price
    #                         else:
    #                             logger.info(f"DW price is { self._price_limit } or less: {current_dw_price}")
    #                     else:
    #                         logger.info(
    #                             f"{dw_sym}: NO LOB"
    #                         )
    #                 res = pd.DataFrame(
    #                     {
    #                         "signal_datetime": date_times,
    #                         "signal": signals,
    #                         "action": actions,
    #                         "ul_mid_time": ul_mid_price_times,
    #                         "ul_mid_price": ul_mid_prices,
    #                         "predicted_ul_mid_price": predicted_ul_mid_prices,
    #                         "current_dw_time": current_dw_ask_price_times,
    #                         "current_dw_ask_price": current_dw_ask_prices, #this is limit buy price
    #                         "dw_price_guideline_given_ul_mid_price": dw_prices_from_guideline_corresp_to_ul_mid_prices,
    #                         "dw_price_guideline_given_predicted_ul_price": dw_prices_from_guideline_corresp_to_predicted_ul_prices, #thiis take profit price
    #                         "ul_stoploss_price": ul_stoploss_price,
    #                         "dw_stoploss_price": dw_stoploss_price,
    #                         'probability':probability
    #                     }
    #                 )
    #
    #                 res["dw_symbol"] = dw_sym
    #                 res = res.loc[res["action"] == 1].reset_index(drop=True)
    #                 # print(res)
    #                 results.append(res)
    #
    #     if len(results) == 0:
    #         self._signals = pd.DataFrame(columns=['signal_datetime', 'signal', 'action', 'ul_mid_time', 'ul_mid_price',
    #                            'predicted_ul_mid_price', 'current_dw_time', 'current_dw_ask_price',
    #                            'dw_price_guideline_given_ul_mid_price',
    #                            'dw_price_guideline_given_predicted_ul_price', 'ul_stoploss_price',
    #                            'dw_stoploss_price','probability', 'dw_symbol'])
    #     else:
    #         self._signals = pd.concat(results).drop_duplicates().reset_index(drop=True)
    #     return self._signals





    def get_signals_intraday_evening(self):
        fn = f"{prediction_path}\\{self._signals_date_assumption.date()}\\{self._symbol}.csv"
        # fn = f"{base_folder}\\production0\\daily_price_prediction\\predictions_1tick\\{trade_date.date()}\\{self._symbol}.csv"

        try:
            df = pd.read_csv(fn, parse_dates=["datetime"])
        except Exception as e:
            logger.debug(e)
            # return
            raise
        max_prob_class = df.loc[:,['-1','0','1']].idxmax(axis=1).iat[0]
        if df[max_prob_class].iat[0] >= 0.4:
            '''just for reading'''
            if max_prob_class == "1" and self._put_call == "C":
                pass
                df =df

            elif max_prob_class == "-1" and self._put_call == "P":
                pass
                df = df
            else:
                df = pd.DataFrame([],columns=df.columns)
        else:
            df = pd.DataFrame([],columns=df.columns)



        logger.debug(f"for {self._put_call} predicted class shape {df.shape[0]}")

        dw_syms = self.get_dw_symbols()
        results = []

        logger.debug(f"list of dw for {self._symbol} => {dw_syms} isEmpty:{ df.empty}")

        if not df.empty:
            for dw_sym in dw_syms:

                logger.debug(f"DW sym: {dw_sym}")

                if not dw_sym in self._SETSMART_DW_SPEC.index:
                    logger.debug(f"dw:{dw_sym} not in DW SPEC in SETSMART skipping")
                    continue
                # else:

                if self._SETSMART_DW_SPEC.loc[dw_sym, "IssueBroker"] in self._valid_issuer_list:
                        print(f"{dw_sym} valid issuer")
                        logger.debug(f"{dw_sym} valid issuer")
                else:
                    logger.debug(f"{dw_sym} not a valid issuer")
                    continue


                logger.info(f"start:{dw_sym}")
                date_times = df["datetime"]
                signals = df["predicted_class"]
                probability = df.at[0,str(int(signals.values[0]))]
                actions = np.zeros(len(signals))
                ul_mid_prices = np.zeros(len(signals))
                predicted_ul_mid_prices = np.zeros(len(signals))
                current_dw_ask_prices = np.zeros(len(signals))
                dw_prices_from_guideline_corresp_to_ul_mid_prices = np.zeros(len(signals))
                dw_prices_from_guideline_corresp_to_predicted_ul_prices = np.zeros(
                    len(signals)
                )
                dw_prices_from_guideline_corresp_to_stoplosses_ul_prices =np.zeros(
                        len(signals)
                    )
                ul_stoploss_price = np.zeros(len(signals))
                dw_stoploss_price = np.zeros(len(signals))
                dw_stopprofit_price = np.zeros(len(signals))
                ul_mid_price_times = [pd.Timestamp(0) for i in range(len(signals))]
                current_dw_ask_price_times = [pd.Timestamp(0) for i in range(len(signals))]

                logger.debug(f"list of dt {date_times}")
                for ii, signal_dt in zip(range(len(date_times)), date_times):

                    # get_prev_trade_date(signal_dt)
                    # prev_date = get_prev_trade_date(signal_dt)
                    df_dw_price_guideline = get_dw_price_guideline(dw_sym, self.price_guideline_date)
                    # df_dw_price_guideline = get_dw_price_guideline(dw_sym, signal_dt)
                    df_dw_price_guideline = df_dw_price_guideline.sort_values('spot')
                    if not df_dw_price_guideline.empty:
                        # ul_dt = pd.Timestamp(signal_dt.date()) + pd.Timedelta(hours=10)

                        try:
                            df_ul = self.get_ul_LOB()

                        except IndexError as e:
                            logger.debug(f"{dw_sym} get UL LOB is empty")
                            continue
                        except Exception as e:
                            print(e)
                            raise e

                        if len(df_ul) == 0:
                            logger.debug(f"{dw_sym} get UL LOB is empty")
                            continue


                        # prev_close_bid = ohlcv.at[0, 'Z_LAST_BID']
                        # prev_close_ask = ohlcv.at[0, 'Z_LAST_OFFER']

                        logger.info(f"UL LOB: {df_ul} b1:{df_ul[self._RTD_class.redis_sub.LOB_flat_column_dict['b1']] } a1:{ df_ul[self._RTD_class.redis_sub.LOB_flat_column_dict['a1']]}")
                        # current_mid_price = (df_ul[REDIS_DB.LOB_flat_coldict['b1']] + df_ul[REDIS_DB.LOB_flat_coldict['a1']]) / 2.0

                        if self._put_call == "C":
                            # pass
                            # ul_mid_price = prev_close_ask
                            current_p = df_ul[self._RTD_class.redis_sub.LOB_flat_column_dict['a1']]

                        else:
                            # pass
                            # ul_mid_price = prev_close_bid
                            current_p = df_ul[self._RTD_class.redis_sub.LOB_flat_column_dict['b1']]
                            # n_tick_from_prev = SETeq_TickRule.get_n_ticks(current_p,ul_mid_price)

                        # ul_mid_price = (df_ul[REDIS_DB.LOB_flat_coldict['a1']])
                        # ul_mid_price = df_ul["Mid"].values[-1]
                        # ul_mid_price = df_ul["MDAsk1Price"].values[-1]
                        logger.info(f"UL Price {self._symbol} current_p:{current_p}")
                        ul_mid_price_time = self._RTD_class.redis_sub.convert_timestamp(df_ul[self._RTD_class.redis_sub.LOB_flat_column_dict['tss']])
                        # ul_mid_price_time = pd.Timestamp(df_ul[self._RTD_class.redis_sub.LOB_flat_column_dict['tss']],unit='ms')
                        ul_mid_price_time = ul_mid_price_time.tz_localize("UTC").tz_convert("ASIA/BANGKOK").tz_localize(None)

                        logger.info(f"ul_mid_price_time {ul_mid_price_time}")
                        if self._put_call == "C":
                            predicted_ul_price = (
                                    current_p
                                    + self._tick_size_multiple
                                    * SETeq_TickRule.get_up_tick(current_p) #.values[-1]
                            )

                            stop_loss_ul_price = (
                                    current_p
                                    - self._tick_size_multiple *10
                                    * SETeq_TickRule.get_up_tick(current_p) #.values[-1]
                            )
                        else:
                            predicted_ul_price = (
                                    current_p
                                    - self._tick_size_multiple
                                    * SETeq_TickRule.get_dw_tick(current_p) #.values[-1]
                            )

                            stop_loss_ul_price = (
                                    current_p
                                    + self._tick_size_multiple *10
                                    * SETeq_TickRule.get_up_tick(current_p) #.values[-1]
                            )

                        logger.info(f"predicted_ul_price {predicted_ul_price}")

                        diffs = df_dw_price_guideline["spot"] - current_p
                        idx_mid = abs(diffs).idxmin()
                        dw_price_from_guideline_corresp_to_ul_mid_price = df_dw_price_guideline.loc[
                            idx_mid, self.col
                        ]


                        logger.info(f"dw_price_from_guideline_corresp_to_ul_mid_price: {dw_price_from_guideline_corresp_to_ul_mid_price}")

                        diffs = df_dw_price_guideline["spot"] - predicted_ul_price
                        idx = abs(diffs).idxmin()
                        dw_price_from_guideline_corresp_to_predicted_ul_price = df_dw_price_guideline.loc[
                            idx, self.col_tp1
                        ]
                        logger.info(f"dw_price_from_guideline_corresp_to_predicted_ul_price: {dw_price_from_guideline_corresp_to_predicted_ul_price}")


                        diffs = df_dw_price_guideline["spot"] - stop_loss_ul_price
                        idx = abs(diffs).idxmin()
                        dw_price_at_ul_stoploss = df_dw_price_guideline.loc[
                            idx, self.col_tp1
                        ]
                        logger.info(f"dw_price_at_ul_stoploss: {dw_price_at_ul_stoploss}")

                        UpSide_expectation =dw_price_from_guideline_corresp_to_predicted_ul_price - dw_price_from_guideline_corresp_to_ul_mid_price

                        UpSide_Sense = UpSide_expectation / (self._tick_size_multiple* SETeq_TickRule.get_up_tick(dw_price_from_guideline_corresp_to_ul_mid_price))

                        DownSide_expectation =dw_price_at_ul_stoploss - dw_price_from_guideline_corresp_to_ul_mid_price

                        DownSide_Sense = DownSide_expectation / (self._tick_size_multiple* SETeq_TickRule.get_dw_tick(dw_price_from_guideline_corresp_to_ul_mid_price))


                        logger.debug(f"UpSide_Sense:{UpSide_Sense} DownSide_Sense:{DownSide_Sense}")
                        SensitivityCondition = (UpSide_Sense>=1) and (DownSide_Sense <= -1)


                        print(f"{dw_sym}")
                        dw_LOB = self.get_last_dw_orderbook(product_type='eq',symbol=dw_sym,sub_type='orderbook',
                                                            init_from_beginning=False)

                        logger.debug(f"{dw_sym} dw_LOB:{dw_LOB}")

                        if len(dw_LOB)> 0 and SensitivityCondition:
                            # print(df_dw[['UpdateTime', 'MDBid1Price', 'MDAsk1Price']])
                            current_dw_price = float(dw_LOB[self._RTD_class.redis_sub.LOB_flat_column_dict['a1']]
                            )
                            # DW_LOB_price = dw_price_from_guideline_corresp_to_ul_mid_price
                            DW_LOB_price = current_dw_price
                            DW_Stop_profit_price = np.round(DW_LOB_price + self._stop_loss_ticks  * SETeq_TickRule.get_dw_tick(current_dw_price),2)

                            if current_dw_price > self._price_limit :
                                if np.isnan(self._stop_loss_ticks):
                                    logger.debug("Stop loss tick isnan")
                                    stop_loss_ul_price = np.nan
                                    stop_loss_price= np.nan
                                else:
                                    stop_loss_price = np.round(current_dw_price - self._stop_loss_ticks * SETeq_TickRule.get_dw_tick(current_dw_price),2)
                                    where_same = df_dw_price_guideline[self.col] == stop_loss_price
                                    # if dw_sym == "KBANK01C2303X":
                                    #     print()
                                    if not where_same.any(): #If no exact price on the PG, need to get first negative
                                        diff_2 = (df_dw_price_guideline[self.col] - stop_loss_price)

                                        if  diff_2[diff_2 < 0].shape[0] == 0:
                                            '''Negative 1 indicates price not found on price guideline... should skip'''
                                            max_of_negative_idx = -1
                                            stop_loss_ul_price = -1
                                            stop_loss_price = -1
                                        else:
                                            max_of_negative_idx = diff_2[diff_2 < 0].idxmax() #First non_positive index

                                            stop_loss_ul_price = df_dw_price_guideline.at[max_of_negative_idx,"spot"]
                                            stop_loss_price = df_dw_price_guideline.at[max_of_negative_idx,self.col]

                                    else:
                                        if self._put_call == "C":
                                            stop_loss_ul_price = df_dw_price_guideline['spot'][where_same].max()
                                        else:
                                            stop_loss_ul_price = df_dw_price_guideline['spot'][where_same].min()


                                # current_dw_time = pd.Timestamp(dw_LOB[REDIS_DB.LOB_flat_coldict['tss']], unit='ms')
                                current_dw_time = self._RTD_class.redis_sub.convert_timestamp(
                                                                dw_LOB[self._RTD_class.redis_sub.LOB_flat_column_dict['tss']])

                                current_dw_time = current_dw_time.tz_localize("UTC").tz_convert("ASIA/BANGKOK").tz_localize(
                                    None)

                                # logger.info(
                                #     f"UL price at {ul_mid_price_time} = {ul_mid_price}"
                                # )
                                logger.info(
                                    f"curent {dw_sym} ask price {current_dw_price} at {current_dw_time}"
                                )
                                logger.info(
                                    f"{dw_sym} dw_price_from_guideline_corresp_to_ul_mid_price {dw_price_from_guideline_corresp_to_ul_mid_price}"
                                )
                                logger.info(
                                    f"{dw_sym} dw_price_from_guideline_corresp_to_predicted_ul_price {dw_price_from_guideline_corresp_to_predicted_ul_price}"
                                )

                                logger.info(f"Condition 1 {np.sign(dw_price_from_guideline_corresp_to_ul_mid_price- dw_price_from_guideline_corresp_to_predicted_ul_price)} == -1"
                                             f" Condition 2 {np.sign(current_dw_price- dw_price_from_guideline_corresp_to_predicted_ul_price)} == -1"
                                            f" StopLoss Price: {stop_loss_ul_price}  ")


                                if  (
                                    (
                                        np.sign(
                                            dw_price_from_guideline_corresp_to_ul_mid_price
                                            - dw_price_from_guideline_corresp_to_predicted_ul_price
                                        )
                                        == -1
                                    )
                                    # &
                                    # (
                                    #     np.sign(
                                    #         ul_mid_price
                                    #         - dw_price_from_guideline_corresp_to_predicted_ul_price
                                    #     )
                                    #     == -1
                                    # )
                                    ) \
                                        and not (stop_loss_ul_price <= 0) :

                                    #TODO check Risk versus reward

                                    logger.debug(f"Passed condition for profit... addding:{dw_sym}")
                                    actions[ii] = 1
                                    ul_mid_price_times[ii] = ul_mid_price_time
                                    ul_mid_prices[ii] = current_p
                                    predicted_ul_mid_prices[ii] = predicted_ul_price
                                    current_dw_ask_price_times[ii] = current_dw_time
                                    current_dw_ask_prices[ii] = DW_LOB_price
                                    dw_prices_from_guideline_corresp_to_ul_mid_prices[
                                        ii
                                    ] = dw_price_from_guideline_corresp_to_ul_mid_price
                                    dw_prices_from_guideline_corresp_to_predicted_ul_prices[
                                        ii
                                    ] = dw_price_from_guideline_corresp_to_predicted_ul_price

                                    dw_prices_from_guideline_corresp_to_stoplosses_ul_prices[ii] = dw_price_at_ul_stoploss

                                    ul_stoploss_price[ii] = stop_loss_ul_price
                                    dw_stoploss_price[ii] = stop_loss_price
                                    dw_stopprofit_price[ii] = DW_Stop_profit_price
                            else:
                                logger.info(f"DW price is { self._price_limit }or less: {current_dw_price}")
                        else:
                            logger.info(
                                f"{dw_sym}: NO LOB"
                            )
                    res = pd.DataFrame(
                        {
                            "signal_datetime": date_times,
                            "signal": signals,
                            "action": actions,
                            "ul_mid_time": ul_mid_price_times,
                            "ul_mid_price": ul_mid_prices,
                            "predicted_ul_mid_price": predicted_ul_mid_prices,
                            "current_dw_time": current_dw_ask_price_times,
                            "current_dw_ask_price": current_dw_ask_prices,
                            "dw_price_guideline_given_ul_mid_price": dw_prices_from_guideline_corresp_to_ul_mid_prices,
                            "dw_price_guideline_given_predicted_ul_price": dw_prices_from_guideline_corresp_to_predicted_ul_prices, #thiis take profit price
                            "ul_stoploss_price": ul_stoploss_price,
                            "dw_stoploss_price": dw_stoploss_price,
                            "dw_stopprofit_price": dw_stopprofit_price,
                            'probability':probability
                        }
                    )

                    res["dw_symbol"] = dw_sym
                    res = res.loc[res["action"] == 1].reset_index(drop=True)
                    # print(res)
                    results.append(res)
        # self._signals
        if len(results) == 0:
            _signals = pd.DataFrame(columns=self.signal_columns)
        else:
            _signals = pd.concat(results).drop_duplicates().reset_index(drop=True)

        if _signals.empty:
            self._signals = pd.DataFrame(columns=self.signal_columns)

        if not _signals.empty:
            self._signals = _signals

        return self._signals


    @property
    def signals(self):
        if self._signals is None:
            # self.get_signals_intraday()
            raise Exception("Signal is None")
        return self._signals

    def write_signal_file(self):

        date_path = os.path.join(signal_cache_path,f"{self._trade_date.date()}")
        if not (os.path.exists(date_path) and os.path.isdir(date_path)):
            os.makedirs(date_path)

        ul_path = os.path.join(date_path,f"{self._symbol}")
        if not (os.path.exists(ul_path) and os.path.isdir(ul_path)):
            os.makedirs(ul_path)

        if not self._signals.empty:
            self._signals.to_csv(os.path.join(ul_path,f"{self._put_call}.csv"))

    def read_signal_file(self,trade_date):

        date_path = os.path.join(signal_cache_path, f"{trade_date.date()}")
        ul_path = os.path.join(date_path, f"{self._symbol}")
        file_Path = os.path.join(ul_path,f"{self._put_call}.csv")

        if os.path.isfile(file_Path):
            self._signals = pd.read_csv(file_Path,index_col=0,parse_dates=['signal_datetime'])
            return True
        else:
            return False

    def get_last_dw_orderbook(self,product_type, symbol,sub_type,init_from_beginning=True):
        self._RTD_class.add_product(product_type=product_type, sub_type=sub_type, symbol=symbol,
                                     log_data=False, FakeSub=False, StartStream=False)

        logger.debug(f"init_price_feeds: {product_type} {symbol} {sub_type} init_from_beginning:{init_from_beginning} ")
        RTD_Instrument: REDIS_DB.instrument_class = self._RTD_class.get_symbol(symbol)

        subscription_string = ":".join([product_type, sub_type, symbol])

        logger.debug(f"subscription_string: {subscription_string}")

        #TODO What happens when redis is empty? my guess is you will get []
        last_item = RTD_Instrument.get_last_stream_item(subscription_string, n_count=2)
        logger.debug(f"Init latest {last_item}")

        if len(last_item) > 0:
            transformed = self.RTD_Instrument.transform_data_dict(last_item[0][1],"orderbook")
            LOB = np.hstack(np.array([v for k, v in transformed.items()], dtype=object))
        else:
            LOB = []
            # if len(last_item) >1:
            #     if sub_type == "tick":
            #         # RTD_symbol.open_tick_redis_stream('$')
            #         RTD_Instrument.open_tick_redis_stream(last_item[1][0])
            #     elif sub_type == "orderbook":
            #         RTD_Instrument.open_lob_redis_stream(last_item[1][0])
            # else:
            #     if sub_type == "tick":
            #         RTD_Instrument.open_tick_redis_stream('$')
            #     elif sub_type == "orderbook":
            #         RTD_Instrument.open_lob_redis_stream('$')

        # LOB = RTD_Instrument.get_latest_LOB()
        logger.debug(f"LOB state for {symbol} {LOB}")
        return LOB



    def init_RTD_feed(self,product_type, symbol,sub_type,init_from_beginning=True):
        self._RTD_class.add_product(product_type=product_type, sub_type=sub_type, symbol=symbol,
                                     log_data=False, FakeSub=False, StartStream=False)

        logger.info(f"init_price_feeds: {product_type} {symbol} {sub_type} init_from_beginning:{init_from_beginning} ")
        self.RTD_Instrument: REDIS_DB.instrument_class = self._RTD_class.get_symbol(symbol)

        subscription_string = ":".join([product_type, sub_type, symbol])

        logger.info(f"subscription_string: {subscription_string}")
        if init_from_beginning:
            logger.info("Init from beggining")
            if sub_type == "tick":
                last_id = self.RTD_Instrument.get_all_tick_states()
                self.RTD_Instrument.open_tick_redis_stream(last_id)
            elif sub_type == "orderbook":
                last_id = self.RTD_Instrument.get_all_LOB_states()
                self.RTD_Instrument.open_lob_redis_stream(last_id)
        else:

            #TODO What happens when redis is empty? my guess is you will get []
            last_item = self.RTD_Instrument.get_last_stream_item(subscription_string, n_count=2)
            logger.info(f"Init latest {last_item} len:{len(last_item)}")
            if len(last_item) >1:
                if sub_type == "tick":
                    # RTD_symbol.open_tick_redis_stream('$')
                    self.RTD_Instrument.open_tick_redis_stream(last_item[1][0])
                elif sub_type == "orderbook":
                    self.RTD_Instrument.open_lob_redis_stream(last_item[1][0])
            else:
                if sub_type == "tick":
                    self.RTD_Instrument.open_tick_redis_stream('$')
                elif sub_type == "orderbook":
                    self.RTD_Instrument.open_lob_redis_stream('$')

    def get_ul_LOB(self):
        logger.info("get_ul_LOB")
        return self.RTD_Instrument.get_latest_LOB()



def read_date_cache_signal_folder(target_date):

    date_path = os.path.join(signal_cache_path, f"{target_date.date()}")

    if os.path.isdir(date_path):
        all_signal_dict = {}
        for ul in os.listdir(date_path):
            ul_path = os.path.join(date_path, f"{ul}")
            for file in os.listdir(ul_path):

                PC = os.path.splitext(file)[0]
                file_Path = os.path.join(ul_path,f"{file}")

                if os.path.isfile(file_Path):
                    _signals = pd.read_csv(file_Path,index_col=0,parse_dates=['signal_datetime'])

                    if ul in all_signal_dict:
                        all_signal_dict[ul][PC] = pd.concat([all_signal_dict[ul],_signals])
                    else:
                        all_signal_dict[ul] = {PC:_signals}

        return all_signal_dict
    else:
        return {}

def check_dw(_symbol,price_guideline_date,_put_call):
    df = get_dw_price_guideline(_symbol, price_guideline_date)
    '''we trading at the end of day, no need to get previous price guideline '''
    # df = get_dw_price_guideline(self._symbol, self._trade_date )
    df = df.loc[~df['symbol'].str.contains(_symbol + '24')]

    _valid_issuer_list = ["01","19","13"]


    put_call_index = len(_symbol) + 2
    df = df.loc[df.symbol.str[put_call_index] == _put_call]

    return  df["symbol"].unique()



if __name__ == "__main__":
    # pass

    calls_dw = check_dw(_symbol="PTTEP",price_guideline_date=pd.Timestamp.now().normalize(),_put_call="C")
    puts_dw = check_dw(_symbol="PTTEP",price_guideline_date=pd.Timestamp.now().normalize(),_put_call="P")

    print()