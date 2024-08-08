import sys, os
import threading
import pandas as pd
import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from GQDatabaseSQL import REDIS_DB, SETSMART
import time
from projUtils import pyqt_utils
from classes import TelegramBOT

# sys.path.append("C:\\gitrepos\\python\\GQOrderInterfacePy")
from GQOrderInterfacePy import GQ_OMS_Interface
from GQOrderInterfacePy.Orders import OrderClass
import configparser
from TFEX_Utils import TFEX_Utils
# Press the green button in the gutter to run the script.
import logging
from myGQLib.Classes import Instrument, GQTimer

import telegram
import yaml
from yaml import CLoader as Loader
import datetime

from Aam import deploy

DB_Config = configparser.ConfigParser()
DB_Config.read(os.path.join("C:\Config", "DatabaseConfig.ini"))
HZ_Market_ID_MAP = {"SET": "XBKK", "TFEX": "TFEX"}

LogPath = os.getcwd() + "\\logs\\dw"
LogPath = os.path.join(LogPath, pd.Timestamp.now().date().strftime("%Y-%m-%d"))
if not os.path.isdir(LogPath):
    os.mkdir(LogPath)
# LogPath = os.path.join(LogPath,pd.Timestamp.now().date().strftime("%Y-%m-%d"))
if not os.path.isdir(LogPath):
    os.mkdir(LogPath)
formatter_log_scripts = logging.Formatter(
    "%(asctime)s - %(filename)s - %(threadName)s - %(funcName)s() %(lineno)d - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(
    os.path.join(LogPath, "spam.log")
)  # This will log all logging levels
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter_log_scripts)
logger.addHandler(fh)


base_folder = "K:\\aam_dw"


class signal_class():
    def __init__(self, symbol, limit_buy_price, quantity, trigger_stop_profit_price, stop_profit_price,
                 trigger_stop_loss_price, stop_loss_price,signal_date,current_qty=0,lower_bound_price=0):
        logger.debug(f"Init Signal Class for: {symbol} limit_buy_price:{limit_buy_price} quantity:{quantity} "
                     f"stop_profit_price:{stop_profit_price} trigger_stop_profit_price:{trigger_stop_profit_price} "
                     f"stop_loss_price:{stop_loss_price} trigger_stop_loss_price:{trigger_stop_loss_price}")
        self.symbol = symbol
        self._signal_entry_date = signal_date
        self.quantity = np.round(quantity, 0)
        self.limit_buy_price = np.round(limit_buy_price, 2)
        self.lower_bound_price = np.round(lower_bound_price, 2)
        self.trigger_stop_profit_price = np.round(trigger_stop_profit_price, 2)
        self.stop_profit_price = np.round(stop_profit_price, 2)
        self.trigger_stop_loss_price = np.round(trigger_stop_loss_price, 2)
        self.stop_loss_price = np.round(stop_loss_price, 2)
        self.AutoLimitOrder = None
        self.TimeScheduleOrder = None
        self._current_qty = current_qty

    # def obsolete_signal(self):
    def __str__(self):
        return (f"Init Signal Class for: {self.symbol} limit_buy_price:{self.limit_buy_price} quantity:{self.quantity} "
                     f"stop_profit_price:{self.stop_profit_price} trigger_stop_profit_price:{self.trigger_stop_profit_price} "
                     f"stop_loss_price:{self.stop_loss_price} trigger_stop_loss_price:{self.trigger_stop_loss_price}"
                f"_signal_entry_date:{self._signal_entry_date}")

    def add_order_class(self, _orderclass):
        self.AutoLimitOrder = _orderclass


    def isSignalLive(self):
        if self._signal_entry_date <= pd.Timestamp.now().normalize():
            return False
        else:
            return True


    def write_signal_class(self,base_path):
        pass

'''
Signal DF hardcoded columns index
'''

signal_columns = deploy.DwBackTest.signal_columns

signal_column_dict = {item[1]: item[0] for item in enumerate(signal_columns)}

expected_symbol = ["FORTH","JMART","JMT","KCE","STA"]

def find_last_row_with_widget(grid_layout: QGridLayout) -> int:
    last_row = -1
    for i in range(grid_layout.count()):
        widget = grid_layout.itemAt(i).widget()
        if widget is not None and not widget.isHidden():
            row, _, _, _ = grid_layout.getItemPosition(i)
            last_row = max(last_row, row)
    return last_row


class main(QMainWindow):
    global bot

    def __init__(self, appconfig,bot, tick_diff_limit=5, qty_size=10, order_interval_limit=10,
                 loop_interval=10, UAT =True):
        super(main, self).__init__()
        self.config = appconfig
        self.BOT_Helper =bot
        # self.BOT_Helper.add_group(group_name, group_id)

        self.symbol_dict = {}
        self.Holiday_class = TFEX_Utils.SETSMART_Holidays()

        sub = dict(self.config['REDIS_PROD_MME_CLOUD'])
        # sub = dict(DB_Config["REDIS_UAT"])
        sub['db'] = 0
        # sub['message_type'] = self.config['REDIS']['message_type']
        pub = dict(self.config['REDIS_UAT_MME_CLOUD'])
        pub['db'] = 0
        # pub['message_type'] =self.config['REDIS']['message_type']
        self.RTD_Manager = REDIS_DB.RTD_Manager(sub=sub, pub=pub,
                                                verbose=False,
                                                block_writes=True)

        if UAT:
            login_config = yaml.load(open('./config/config_uat.yaml'), Loader=Loader)
        else:
            login_config = yaml.load(open('./config/config.yaml'), Loader=Loader)


        dst_config = yaml.load(open('./config/dst_config.yaml'), Loader=Loader)

        account_info = {'account_no':login_config['account_no']}

        self.mOrderInterface = GQ_OMS_Interface.interface.DST_Interface(account_info=account_info, config=login_config,
                                                                        dst_config=dst_config)

        self.market = HZ_Market_ID_MAP["SET"]
        self.mOrderManager = OrderClass.OrdersManager(self.mOrderInterface, self.RTD_Manager)
        self.Portfolio_Class = OrderClass.Portfolio(mConnector=self.mOrderInterface, market=self.market)

        self.run_date = pd.Timestamp(pd.Timestamp.now().date())
        # self.expected_signal_date = deploy.get_next_trading_date(self.run_date)
        # self.run_date = pd.Timestamp("20230113")
        self._signal_update = False
        self.allow_trade = False
        self.Infinite_loop = True
        self.HasInit = False
        self.last_timestamp = None
        self.current_status = None
        self.update_table_loop  =False
        self.qty_size = qty_size
        self.order_interval_limit = 10
        self.loop_interval = loop_interval
        self.n_rows = 10
        self.tick_diff_limit = tick_diff_limit
        self._verbose = False
        self.cache_data = pd.DataFrame([])
        self.trading_Qty = int(500)
        self.trading_qty_max = 2000
        # self.trading_value_limit = 4000
        self.stop_loss_ticks = 10
        self.tick_size_multiple = 1

        self.skip_ul_sym_list = []
        self.valid_symbols_manual = []
        self.list_of_ul = []
        self.dict_of_ul_classes = {}
        self.symbols_signal_dict = {}
        self.qty_weighted_dict = {}
        self.list_of_trading_instruments = []

        self.DW_spec = SETSMART.get_DW_Specs(Last_trade_date_gr=self.run_date)

        self.DW_spec['Instrument'] = [symbol.rstrip() for symbol in self.DW_spec['Instrument'].values]
        self.DW_spec['UL'] = [symbol.rstrip().lstrip() for symbol in self.DW_spec['UL'].values]
        self.DW_spec.set_index('Instrument', inplace=True)
        logger.info(f"Init Main Complete")

        self.init_positions()

        '''
        There are 5 events to execute
        1) Take profit during opening
            -> done -> run place_limit_order_at_take_profit
        2) if position exists and signal is old, create bracket stop order 
            -> run morning_process -> start execution [checks for positions in the previous signals] 
                if exist, and old signal (prev signal) set up bracket order
        3) at 16:10: Read signal. Enter with bracket stop order
            -> read new signal
        4) exit old positions
        5) Take Profit during closing auction
        '''

        self.init_ui()


        self.opening_auction_time_event = datetime.datetime(
            datetime.datetime.today().year,
            datetime.datetime.today().month,
            datetime.datetime.today().day,
            9,
            50,
            00,
        )
        self.morning_session_time = datetime.datetime(
            datetime.datetime.today().year,
            datetime.datetime.today().month,
            datetime.datetime.today().day,
            10,
            0,
            2,
        )
        self.new_signal_time = datetime.datetime(
            datetime.datetime.today().year,
            datetime.datetime.today().month,
            datetime.datetime.today().day,
            10,
            31,
            0,
        )
        self.closing_auction_time_start = datetime.datetime(
            datetime.datetime.today().year,
            datetime.datetime.today().month,
            datetime.datetime.today().day,
            16,
            30,
            00,
        )
        self.closing_auction_time_event = datetime.datetime(
            datetime.datetime.today().year,
            datetime.datetime.today().month,
            datetime.datetime.today().day,
            16,
            32,
            00,
        )

        self._stop_algo_time_event = datetime.datetime(
            datetime.datetime.today().year,
            datetime.datetime.today().month,
            datetime.datetime.today().day,
            16,
            41,
            00,
        )




        self.populate_signal_table()

        # self.get_time_to_exit_position()
        self.OpenningAuctionTakeProfit()
        self.morning_start_timer()
        self.try_read_signal_timer() #Afternoon read and execution
        self.ClosingAuctionAction()
        self.get_close_app_timer()
        self.start_UI_loop()
        # self.start_execution()


    def init_positions(self):
        '''
        This should run in the morning before 10:00
        > to run take profit
        >
        '''
        prev_date = deploy.get_prev_trade_date(self.run_date)
        '''folders are +1 trading date'''
        prev_days_dict = deploy.read_date_cache_signal_folder(prev_date)
        # if len(prev_days_dict) == 0:
        #     print()
        self.current_position_dict = {}

        for ul,PC_dict in prev_days_dict.items():
            for pc, signal_df in PC_dict.items():
                for i in range(signal_df.shape[0]):
                    dw_symbol = signal_df.iat[i,signal_df.columns.get_loc('dw_symbol')]
                    try:
                        current_position = self.Portfolio_Class.get_positions_from_connector(dw_symbol)
                        print(current_position)
                    except Exception as e:
                        logger.debug(f"Request Error:{e} moving on")
                        continue
                    if ul not in self.current_position_dict:
                        self.current_position_dict[ul] = {'P':0,'C':0}

                    self.current_position_dict[ul][pc] = current_position

    def init_ui(self):

        self.Main_Widget = QWidget()

        self.pushButton1 = QPushButton(self.Main_Widget)
        self.pushButton1.setText("Read Signal")
        self.pushButton1.clicked.connect(self.on_button1_clicked)

        self.pushButton2 = QPushButton(self.Main_Widget)
        self.pushButton2.setText("Generate Signal Class")
        self.pushButton2.clicked.connect(self.on_button2_clicked)

        self.pushButton3 = QPushButton(self.Main_Widget)
        self.pushButton3.setText("Execute")
        self.pushButton3.clicked.connect(self.on_button3_clicked)

        self.pushButton4 = QPushButton(self.Main_Widget)
        self.pushButton4.setText("Try Exit All")
        self.pushButton4.clicked.connect(self.on_button4_clicked)

        self.pushButton5 = QPushButton(self.Main_Widget)
        self.pushButton5.setText("Read SigClassCache")
        self.pushButton5.clicked.connect(self.on_button5_clicked)

        self.main_Layout = QGridLayout(self.Main_Widget)
        self.main_layout_grid_rows_count = {}

        row_1_widget = QWidget()
        row_1_grid = QGridLayout(row_1_widget)
        row_1_grid.addWidget(self.pushButton1,0,0)
        row_1_grid.addWidget(self.pushButton5,0,1)

        # Tables for input
        '''Table'''

        self.SendingOrderTable: QTableWidget = QTableWidget()
        self.LiveOrderTable: QTableWidget = QTableWidget()

        self.SendOrderButton = QPushButton(self.Main_Widget)
        self.SendOrderButton.setText("Send Order")
        self.SendOrderButton.clicked.connect(self.on_send_order_clicked)


        self.AlgoOrderTables = QTableWidget(self.Main_Widget)

        self.PositionTable = QTableWidget(self.Main_Widget)
        self.DST_Order_Table = QTableWidget(self.Main_Widget)

        self.SignalTable = QTableWidget(self.Main_Widget)

        self.order_reply_label = QLabel(self.Main_Widget)

        self.run_date_label = QLabel(self.Main_Widget)
        self.run_date_label.setText(self.run_date.strftime("%Y-%m-%d"))

        self.pushButton6 = QPushButton(self.Main_Widget)
        self.pushButton6.setText("Stop Algo")
        self.pushButton6.clicked.connect(self.on_button6_clicked)

        # self.pushButton8 = QPushButton(self.Main_Widget)
        # self.pushButton8.setText("Run Afternoon")
        # self.pushButton8.clicked.connect(self.try_read_signal_execute)

        #
        # self.pushButton7 = QPushButton(self.Main_Widget)
        # self.pushButton7.setText("Clear all orders from symbol in signal")
        # self.pushButton7.clicked.connect(self.cancel_all_symbol_order)
        # self.main_Layout.addWidget(self.pushButton7, 9, 0)

        self.append_widget_to_column(self.main_Layout,0,row_1_widget,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,0,self.pushButton2,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,0,self.pushButton3,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,0,self.pushButton4,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,0,self.PositionTable,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,0,self.DST_Order_Table,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,0,self.AlgoOrderTables,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,0,self.SignalTable,self.main_layout_grid_rows_count)

        self.append_widget_to_column(self.main_Layout,0,self.SendingOrderTable,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,0,self.SendOrderButton,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,0,self.LiveOrderTable,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,0,self.order_reply_label,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,1,self.pushButton6,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,1,self.run_date_label,self.main_layout_grid_rows_count)


        self.blank = QLabel(self.Main_Widget)
        self.append_widget_to_column(self.main_Layout,1,self.blank,self.main_layout_grid_rows_count)
        self.append_widget_to_column(self.main_Layout,1,self.blank,self.main_layout_grid_rows_count)


        label = QLabel(self.Main_Widget)
        label.setText("Position")
        self.append_widget_to_column(self.main_Layout,1,label,self.main_layout_grid_rows_count)

        label = QLabel(self.Main_Widget)
        label.setText("Order Status")
        self.append_widget_to_column(self.main_Layout,1,label,self.main_layout_grid_rows_count)

        label = QLabel(self.Main_Widget)
        label.setText("AlgoOrderTables")
        self.append_widget_to_column(self.main_Layout,1,label,self.main_layout_grid_rows_count)

        label = QLabel(self.Main_Widget)
        label.setText("Signal Tables")
        self.append_widget_to_column(self.main_Layout,1,label,self.main_layout_grid_rows_count)

        label = QLabel(self.Main_Widget)
        label.setText("Sender Order")
        self.append_widget_to_column(self.main_Layout,1,label,self.main_layout_grid_rows_count)


        # self.append_widget_to_column(self.main_Layout,1,self.pushButton6,self.main_layout_grid_rows_count)
        # self.append_widget_to_column(self.main_Layout,1,self.pushButton6,self.main_layout_grid_rows_count)
        # self.main_Layout.addWidget(row_1_widget, 0, 0)
        # self.main_Layout.addWidget(self.pushButton2, 1, 0)
        # self.main_Layout.addWidget(self.pushButton3, 2, 0)
        # self.main_Layout.addWidget(self.pushButton4, 3, 0)
        # self.main_Layout.addWidget(self.pushButton6, 8, 1)
        # self.main_Layout.addWidget(self.PositionTable, 7, 0)
        # self.main_Layout.addWidget(self.AlgoOrderTables, 6, 0)
        # self.main_Layout.addWidget(self.SendingOrderTable, 4, 0)
        # self.main_Layout.addWidget(self.SendOrderButton, 5, 0)
        # self.main_Layout.addWidget(self.SignalTable, 8, 0)
        # self.main_Layout.addWidget(self.order_reply_label, 9, 0)
        # self.main_Layout.addWidget(self.run_date_label, 9, 0)
        # self.main_Layout.addWidget(self.pushButton8, 0, 1)




        self.setCentralWidget(self.Main_Widget)
        self.init_order_table()

        self.show()


    def get_app_state(self):
        pd.Timestamp.now()


    def append_widget_to_column(self, layout: QGridLayout, column, widget,row_count_dict):

        if isinstance(layout, QGridLayout):

            if not column in row_count_dict:
                row_count_dict[column] = -1

            layout.addWidget(widget, row_count_dict[column]+1, column)
            row_count_dict[column] = row_count_dict[column] + 1

    def init_order_table(self):

        self.algo_order_table_column = ['symbol', 'order_type', 'direction', 'order_price',
                                   'stop_price','qty','filled', 'bid','ask']
        self.sendorder_columns = ["symbol", "Price", "Qty"]

        self.DST_Order_Table_columns = ['symbol','status','iOrdNo','price','Qty','side']
        self.DST_Order_Table_columns_dict = {k:i for i,k in enumerate(self.DST_Order_Table_columns)}

        self.position_table_columns = ["symbol", "Qty"]
        self.position_table_columns_dict = {k:i for i,k in enumerate(self.position_table_columns)}

        self.signal_table_column =['symbol','signal_datetime','ul_stoploss_price','order_price','dw_stoploss_price','dw_take_profit_price']

        self.signal_table_column_dict = {k:i for i,k in enumerate(self.signal_table_column)}


        self.LiveOrderTable_columns = ["symbol", "Price", "Qty","DST_ID"]
        self.LiveOrderTable_columns_columns_dict = {k:i for i,k in enumerate(self.LiveOrderTable_columns)}


        self.table_columns_map = {colname: i for i, colname in enumerate(self.sendorder_columns)}

        self.PositionTable.clear()
        self.AlgoOrderTables.clear()
        self.SendingOrderTable.clear()
        self.SignalTable.clear()

        self.PositionTable.setColumnCount(len(self.position_table_columns))
        self.PositionTable.setHorizontalHeaderLabels(self.position_table_columns)
        self.PositionTable.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.AlgoOrderTables.setColumnCount(len(self.algo_order_table_column))
        self.AlgoOrderTables.setHorizontalHeaderLabels(self.algo_order_table_column)
        self.AlgoOrderTables.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.SendingOrderTable.setColumnCount(len(self.sendorder_columns))
        self.SendingOrderTable.setHorizontalHeaderLabels(self.sendorder_columns)
        self.SendingOrderTable.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.SendingOrderTable.insertRow(0)


        self.SignalTable.setColumnCount(len(self.signal_table_column))
        self.SignalTable.setHorizontalHeaderLabels(self.signal_table_column)
        self.SignalTable.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.DST_Order_Table.setColumnCount(len(self.DST_Order_Table_columns))
        self.DST_Order_Table.setHorizontalHeaderLabels(self.DST_Order_Table_columns)
        self.DST_Order_Table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)

        self.LiveOrderTable.setColumnCount(len(self.LiveOrderTable_columns))
        self.LiveOrderTable.setHorizontalHeaderLabels(self.LiveOrderTable_columns)
        self.LiveOrderTable.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)


    def generate_dw_signal(self, run_date, tick_size_multiple=1, start_minute=2, read_cache=False):
        # logger.info(f"read_signal function start")
        # signal_file_tmp = pd.DataFrame(["EA24C2212A",0.23,5,3],columns=['Symbol','Price','ProfitTick','LossTick'])

        if read_cache:
            # file_path = f"{deploy.prediction_path}\\{run_date.date()}"
            file_path = os.path.join(deploy.signal_cache_path,f"{run_date.date()}")
        else:
            file_path = f"{deploy.prediction_path}\\{run_date.date()}"

        logger.info(f"read_signal function start at: {file_path}")
        self.list_of_ul = [os.path.splitext(item)[0] for item in os.listdir(file_path)]
        for ul_sym in self.list_of_ul:
            logger.info(f"UL:{ul_sym}")

            if ul_sym in self.skip_ul_sym_list:
                logger.info(f"{ul_sym} in skip list... we skip")
                continue

            if ul_sym in self.dict_of_ul_classes:
                ul_dw_PC_dict = self.dict_of_ul_classes[ul_sym]

                if read_cache:
                    ul_dw_PC_dict['P'].read_signal_file(run_date)
                    ul_dw_PC_dict['C'].read_signal_file(run_date)
                else:
                    ul_dw_PC_dict['P'].get_signals_intraday_morning()
                    ul_dw_PC_dict['C'].get_signals_intraday_morning()
            else:
                logger.info(f"ul sym no in dict of ul classes")

                stop_loss_ticks = self.stop_loss_ticks

                self.dict_of_ul_classes[ul_sym] = {"P": deploy.DwBackTest(symbol=ul_sym, trade_date=run_date,
                                                                          RTD_Class=self.RTD_Manager,
                                                                          SETSMART_DW_SPEC=self.DW_spec,
                                                                          put_call="P",
                                                                          tick_size_multiple=tick_size_multiple,
                                                                          start_minute=start_minute,
                                                                          stop_loss_ticks=stop_loss_ticks,
                                                                          outlook_email=None,
                                                                          use_prev_day=False
                                                                          ),
                                                   "C": deploy.DwBackTest(symbol=ul_sym, trade_date=run_date,
                                                                          RTD_Class=self.RTD_Manager,
                                                                          SETSMART_DW_SPEC=self.DW_spec,
                                                                          put_call="C",
                                                                          tick_size_multiple=tick_size_multiple,
                                                                          start_minute=start_minute,
                                                                          stop_loss_ticks=stop_loss_ticks,
                                                                          outlook_email=None,
                                                                          use_prev_day=False
                                                                          )
                                                   }


                if read_cache:
                    self.dict_of_ul_classes[ul_sym]['P'].read_signal_file(run_date)
                    self.dict_of_ul_classes[ul_sym]['C'].read_signal_file(run_date)
                else:

                    self.dict_of_ul_classes[ul_sym]['P'].get_signals_intraday_morning()
                    self.dict_of_ul_classes[ul_sym]['C'].get_signals_intraday_morning()
                    self.dict_of_ul_classes[ul_sym]['P'].write_signal_file()
                    self.dict_of_ul_classes[ul_sym]['C'].write_signal_file()
                # return signal_file_tmp

        # pass
        logger.info(f"read signal complete")

    def create_signal_class(self):
        '''
        Design:
        1) Enter Limit Order
        2) if get filled or partial update, send on stop order for the amount
            2.a) if partial, will need to increase qty of stop order
        3) Not sure if we read at lunch?
        4) at 3:55pm may need to consider off loading positions at bid best i could
            4.a) Let's just keep running at expoenential 5min interval
        '''
        # self.get_weights_simple()

        logger.info("Start creating signal class for all")

        for ul_sym, PC_pair in self.dict_of_ul_classes.items():
            print(ul_sym)

            logger.info(f"ul_sym :{ul_sym}")
            if not PC_pair['P']._signals is None and not PC_pair['P'].signals.empty:

                logger.info(f"Put signal not empty")
                # put_symbols = PC_pair['P'].signals['dw_symbol']

                for i in range(PC_pair['P'].signals.shape[0]):
                    # rowItem = PC_pair['P'].signals.iloc[i]

                    dw_sym = PC_pair['P'].signals.iat[i, signal_column_dict['dw_symbol']]
                    if len(self.valid_symbols_manual) > 0:
                        if not dw_sym in self.valid_symbols_manual:
                            continue
                    # limit_buy_price = PC_pair['P'].signals.iat[i,signal_column_dict['current_dw_ask_price']]
                    limit_buy_price = PC_pair['P'].signals.iat[
                        i, signal_column_dict['dw_price_guideline_given_ul_mid_price']]
                    stop_loss_price = PC_pair['P'].signals.iat[i, signal_column_dict['dw_stoploss_price']]
                    trigger_stop_loss_price = PC_pair['P'].signals.iat[i, signal_column_dict['ul_stoploss_price']]
                    # stop_loss_price = limit_buy_price - np.round(deploy.SETeq_TickRule.get_dw_tick(limit_buy_price)*2,2)
                    stop_profit_price = PC_pair['P'].signals.iat[
                        i, signal_column_dict['dw_price_guideline_given_predicted_ul_price']]
                    signal_date = PC_pair['P'].signals.iat[
                        i, signal_column_dict['signal_datetime']]

                    lower_bound_price = PC_pair['P'].signals.iat[i, signal_column_dict['dw_price_lower_bound_array']]

                    if np.isnan(limit_buy_price) or np.isnan(stop_profit_price):
                        continue

                    current_pos_dict = self.current_position_dict.get(ul_sym,{'P':0,'C':0})
                    # current_pos_dict.get('P',)

                    self.add_instrument(symbol=dw_sym, market="XBKK", product_type="eq", pc="P")
                    self.symbols_signal_dict[dw_sym] = signal_class(symbol=dw_sym, limit_buy_price=limit_buy_price,
                                                                    quantity=self.trading_Qty,
                                                                    # quantity=self.qty_weighted_dict[ul_sym],
                                                                    trigger_stop_loss_price=trigger_stop_loss_price,
                                                                    stop_loss_price=stop_loss_price,
                                                                    trigger_stop_profit_price=stop_profit_price,
                                                                    stop_profit_price=stop_profit_price,
                                                                    signal_date=signal_date,
                                                                    current_qty=current_pos_dict['P'],
                                                                    lower_bound_price=lower_bound_price,
                                                                    )

            if not PC_pair['C']._signals is None and not PC_pair['C'].signals.empty:

                logger.info(f"Call signal not empty")
                for i in range(PC_pair['C'].signals.shape[0]):
                    dw_sym = PC_pair['C'].signals.iat[i, signal_column_dict['dw_symbol']]
                    if len(self.valid_symbols_manual) > 0:
                        if not dw_sym in self.valid_symbols_manual:
                            continue
                    # limit_buy_price = PC_pair['C'].signals.iat[i,signal_column_dict['current_dw_ask_price']]
                    limit_buy_price = PC_pair['C'].signals.iat[
                        i, signal_column_dict['dw_price_guideline_given_ul_mid_price']]
                    stop_loss_price = PC_pair['C'].signals.iat[i, signal_column_dict['dw_stoploss_price']]
                    trigger_stop_loss_price = PC_pair['C'].signals.iat[i, signal_column_dict['ul_stoploss_price']]
                    stop_profit_price = PC_pair['C'].signals.iat[
                        i, signal_column_dict['dw_price_guideline_given_predicted_ul_price']]

                    signal_date = PC_pair['C'].signals.iat[
                        i, signal_column_dict['signal_datetime']]

                    lower_bound_price = PC_pair['C'].signals.iat[i, signal_column_dict['dw_price_lower_bound_array']]

                    if np.isnan(limit_buy_price) or np.isnan(stop_profit_price):
                        continue

                    current_pos_dict = self.current_position_dict.get(ul_sym,{'P':0,'C':0})

                    self.add_instrument(symbol=dw_sym, market="XBKK", product_type="eq", pc="C")
                    self.symbols_signal_dict[dw_sym] = signal_class(symbol=dw_sym, limit_buy_price=limit_buy_price,
                                                                    quantity=self.trading_Qty,
                                                                    # quantity=self.qty_weighted_dict[ul_sym],
                                                                    trigger_stop_loss_price=trigger_stop_loss_price,
                                                                    stop_loss_price=stop_loss_price,
                                                                    trigger_stop_profit_price=stop_profit_price,
                                                                    stop_profit_price=stop_profit_price,
                                                                    signal_date=signal_date,
                                                                    current_qty=current_pos_dict['P'],
                                                                    lower_bound_price=lower_bound_price)

        self.populate_signal_table()
        print("Done")

    def get_weights_simple(self):

        if len(self.dict_of_ul_classes) == 0:
            print(f"dict_of_ul_classes is empty! Please check!")

        self.probability_dict = {}
        for ul_sym, PC_pair in self.dict_of_ul_classes.items():

            for PC_s, DwBackTest_obj in PC_pair.items():
                if not DwBackTest_obj._signals is None and not DwBackTest_obj.signals.empty:
                    # probability =
                    self.probability_dict[ul_sym] = [DwBackTest_obj.signals.at[0, 'probability'],
                                                     DwBackTest_obj.signals.shape[0]]

        if len(self.probability_dict) > 0:
            matrix = np.array(list(self.probability_dict.values()))
            # print()
            rounded_qty_each_dw_per_ul = np.round(((matrix[:, 0]) * self.trading_qty_max) / matrix[:, 1], -2)

            # self.symbols_signal_dict
            for i in range(len(self.probability_dict)):
                self.qty_weighted_dict[list(self.probability_dict.keys())[i]] = rounded_qty_each_dw_per_ul[i]
                # v

    def fast_offload(self):
        logger.debug("fast_offload triggered")
        try:
            self.fast_off_load()
        except Exception as e:
            print(e)
            self.send_notification(text=f"Off load failed {e} try again")

            repeat_fast_offload = GQTimer.MyTimerThread(self.fast_offload,np.random.exponential(scale=10))
            repeat_fast_offload.start()

    def cancel_all_symbol_order(self):
        for symbol, signal_class_obj in self.symbols_signal_dict.items():
            logger.debug(f"symbol:{symbol}")
            self.mOrderInterface.clear_ActiveOrdersSym(symbol, market= self.market, maxtimeout=5)



    def fast_off_load(self):

        logger.info("fast_off_load() initiated")
        for symbol, signal_class_obj in self.symbols_signal_dict.items():
            logger.debug(f"symbol:{symbol}")
            if not signal_class_obj.isSignalLive() and signal_class_obj.AutoLimitOrder:
                logger.debug(f"AutoLimitOrder exist... deleting")

                if not signal_class_obj.AutoLimitOrder is None:
                    signal_class_obj.AutoLimitOrder.delete_order(clear_all_sym=False)
                if not signal_class_obj.TimeScheduleOrder is None:
                    signal_class_obj.TimeScheduleOrder.delete_order()

                current_pos = self.Portfolio_Class.get_positions_from_connector(symbol)['q']
                if symbol in self.symbol_dict:
                    ainstr = self.symbol_dict[symbol]
                elif symbol in self.DW_spec.index:
                    pc = self.DW_spec.loc[symbol, 'PutOrCall']
                    self.add_instrument(symbol, market=self.market, product_type='eq', pc=pc)
                    ainstr = self.symbol_dict[symbol]
                else:
                    logger.debug(f"{symbol} no in dictionary nor Setsmart table skip...")
                    self.send_notification(f"{symbol} with {current_pos} is no in dictionary nor Setsmart table Cannot Send ORDER!!!")
                    continue
                tick_limit = 1

                self.Instrument.get_underlying_instrument()
                signal_class_obj:signal_class


                Best_Price_Order = self.mOrderManager.place_best_order(instrument=ainstr,
                                                                       quantity=-current_pos,
                                                                       tick_limit=tick_limit,
                                                                       signal_price=None,
                                                                       manual_order_price=None, retry_max=50,
                                                                       TG_BOT_class_instance=self.BOT_Helper,
                                                                       check_LOB_Tick_state=False,
                                                                       execute_now=True)


        # # self.Portfolio_Class.get_positions_from_deal(accountno="6502907")
        # for symbol, signal_class_obj in self.symbols_signal_dict.items():
        #     current_pos = self.Portfolio_Class.get_positions_from_connector(symbol)
        #
        #     logger.debug(f"{symbol} current pos info: {current_pos}")
        #     #
        #     tick_limit = None
        #     if np.abs(current_pos['q']) > 0:
        #         if symbol in self.symbol_dict:
        #             ainstr = self.symbol_dict[symbol]
        #         elif symbol in self.DW_spec.index:
        #             pc = self.DW_spec.loc[symbol, 'PutOrCall']
        #             self.add_instrument(symbol, market=self.market, product_type='eq', pc=pc)
        #             ainstr = self.symbol_dict[symbol]
        #         else:
        #             logger.debug(f"{symbol} no in dictionary nor Setsmart table skip...")
        #             continue


    def liquidate(self,symbol,signal_class_obj):
        target_finish_time = datetime.datetime(
                datetime.datetime.today().year,
                datetime.datetime.today().month,
                datetime.datetime.today().day,
                16,
                25,
                00,
            )
        logger.debug(f" liquidating symbol:{symbol}")
        if not signal_class_obj.isSignalLive() and signal_class_obj.AutoLimitOrder:
            logger.debug(f"AutoLimitOrder exist... deleting")
            signal_class_obj.AutoLimitOrder.delete_order(clear_all_sym=False)

            current_pos = self.Portfolio_Class.get_positions_from_connector(symbol)['q']
            if symbol in self.symbol_dict:
                ainstr = self.symbol_dict[symbol]
            elif symbol in self.DW_spec.index:
                pc = self.DW_spec.loc[symbol, 'PutOrCall']
                self.add_instrument(symbol, market=self.market, product_type='eq', pc=pc)
                ainstr = self.symbol_dict[symbol]
            else:
                logger.debug(f"{symbol} no in dictionary nor Setsmart table skip...")
                self.send_notification(
                    f"{symbol} with {current_pos} is no in dictionary nor Setsmart table Cannot Send ORDER!!!")
                return
            tick_limit = 1

            self.Instrument.get_underlying_instrument()
            signal_class_obj: signal_class

            tdelta = target_finish_time - pd.Timestamp.now()

            logger.debug(f" liquidating symbol tdelta:{tdelta}")
            if tdelta > pd.Timedelta(0):

                logger.debug(f" liquidating symbol sending time schedule Order {symbol} current_pos:{current_pos}")
                Order = self.mOrderManager.create_TimeScheduledBestOrder(instrument=symbol,
                                                                         quantity=-current_pos,
                                                                         tick_limit=tick_limit,
                                                                         signal_price=signal_class_obj.stop_loss_price,
                                                                         scale_tick_limit='expo'
                                                                         )
                signal_class_obj.TimeScheduleOrder = Order


                logger.debug(f" starting schedule Order in thread :{symbol} current_pos:{current_pos}")

                threading.Thread(target=self.execute_order_helper, args=(Order, {
                    'limit_time_in_second': tdelta.total_seconds(),
                    'seconds_preempted': 10,
                    'limit_seconds_buffer': 5,
                    'number_of_orders': 40
                })).start()
            else:
                tick_limit = 2
                logger.debug(f" liquidating {symbol} with best price order current_pos:{current_pos} with ticklimit:{tick_limit}")
                Best_Price_Order = self.mOrderManager.place_best_order(instrument=ainstr,
                                                                       quantity=-current_pos,
                                                                       tick_limit=tick_limit,
                                                                       signal_price=None,
                                                                       manual_order_price=None, retry_max=50,
                                                                       TG_BOT_class_instance=self.BOT_Helper,
                                                                       check_LOB_Tick_state=False,
                                                                       execute_now=True)

    def execute_order_helper(self,order_object,params):

        try:
            if len(params) > 0:
                order_object.trigger_execute(params)
            else:
                order_object.trigger_execute()
        # except mGQConnector.RequestTimeOutException as TimeOut:
        #     self.send_notification(text=f"{__name__} Order Request TimeOUT!! must take action immediately {TimeOut}",
        #                            subject=f"{__name__} order request timeout")
        except OrderClass.RetryExceedMax as RE:
            print(RE)
            logger.debug(f"RetryMax Exception {RE}")
            if RE.type == 1:

                # self.try_STOP_CANCEL_all()

                # self.send_notification(text=f"{__name__} RetryExceedMax type 1: unknown reason for retrying. Best Price feed data may not match market actual prices."
                #                             f" If Signal side is NONE target_qty=[0,0] please exit immediately",
                #                        subject=f"{__name__} RetryExceedMax type 1")
                raise RE
            else:
                logger.debug("Just type 0, not a big deal loop through again")
        except OrderClass.order_message_error as OE:
            if OE.get_error_type() == OrderClass.order_error.PRICE_OUTSIDE_LIMIT:
                logger.debug(f"{OE}")

                # self.try_STOP_CANCEL_all()
                # self.send_notification(text=f"{__name__} Price OUTSIDELIMIT range, please check If Signal side is NONE target_qty=[0,0] please exit immediately",
                #                        subject=f"{__name__} Orders Price OUTSIDELIMIT")
                raise OE
            elif OE.get_error_type() == OrderClass.order_error.ORDER_EXECEED_LINE_LIMIT:
                self.try_STOP_CANCEL_all()
                logger.debug(f"{OE}")
                # self.send_notification(text=f"{__name__} ORDER_EXECEED_LINE_LIMIT!!! ",
                #                        subject=f"{__name__} ORDER_EXECEED_LINE_LIMIT")

                raise OE
            elif OE.get_error_type() == OrderClass.order_error.UNKNOWN_ERROR_1:
                self.try_STOP_CANCEL_all()
                logger.debug(f"{OE}")
                # self.send_notification(text=f"{__name__} Order UNKNOWN_ERROR_1",
                #                        subject=f"{__name__} Order UNKNOWN_ERROR_1")
                raise OE
        except OrderClass.DATA_STALE as STALE_ERR:
            if STALE_ERR.stale_level >= 2:
                # self.try_STOP_CANCEL_all()
                logger.debug(f"{STALE_ERR}")
                # self.send_notification(text=f"{__name__} STALE_ERR: {STALE_ERR}",
                #                        subject=f"{__name__} STALE_ERR")
                raise STALE_ERR
        except Exception as e:
            logger.debug(f"{e}")
            # self.send_notification(text=f"{__name__} Unknown Error {e}",
            #                        subject=f"{__name__} Unknown Error")

    def add_fake_signal(self, symbol, quantity, limit_buy_price, stop_profit_price, stop_loss_price):

        self.symbols_signal_dict[symbol] = signal_class(symbol=symbol, limit_buy_price=limit_buy_price,
                                                        quantity=quantity,
                                                        stop_profit_price=stop_profit_price,
                                                        stop_loss_price=stop_loss_price)

    def start_execution(self):
        print("Start execution")
        logger.info("start_execution")

        # self.Portfolio_Class.get_positions_from_deal(accountno="6502907")

        #check whether today's signal has been

        for symbol, signal_class_obj in self.symbols_signal_dict.items():

            signal_class_obj:signal_class

            logger.info(f"\t{symbol} {signal_class_obj}")
            if not signal_class_obj.isSignalLive() and self._signal_update:
                #Signal _signal_update is to tell that evening signal has been read
                logger.info(f"\t{symbol} {signal_class_obj} signal is not live and update is True... liquidate")
                self.liquidate(symbol, signal_class_obj)
            else:
                OrderObj = self.send_Stop_send_Limit_with_bracket_order(symbol=symbol, signal_obj=signal_class_obj,tick_limit=None)
                if OrderObj:
                    logger.debug(f"{symbol} order compelte")
                    signal_class_obj.add_order_class(OrderObj)

    def add_instrument(self, symbol, market, product_type, pc=""):
        if symbol in self.DW_spec.index:  # making sure it in set smart
            print(f"creating class for:{symbol}")

            logger.debug(f"adding Instrument:{symbol} but first check if UL is already added")

            ul_sym = self.DW_spec.loc[symbol, 'UL']

            if not ul_sym in self.symbol_dict:
                logger.debug("UL already init on RTD {}")
                mInstr = Instrument.Instrument(symbol=ul_sym, market=market, product_class=product_type, sub_type="eq")
                mInstr.set_RTD_Manager(self.RTD_Manager)
                mInstr.init_price_feeds(product_type=product_type, symbol=ul_sym, sub_type='orderbook',
                                        init_from_beginning=False,StartStream=False)
                mInstr.init_price_feeds(product_type=product_type, symbol=ul_sym, sub_type='tick',
                                        init_from_beginning=False,StartStream=False)
                self.symbol_dict[ul_sym] = mInstr

            DW_mInstr = Instrument.DW_instrument(symbol=symbol, market=market, product_class=product_type,
                                                 putcall=pc, sub_type="eq")
            DW_mInstr.set_RTD_Manager(self.RTD_Manager)
            DW_mInstr.init_price_feeds(product_type=product_type, symbol=symbol, sub_type='orderbook',
                                       init_from_beginning=False,StartStream=False)
            DW_mInstr.init_price_feeds(product_type=product_type, symbol=symbol, sub_type='tick',
                                       init_from_beginning=False,StartStream=False)
            DW_mInstr.set_underlying_instrument(self.symbol_dict[ul_sym])
            self.symbol_dict[symbol] = DW_mInstr

        # mInstr.set_RTD_class(aRTD)
        # self.list_of_trading_instruments.append(symbol)


    def send_Stop_send_Limit_with_bracket_order(self, symbol, signal_obj, tick_limit):
        target_quantity = signal_obj.quantity
        order_price = signal_obj.limit_buy_price
        stop_profit_trigger_price = signal_obj.trigger_stop_profit_price
        stop_profit_price = signal_obj.stop_profit_price
        trigger_price = signal_obj.limit_buy_price
        stop_loss_trigger_price = signal_obj.trigger_stop_loss_price
        stop_loss_price = signal_obj.stop_loss_price
        lower_bound_price = signal_obj.lower_bound_price


        stop_condition = "<="
        trigger_price_type = "A"

        logger.info(
            f"send_Stop_Limit_with_bracket_order for: symbol:{symbol} target_quantity:{target_quantity} order_price:{order_price} "
            f"stop_profit_trigger_price:{stop_profit_trigger_price} stop_profit_price:{stop_profit_price} "
            f"stop_loss_trigger_price:{stop_loss_trigger_price} stop_loss_price:{stop_loss_price} "
            f"stop_condition:{stop_condition} trigger_price_type:{trigger_price_type}"
        )


        self.mOrderInterface.clear_ActiveOrdersSym(symbol, market=self.market, maxtimeout=10)

        if not signal_obj.isSignalLive():
            current_position = signal_obj._current_qty
            target_quantity = current_position
            # target_quantity = self.trading_Qty
        else:
            current_position = self.Portfolio_Class.get_positions_from_connector(symbol)['q']

        ainstr = self.symbol_dict[symbol]
        # current_position = self.Portfolio_Class.get_position_of_symbol(symbol)
        # current_position = 0
        logger.info(f"current existing position:{current_position} and target_quantity: {target_quantity}")

        if (current_position) == 0:
            if self.check_todays_deal(symbol) or not signal_obj.isSignalLive():
                # Already traded today, skip
                # Already passed limit entry time
                logger.debug(f"{symbol} traded today, skip Already passed limit entry time: {signal_obj._signal_entry_date}")
                return

        try:
            '''Deleting all existing algo orders within the process'''
            list_of_order_class = self.mOrderManager.get_symbols_order_collection(symbol)
            for a_order in list_of_order_class:
                a_order.delete_order()
                logger.info(f"Deleted order {a_order}")

            # SmartBracketOrder = self.mOrderManager.create_trigger_limit_with_auto_bracket_on_full(instrument=ainstr,
            #                                                                                       target_quantity=target_quantity,
            #                                                                                       current_quantity=current_position,
            #                                                                                       order_price=order_price,
            #                                                                                       signal_price=None,
            #                                                                                       trigger_price=trigger_price,
            #                                                                                       stop_condition=stop_condition,
            #                                                                                       stop_profit_trigger_price=stop_profit_trigger_price,
            #                                                                                       stop_profit_price=stop_profit_price,
            #                                                                                       stop_loss_trigger_price=stop_loss_trigger_price,
            #                                                                                       stop_loss_price=stop_loss_price,
            #                                                                                       tick_limit=tick_limit,
            #                                                                                       manual_order_price=None,
            #                                                                                       TG_BOT_class_instance=self.BOT_Helper,
            #                                                                                       check_LOB_Tick_state=False
            #
            #                                                                                       )

            SmartBracketOrder = self.mOrderManager.create_trigger_limit_with_auto_bracket_on_full(instrument=ainstr,
                                                                                                  order_price=order_price,
                                                                                                  target_quantity=target_quantity,
                                                                                                  trigger_price=trigger_price,
                                                                                                  stop_condition=stop_condition,
                                                                                                  current_quantity=current_position,
                                                                                                  lower_bound_price=lower_bound_price,
                                                                                                  signal_price=None,
                                                                                                  trigger_price_type=trigger_price_type,
                                                                                                  stop_profit_trigger_price=stop_profit_trigger_price,
                                                                                                  stop_profit_price=stop_profit_price,
                                                                                                  stop_loss_trigger_price=stop_loss_trigger_price,
                                                                                                  stop_loss_price=stop_loss_price,
                                                                                                  tick_limit=tick_limit,
                                                                                                  manual_order_price=None,
                                                                                                  TG_BOT_class_instance=self.BOT_Helper,
                                                                                                  check_LOB_Tick_state=False

                                                                                                  )
            return SmartBracketOrder
            # self.

        except OrderClass.OrderInvalid as e:
            logger.debug(f"Order Invalid Type: {e.type}")
            if e.type == 0:
                return None
            elif e.type == 1:
                self.send_notification(text=str(
                    e) + f" send_Limit_with_bracket_order for: symbol:{symbol} target_quantity:{target_quantity} order_price:{order_price} "
                         f"stop_profit_trigger_price:{stop_profit_trigger_price} stop_profit_price:{stop_profit_price} "
                         f"stop_loss_trigger_price:{stop_loss_trigger_price} stop_loss_price:{stop_loss_price}")

                if current_position > 0:
                    '''
                    Exit
                    '''
                    try:
                        Best_Price_Order = self.mOrderManager.place_best_order(instrument=ainstr,
                                                                               quantity=-current_position,
                                                                               tick_limit=tick_limit,
                                                                               signal_price=None,
                                                                               manual_order_price=None, retry_max=5,
                                                                               TG_BOT_class_instance=self.BOT_Helper,
                                                                               check_LOB_Tick_state=False)

                    except OrderClass.RetryExceedMax as e:
                        if e.type == 1:
                            print("Failed to sell after repeating... skipping")
                            pass
                return None
            elif e.type == 2:
                self.send_notification(text=str(
                    e) + f" send_Limit_with_bracket_order for: symbol:{symbol} target_quantity:{target_quantity} order_price:{order_price} "
                         f"stop_profit_trigger_price:{stop_profit_trigger_price} stop_profit_price:{stop_profit_price} "
                         f"stop_loss_trigger_price:{stop_loss_trigger_price} stop_loss_price:{stop_loss_price}")

                if current_position > 0:
                    '''
                    Exit
                    '''
                    Best_Price_Order = self.mOrderManager.place_best_order(instrument=ainstr,
                                                                           quantity=-current_position,
                                                                           tick_limit=tick_limit,
                                                                           signal_price=None,
                                                                           manual_order_price=None, retry_max=10,
                                                                           TG_BOT_class_instance=self.BOT_Helper,
                                                                           check_LOB_Tick_state=False)

                return None
            pass
        except OrderClass.order_message_error as OE:
            if OE.get_error_type() == OrderClass.order_error.UNABLE_TO_FIND_PRODUCT:
                logger.debug(f"{OE}")
                self.send_notification(text=str(
                    OE) + f" send_Limit_with_bracket_order for: symbol:{symbol} target_quantity:{target_quantity} order_price:{order_price} "
                          f"stop_profit_trigger_price:{stop_profit_trigger_price} stop_profit_price:{stop_profit_price} "
                          f"stop_loss_trigger_price:{stop_loss_trigger_price} stop_loss_price:{stop_loss_price}")

                return None
            else:
                raise OE
        except OrderClass.LOB_EMPTY_RETRY_EXCEED as OE:
            logger.debug("LOB_EMPTY_RETRY_EXCEED Skipping")
        except Exception as e:
            logger.debug(f"Crashed {e}")
            raise e


    def send_Limit_with_bracket_order(self, symbol, signal_obj, tick_limit):

        target_quantity = signal_obj.quantity
        order_price = signal_obj.limit_buy_price
        stop_profit_trigger_price = signal_obj.trigger_stop_profit_price
        stop_profit_price = signal_obj.stop_profit_price
        stop_loss_trigger_price = signal_obj.trigger_stop_loss_price
        stop_loss_price = signal_obj.stop_loss_price
        logger.info(
            f"send_Limit_with_bracket_order for: symbol:{symbol} target_quantity:{target_quantity} order_price:{order_price} "
            f"stop_profit_trigger_price:{stop_profit_trigger_price} stop_profit_price:{stop_profit_price} "
            f"stop_loss_trigger_price:{stop_loss_trigger_price} stop_loss_price:{stop_loss_price}")


        self.mOrderInterface.clear_ActiveOrdersSym(symbol, market=self.market, maxtimeout=10)

        if not signal_obj.isSignalLive():
            current_position = signal_obj._current_qty
            target_quantity = current_position
            # target_quantity = self.trading_Qty
        else:
            current_position = self.Portfolio_Class.get_positions_from_connector(symbol)['q']

        ainstr = self.symbol_dict[symbol]
        # current_position = self.Portfolio_Class.get_position_of_symbol(symbol)
        # current_position = 0
        logger.info(f"current existing position:{current_position} and target_quantity: {target_quantity}")

        if (current_position) == 0:
            if self.check_todays_deal(symbol) or not signal_obj.isSignalLive():
                # Already traded today, skip
                # Already passed limit entry time
                logger.debug(f"{symbol} traded today, skip Already passed limit entry time: {signal_obj._signal_entry_date}")
                return

        try:
            '''Deleting all existing algo orders within the process'''
            list_of_order_class = self.mOrderManager.get_symbols_order_collection(symbol)
            for a_order in list_of_order_class:
                a_order.delete_order()
                logger.info(f"Deleted order {a_order}")

            SmartBracketOrder = self.mOrderManager.create_limit_with_auto_bracket_on_full(instrument=ainstr,
                                                                                          target_quantity=target_quantity,
                                                                                          current_quantity=current_position,
                                                                                          order_price=order_price,
                                                                                          signal_price=None,
                                                                                          stop_profit_trigger_price=stop_profit_trigger_price,
                                                                                          stop_profit_price=stop_profit_price,
                                                                                          stop_loss_trigger_price=stop_loss_trigger_price,
                                                                                          stop_loss_price=stop_loss_price,
                                                                                          tick_limit=tick_limit,
                                                                                          manual_order_price=None,
                                                                                          TG_BOT_class_instance=self.BOT_Helper,
                                                                                          check_LOB_Tick_state=False)


            return SmartBracketOrder
            # self.

        except OrderClass.OrderInvalid as e:
            logger.debug(f"Order Invalid Type: {e.type}")
            if e.type == 0:
                return None
            elif e.type == 1:
                self.send_notification(text=str(
                    e) + f" send_Limit_with_bracket_order for: symbol:{symbol} target_quantity:{target_quantity} order_price:{order_price} "
                         f"stop_profit_trigger_price:{stop_profit_trigger_price} stop_profit_price:{stop_profit_price} "
                         f"stop_loss_trigger_price:{stop_loss_trigger_price} stop_loss_price:{stop_loss_price}")

                if current_position > 0:
                    '''
                    Exit
                    '''
                    try:
                        Best_Price_Order = self.mOrderManager.place_best_order(instrument=ainstr,
                                                                               quantity=-current_position,
                                                                               tick_limit=tick_limit,
                                                                               signal_price=None,
                                                                               manual_order_price=None, retry_max=5,
                                                                               TG_BOT_class_instance=self.BOT_Helper,
                                                                               check_LOB_Tick_state=False)

                    except OrderClass.RetryExceedMax as e:
                        if e.type == 1:
                            print("Failed to sell after repeating... skipping")
                            pass
                return None
            elif e.type == 2:
                self.send_notification(text=str(
                    e) + f" send_Limit_with_bracket_order for: symbol:{symbol} target_quantity:{target_quantity} order_price:{order_price} "
                         f"stop_profit_trigger_price:{stop_profit_trigger_price} stop_profit_price:{stop_profit_price} "
                         f"stop_loss_trigger_price:{stop_loss_trigger_price} stop_loss_price:{stop_loss_price}")

                if current_position > 0:
                    '''
                    Exit
                    '''
                    Best_Price_Order = self.mOrderManager.place_best_order(instrument=ainstr,
                                                                           quantity=-current_position,
                                                                           tick_limit=tick_limit,
                                                                           signal_price=None,
                                                                           manual_order_price=None, retry_max=10,
                                                                           TG_BOT_class_instance=self.BOT_Helper,
                                                                           check_LOB_Tick_state=False)

                return None
            pass
        except OrderClass.order_message_error as OE:
            if OE.get_error_type() == OrderClass.order_error.UNABLE_TO_FIND_PRODUCT:
                logger.debug(f"{OE}")
                self.send_notification(text=str(
                    OE) + f" send_Limit_with_bracket_order for: symbol:{symbol} target_quantity:{target_quantity} order_price:{order_price} "
                          f"stop_profit_trigger_price:{stop_profit_trigger_price} stop_profit_price:{stop_profit_price} "
                          f"stop_loss_trigger_price:{stop_loss_trigger_price} stop_loss_price:{stop_loss_price}")

                return None
            else:
                raise OE
        except OrderClass.LOB_EMPTY_RETRY_EXCEED as OE:
            logger.debug("LOB_EMPTY_RETRY_EXCEED Skipping")
        except Exception as e:
            logger.debug(f"Crashed {e}")
            raise e

    def send_notification(self, text):

        # outlook_email.send_email(subject, message=text)

        bot.send_message(text=text, chat_id=group_id)
        # pass
        '''

        Telegram mesage here
        '''

    def get_time_to_exit_position_FAST(self):
        ending_dt = datetime.datetime(
            datetime.datetime.today().year,
            datetime.datetime.today().month,
            datetime.datetime.today().day,
            16,
            25,
            00,
        )

        next_ts_timedelta = ending_dt - pd.Timestamp.now()
        print(f"time_to_exit_position at {ending_dt} => {next_ts_timedelta.total_seconds()}")
        logger.debug(f"time_to_exit_position at {ending_dt} => {next_ts_timedelta.total_seconds()}")
        if next_ts_timedelta.total_seconds() > 0:
            repeat_fast_offload = GQTimer.MyTimerThread(self.fast_offload,next_ts_timedelta.total_seconds())
            repeat_fast_offload.start()


    def get_close_app_timer(self):
        ending_dt = datetime.datetime(
            datetime.datetime.today().year,
            datetime.datetime.today().month,
            datetime.datetime.today().day,
            16,
            45,
            00,
        )

        next_ts_timedelta = ending_dt - pd.Timestamp.now()
        print(f"time_to_close_app at {ending_dt} => {next_ts_timedelta.total_seconds()}")
        logger.debug(f"time_to_close_app at {ending_dt} => {next_ts_timedelta.total_seconds()}")
        if next_ts_timedelta.total_seconds() > 0:

            close_app_timer = GQTimer.MyTimerThread(self.stop_algo,next_ts_timedelta.total_seconds())
            close_app_timer.start()

    def morning_start_timer(self):


        next_ts_timedelta = self.morning_session_time - pd.Timestamp.now()
        print(f"time_to_run at {self.morning_session_time } => {next_ts_timedelta.total_seconds()}")
        logger.debug(f"time_to_run at {self.morning_session_time } => {next_ts_timedelta.total_seconds()}")
        if next_ts_timedelta.total_seconds() > 0:
            MorningStartTimer = GQTimer.MyTimerThread(self.morning_process,next_ts_timedelta.total_seconds())
            MorningStartTimer.start()
        else:
            if pd.Timestamp.now() >= self.morning_session_time and pd.Timestamp.now() < (self.new_signal_time -pd.Timedelta(10,'min')) :
                logger.debug(
                    f"after initial opening_auction_time_event time but still new signal arrival {pd.Timestamp.now()}")
                self.morning_process()

    def try_read_signal_timer(self):
        next_ts_timedelta =  self.new_signal_time - pd.Timestamp.now()
        print(f"time_to_run at { self.new_signal_time } => {next_ts_timedelta.total_seconds()}")
        logger.debug(f"time_to_run at { self.new_signal_time } => {next_ts_timedelta.total_seconds()}")
        if next_ts_timedelta.total_seconds() > 0:
            Try_read_Signal_Timer = GQTimer.MyTimerThread(self.try_read_signal_execute,wait_time=next_ts_timedelta.total_seconds())
            Try_read_Signal_Timer.start()
        else:
            self.try_read_signal_execute()
            # if pd.Timestamp.now() <= (self.closing_auction_time_start   -pd.Timedelta(5,'min')):
            #     logger.debug(
            #         f"after initial opening_auction_time_event time but still new signal arrival {pd.Timestamp.now()}")
            #     self.try_read_signal_execute()

    def check_signals_ready_as_expected(self):

        # expected_signal_date = deploy.get_next_trading_date(self.run_date)
        expected_signal_date = self.run_date

        expected_path = os.path.join(deploy.prediction_path,f"{expected_signal_date.date()}")
        if os.path.isdir(expected_path ):
            list_of_files_sym = []
            for file in os.listdir(expected_path):
                f_symbols = os.path.splitext(file)[0]
                list_of_files_sym.append(f_symbols)


            logger.debug(f"list of symbols found: {list_of_files_sym}")
            matching_syms = np.intersect1d(expected_symbol, list_of_files_sym)
            logger.debug(f"matching_syms {matching_syms}")
            if len(matching_syms)>0:
                print("matching_syms")
                print(matching_syms)
                return True
            else:
                return False
        else:
            return False

    def try_read_signal_execute(self):
        # raise NotImplementedError("")
        print("try_read_signal_execute")
        logger.debug("try_read_signal_execute")
        max_it = 30
        count = 0
        while (not self.check_signals_ready_as_expected()):
            if count > max_it:
                logger.debug(f"Signal NOT FOUND at: {pd.Timestamp.now()}")
                return
                # raise Exception(f"Signal NOT FOUND at: {pd.Timestamp.now()}")

            time.sleep(1)
            count+=1


        logger.debug("new signal Found!")
        self.generate_dw_signal(self.run_date,tick_size_multiple=self.tick_size_multiple,start_minute=0,read_cache=False)
        self.create_signal_class()
        self._signal_update = True
        self.start_execution()


    def schedule_session_take_profit(self, hours, minutes, seconds=0):
        ending_dt = datetime.datetime(
            datetime.datetime.today().year,
            datetime.datetime.today().month,
            datetime.datetime.today().day,
            hours,
            minutes,
            seconds,
        )

        next_ts_timedelta = ending_dt - pd.Timestamp.now()
        print(f"Time To take profit at {ending_dt} => {next_ts_timedelta.total_seconds()}")
        logger.debug(f"Time To take profit at {ending_dt} => {next_ts_timedelta.total_seconds()}")
        if next_ts_timedelta.total_seconds() > 0:
            Try_Take_Profit = GQTimer.MyTimerThread(self.try_take_profit,wait_time=next_ts_timedelta.total_seconds())
            Try_Take_Profit.start()

    def try_take_profit(self):
        for symbol, signal_class_obj in self.symbols_signal_dict.items():
            logger.debug(f"symbol:{symbol}")
            if signal_class_obj.AutoLimitOrder:
                logger.debug(f"AutoLimitOrder exist... run take profit")
                signal_class_obj.AutoLimitOrder.take_profit(percent_exit=0.1)


    def ClosingAuctionAction(self):

        next_ts_timedelta = self.closing_auction_time_event - pd.Timestamp.now()
        print(f"schedule ClosingAuctionAction at {self.closing_auction_time_event} => {next_ts_timedelta.total_seconds()}")
        logger.debug(f"schedule ClosingAuctionAction at {self.closing_auction_time_event} => {next_ts_timedelta.total_seconds()}")
        if next_ts_timedelta.total_seconds() > 0:
            ClosingAuctionActionTimer = GQTimer.MyTimerThread(self.place_limit_order_at_take_profit,wait_time=next_ts_timedelta.total_seconds())
            ClosingAuctionActionTimer.start()
        else:
            logger.debug("Closing Auction Event passed... do nothing")
            pass



    def OpenningAuctionTakeProfit(self):

        next_ts_timedelta = self.opening_auction_time_event - pd.Timestamp.now()
        print(f"schedule OpenningAuctionTakeProfit at {self.opening_auction_time_event} => {next_ts_timedelta.total_seconds()}")
        logger.debug(f"schedule OpenningAuctionTakeProfit at {self.opening_auction_time_event} => {next_ts_timedelta.total_seconds()}")
        if next_ts_timedelta.total_seconds() > 0:

            OpenningAuctionTimer = GQTimer.MyTimerThread(self.place_limit_order_at_take_profit,next_ts_timedelta.total_seconds())
            OpenningAuctionTimer.start()
        else:
            if pd.Timestamp.now() > self.opening_auction_time_event and pd.Timestamp.now() < self.morning_session_time:

                logger.debug(f"after initial OpenningAuctionTakeProfit time but still before morning session {pd.Timestamp.now() }" )
                self.place_limit_order_at_take_profit()


    # def stop_algo_Timer(self):
    #
    #     next_ts_timedelta = self._stop_algo_time_event - pd.Timestamp.now()
    #     print(
    #         f"schedule Stop Algo at at {self._stop_algo_time_event} => {next_ts_timedelta.total_seconds()}")
    #     logger.debug(
    #         f"schedule Stop Algo at {self._stop_algo_time_event} => {next_ts_timedelta.total_seconds()}")
    #     if next_ts_timedelta.total_seconds() > 0:
    #         stopAlgoTimer = GQTimer.MyTimerThread(self.stop_algo,next_ts_timedelta.total_seconds())
    #         stopAlgoTimer.start()
    #     else:
    #         logger.debug("Closing Auction Event passed... do nothing")
    #         pass

    def place_limit_order_at_take_profit(self):
        print("place_limit_order_at_take_profit")
        logger.debug("place_limit_order_at_take_profit")
        for symbol, signal_class_obj in self.symbols_signal_dict.items():
            # if not signal_class_obj.isSignalLive():
            #     current_position = signal_class_obj._current_qty
            # else:
            current_position = self.Portfolio_Class.get_positions_from_connector(symbol)['q']
            # current_position = 100

            logger.debug(f"{symbol} current_position:{current_position}")

            list_of_order_class = self.mOrderManager.get_symbols_order_collection(symbol)
            for a_order in list_of_order_class:
                if a_order.status == 2 or a_order.status == 4:
                    logger.debug(f"Algoorder exists: {a_order} deleting")
                    a_order.delete_order()

            logger.debug(f"Clear All Orders for symbols")
            self.mOrderInterface.clear_ActiveOrdersSym(symbol, market= self.market, maxtimeout=10)

            logger.debug(f"placing LO to take profit during auction")
            limit_order = self.mOrderManager.place_order(market=self.market, symbol=symbol,
                                                              quantity=-current_position,
                                                              order_price=signal_class_obj.stop_profit_price,
                                                               TG_BOT_class_instance=self.BOT_Helper,
                                                              otype="LIMIT")


    def morning_process(self):
        '''Start Algo on previous, existing signal e.i. start bracket order'''

        logger.debug(f"morning_process")
        self.start_execution()

    def stop_algo(self):
        print("Exiting Algo")
        self.cancel_all_symbol_order()
        self.RTD_Manager.stop_RTD_classes()
        logger.debug("Exiting Algo")
        self.close()
        QApplication.instance().quit()
        # self.quit()
        # sys.exit(0)

    def on_button1_clicked(self):
        self.generate_dw_signal(self.run_date, tick_size_multiple=self.tick_size_multiple)
        print("Button1 done")

    def on_button2_clicked(self):
        self.create_signal_class()
        print("Button2 done")

    def on_button3_clicked(self):
        self.start_execution()
        print("Button3 done")

    def on_button4_clicked(self):
        self.fast_offload()

    def on_button5_clicked(self):
        print("Reading signal from Cache")
        self.generate_dw_signal(self.run_date, read_cache=True, tick_size_multiple=self.tick_size_multiple)

    def on_button6_clicked(self):
        # self.run_get_deals()
        self.stop_algo()


    def run_get_deals(self):
        reply = self.mOrderInterface.get_todays_deal_by_symbol(symbol="JMART19P2304A", market="XBKK")
        print(reply)
        logger.debug(f"run_get_deals: {reply}")

    def check_todays_deal(self, symbol):
        buy_deal_exist = False
        net_fill = 0
        reply = self.mOrderInterface.get_todays_deal_by_symbol(symbol=symbol, market="XBKK")
        if reply['reply'] == "valid":
            for trade_data in reply['msg']['data']:
                pass

                side_m = -1 if trade_data['s'] == "SELL" else 1
                net_fill += trade_data['q'] * side_m

                if side_m > 0:
                    buy_deal_exist = True

        if net_fill != 0:
            raise
        else:
            return buy_deal_exist



    def on_send_order_clicked(self):

        try:
            symbol = self.SendingOrderTable.item(0, self.table_columns_map['symbol']).text()
            price = float(self.SendingOrderTable.item(0, self.table_columns_map['Price']).text())
            qty = float(self.SendingOrderTable.item(0, self.table_columns_map['Qty']).text())

            reply = self.mOrderInterface.place_order(symbol, self.market, price=price,
                                                     quantity=qty, otype="LIMIT",
                                                     blocking=True)
            print(f"Order Reply {reply}")
            logger.debug(f"Order Reply {reply}")

            self.order_reply_label.setText(str(reply))
            self.SendingOrderTable.clear()

        except Exception as e:
            print(f"Send Order Fail {e}")
            self.order_reply_label.setText( f"Send Order Fail {e}")
            logger.debug(f"Send Order Fail {e}")


    def start_UI_loop(self):
        self.dst_loop_thread = threading.Thread(target=self.update_table_ui_timer_loop,daemon=True)
        self.dst_loop_thread.start()

    def update_position_table(self):
        current_positions = self.Portfolio_Class.get_positions()

        # self.PositionTable.clear()
        self.PositionTable.setRowCount(len(current_positions))
        row_int = 0
        for sym, posinfo in current_positions.items():
            self.PositionTable.setItem(
                row_int,
                self.position_table_columns_dict['symbol'],
                QTableWidgetItem(sym)
            )
            self.PositionTable.setItem(
                row_int,
                self.position_table_columns_dict['Qty'],
                pyqt_utils.TableFloatItemSetHelper(Value=posinfo['q'])
            )
            row_int+=1


    def update_DST_order_table(self):
        All_Orderrs = self.mOrderInterface.getAllOrders()['data']

        # self.DST_Order_Table.clear()
        self.DST_Order_Table.setRowCount(len(All_Orderrs))
        row_int = 0
        for posinfo in All_Orderrs:
            self.DST_Order_Table.setItem(
                row_int,
                self.DST_Order_Table_columns_dict['symbol'],
                QTableWidgetItem(posinfo['symbol'])
            )
            self.DST_Order_Table.setItem(
                row_int,
                self.DST_Order_Table_columns_dict['status'],
                QTableWidgetItem(posinfo['status'])
            )
            self.DST_Order_Table.setItem(
                row_int,
                self.DST_Order_Table_columns_dict['iOrdNo'],
                QTableWidgetItem(posinfo['oid'])
            )
            self.DST_Order_Table.setItem(
                row_int,
                self.DST_Order_Table_columns_dict['price'],
                pyqt_utils.TableFloatItemSetHelper(Value=posinfo['p'])
            )
            self.DST_Order_Table.setItem(
                row_int,
                self.DST_Order_Table_columns_dict['Qty'],
                pyqt_utils.TableFloatItemSetHelper(Value=posinfo['q'])
            )
            self.DST_Order_Table.setItem(
                row_int,
                self.DST_Order_Table_columns_dict['side'],
                QTableWidgetItem(posinfo['s'])
            )
            row_int+=1

    def update_table_ui_timer_loop(self):
        self.update_table_loop = True

        while self.update_table_loop:

            self.update_position_table()
            self.update_DST_order_table()

            time.sleep(5)


    def populate_signal_table(self):
        '''
        self.signal_table_column =['symbol','signal_datetime','signal',
        'ul_stoploss_price','order_price','dw_stoploss_price',
        'dw_take_profit_price']


        :return:
        '''
        # self.SignalTable.clear()
        logger.debug(f"populate_signal_table len signal dict{len(self.symbols_signal_dict)} ")
        logger.debug(self.symbols_signal_dict)

        self.SignalTable.setRowCount(len(self.symbols_signal_dict))
        row_int = 0
        for sym,signal_obj in self.symbols_signal_dict.items():
            signal_obj: signal_class
            self.SignalTable.setItem(
                row_int,
                self.signal_table_column_dict['symbol'],
                QTableWidgetItem(sym)
            )
            self.SignalTable.setItem(
                row_int,
                self.signal_table_column_dict['signal_datetime'],
                QTableWidgetItem(signal_obj._signal_entry_date.strftime("%Y-%m-%d"))
            )
            self.SignalTable.setItem(
                row_int,
                self.signal_table_column_dict['signal_datetime'],
                QTableWidgetItem(signal_obj._signal_entry_date.strftime("%Y-%m-%d"))
            )
            self.SignalTable.setItem(
                row_int,
                self.signal_table_column_dict['order_price'],
                pyqt_utils.TableFloatItemSetHelper(Value=signal_obj.limit_buy_price)
            )

            self.SignalTable.setItem(
                row_int,
                self.signal_table_column_dict['ul_stoploss_price'],
                pyqt_utils.TableFloatItemSetHelper(Value=signal_obj.trigger_stop_loss_price)
            )

            self.SignalTable.setItem(
                row_int,
                self.signal_table_column_dict['dw_stoploss_price'],
                pyqt_utils.TableFloatItemSetHelper(Value=signal_obj.stop_loss_price)
            )

            self.SignalTable.setItem(
                row_int,
                self.signal_table_column_dict['dw_take_profit_price'],
                pyqt_utils.TableFloatItemSetHelper(Value=signal_obj.stop_profit_price)
            )

            row_int+=1

        logger.debug(f"populate_signal_table complete")


    def get_all_DST_orders(self):

        self.LiveOrderTable.clear()


if __name__ == '__main__':
    """
    How to handle on restart
    On restart (how to know?)

    For each symbols in the position.
    1) get position
    2) if not fully filled limit order auto stop
    2a) For already filled position create stop 
    """
    bot_config = yaml.load(open('process_conf.yaml'), Loader=Loader)

    telegram_group = "telegramTest"
    bot = telegram.Bot(bot_config[telegram_group]['api'])
    group_id = bot_config[telegram_group]['group_id']
    group_name = bot_config[telegram_group]['group_name']

    app = QApplication(sys.argv)
    Appconfig = configparser.ConfigParser()
    Appconfig.read("AppConfig.ini")

    # endpoint = Appconfig['ENDPOINTS']['PROD']

    TG_BOT = TelegramBOT.TG_BOT_Helper(bot)
    TG_BOT.add_group(group_name, group_id)
    TG_BOT.set_default_group_name(group_name)
    TG_BOT.start_loop()
    TG_BOT.push_message_to_queue("Starting DW")

    mMain = main(Appconfig,bot=TG_BOT,UAT=True)

    sys.exit(app.exec_())

    # mMain.try_offload()


