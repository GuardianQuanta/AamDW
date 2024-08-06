import sys, os
import threading
import pandas as pd
import numpy as np


from GQDatabaseSQL import REDIS_DB, SETSMART
import time

sys.path.append("C:\\gitrepos\\python\\GQOrderInterfacePy")
from GQOrderInterfacePy import GQ_OMS_Interface
from GQOrderInterfacePy.Orders import OrderClass
import configparser
from TFEX_Utils import TFEX_Utils
# Press the green button in the gutter to run the script.
import logging
from myGQLib.Classes import outlook_email, Instrument

import telegram
import yaml
from yaml import CLoader as Loader
import datetime


DB_Config = configparser.ConfigParser()
DB_Config.read(os.path.join("C:\Config", "DatabaseConfig.ini"))
if __name__ =="__main__":


    # GQ_OMS_Interface.interface
    #
    # mConnector = GQ_OMS_Interface.gqwebsocket.GQConnector.GQConnector(self.config['WS_ENDPOINTS']['UAT_CLOUD'])
    market = "XBKK"


    param = {
        'account_no': '0209001'
    }

    config = yaml.load(open('.\\config\\config_uat.yaml'), Loader=Loader)

    dst_config = yaml.load(open('.\\config\\dst_config.yaml'), Loader=Loader)
    dst_client = GQ_OMS_Interface.interface.DST_Interface(account_info=param,config=config,dst_config=dst_config)

    sub = dict(DB_Config['REDIS_PROD_MME_CLOUD'])
    # sub = dict(DB_Config["REDIS_UAT"])
    sub['db'] = 0
    sub['message_type'] = 'MME2'
    pub = dict(DB_Config['REDIS_PROD_MME_CLOUD'])
    pub['db'] = 0
    pub['message_type'] = 'MME2'
    RTD_Manager = REDIS_DB.RTD_Manager(sub=sub, pub=pub,
                                            verbose=False,
                                            block_writes=True)
    mOrderManager = OrderClass.OrdersManager(dst_client, RTD_Manager)

    ul_sym = "PTT"
    product_type = "eq"
    mInstr = Instrument.Instrument(symbol=ul_sym, market=market, product_class=product_type, sub_type="eq")
    mInstr.set_RTD_Manager(RTD_Manager)
    mInstr.init_price_feeds(product_type=product_type, symbol=ul_sym, sub_type='orderbook',
                            init_from_beginning=False,StartStream=False)
    mInstr.init_price_feeds(product_type=product_type, symbol=ul_sym, sub_type='tick',
                            init_from_beginning=False,StartStream=False)
    # symbol_dict[ul_sym] = mInstr

    for i in range(10):
        start_t = pd.Timestamp.now()
        print(f"Creating: {start_t}")
        Best_Price_Order = mOrderManager.place_best_order(instrument=mInstr,
                                                               quantity=100,
                                                               tick_limit=None,
                                                               signal_price=None,
                                                               manual_order_price=None, retry_max=50,
                                                               TG_BOT_class_instance=None,
                                                               check_LOB_Tick_state=False,
                                                                mean_wait_time=8
                                                            )
        end_t = pd.Timestamp.now()

        print(f"Complete: {end_t}  { (end_t - start_t).total_seconds()} seconds")
        print(Best_Price_Order)