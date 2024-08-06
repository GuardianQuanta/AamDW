
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

    config = yaml.load(open('config.yaml'), Loader=Loader)

    dst_config = yaml.load(open('dst_config.yaml'), Loader=Loader)
    dst_client = GQ_OMS_Interface.interface.DST_Interface(account_info=param,config=config,dst_config=dst_config)
    dst_client.start_deal_loop_thread()
    # sub = dict(DB_Config['REDIS_PROD_MME_CLOUD'])
    sub = dict(DB_Config['REDIS_UAT_CLOUD'])
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



    dw_sym = "PTT"
    product_type = "eq"
    dw_mInstr = Instrument.DW_instrument(symbol=dw_sym, market=market, product_class=product_type, sub_type="eq",putcall='c')
    dw_mInstr.set_RTD_Manager(RTD_Manager)
    dw_mInstr.init_price_feeds(product_type=product_type, symbol=ul_sym, sub_type='orderbook',
                            init_from_beginning=False,StartStream=False)
    dw_mInstr.init_price_feeds(product_type=product_type, symbol=ul_sym, sub_type='tick',
                            init_from_beginning=False,StartStream=False)

    dw_mInstr.set_underlying_instrument(mInstr)

    # symbol_dict[ul_sym] = mInstr

    # mOrderManager.create_limit_with_auto_bracket_on_full()

    SmartBracketOrder = mOrderManager.create_limit_with_auto_bracket_on_full(instrument=dw_mInstr,
                                                         target_quantity=400,
                                                         current_quantity=0,
                                                         order_price=34.5,
                                                         signal_price=None,
                                                         stop_profit_trigger_price=35,
                                                        stop_profit_price=35,
                                                         stop_loss_trigger_price=33,
                                                         stop_loss_price=33,
                                                         tick_limit=None,manual_order_price=None,
                                                         TG_BOT_class_instance=None,check_LOB_Tick_state=False)
        # pass

    # print()

    while True:
        time.sleep(5)

    # print(Best_Price_Order)