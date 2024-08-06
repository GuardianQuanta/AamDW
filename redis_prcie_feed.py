import sys,os
import numpy as np
import pandas as pd
import configparser

from myGQLib.Classes import outlook_email, Instrument
from GQDatabaseSQL import REDIS_DB
DB_Config = configparser.ConfigParser()
DB_Config.read(os.path.join("C:\Config", "DatabaseConfig.ini"))

Appconfig = configparser.ConfigParser()
Appconfig.read("AppConfig.ini")
if __name__ == "__main__":

    pass

    sub = dict(DB_Config['REDIS_UAT_CLOUD'])
    # sub = dict(DB_Config["REDIS_UAT"])
    sub['db'] = 0
    sub['message_type'] = Appconfig['REDIS']['message_type']
    pub = dict(DB_Config['REDIS_UAT_CLOUD'])
    pub['db'] = 0
    pub['message_type'] = Appconfig['REDIS']['message_type']
    RTD_Manager = REDIS_DB.RTD_Manager(sub=sub, pub=pub,
                                            verbose=False,
                                            block_writes=False)

    RTD_Manager.redis_pub.start_writing_stream_on_Thread()

    ul_sym = "PTT"
    product_type = "eq"
    market = "XBKK"
    mInstr = Instrument.Instrument(symbol=ul_sym, market=market, product_class=product_type, sub_type="eq")
    mInstr.set_RTD_Manager(RTD_Manager)
    mInstr.init_price_feeds(product_type=product_type, symbol=ul_sym, sub_type='orderbook',
                            init_from_beginning=False)
    mInstr.init_price_feeds(product_type=product_type, symbol=ul_sym, sub_type='tick',
                            init_from_beginning=False)


    # value_to_write = {parameters_df.columns[r]: str(params_array[i, r])
    #                   for r in range(params_array.shape[1] - 1)}
    # value_to_write['Maturity'] = pd.to_datetime(params_array[i, -1]).strftime("%Y-%m-%d")

    machine_time_utc = pd.Timestamp.now().tz_localize("ASIA/BANGKOK").tz_convert("UTC")
    tdiff = machine_time_utc - pd.Timestamp.now().normalize().tz_localize("ASIA/BANGKOK").tz_convert("UTC")

    id = int(machine_time_utc.timestamp() * 1e6)
    # key = "SVI_PARAM:" + parameters_df.index[i]

    key = f"{product_type}:orderbook:{ul_sym}"

    best_bid =33
    best_ask = 34.5
    value_to_write = {'aq': str([50000]),
                                 'bq': str([50000]),
                                 'bt': str([id]),
                                 'a': str([best_ask]),
                                 'maxLevel': str(10),
                                 'b':  str([best_bid]),
                                 'tss':  str(id),
                                 'tsb':  str(id),
                                 'at':  str([id]),
                                 'id': str(int(tdiff.total_seconds()*100)),
                                     }

    print(f"key:{key} id:{id}")
    print(value_to_write)
    RTD_Manager.queue_up_write_to_stream(key=key, id=id, value=value_to_write)
    RTD_Manager.stop_RTD_classes()