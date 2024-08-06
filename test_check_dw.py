import sys, os
import threading
import pandas as pd
import numpy as np



from Aam import deploy
from GQDatabaseSQL import REDIS_DB, SETSMART

pg_date = deploy.get_prev_trade_date(pd.Timestamp.now().normalize())
calls_dw = deploy.check_dw(_symbol="PTTEP", price_guideline_date=pg_date, _put_call="C")
puts_dw = deploy.check_dw(_symbol="PTTEP", price_guideline_date=pg_date, _put_call="P")
# deploy.check_dw()
# 3
DW_spec = SETSMART.get_DW_Specs(Last_trade_date_gr=pd.Timestamp.now().normalize())

DW_spec.set_index("Instrument", inplace=True)
_valid_issuer_list = ["01","19","13"]


calls_dw_array = []
puts_dw_array = []

for dw_sym in calls_dw:
    try:
        if DW_spec.loc[dw_sym, "IssueBroker"] in _valid_issuer_list:
            print(f"{dw_sym} valid issuer")
            calls_dw_array.append(dw_sym)
        else:
            continue
    except KeyError as e:
        print(e)


for dw_sym in puts_dw:
    try:
        if DW_spec.loc[dw_sym, "IssueBroker"] in _valid_issuer_list:
            print(f"{dw_sym} valid issuer")
            puts_dw_array.append(dw_sym)
        else:
            continue
    except KeyError as e:
        print(e)

print(calls_dw_array)
print(f"call count:{len(calls_dw_array)}")

print(puts_dw_array)
print(f"put count:{len(puts_dw_array)}")