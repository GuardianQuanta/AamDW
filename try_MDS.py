import pandas as pd
from GQDatabaseSQL.MDSDB import  MDSDB

df = MDSDB.get_tfex_ticker_between(_date1=pd.Timestamp("2024-07-01"),_date2=pd.Timestamp("2024-07-05"),
                              _symbol='S50___')

print(df)