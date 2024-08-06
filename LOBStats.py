import pandas as pd
import os,sys
import datetime
import numpy as np

query = f"""
INSERT INTO mdsmmedb
SELECT MAX(TssTime) as TssTime, 
    SecName, 
    AVG(MDBid1Price) as Bid1Price,
    AVG(MDAsk1Price) as Ask1Price, 
    SUM(MDBid1Size) as Bid1Size, 
    SUM(MDAsk1Size) as Ask1Size,
    AVG((MDBid1Price + MDAsk1Price)/2) as Mid, 
    AVG((MDAsk1Price - MDBid1Price)/2) as Sprd1, 
    COUNT(SecName) as TickCount
FROM 
     mdsmmedb.mds_equityorderbook 	 
WHERE 
    TssTime BETWEEN CONCAT("str_date", ' 10:00:00.000') 
    AND CONCAT("str_date", ' 16:10:00.000')
    AND MDAsk1Price BETWEEN 0 AND 50000 
    AND MDBid1Price BETWEEN 0 AND 50000 
    AND MDAsk1Price > MDBid1Price
GROUP BY 
    SecName,
    YEAR(TssTime),
    MONTH(TssTime),
    DAY(TssTime),
    (HOUR(TssTime) / H)
ORDER BY 
    SecName, TssTime ASC;
"""


df = pd.read_feather("PTTEP_LOB.feather")

print()

df['Extime'] = pd.to_datetime(df['TssNanos'],unit='ns')
df['index'] = pd.to_datetime(df['TssNanos'],unit='ns')
df = df[ (df['Extime'].dt.time<= datetime.time(9,10)) & (df['Extime'].dt.time >= datetime.time(3,0))]
df = df[ df['MDAsk1Price']> df['MDBid1Price']]
df['MidPrice'] = (df['MDAsk1Price'] + df['MDBid1Price'])/(
            (~df['MDAsk1Price'].isna()).astype(float)+ (~ df['MDBid1Price'].isna()).astype(float) )
df['Sprd1'] = (df['MDAsk1Price'] - df['MDBid1Price'])/ (
            df['MDAsk1Price'].where(df['MDAsk1Price'].isna(),1)+ df['MDBid1Price'].where(df['MDBid1Price'].isna(),1) )


df.set_index("index",inplace=True)
resample = df.groupby('SecName').resample("6H").agg({
    "MDBid1Price":['mean'],
    "MDAsk1Price":['mean'],
    "MDBid1Size":['sum'],
    "MDAsk1Size":['sum'],
    "MidPrice":['mean'],
    "Sprd1":['mean'],
    "SecName":['count'],
    "Extime":['last']
})
resample.reset_index(inplace=True)
print(resample.T)
