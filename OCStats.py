import pandas as pd
import os,sys
import datetime
import numpy as np

query = f"""
    INSERT INTO mdsmmedb.table_name
    SELECT MAX(TssTime) as TssTime, 
        SecName, 
        AVG(MDBid1Price) as Bid, 
        AVG(MDAsk1Price) as Ask, 
        AVG((MDBid1Price + MDAsk1Price)/2) as Mid,
        'O' as OpenClose
    FROM mdsmmedb.mds_equityorderbook 
    WHERE 
        TssTime BETWEEN 'str_date 10:05:00'  AND  'str_date 10:35:00' AND
        MDAsk1Price BETWEEN 0 AND 50000 AND
        MDBid1Price BETWEEN 0 AND 50000 AND
        MDAsk1Price > MDBid1Price
    GROUP BY 
        SecName,
        YEAR(TssTime),
        MONTH(TssTime),
        DAY(TssTime)
    UNION
    SELECT MAX(TssTime) as TssTime, 
        SecName, 
        AVG(MDBid1Price) as Bid, 
        AVG(MDAsk1Price) as Ask, 
        AVG((MDBid1Price + MDAsk1Price)/2) as Mid,
        'C' as OpenClose
    FROM mdsmmedb.mds_equityorderbook
    WHERE 

        TssTime BETWEEN 'str_date 16:00:00'  AND  'str_date 16:10:00' AND
        MDAsk1Price BETWEEN 0 AND 50000 AND
        MDBid1Price BETWEEN 0 AND 50000 AND
        MDAsk1Price > MDBid1Price
    GROUP BY 
        SecName,
        YEAR(TssTime),
        MONTH(TssTime),
        DAY(TssTime)
    ORDER BY TssTime ASC;
    """


df = pd.read_feather("PTTEP_LOB.feather")

# print()
'''
Morning
'''
morning_start, morning_end = '3:05', '3:35'
evening_start, evening_end = '9:00', '9:10'

df['Extime'] = pd.to_datetime(df['TssNanos'],unit='ns')


# df = df[ (df['Extime'].dt.time<= datetime.time(9,10)) & (df['Extime'].dt.time >= datetime.time(3,0))]
df = df[ df['MDAsk1Price']> df['MDBid1Price']]
df = df[ (df['MDAsk1Price']> 0) & (df['MDAsk1Price']< 50000)]
df = df[ (df['MDBid1Price']> 0) & (df['MDBid1Price']< 50000)]
df['MidPrice'] = (df['MDAsk1Price'] + df['MDBid1Price'])/(
            (~df['MDAsk1Price'].isna()).astype(float)+ (~ df['MDBid1Price'].isna()).astype(float) )

# Create masks for the time ranges
morning_mask = df['Extime'].dt.time.between(pd.to_datetime(morning_start).time(), pd.to_datetime(morning_end).time())
evening_mask = df['Extime'].dt.time.between(pd.to_datetime(evening_start).time(), pd.to_datetime(evening_end).time())

# Create a new column 'time_range' based on the masks
df.loc[morning_mask, 'OpenClose'] = 'O'
df.loc[evening_mask, 'OpenClose'] = 'C'

result = df.groupby(['SecName', 'OpenClose']).agg({
    "MDBid1Price":['mean'],
    "MDAsk1Price":['mean'],
    "MidPrice":['mean'],
    "Extime": ['last']
}).reset_index()

print(result.T)
