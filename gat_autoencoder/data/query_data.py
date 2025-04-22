import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from utils.query_snowflake import query_snowflake_pd


def change_col_order(df1):
    new_order = [
        'POSTCODE', 'SUBURB_NAME', 'STREET_NAME', 'STREET_TYPE', 'STREET_NUMBER',

        "CL_FLOOR_AREA",
        "NEARMAP_PARCEL_AREA_SQM",
        "NEARMAP_BUILDING_COUNT",
        "NEARMAP_BUILDING_TOTAL_AREA_SQM",
        "BUILDING_SIZE",
        "ROOF_HEIGHT",
        "EAVE_HEIGHT",
        "ESTIMATED_LEVELS",
        "AVG_BED",
        "AVG_BATH",
        "AVG_CAR",
        "PRIMARY_ROOF_MATERIAL_-1",
        "PRIMARY_ROOF_MATERIAL_FIBERGLASS/PLASTIC",
        "PRIMARY_ROOF_MATERIAL_FLAT CONCRETE",
        "PRIMARY_ROOF_MATERIAL_METAL",
        "PRIMARY_ROOF_MATERIAL_TILE",
        "ROOF_SHAPE_-1",
        "ROOF_SHAPE_FLAT",
        "ROOF_SHAPE_GABLED",
        "ROOF_SHAPE_HIPPED",
        "ROOF_SHAPE_MANSARD",
        "ROOF_SHAPE_MIXED",
        "ROOF_SHAPE_SHED",
        "CL_PROPERTY_TYPE_CATEGORY_Apartment/Unit",
        "CL_PROPERTY_TYPE_CATEGORY_House",
        "CL_PROPERTY_TYPE_CATEGORY_Rural/Farming",
        "CL_PROPERTY_TYPE_CATEGORY_Townhouse"
    ]
    df1 = df1[new_order]
    return df1


df = query_snowflake_pd(
    '''

    select
 
     SVP_ADDRESS_ID,
     postcode,
     suburb_name,
     street_name,
     street_type,
     street_number,
     
     
     CL_FLOOR_AREA,
     NEARMAP_PARCEL_AREA_SQM ,
     NEARMAP_BUILDING_COUNT ,
     NEARMAP_BUILDING_TOTAL_AREA_SQM ,
    
     BUILDING_SIZE ,
     ROOF_HEIGHT ,
     EAVE_HEIGHT ,
     ESTIMATED_LEVELS ,

     CL_PROPERTY_TYPE_CATEGORY,

    "PRIMARY_ROOF_MATERIAL_-1",
	"PRIMARY_ROOF_MATERIAL_FIBERGLASS/PLASTIC",
	"PRIMARY_ROOF_MATERIAL_FLAT CONCRETE",
	PRIMARY_ROOF_MATERIAL_METAL,
	PRIMARY_ROOF_MATERIAL_TILE,

    "ROOF_SHAPE_-1",
	ROOF_SHAPE_FLAT,
	ROOF_SHAPE_GABLED,
	ROOF_SHAPE_HIPPED,
	ROOF_SHAPE_MANSARD,
	ROOF_SHAPE_MIXED,
	ROOF_SHAPE_SHED,

     -- BBC ??
     PP_BED_ROOM_NUMBER, 
     CL_BED_ROOM_NUMBER, 
     REA_BED_ROOM_NUMBER,
     PP_BATH_NUMBER, 
     CL_BATH_NUMBER, 
     REA_BATH_NUMBER,
     PP_CARPARK_NUMBER, 
     CL_CARPARK_NUMBER, 
     REA_CARPARK_NUMBER
     
FROM 
    ZZ_YIJU_WEI_DEV.PROPERTY_TYPE.CL_NE_HPG_FEATURES
WHERE 
    CL_FLOOR_AREA <> -1 and CL_FLOOR_AREA < 2000
    AND NEARMAP_PARCEL_AREA_SQM <> -1
    AND NEARMAP_BUILDING_COUNT <> -1 and NEARMAP_BUILDING_COUNT < 20
    AND NEARMAP_BUILDING_TOTAL_AREA_SQM <> -1 and NEARMAP_BUILDING_TOTAL_AREA_SQM < 15000
    AND BUILDING_SIZE <> -1 and BUILDING_SIZE < 7500
    AND ROOF_HEIGHT <> -1
    AND EAVE_HEIGHT <> -1
    AND ESTIMATED_LEVELS <> -1       
    AND CL_PROPERTY_TYPE_CATEGORY is not null
    ;

    ''', role="ROLE_PROPERTY_DATA_RESTRICTED")

df[['PP_BED_ROOM_NUMBER', 'CL_BED_ROOM_NUMBER', 'REA_BED_ROOM_NUMBER']] = df[
    ['PP_BED_ROOM_NUMBER', 'CL_BED_ROOM_NUMBER', 'REA_BED_ROOM_NUMBER']].replace({-1: np.nan})
df[['PP_BATH_NUMBER', 'CL_BATH_NUMBER', 'REA_BATH_NUMBER']] = df[
    ['PP_BATH_NUMBER', 'CL_BATH_NUMBER', 'REA_BATH_NUMBER']].replace({-1: np.nan})
df[['PP_CARPARK_NUMBER', 'CL_CARPARK_NUMBER', 'REA_CARPARK_NUMBER']] = df[
    ['PP_CARPARK_NUMBER', 'CL_CARPARK_NUMBER', 'REA_CARPARK_NUMBER']].replace({-1: np.nan})

df['AVG_BED'] = df[['PP_BED_ROOM_NUMBER', 'CL_BED_ROOM_NUMBER', 'REA_BED_ROOM_NUMBER']].mean(axis=1)
df['AVG_BATH'] = df[['PP_BATH_NUMBER', 'CL_BATH_NUMBER', 'REA_BATH_NUMBER']].mean(axis=1)
df['AVG_CAR'] = df[['PP_CARPARK_NUMBER', 'CL_CARPARK_NUMBER', 'REA_CARPARK_NUMBER']].mean(axis=1)

# df[['AVG_BED', 'AVG_BATH', 'AVG_CAR']] = df[['AVG_BED', 'AVG_BATH', 'AVG_CAR']].replace(0, np.nan)


df = df.drop(columns=['PP_BED_ROOM_NUMBER', 'CL_BED_ROOM_NUMBER', 'REA_BED_ROOM_NUMBER',
                      'PP_BATH_NUMBER', 'CL_BATH_NUMBER', 'REA_BATH_NUMBER',
                      'PP_CARPARK_NUMBER', 'CL_CARPARK_NUMBER', 'REA_CARPARK_NUMBER', ])

df = df.dropna(subset=['AVG_BED', 'AVG_BATH', 'AVG_CAR'])

df = df[(df['AVG_BED'] < 15) & (df['AVG_BATH'] < 15) & (df['AVG_CAR'] < 10)]

df['PRIMARY_ROOF_MATERIAL_-1'] = df['PRIMARY_ROOF_MATERIAL_-1'].replace(-1, 1)

df.replace(-1, np.nan, inplace=True)

df.set_index('SVP_ADDRESS_ID')

columns_normalize = [
    'CL_FLOOR_AREA',
    'NEARMAP_PARCEL_AREA_SQM',
    'NEARMAP_BUILDING_COUNT',
    'NEARMAP_BUILDING_TOTAL_AREA_SQM',
    'BUILDING_SIZE',
    'ROOF_HEIGHT',
    'EAVE_HEIGHT',
    'ESTIMATED_LEVELS',
    'AVG_BED',
    'AVG_BATH',
    'AVG_CAR'
]

print('col normalize all size:', len(columns_normalize))

column_types = df[columns_normalize].dtypes
string_columns = column_types[column_types == 'object'].index.tolist()

df[columns_normalize] = np.log1p(df[columns_normalize])

scaler = MinMaxScaler()
df[columns_normalize] = scaler.fit_transform(df[columns_normalize])
dump(scaler, './scaler_new.pkl')

columns_get_onehot = ['CL_PROPERTY_TYPE_CATEGORY']
df = pd.get_dummies(df, columns=columns_get_onehot)

df = df.sort_values(by=['POSTCODE', 'SUBURB_NAME', 'STREET_NAME', 'STREET_TYPE', 'STREET_NUMBER'])
# df = df.drop(columns=['POSTCODE', 'SUBURB_NAME', 'STREET_NAME', 'STREET_TYPE', 'STREET_NUMBER', 'SVP_ADDRESS_ID'])

df = change_col_order(df)

print(df.isnull().sum())
print(list(df.columns))

df.to_hdf('./log_minmax_clean_df_sorted_2.hdf', key='s')
print('write success.')
