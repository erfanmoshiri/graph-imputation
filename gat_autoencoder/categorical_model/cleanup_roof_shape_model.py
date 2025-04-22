import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df1 = pd.read_hdf('../data/log_minmax_clean_df_sorted_3.hdf')

df = df1[
    (df1['ROOF_SHAPE_-1'] == 0) &
    (df1['ROOF_SHAPE_MANSARD'] == 0) &
    (df1['ROOF_SHAPE_MIXED'] == 0) &
    (df1['ROOF_SHAPE_SHED'] == 0)
    ]

# Assuming 'df' is your DataFrame
X = df[['CL_FLOOR_AREA', 'NEARMAP_PARCEL_AREA_SQM', 'NEARMAP_BUILDING_COUNT', 'NEARMAP_BUILDING_TOTAL_AREA_SQM',
        'BUILDING_SIZE', 'ROOF_HEIGHT', 'EAVE_HEIGHT', 'ESTIMATED_LEVELS', 'AVG_BED', 'AVG_BATH', 'AVG_CAR',
        'PRIMARY_ROOF_MATERIAL_-1', 'PRIMARY_ROOF_MATERIAL_FIBERGLASS/PLASTIC', 'PRIMARY_ROOF_MATERIAL_FLAT CONCRETE',
        'PRIMARY_ROOF_MATERIAL_METAL', 'PRIMARY_ROOF_MATERIAL_TILE', 'CL_PROPERTY_TYPE_CATEGORY_Apartment/Unit',
        'CL_PROPERTY_TYPE_CATEGORY_House', 'CL_PROPERTY_TYPE_CATEGORY_Rural/Farming',
        'CL_PROPERTY_TYPE_CATEGORY_Townhouse']]

# Assuming target labels are already one-hot encoded for the 3 classes
y = df[['ROOF_SHAPE_FLAT', 'ROOF_SHAPE_GABLED', 'ROOF_SHAPE_HIPPED']]
y_single = y.idxmax(axis=1)
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_single, test_size=0.2, random_state=42)

# Initialize Logistic Regression model with multinomial for multi-class classification
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

##############


# Assume df is the original DataFrame, and filtered_df is the filtered DataFrame for prediction
# filtered_df should have the same columns that were used for training the model

# Step 1: Filter the subset of rows that match the condition into df1_small
df1_small = df1[
    (df1['ROOF_SHAPE_-1'] != 0) |
    (df1['ROOF_SHAPE_MANSARD'] != 0) |
    (df1['ROOF_SHAPE_MIXED'] != 0) |
    (df1['ROOF_SHAPE_SHED'] != 0)
    ].copy().reset_index(drop=True)  # Use .copy() to avoid modifying df1_small when modifying df1

# Step 2: Remove the filtered rows from df1
df1_big = df1[
    ~((df1['ROOF_SHAPE_-1'] != 0) |
      (df1['ROOF_SHAPE_MANSARD'] != 0) |
      (df1['ROOF_SHAPE_MIXED'] != 0) |
      (df1['ROOF_SHAPE_SHED'] != 0))
].copy().reset_index(drop=True)  # Use .copy() to avoid any SettingWithCopyWarning

df1_small_feat = df1_small[['CL_FLOOR_AREA', 'NEARMAP_PARCEL_AREA_SQM', 'NEARMAP_BUILDING_COUNT',
                            'NEARMAP_BUILDING_TOTAL_AREA_SQM', 'BUILDING_SIZE', 'ROOF_HEIGHT',
                            'EAVE_HEIGHT', 'ESTIMATED_LEVELS', 'AVG_BED', 'AVG_BATH', 'AVG_CAR',
                            'PRIMARY_ROOF_MATERIAL_-1', 'PRIMARY_ROOF_MATERIAL_FIBERGLASS/PLASTIC',
                            'PRIMARY_ROOF_MATERIAL_FLAT CONCRETE', 'PRIMARY_ROOF_MATERIAL_METAL',
                            'PRIMARY_ROOF_MATERIAL_TILE', 'CL_PROPERTY_TYPE_CATEGORY_Apartment/Unit',
                            'CL_PROPERTY_TYPE_CATEGORY_House', 'CL_PROPERTY_TYPE_CATEGORY_Rural/Farming',
                            'CL_PROPERTY_TYPE_CATEGORY_Townhouse']]

# Filter the DataFrame based on some condition and preserve original indices
# filtered_df = df1.loc[
#     (df1['ROOF_SHAPE_-1'] != 0) |
#     (df1['ROOF_SHAPE_MANSARD'] != 0) |
#     (df1['ROOF_SHAPE_MIXED'] != 0) |
#     (df1['ROOF_SHAPE_SHED'] != 0)
#     ,
#     list(filtered_df.columns)
# ]

# Predict the labels in one-hot encoded form
y_pred_filtered = model.predict(df1_small_feat)
y_pred_onehot = pd.get_dummies(y_pred_filtered)

predicted_onehot_df = pd.DataFrame(y_pred_onehot, columns=['ROOF_SHAPE_FLAT', 'ROOF_SHAPE_GABLED', 'ROOF_SHAPE_HIPPED'])

df1_small[['ROOF_SHAPE_FLAT', 'ROOF_SHAPE_GABLED', 'ROOF_SHAPE_HIPPED']] = predicted_onehot_df

df_full_back = pd.concat([df1_small, df1_big], axis=0).sort_values(
    by=['POSTCODE', 'SUBURB_NAME', 'STREET_NAME', 'STREET_TYPE', 'STREET_NUMBER']).reset_index(drop=True)

df_full_back = df_full_back.drop(columns=[
    'ROOF_SHAPE_-1',
    'ROOF_SHAPE_MANSARD',
    'ROOF_SHAPE_MIXED',
    'ROOF_SHAPE_SHED'
])

# validate:
test = df_full_back[
    (df_full_back['ROOF_SHAPE_FLAT'] != 0) |
    (df_full_back['ROOF_SHAPE_GABLED'] != 0) |
    (df_full_back['ROOF_SHAPE_HIPPED'] != 0)
    ]

df_full_back.to_hdf('../data/log_minmax_clean_df_sorted_4.hdf', key='s')

# Now df1 has the updated one-hot encoded columns for the filtered rows
