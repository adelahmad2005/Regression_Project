from utility import *
from encoding import *

# Normalizing the numeric columns using Z-score normalization
df_numeric = movies_overseas_df_encoded[numeric_columns]
df_cat = movies_overseas_df_encoded.drop(numeric_columns, axis=1)
y = movies_overseas_df_encoded[OVERSEAS_MILL_COLUMN_NAME]

# Normalizing the numerical data with Z-Score
z_scaler = StandardScaler()
df_numeric_z = z_scaler.fit_transform(df_numeric)
df_numeric_z = pd.DataFrame(df_numeric_z, columns=df_numeric.columns)

# Concatenating the encoded and normalized data to form a full encoded and normalized data frame
df_full_z = pd.concat([df_cat, df_numeric_z], axis=1)

# Normalizing the numerical data with Min-Max scaler
min_max_scaler = MinMaxScaler()
df_numeric_mm = min_max_scaler.fit_transform(df_numeric)
df_numeric_mm = pd.DataFrame(df_numeric_mm, columns=df_numeric.columns)

# Concatenating the encoded and normalized data to form a full encoded and normalized data frame
df_full_mm = pd.concat([df_cat, df_numeric_mm], axis=1)

# Splitting the data into dependent and independent variables
x_z = df_full_z.drop(OVERSEAS_MILL_COLUMN_NAME, axis=1)
x_mm = df_full_mm.drop(OVERSEAS_MILL_COLUMN_NAME, axis=1)
y_z = df_full_z[OVERSEAS_MILL_COLUMN_NAME]
y_mm = df_full_mm[OVERSEAS_MILL_COLUMN_NAME]
