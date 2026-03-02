from utility import *


# Handling low-cardinality categorical data with one-hot encoding
one_hot_cols = ["Day of Week", "Genre"]
one_hot_encoder = OneHotEncoder()
one_hot_encoded_data = one_hot_encoder.fit_transform(
    movies_overseas_df_encoded[one_hot_cols]
).toarray()
one_hot_encoded_df = pd.DataFrame(
    one_hot_encoded_data,
    columns=one_hot_encoder.get_feature_names_out(),
    index=movies_overseas_df_encoded.index,
)
movies_overseas_df_encoded = pd.concat(
    [movies_overseas_df_encoded.drop(one_hot_cols, axis=1), one_hot_encoded_df], axis=1
)

# Handling low-cardinality categorical data with Frequency Encoding
frequency_encoder = {
    col: movies_overseas_df_encoded[col].value_counts(normalize=True)
    for col in ["Studio", "Director"]
}

# Applying frequency encoding by mapping the frequency values
for col in ["Studio", "Director"]:
    movies_overseas_df_encoded[col + "_encoded"] = movies_overseas_df_encoded[col].map(
        frequency_encoder[col]
    )

# Create a DataFrame with the frequency encoded data
frequency_encoded_df = movies_overseas_df_encoded[
    [col + "_encoded" for col in ["Studio", "Director"]]
]

# Concatenate the new frequency encoded data with the original DataFrame, dropping the original columns
movies_overseas_df_encoded = pd.concat(
    [
        movies_overseas_df_encoded.drop(["Studio", "Director"], axis=1),
        frequency_encoded_df,
    ],
    axis=1,
)


# omit the 'Movie Title' column
movies_overseas_df_encoded.drop("Movie Title", axis=1, inplace=True)

# IMDb ratings (scaled out of 10)
imdb_bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10]
imdb_midpoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# MovieLens ratings (scaled out of 5)
movielens_bins = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5]
movielens_midpoints = [0, 1, 2, 3, 4, 5]

# Apply the custom bins using pandas.cut with numeric midpoints as labels
movies_overseas_df_encoded[IMBD_RATING_COLUMN_NAME] = pd.cut(
    movies_overseas_df_encoded[IMBD_RATING_COLUMN_NAME],
    bins=imdb_bins,
    labels=imdb_midpoints,
    include_lowest=True,
    right=False,
)

movies_overseas_df_encoded[MOVIELENS_RATING_COLUMN_NAME] = pd.cut(
    movies_overseas_df_encoded[MOVIELENS_RATING_COLUMN_NAME],
    bins=movielens_bins,
    labels=movielens_midpoints,
    include_lowest=True,
    right=False,
)

# Convert cut columns to float as pd.cut assigns categorical dtype by default
movies_overseas_df_encoded[IMBD_RATING_COLUMN_NAME] = movies_overseas_df_encoded[
    IMBD_RATING_COLUMN_NAME
].astype(float)
movies_overseas_df_encoded[MOVIELENS_RATING_COLUMN_NAME] = movies_overseas_df_encoded[
    MOVIELENS_RATING_COLUMN_NAME
].astype(float)
