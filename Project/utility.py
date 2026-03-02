import pandas as pd
import os
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVC, SVR
import category_encoders as ce
from category_encoders import BinaryEncoder

import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    LabelEncoder,
    KBinsDiscretizer,
    OneHotEncoder,
    MinMaxScaler,
    StandardScaler,
    LabelEncoder,
)

# Reading the data
movies_overseas_df = pd.read_csv("CSVs/movies_overseas.csv", encoding="windows-1252")

# Defining constants
RELEASE_DATE_COLUMN_NAME = "Release Date"
IMBD_RATING_COLUMN_NAME = "IMDb Rating"
MOVIELENS_RATING_COLUMN_NAME = "MovieLens Rating"
OVERSEAS_MILL_COLUMN_NAME = "Overseas ($mill)"

# Replace commas and convert to float
for col in movies_overseas_df.columns[6:]:
    movies_overseas_df[col] = (
        movies_overseas_df[col].astype(str).str.replace(",", "").astype(float)
    )

# Convert 'Release Date' into separate year, month, day columns
movies_overseas_df[RELEASE_DATE_COLUMN_NAME] = pd.to_datetime(
    movies_overseas_df[RELEASE_DATE_COLUMN_NAME], dayfirst=True
)
movies_overseas_df["Year"] = movies_overseas_df[RELEASE_DATE_COLUMN_NAME].dt.year
movies_overseas_df["Month"] = movies_overseas_df[RELEASE_DATE_COLUMN_NAME].dt.month
movies_overseas_df["Day"] = movies_overseas_df[RELEASE_DATE_COLUMN_NAME].dt.day
movies_overseas_df.drop(RELEASE_DATE_COLUMN_NAME, axis=1, inplace=True)

# Copying the dataframe for encoding
movies_overseas_df_encoded = movies_overseas_df.copy()


# Function to save the data to a csv file
def save_csv(data, title, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{title.replace(' ', '_')}.csv")
    data.to_csv(filepath, index=True)
    print(f"{title} has been saved to {directory}")


# Function to save the image to a directory
def save_and_close_image(title, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{title.replace(' ', '_')}.png")
    plt.savefig(filepath)
    plt.close()
    print(f"{title} has been saved to {directory}")


# Graphical Summaries functions
# Histogram
def plot_histogram(column, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(movies_overseas_df[column], kde=False, bins="auto")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(title)
    save_and_close_image(title, "graphs")


# Bar Graph
def plot_bar(column, title):
    plt.figure(figsize=(10, 6))
    value_counts = movies_overseas_df[column].value_counts()
    if len(value_counts) > 20:
        value_counts = value_counts.head(20)
    value_counts.plot(kind="bar")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_and_close_image(title, "graphs")


# Scatter Plot
def plot_scatter(x, y, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=movies_overseas_df, x=x, y=y)
    plt.title(title)
    save_and_close_image(title, "graphs")


# Pie Chart
def plot_pie(column, title):
    plt.figure(figsize=(10, 6))
    movies_overseas_df[column].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.title(title)
    plt.ylabel("")
    save_and_close_image(title, "graphs")


# Box Plot
def plot_box(column, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=movies_overseas_df, y=column)
    plt.title(title)
    save_and_close_image(title, "graphs")


# Pair Plot
def plot_pair(columns, title):
    plt.figure(figsize=(10, 6))
    sns.pairplot(movies_overseas_df_encoded[columns])
    plt.suptitle(title)
    save_and_close_image(title, "graphs")


# Heat Map
def plot_heatmap(columns, title):
    plt.figure(figsize=(10, 8))
    corr = movies_overseas_df_encoded[columns].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title(title)
    save_and_close_image(title, "graphs")


# Definig numeric columns
numeric_columns = [
    "Adjusted Gross ($mill)",
    "Budget ($mill)",
    "Gross ($mill)",
    "IMDb Rating",
    "MovieLens Rating",
    "Runtime (min)",
    "Overseas ($mill)",
    "Year",
    "Month",
    "Day",
]


# Regression function definition
def regm(x, y, normalization):
    results = []
    for r in [1, 20, 40]:
        models = [
            LinearRegression(),
            SVR(kernel="linear"),
            SVR(kernel="poly"),
            SVR(kernel="rbf"),
            MLPRegressor(max_iter=1000, random_state=r),
        ]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=r
        )
        for model in models:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            MSE = mean_squared_error(y_test, y_pred)
            r_2 = r2_score(y_test, y_pred)
            r_2_scorefn = model.score(x_test, y_test)

            if isinstance(model, SVR):
                model_name = f"SVR {model.kernel.title()}"
            else:
                model_name = model.__class__.__name__

            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_test, color="blue", marker="o", label="Actual")
            plt.scatter(y_test, y_pred, color="orange", marker="o", label="Predicted")
            plt.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                "--",
                lw=2,
                color="red",
                label="Ideal Prediction Line",
            )
            title = f"{model_name} - {normalization} (Random State: {r})"
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(title)
            plt.legend()
            plt.grid(True)

            directory = "regression"
            if not os.path.exists(directory):
                os.makedirs(directory)
            filepath = os.path.join(
                directory,
                f"{model_name.replace(' ', '_')}_Random_State_{r}_{normalization}.png",
            )
            plt.savefig(filepath)
            plt.close()
            print(
                f"{model_name} - {normalization} (Random State: {r}) has been saved to {directory}"
            )

            results.append(
                {
                    "Normalization": normalization,
                    "Model": model_name,
                    "Random State": r,
                    "MSE": round(MSE, 2),
                    "R^2": round(r_2, 2),
                    "R^2 using score function": round(r_2_scorefn, 2),
                }
            )
    return pd.DataFrame(results)
