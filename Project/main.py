from utility import *
from encoding import *
from normalization import *


# Numerical Summary
numerical_summary = movies_overseas_df.describe(include="all")
mode_row = movies_overseas_df.iloc[:, 6:].mode().iloc[0]
mode_row.name = "mode"
median_row = movies_overseas_df.iloc[:, 6:].median()
median_row.name = "median"
range_row = numerical_summary.loc["max"] - numerical_summary.loc["min"]
range_row.name = "range"
variance_row = movies_overseas_df.iloc[:, 6:].var()
variance_row.name = "variance"
numerical_summary = pd.concat(
    [numerical_summary, pd.DataFrame([range_row, variance_row, mode_row, median_row])]
)

# Saving the numerical summary to a CSV file
save_csv(numerical_summary, "Numerical Summary", "CSVs/CSV_RES")
print("\n")

# Histograms
for column in numeric_columns:
    plot_histogram(column, f"Histogram of {column}")
print("\n")

# Bar Graphs
plot_bar("Director", "Bar Graph of Directors")
plot_bar("Genre", "Bar Graph of Genres")
plot_bar("Studio", "Bar Graph of Studios")
plot_bar("Day of Week", "Bar Graph of Day of Week")
print("\n")

# Scatter Plots
plot_scatter("Budget ($mill)", "Overseas ($mill)", "Budget vs Overseas Revenue")
plot_scatter("IMDb Rating", "Gross ($mill)", "IMDb Rating vs Gross Revenue")
print("\n")

# Pie Charts
plot_pie("Genre", "Pie Chart of Genres")
print("\n")

# Box Plots
for column in numeric_columns:
    plot_box(column, f"Distribution of {column}")
print("\n")

# Pair Plot and Heat Map
plot_pair(numeric_columns, "Pair Plot of Numeric Variables")
plot_heatmap(numeric_columns, "Heatmap of Numeric Variables")
print("\n")

# Outing the normalized and encoded independent and dependent dataframes to a CSV file
save_csv(x_z, "Independent_Z_norm", "CSVs/CSV_RES")
save_csv(x_mm, "Independent_minmax_norm", "CSVs/CSV_RES")
save_csv(y_z, "Dependent_Z_norm", "CSVs/CSV_RES")
save_csv(y_mm, "Dependent_minmax_norm", "CSVs/CSV_RES")
print("\n")


# Calculating results for both normalizations
results_z = regm(x_z, y_z, "Z-Score Normalization")
results_mm = regm(x_mm, y_mm, "Min-Max Normalization")

# Consolidating and displaying the results
final_results = pd.concat([results_z, results_mm], axis=0).reset_index(drop=True)
save_csv(final_results, "regression_MSE_R2", "CSVs/CSV_RES")
