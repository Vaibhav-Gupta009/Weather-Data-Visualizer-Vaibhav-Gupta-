import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Task 1: Load and inspect 
# ----------------------------

# change the file name here if your CSV is different
file_name = "DailyDelhiClimateTrain.csv"

df = pd.read_csv(file_name)

print("First 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe())

# ----------------------------
# Task 2: Clean and process
# ----------------------------

# convert date to datetime
df["date"] = pd.to_datetime(df["date"])

# keep only useful columns (these exist in the Delhi climate file) [web:24]
df = df[["date", "meantemp", "humidity", "wind_speed", "meanpressure"]]

# handle missing values (if any): fill numeric NaN with column mean
df["meantemp"] = df["meantemp"].fillna(df["meantemp"].mean())
df["humidity"] = df["humidity"].fillna(df["humidity"].mean())
df["wind_speed"] = df["wind_speed"].fillna(df["wind_speed"].mean())
df["meanpressure"] = df["meanpressure"].fillna(df["meanpressure"].mean())

# sort by date
df = df.sort_values("date")

# create year and month columns
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# ----------------------------
# Task 3: Statistics with NumPy
# ----------------------------

# daily stats for whole period (mean, min, max, std)
temps = df["meantemp"].values
hums = df["humidity"].values

print("\nTemperature stats (°C):")
print("Mean:", np.mean(temps))
print("Min :", np.min(temps))
print("Max :", np.max(temps))
print("Std :", np.std(temps))

print("\nHumidity stats (%):")
print("Mean:", np.mean(hums))
print("Min :", np.min(hums))
print("Max :", np.max(hums))
print("Std :", np.std(hums))

# monthly statistics (groupby year+month)
monthly_stats = df.groupby(["year", "month"]).agg(
    temp_mean=("meantemp", "mean"),
    temp_min=("meantemp", "min"),
    temp_max=("meantemp", "max"),
    hum_mean=("humidity", "mean"),
)
print("\nMonthly stats (first 5 rows):")
print(monthly_stats.head())

# yearly statistics
yearly_stats = df.groupby("year").agg(
    temp_mean=("meantemp", "mean"),
    temp_min=("meantemp", "min"),
    temp_max=("meantemp", "max"),
    hum_mean=("humidity", "mean"),
)
print("\nYearly stats:")
print(yearly_stats)

# ----------------------------
# Task 4: Visualizations
# ----------------------------

# 1) Line chart: daily temperature
plt.figure()
plt.plot(df["date"], df["meantemp"])
plt.title("Daily Mean Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig("plot_daily_temperature.png")
plt.close()

# 2) Bar chart: monthly average temperature
month_avg = df.groupby("month")["meantemp"].mean()

plt.figure()
plt.bar(month_avg.index, month_avg.values)
plt.title("Average Temperature by Month")
plt.xlabel("Month")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig("plot_monthly_temp_bar.png")
plt.close()

# 3) Scatter plot: humidity vs temperature
plt.figure()
plt.scatter(df["meantemp"], df["humidity"])
plt.title("Humidity vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")
plt.tight_layout()
plt.savefig("plot_humidity_vs_temp.png")
plt.close()

# 4) Combined figure: line + scatter in subplots
plt.figure(figsize=(10, 6))

# top: temperature line
plt.subplot(2, 1, 1)
plt.plot(df["date"], df["meantemp"])
plt.title("Daily Mean Temperature")

# bottom: humidity vs temperature scatter
plt.subplot(2, 1, 2)
plt.scatter(df["meantemp"], df["humidity"])
plt.title("Humidity vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Humidity (%)")

plt.tight_layout()
plt.savefig("plot_combined.png")
plt.close()

# ----------------------------
# Task 5: Grouping and aggregation
# ----------------------------

# simple seasons function
def get_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Summer"
    elif m in [6, 7, 8]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

df["season"] = df["month"].apply(get_season)

season_stats = df.groupby("season").agg(
    temp_mean=("meantemp", "mean"),
    temp_min=("meantemp", "min"),
    temp_max=("meantemp", "max"),
    hum_mean=("humidity", "mean"),
)
print("\nSeason stats:")
print(season_stats)

# ----------------------------
# Task 6: Export
# ----------------------------

# export cleaned data
df.to_csv("weather_cleaned_basic.csv", index=False)

# export stats
monthly_stats.to_csv("weather_monthly_stats_basic.csv")
yearly_stats.to_csv("weather_yearly_stats_basic.csv")
season_stats.to_csv("weather_season_stats_basic.csv")

print("\nSaved files:")
print("weather_cleaned_basic.csv")
print("weather_monthly_stats_basic.csv")
print("weather_yearly_stats_basic.csv")
print("weather_season_stats_basic.csv")
print("plot_daily_temperature.png")
print("plot_monthly_temp_bar.png")
print("plot_humidity_vs_temp.png")
print("plot_combined.png")
