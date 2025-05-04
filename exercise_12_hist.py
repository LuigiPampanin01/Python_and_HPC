import pandas as pd
import matplotlib.pyplot as plt

# Replace 'results.out' with your actual filename if different
csv_file = 'batch_outputs/Final_12_24885350.out'

# Load data
df = pd.read_csv(csv_file, header=0, nrows=4571)

# a) Histogram for mean temperatures
plt.figure()
plt.hist(df['mean_temp'], bins=30)
plt.xlabel('Mean Temperature (°C)')
plt.ylabel('Number of Buildings')
plt.title('Distribution of Mean Building Temperatures')
plt.tight_layout()
plt.show()
plt.savefig('mean_temp_histogram.png')

# b) Average mean temperature
avg_mean_temp = df['mean_temp'].mean()

# c) Average temperature standard deviation
avg_std_temp = df['std_temp'].mean()

# d) Number of buildings with ≥50% area above 18°C
count_above_50 = (df['pct_above_18'] >= 50).sum()

# e) Number of buildings with ≥50% area below 15°C
count_below_50 = (df['pct_below_15'] >= 50).sum()

# Print summary
print(f"Average mean temperature: {avg_mean_temp:.2f} °C")
print(f"Average temperature standard deviation: {avg_std_temp:.2f} °C")
print(f"Buildings with ≥50% of area above 18°C: {count_above_50}")
print(f"Buildings with ≥50% of area below 15°C: {count_below_50}")