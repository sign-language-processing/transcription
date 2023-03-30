from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

num_shapes = Counter({
    1: 484071,
    2: 170407,
    3: 23588,
    4: 8396,
    0: 6394,
    5: 5029,
    6: 3777,
    7: 1886,
    8: 881,
    9: 421,
    10: 151,
    11: 58,
    12: 24,
    22: 16,
    13: 14,
    14: 7,
    23: 5,
    15: 5,
    16: 4,
    38: 2,
    56: 2,
    49: 1,
    20: 1,
    29: 1,
    59: 1,
    46: 1,
    19: 1,
    31: 1,
    58: 1,
    30: 1,
    18: 1,
    40: 1,
    17: 1,
    44: 1
})

# Get the keys and values from the counter
keys = list(range(5))
total = sum(num_shapes.values())
values = [num_shapes[k] / total * 100 for k in keys]

plt.figure(figsize=(6, 3))

# Set the font to Times Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Create a bar chart
plt.bar(keys, values, color='C3')

# Add labels and title
# plt.xlabel('# Hand Shapes', fontsize=18)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

# Add horizontal gridlines
plt.gca().yaxis.grid(True, linestyle='--', alpha=0.7)

# Format the y-axis labels as percentages with larger font size
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tick_params(axis='both', which='major', labelsize=18)

plt.subplots_adjust(bottom=0.1, top=0.98)

plt.savefig('signbank.pdf')

# # Show the chart
# plt.show()
