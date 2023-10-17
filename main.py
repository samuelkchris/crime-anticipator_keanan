import matplotlib.pyplot as plt

# Sample crime data
crimes = ['Robbery', 'Assault', 'Burglary', 'Theft', 'Fraud']
crime_counts = [25, 40, 15, 50, 30]

# Define colors based on crime count
colors = []
for count in crime_counts:
    if count > 30:
        colors.append('red')
    elif count > 20:
        colors.append('yellow')
    else:
        colors.append('green')

# Plotting the graph with colors
plt.bar(crimes, crime_counts, color=colors)
plt.xlabel('Crimes')
plt.ylabel('Crime Count')
plt.title('Crime Count by Type')
plt.xticks(rotation=45)
plt.savefig('static/images/crime_graph.png')
plt.show()
