import json
import matplotlib.pyplot as plt

with open('double_slit_results.json', 'r') as f:
    data = json.load(f)

edges = data['histogram_edges']
bins = data['histogram_bins']
centers = [(edges[i] + edges[i+1])/2 for i in range(len(bins))]

plt.figure(figsize=(10, 6))
plt.bar(centers, bins, width=(edges[1]-edges[0]), color='blue', alpha=0.7)
plt.title('Pilot Wave Double Slit')
plt.xlabel('Y Position on Detector Screen')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.savefig('pilot_wave_histogram.png')
print('Saved pilot_wave_histogram.png')
