# Make Predictions with k-nearest neighbors on the Iris Flowers Dataset
from csv import reader
from math import sqrt

# Load a CSV file
def load_csvfile(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_read = reader(file)
		for row in csv_read:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(trainset, test_row, k_neighbors):
	distances = list()
	for train_row in trainset:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(k_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a prediction with neighbors
def predict_classification(trainset, test_row, k_neighbors):
	neighbors = get_neighbors(trainset, test_row, k_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

def main():
    # Make a prediction with KNN on Iris Dataset
    filename = 'iris.csv'
    dataset = load_csvfile(filename)
    for i in range(len(dataset[0])-1):
            str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    # define model parameter
    k_neighbors = 5
    # define a new record
    test_set = [[5.7,2.9,4.2,1.3],[4.4,4.2,1.1,2.5],[5.9,3.9,4.2,1.2],[6.8,3.2,5.9,2.3],[6.6,3.3,1.1,1.0],[6.4,2.6,5.1,1.8]]
    # predict the label
    for i in range(len(test_set)):
        Prediction = predict_classification(dataset, test_set[i], k_neighbors)
        print('Test data=%s' % (test_set[i]))
        print('Predicted value: %s' % (Prediction))
        print()

main()
