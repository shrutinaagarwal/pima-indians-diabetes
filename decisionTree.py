from random import seed
from random import randrange
from csv import reader
 
# Load a CSV file
def load_csv(filename):
	file = open(filename, "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores
 
#split the dataset based on feature and its value
def test_split(index,value,dataset):
    left,right = list(),list()
    for row in dataset:
        if row[index]<value:
            left.append(row)
        else:
            right.append(row)
    return left,right

# calculating the gini index for a split
def gini_index(groups,y):
    gini = 0.0
    for y in y:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            p = [row[-1] for row in group].count(y)/float(size)
            gini +=(p *(1.0-p))
    return gini

# selecting the best split point
def get_split(dataset):
    y = list(set(row[-1]for row in dataset))
    s_index, s_value, s_score, s_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, y)
            if gini < s_score:
                s_index, s_value, s_score, s_groups = index, row[index], gini, groups
    return {'index': s_index, 'value': s_value, 'groups': s_groups}

# creating terminal nodes
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key = outcomes.count)

# creating child nodes
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    #check if there is either left or right child
    if not left or not right:
         node['left']=node['right']= to_terminal(left+right)
         return
    #check for max depth
    if depth>=max_depth:
        node['left'], node['right'] = to_terminal(left),to_terminal(right)
        return
    #process left child
    if len(left)<= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] =get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    #process right child
    if len(right)<= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] =get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

# building the decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root,max_depth,min_size,1)
    return root
# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))
 
# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']
 
# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)
 
# Test CART on Indian Diabetes dataset
seed(1)
# load and prepare data
filename = 'pima-indians-diabetes.data.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
#for last char row
for row in dataset:
    if row[len(dataset[0])-1].strip()=='R':
        row[len(dataset[0])-1] = 0
    else:
        row[len(dataset[0])-1] = 1
# evaluate algorithm
n_folds = 5
max_depth = 10
min_size = 10
scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)

print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
