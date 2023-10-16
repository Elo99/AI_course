from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train your custom Decision Tree Classifier
clf_custom = DecisionTreeClassifier(max_depth=2)
clf_custom.fit(X_train, y_train)

# Visualize the custom Decision Tree using graphviz
dot_data = export_graphviz(clf_custom, out_file=None,
                            feature_names=iris.feature_names,
                            class_names=iris.target_names,
                            filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")  # This will save the tree as a PDF file
graph.view("iris_decision_tree")    # This will open the PDF in a viewer


