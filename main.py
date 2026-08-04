
import sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    # Write your primary logic here
    print(sklearn.__version__)

    # Load dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create model
    model = KNeighborsClassifier(n_neighbors=3)

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)

    print("Accuracy:", accuracy)            

    print("This is the end of the program ")

if __name__ == "__main__":
    main()