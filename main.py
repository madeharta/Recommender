
import sklearn
import mlflow
#import skops.io as sio

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def main():
    # Write your primary logic here
    print(sklearn.__version__)
    mlflow.set_experiment("demo")

    with mlflow.start_run():

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

        #print("Let's explore the data set feature here : ")
        #print("Data : ", iris.data)
        #print("Target : ", iris.target)
        #print("Feature Names : ", iris.feature_names)
        #print("Target Names : ", iris.target_names)
        #print("Description : ", iris.DESCR)
        # Train
        model.fit(X_train, y_train)

        # Evaluate
        accuracy = model.score(X_test, y_test)

        print("Accuracy:", accuracy)            

        # Here we will log the model to MLFLOW
    
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("Accuracy", accuracy)

        #trusted_types = sio.get_untrusted_types(file="model.skops")

        #print(trusted_types)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="KNN_algorithm",
            serialization_format="pickle"
        )

        print("Now make a new prediction using the model")

        new_flower = [[100.8, 3.0, 4.2, 1.2]]

        prediction = model.predict(new_flower)

        print("New prediction for the ", new_flower, "is : ", prediction)

if __name__ == "__main__":
    main()