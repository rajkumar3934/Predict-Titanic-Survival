# Predict-Titanic-Survival
This project uses machine learning techniques to predict the survival of passengers on the Titanic. The dataset contains information about passengers, including their sex, age, class, and survival outcome. Logistic regression is employed as the predictive model.

## Dataset

The dataset used for this project is sourced from the Titanic Kaggle competition and consists of information about Titanic passengers, including their sex, age, class, and survival outcome. The dataset is stored in the `passengers.csv` file.

## Dependencies

To run the project, you'll need the following dependencies:

- pandas
- numpy
- matplotlib
- scikit-learn

## Installation

To use this script, first clone the repository:
```shell
$ git clone https://github.com/rajkumar3934/titanic-survival-prediction.git
```

Change into the project directory:

$ cd Predict-Titanic-Survival

Then, install the required dependencies:

```shell
$ pip install pandas numpy matplotlib scikit-learn
```

## Usage

To run the script, simply execute the `titanic_prediction.py` file:
```shell
$ python titanic_prediction.py
```
The script will train the logistic regression model on the dataset, evaluate its performance, and provide survival predictions for sample passengers.

## Results

The trained logistic regression model achieves an accuracy of 79% on the training data and 78% on the test data. The coefficients of the model indicate the impact of each feature on the survival prediction.

Here is an example of the coefficients:

[('Sex', 1.1875166993955768), ('Age', -0.3491925085457576), ('FirstClass', 0.9232550433575493), ('SecondClass', 0.5099535431171025)]

## Sample Predictions

The model predicts the survival outcome for the following sample passengers:

- Jack: [0.0, 20.0, 0.0, 0.0]
- Rose: [1.0, 17.0, 1.0, 0.0]
- You: [0.0, 24.0, 0.0, 1.0]

The predictions are as follows:

- Jack: Not Survived
- Rose: Survived
- You: Not Survived

## Contributing

If you'd like to contribute to this project, please fork the repository and create a new branch for your changes. Then, submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

