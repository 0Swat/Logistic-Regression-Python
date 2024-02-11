
# Project Title: Logistic Regression for Survival Prediction

## Description

This project demonstrates the implementation of logistic regression for predicting survival outcomes. Using the Haberman dataset, it explores both manual logistic regression implementation and the utilization of `scikit-learn`'s logistic regression model. The manual implementation includes a custom function for the sigmoid activation, computation of log likelihood, and the gradient descent algorithm for optimizing the model's parameters. The project also showcases the evaluation of the model's performance through classification reports, confusion matrices, and visualization of the latter using seaborn and matplotlib.

## Features

-   Custom implementation of logistic regression.
-   Use of `scikit-learn` for logistic regression.
-   Calculation of log likelihood for model evaluation.
-   Performance evaluation using classification reports and confusion matrices.
-   Visualization of confusion matrices.

## Requirements

-   Python 3.x
-   Numpy
-   Pandas
-   Scikit-learn
-   Seaborn
-   Matplotlib

Ensure these Python libraries are installed and available in your environment.

## Installation

No specific installation steps are required beyond setting up the Python environment and installing the necessary libraries. Clone this repository or download the script to your local machine to get started.

## Usage

The script can be executed as a standalone Python program. Ensure you have the Haberman dataset (`haberman.csv`) available in the same directory as the script, or adjust the script to point to the location of your dataset.

shell

`python logistic_regression.py` 

The script performs the following operations:

1.  Loads the Haberman dataset.
2.  Preprocesses the dataset for logistic regression.
3.  Implements logistic regression manually, including optimization of model parameters.
4.  Utilizes `scikit-learn`'s logistic regression to train and test on the dataset.
5.  Evaluates and visualizes the model's performance.

## Contributing

Contributions, suggestions, and issues are welcome. Please feel free to fork the repository, make your changes, and submit a pull request or open an issue for discussion.

## License

This project is open-source and available under the MIT License.

## Acknowledgments

-   The Haberman dataset is utilized for demonstrating the logistic regression model's application in survival prediction.
-   The `scikit-learn` library is used for logistic regression and model evaluation.
-   Visualization is enhanced by the `seaborn` and `matplotlib` libraries.

## Report
The entire work is described in more detail in the report "ZMO_SwatO_Regresje.pdf"
