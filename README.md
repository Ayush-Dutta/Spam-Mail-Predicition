# Spam Detection Project

This project is a machine learning application that classifies email messages as either "spam" or "ham" (non-spam). The classification is performed using a logistic regression model trained on a dataset of labeled emails.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Prediction](#prediction)
- [Contributing](#contributing)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/spam-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd spam-detection
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Load and preprocess the dataset:
    ```python
    import pandas as pd

    Raw = pd.read_csv('/path/to/Data.csv')
    Data = Raw.where((pd.notnull(Raw)), '')
    ```

2. Train the model:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    X = Data['Message']
    Y = Data['Category']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_features, Y_train)
    ```

3. Evaluate the model:
    ```python
    from sklearn.metrics import accuracy_score

    prediction_on_training_data = model.predict(X_train_features)
    accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

    prediction_on_test_data = model.predict(X_test_features)
    accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
    ```

4. Make predictions:
    ```python
    input_mail = ["Your example email here..."]
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)

    if prediction[0] == 1:
        print('Ham mail')
    else:
        print('Spam mail')
    ```

## Dataset

The dataset used in this project can be downloaded from [Kaggle](https://www.kaggle.com/datasets/mohinurabdurahimova/maildataset). It consists of email messages labeled as "spam" or "ham".

## Project Structure

## Model Training and Evaluation

1. Preprocess the data:
    - Replace null values with empty strings.
    - Label "spam" mails as 0 and "ham" mails as 1.

2. Split the data into training and testing sets:
    ```python
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    ```

3. Transform the text data into feature vectors using TF-IDF vectorization.

4. Train a logistic regression model on the training data.

5. Evaluate the model on both training and testing data to check for overfitting and generalization.

## Prediction

To predict whether a new email is spam or ham, use the trained model:
```python
input_mail = ["WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only"]
input_data_features = feature_extraction.transform(input_mail)
prediction = model.predict(input_data_features)

if prediction[0] == 1:
    print('Ham mail')
else:
    print('Spam mail')
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes.

