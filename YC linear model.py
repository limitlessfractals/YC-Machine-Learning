import pandas
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold
import numpy as np
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


# We can use the pandas library in python to read in the csv file.
# This creates a pandas dataframe and assigns it to the startup variable.
startup = pandas.read_csv("/home/limitless/Documents/machine learning startup/Startups.csv")


# The startup variable is available here.
#startup.loc["Y Combinator Year"] = startup.loc["Y Combinator Year"] - 2000

# Status to numbers.
startup.loc[startup["Status"] == "Exited", "Status"] = 2
startup.loc[startup["Status"] == "Operating", "Status"] = 1
startup.loc[startup["Status"] == "Dead", "Status"] = 0


# Session to numbers.
startup.loc[startup["Y Combinator Session"] == "Winter", "Y Combinator Session"] = 0
startup.loc[startup["Y Combinator Session"] == "Summer", "Y Combinator Session"] = 1

# Find all the unique values for State.

startup["Headquarters (US State)"] = startup["Headquarters (US State)"].fillna(11)
startup.loc[startup["Headquarters (US State)"] == "California", "Headquarters (US State)"] = 1
startup.loc[startup["Headquarters (US State)"] == "New York", "Headquarters (US State)"] = 2
startup.loc[startup["Headquarters (US State)"] == "New Jersey", "Headquarters (US State)"] = 3
startup.loc[startup["Headquarters (US State)"] == "Massachusetts", "Headquarters (US State)"] = 4
startup.loc[startup["Headquarters (US State)"] == "Illinois", "Headquarters (US State)"] = 5
startup.loc[startup["Headquarters (US State)"] == "Colorado", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Washington", "Headquarters (US State)"] = 7
startup.loc[startup["Headquarters (US State)"] == "Arkansas", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Tennessee", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Texas", "Headquarters (US State)"] = 8
startup.loc[startup["Headquarters (US State)"] == "Giorgia", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Maryland", "Headquarters (US State)"] = 6
startup.loc[startup["Headquarters (US State)"] == "Michigan", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Minnesota", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Florida", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Delaware", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Ohio", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Pennsylvania", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Utah", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Oregon", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "California, New York", "Headquarters (US State)"] = 1
startup.loc[startup["Headquarters (US State)"] == "Virginia", "Headquarters (US State)"] = 10
startup.loc[startup["Headquarters (US State)"] == "Rhode Island", "Headquarters (US State)"] = 10



#startup.to_csv("/home/limitless/Documents/machine learning startup/testing1.csv", index = False)






#splitting data frame into training and test samples 
startup_train, startup_test = train_test_split(startup, test_size = 0.2)


# The columns we'll use to predict the target
predictors = ["Y Combinator Year", "Y Combinator Session", "Headquarters (US State)"]

# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the startup dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(startup.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (startup[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = startup["Status"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(startup[predictors].iloc[test,:])
    predictions.append(test_predictions)



# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions == startup["Status"])/len(predictions)
print(accuracy)

# Initialize our algorithm
alg = linear_model.LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, startup[predictors], startup["Status"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())



# Train the algorithm using all the training data
alg.fit(startup[predictors], startup["Status"])

# Make predictions using the test set.
predictions = alg.predict(startup_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "Company": startup_test["Company"],
        "Status": predictions
    })
submission.to_csv("/home/limitless/Documents/machine learning startup/Startups_pred_lin1.csv", index = False)

