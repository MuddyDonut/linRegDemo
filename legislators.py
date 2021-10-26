# pip install pandas sklearn

import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import sys

def main():

    # grabds data file and user inputs
    year = int(sys.argv[1])
    fileLocation = str(sys.argv[2])
    f = open(fileLocation)
    data = json.load(f)

    # create dicts for male and female times in office
    femYear = {}
    menYear = {}
    dataFill(data, femYear, "F", year)
    dataFill(data, menYear, "M", year)

    # creating data frame based on yearly aggrgations of both genders
    modelData = pd.DataFrame(femYear, index=[0]).transpose().rename(columns={0:"numFemales", "index":"Year"}).join(pd.DataFrame(menYear, index=[0]).transpose().rename(columns={0:"numMales", "index":"Year"}))
    
    #creating model parameters, splits, and predictions
    y = modelData.numFemales
    X = modelData[['numMales']]

    # sci-kit learn documentation -> https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    # sci-kit learn documentation -> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    reg = linear_model.LinearRegression()
    reg.fit(X_test, y_test)
    y_pred = reg.predict(X_test)
    accuracy = pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
    print(int(abs(reg.intercept_ + reg.coef_ * year)[0]))

def dataFill(data, emptyDict, gender:str, year):
    # this aggregates the terms for each year for both genders, accounts for gaps in terms
    for i in data:
        x = len(i['terms'])
        prevEnd = 0
        counter = 1
        
        if i['bio']['gender'] == gender:
        
            for t in i['terms']:
                start = int(t['start'][:4])
                end = int(t['end'][:4])

                for year in range(start, end + 1):
                    
                    #if the year is already a key value then +1 to the value
                    # if not add the year key and value as 1
                    if year in emptyDict:
                        emptyDict[year] = emptyDict.get(year) + 1
                    else:
                        emptyDict[year] = 1

                #if the same person started a new position the same year, remove duplicate
                if counter > 1 and start == prevEnd:
                    emptyDict[start] = emptyDict.get(start) - 1
                    
                counter +=1
                prevEnd = end

if __name__ == "__main__":
    main()