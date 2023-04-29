import os
import pandas as pd
import numpy as np
from pybaseball import batting_stats
from sklearn.linear_model import *
from sklearn.feature_selection import *
from sklearn.model_selection import *
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

'''
Set Parameters for the range of years of data I want to use. I picked 2015 since that is when Statcast began.
Statcast is considered a pivotal turning point for advanced analytics in the MLB world.
Starting in 2015, all 30 stadiums were installed the technology that tracked player movements and athletic abilities
in real time (via Wikipedia page on Statcast).
'''
START_YEAR = 2015
END_YEAR = 2022

# Downloading the Data
if os.path.exists("mlbStats2015_2022.csv"):
    baseballData = pd.read_csv("mlbStats2015_2022.csv", index_col=0)
else:
    baseballData = batting_stats(START_YEAR, END_YEAR, qual=200)
    baseballData.to_csv("mlbStats2015_2022.csv")

'''
Split the Data based on "IDfg", which is the player's Fangraphs ID(Site where Data was downloaded from).
Each player has their own 'ID'.
'''
baseballData = baseballData.groupby("IDfg",group_keys=False).filter(lambda x: x.shape[0] > 1)

'''
Function that for every player, will add their home run count for the following season as a column. This is what we
want to predict for each year.
'''
def add_next_season_hr(player):
    player = player.sort_values("Season")
    player["Next_HR"] = player["HR"].shift(-1)
    return player

baseballData = baseballData.groupby("IDfg", group_keys=False).apply(add_next_season_hr)

# Counts the number of null values per attribute.
null_count_columns = baseballData.isnull().sum()

# Will get a list of all the columns where the number of null values in the column is 0.
complete_columns = list(baseballData.columns[null_count_columns == 0])

# Will updating the 'baseballData' dataframe to just be the complete columns in addition to Next_wRC(our target value).
baseballData = baseballData[complete_columns + ["Next_HR"]].copy()

'''
Here we are deleting some useless columns in the dataset as they are not important and are most importantly, not numeric.
Age Range will show if a player has a birthday during the season, it will show their age before and after, so for example,
'29-30' if a 29 year old had a birthday on June 1st and turned 30. It is not useful. 
'''
del baseballData["Age Rng"]
del baseballData["Dol"]

'''
This is another string column, but the player a team plays for can be important, so we don't want to delete it, so
I give them all unique ids. One con to this is if a player is traded during the season, they don't have a team, so it's
difficult to gauge value from this.
'''
baseballData["TeamID"] = baseballData["Team"].astype("category").cat.codes

'''
Here, you'll notice that earlier, when I did do Next_HR column, I was pulling next season's homeruns and putting it there.
However, for 2022, there is nothing for 2023, so what are we doing there?. Well, what I decided to do was to just use 2021's
data and push it there. It really isn't too important in the grand scheme as this column for 2022 is pointless, so I just did
this, so the code would actually run cause this column can't run with a null value.
'''
baseballData_full = baseballData.copy()
baseballData["Next_HR"].fillna(method='ffill', inplace=True)
baseballData = baseballData.dropna().copy()


'''
Here is where I select the model to use. 
As you can see, there is a Lasso and then a commented Ridge model.
'''
ls = LassoCV(cv=20, n_alphas=100)
#ls = Ridge(alpha=0.5)

'''
This is the Sequential Feature Selector. This is what is used to find the features in our dataset.
'''
split = TimeSeriesSplit(n_splits=20)
sfs = SequentialFeatureSelector(ls,n_features_to_select=20, direction="forward",cv=split, n_jobs=4)

removed_columns = ["Next_HR", "Name", "Team", "IDfg", "Season"]
selected_columns = baseballData.columns[~baseballData.columns.isin(removed_columns)]

'''
These commented lines below are what I used to get the predictors variable below. 
The reason this is commented is because these lines of code will singlehandedly take 10 minutes to run.
For older computers, it could take even longer, so this is basically doing the dirty work ahead of time.
'''
scaler = MinMaxScaler()
baseballData.loc[:,selected_columns] = scaler.fit_transform(baseballData[selected_columns])
#sfs.fit(baseballData[selected_columns],baseballData["Next_HR"])
#predictors = list(selected_columns[sfs.get_support()])

# All 20 attributes the Sequential Feature Selector picked out.
predictors = ['Age', 'G', 'IBB', 'GDP', 'FB', 'SLv', 'vCH (sc)', 'SL-X (sc)', 'KC-X (sc)',
              'SL-Z (sc)', 'KC-Z (sc)', 'vFS (pi)', 'BB%+', 'ISO+', 'FB%+', 'Hard%+', 'Barrels',
              'maxEV', 'HardHit', 'L-WAR']


# Function that performs the machine learning on the model.
def performTrainTest(baseballData, modelUsed, predictors, start=3, step=1):

    # Store all the predictions in this array.
    all_predictions_stored = []

    # All years of data used
    years = sorted(baseballData["Season"].unique())

    # For loop that basically says, 2015 to 2017 is training data, and 2018 onwards is testing data.
    for i in range(start, len(years), step):
        current_year = years[i]

        train = baseballData[baseballData["Season"] < current_year]
        test = baseballData[baseballData["Season"] == current_year]

        modelUsed.fit(train[predictors], train["Next_HR"])

        preds = modelUsed.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["Next_HR"], preds], axis=1)
        combined.columns = ["actual", "Proj HR"]

        all_predictions_stored.append(combined)
    return pd.concat(all_predictions_stored)

# This is the new dataframe that holds all the data
prediction = performTrainTest(baseballData,ls,predictors)

'''
These 2 lines of code while commented, are actually important.
The first one is what tells us the mean squared error.
The second one is what tells us the weight of each coefficient in the model.
'''
#print(mean_squared_error(prediction["actual"], prediction["Proj HR"]))
#print(pd.Series(ls.coef_, index=predictors).sort_values())

# Combine Dataframes
combinedData = prediction.merge(baseballData_full,left_index=True,right_index=True)

# This function will put the players next season HR in the dataframe
def next_season_homer(player):
    player = player.sort_values("Season")
    player["Next_HR"] = player["HR"].shift(-1)
    return player

# Call the above function
combinedData = combinedData.groupby("IDfg", group_keys=False).apply(next_season_homer)

'''
*********************************************************************************************************************************************************************************************************************
IMPORTANT. Due to the sheer size of the dataframe and how much console space it takes, I only print ONE season at a time.
This means what ever year you set 'szn' to is what year it will show. 
*********************************************************************************************************************************************************************************************************************
'''
print("Enter a year from 2018 to 2022 to see data from: ")
szn = input()


# Validation if input if they aren't a non-year or an invalid year.
try:
    szn = int(szn)
except ValueError:
    print("Input was not a year. Try again with a year from 2018 to 2022 (inclusive)")
    exit(1)

listOfValidSeasons = [2018,2019,2020,2021,2022]
if(szn not in listOfValidSeasons):
    print("You chose an invalid year. Please select a year from 2018 to 2022 (inclusive)")
    exit(2)


'''
This is just cleaning up the data for cleaner visualization when printing the table out. Such as converting columns that
are decimals to integers. I also created a column called 'Proj Rank' to show where they are expected to rank next year.
'''
recentData = combinedData[combinedData["Season"] == szn]
recentData2 = recentData.copy()
recentData2['Proj Rank'] = recentData2['Proj HR'].rank(method='dense',ascending=False)
recentData2['Proj HR'] = recentData2['Proj HR'].round().astype(int)
recentData2['Proj Rank'] = recentData2['Proj Rank'].round().astype(int)

'''
Depending on the year you pick, I print out different values in the table.
The reason being is that 'Next HR' is useless for printing 2022 out because we don't know that home runs for 2023.
However, it is useful for the years before that, so I print it out there.
'''
if(szn == 2022):
    print(recentData2[["Name","Age","Season","HR", "Proj HR",'Proj Rank']].sort_values("Proj Rank",ascending=True).to_string())
else:
    recentData2["Next_HR"].fillna(0, inplace=True)
    recentData2.drop(recentData2.index[recentData2['Next_HR'] == 0], inplace = True)
    recentData2['Next_HR'] = recentData2['Next_HR'].round().astype(int)
    print(recentData2[["Name","Season","HR","Next_HR","Proj HR",'Proj Rank']].sort_values("HR",ascending=False).to_string())


