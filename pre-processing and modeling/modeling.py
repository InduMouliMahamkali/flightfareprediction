### importing all the required modlues
import pandas as pd
from matplotlib import pyplot as plt
from pandas_profiling import ProfileReport
from IPython.core.display import display
import seaborn as sns
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV


# reading data set using the pandas function
df = pd.read_csv("Data_Train_Download.csv")
#displaying the first 5 data
df.head()
# describing the whole data set which show cases the mean, first quatile etc...
df.describe()

# showing the profile report using the pandas profile report function
pf = ProfileReport(df,title="Complete report of original data")
display(df)
# rendering the data report into a html page
pf.to_file('Data_set_report.html')

df.isnull().sum()  # checking for null values
df.dropna(inplace=True)     # droppinng the null values
print(df.isnull().sum())

# creating a function to convert the required columns into the date time format using the pandas datetime module
def change_into_datetime(col):
    df[col] = pd.to_datetime(df[col])


for i in ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']:
    change_into_datetime(i)

df['journey_day'] = df['Date_of_Journey'].dt.day     # extracting the day form the date of journey
df['journey_month'] = df['Date_of_Journey'].dt.month      # extracting the month form the date of journey

print(df.dtypes)

# creating a function to extract hours form the required columns
def extract_hour(data, col):
    data[col + '_hour'] = data[col].dt.hour

# creating the function to extrac minutes from the required columns
def extract_min(data, col):
    data[col + '_min'] = data[col].dt.minute


def drop_col(data, col):
    data.drop(col, axis=1, inplace=True)


extract_hour(df, 'Dep_Time')

# extracting minutes
extract_min(df, 'Dep_Time')

# drop the column
drop_col(df, 'Dep_Time')

extract_hour(df, 'Arrival_Time')

# extracting min
extract_min(df, 'Arrival_Time')

# drop the column
drop_col(df, 'Arrival_Time')

duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:  # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"  # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]  # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))  # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))  # Extracts only minutes from duration

df["Duration_hours"] = duration_hours
df["Duration_mins"] = duration_mins


# getting weekday names from the date of journey
df["weekday_name"] = df[['Date_of_Journey']].apply(lambda x: x.dt.day_name())

# rendering different graphs for data visualizations
sns.catplot(x='Airline', y='Price', data=df, height=6, aspect=2)
sns.catplot(x='Source', y='Price', data=df, height=6, aspect=2)
sns.catplot(x='Destination', y='Price', data=df, height=6, aspect=2)
sns.catplot(x='weekday_name', y='Price', data=df, height=6, aspect=2)
sns.catplot(x='Airline', y='Price', kind='box', data=df, height=6, aspect=2)
sns.catplot(x='weekday_name', y='Price', kind='box', data=df, height=6, aspect=2)
df.drop(["weekday_name", "Date_of_Journey", "Duration", "Additional_Info"], inplace=True, axis=1)


print(df.dtypes)
print(df.describe())

# extracting all the object columns so as to know the description of the such columns
list_col = df.dtypes[df.dtypes == "object"].index
df[list_col].describe()

# dummhy encoding the airline column
Airline = pd.get_dummies(df['Airline'], drop_first=True)
Airline.head()

# dummy encoding the source column
source = pd.get_dummies(df['Source'], drop_first=True)
print(source.head())

# dummy encoding the destionation column
destination = pd.get_dummies(df['Destination'], drop_first=True)
print(destination.head())

# maping the total stops columnn to 0,1,2,3,4 respectively
dict = {'non-stop': 0, '2 stops': 2, '1 stop': 1, '3 stops': 3, '4 stops': 4}
df['Total_Stops'] = df['Total_Stops'].map(dict)

print(df['Total_Stops'])

# concatenating all the newly created columns
df = pd.concat([df, Airline, source, destination], axis=1)

print(df.dtypes)

# dropping all the non required colummns
df.drop(["Airline", 'Source', 'Destination', 'Route'], inplace=True, axis=1)


print(df.columns)


# MODELLING

x = df.drop(['Price'], axis=1)  # creating our feature column

y = df['Price']      # Creating our target column
print(y)

plt.figure(figsize=(20, 20))
sns.heatmap(x.corr(), annot=True, cmap="RdYlGn")



# knowing the important features using extra tree regressor
selection = ExtraTreesRegressor()
selection.fit(x, y)

print(selection.feature_importances_)

plt.figure(figsize=(12, 8))
feat_importances = pd.Series(selection.feature_importances_, index=x.columns)
feat_importances.nlargest(20).plot(kind='barh')



# splitting our data into train test validations
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=126)


# fitting our model into random forest regressor algorithm
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)

y_pred = reg_rf.predict(X_test)  # predicting the values using the test data

reg_rf.score(X_train, y_train)  # knowing the scores

reg_rf.score(X_test, y_test)

sns.distplot(y_test, y_pred)


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")




print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

metrics.r2_score(y_test, y_pred)



# Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# Create the random grid

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# Random search of parameters, using 5 fold cross validation,
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator=reg_rf, param_distributions=random_grid, scoring='neg_mean_squared_error',
                               n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)

rf_random.fit(X_train, y_train)

print(rf_random.best_params_)

prediction = rf_random.predict(X_test)

plt.figure(figsize=(8, 8))
sns.displot(y_test - prediction)


plt.figure(figsize=(8, 8))
plt.scatter(y_test, prediction, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))



# open a file, where you ant to store the data
file = open('flight_rf.pkl', 'wb')

# dump information to that file
pickle.dump(reg_rf, file)

model = open('flight_rf.pkl', 'rb')
forest = pickle.load(model)   # loading our model into the interpreter

y_prediction = forest.predict(X_test)   #tesing the data using the loaded pickel file

metrics.r2_score(y_test, y_prediction)     # knowing the score for the predicted values using the loaded file

plt.show()