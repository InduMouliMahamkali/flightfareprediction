import pandas as pd
import logging as lg

import pickle

lg.basicConfig(filename='data_test_automation.log', level=lg.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %('
                                                                          'message)s', datefmt='%m-%d %H:%M',
               filemode='w')


def automated(a):

    """This function takes the input of the test data file location, and performs all the data processing done
    on the test data set.
    For logs check the data_test_automation.log file in your system"""

    try:

        lg.warning("user gave the input path/file as:"+' '+str(a))
        df=pd.read_excel(a)
        lg.warning("data successfully loaded from the file/path"+' '+str(a))

        lg.info("starting all the pre-processing done for the train dataset")

        df.dropna(inplace=True)
        lg.warning("successfully dropped all null values in the given dataset")

        def change_into_datetime(col):
            df[col]=pd.to_datetime(df[col])

        for i in ['Date_of_Journey','Dep_Time', 'Arrival_Time']:
            change_into_datetime(i)
        lg.info("successfully changed the required columns into datetime format")

        df['journey_day']=df['Date_of_Journey'].dt.day
        lg.info("successfully extracted day from Date_of_journey and creating a separate column for day")
        df['journey_month']=df['Date_of_Journey'].dt.month
        lg.info("successfully extracted month from Date_of_Journey and creating a separate column for month")

        def extract_hour(data,col):
            data[col+'_hour']=data[col].dt.hour
        def extract_min(data,col):
            data[col+'_min']=data[col].dt.minute
        def drop_col(data,col):
            data.drop(col,axis=1,inplace=True)


        extract_hour(df,'Dep_Time')
        lg.info("successfully extracted hours from Dep_Time and dumped the data into new column Dep_Time_hour")
        extract_min(df,'Dep_Time')
        lg.info("successfully extracted minutes from Dep_Time and dumped the data into new column Dep_Time_min")
        drop_col(df,'Dep_Time')
        lg.warning("dropping the original Dep_Time column as we extracted the values form that column")
        extract_hour(df,'Arrival_Time')
        lg.info("successfully extracted hours from Arrival_Time and dumped the data into new column Arrival_Time_hour")
        extract_min(df,'Arrival_Time')
        lg.info("successfully extracted min from Arrival_Time and dumped the data into new column Arrival_Time_min")
        drop_col(df,'Arrival_Time')
        lg.warning("dropping the original Arrival_Time column as we extracted the values form that column")

        duration = list(df["Duration"])

        for i in range(len(duration)):
            if len(duration[i].split()) != 2:
                if "h" in duration[i]:
                    duration[i] = duration[i].strip() + " 0m"
                else:
                    duration[i] = "0h " + duration[i]

        duration_hours = []
        duration_mins = []
        for i in range(len(duration)):
            duration_hours.append(int(duration[i].split(sep = "h")[0]))
            duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))

        df["Duration_hours"] = duration_hours
        lg.info("successfully extracted hours from Duration column and dumped the data into new column Duration_hours")
        df["Duration_mins"] = duration_mins
        lg.info("successfully extracted minutes from Duration column and dumped the data into new column Duration_mins")

        df.drop(["Date_of_Journey","Duration","Additional_Info"], inplace=True,axis=1)
        lg.warning("dropping the Date_of_Journey, Duration, Additional_Info columns as we extracted the required "
                   "information")

        Airline=pd.get_dummies(df['Airline'],drop_first=True)
        lg.info("creating dummy variables for Airline and dropping the first dummy column")

        source=pd.get_dummies(df['Source'],drop_first=True)
        lg.info("creating dummy variables for Source and dropping the first dummy column")

        destination=pd.get_dummies(df['Destination'],drop_first=True)
        lg.info("creating dummy variables for Destination and dropping the first dummy column")

        dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}
        df['Total_Stops']=df['Total_Stops'].map(dict)
        lg.info("successfully mapped the Total_Stops column to 0,1,2,3,4 respectfully")

        df=pd.concat([df, Airline, source, destination], axis = 1)
        lg.warning("concatenating all the newly created columns into the main dataframe")

        df.drop(["Airline", 'Source', 'Destination','Route'],inplace=True,axis=1)
        lg.warning("dropping the categorical columns as we dummy encoded them")

        df['Trujet']=0
        lg.info("adding an extra column as this feature is not there in our test dataset")


        model = open('flight_rf.pkl','rb')
        forest = pickle.load(model)
        lg.info("loading our test model for prediction")

        y_prediction = forest.predict(df)
        lg.info("processing the prediction")

        a=pd.DataFrame(y_prediction)
        lg.info("dumping all our predicted values into a dataframe and showing the results")

        print(a)
        return a

    except Exception as e:
        lg.warning("error occurred during execution, which is:"+' '+str(e))
        return "error occurs is:"+' '+str(e)



a=input(str("give the file path or file name:"))
automated(a)