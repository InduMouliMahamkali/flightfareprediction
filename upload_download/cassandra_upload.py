from flask import Flask, request
import cassandra
import pandas as pd
import logging

print(cassandra.__version__)

logging.basicConfig(filename='data_insertion_log.log', level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M', filemode='w')

app = Flask(__name__)

app.logger.info("initializing connectivity to database")
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config = {
    'secure_connect_bundle': 'C:\\Users\\Mouli\\Documents\\ineuron\\my poroject\\flight '
                             'fare\\secure-connect-ineuron.zip'}
auth_provider = PlainTextAuthProvider('BLxdawGCntYwfTBRxSICRyYe',
                                      'UWQ_eJ-oN5ZRksyr4Zkc8-b9Gg,XoFPk+zeJkcAL,'
                                      'KgF.SdjfZ4_CdufCut1yj+.CTkf3iB5PaLWr36Z2hmGMp19a-96+ZjnXcFqlCZ+rsPe6PRPoa'
                                      '-_ZCrP.zRJslr9')
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()
app.logger.info("connection established successfully")


@app.route('/cassandra_create_table', methods=['POST'])  # for calling the API from Postman/SOAPUI
def create_cass_table():
    try:
        app.logger.info("initializing to create a table in given keyspace")
        if request.method == 'POST':
            qurey = "CREATE TABLE train_data.data(ID int PRIMARY KEY,Airline text, Date_of_Journey text, Source text, " \
                    "Destination text, Route text,Dep_Time text, Arrival_Time text, Duration text, Total_Stops text," \
                    "Additional_Info text, Price int); "
            session.execute(qurey)
            app.logger.info("successfully created table in the required keyspace")
            return " Table is created i.e.,{qurey}"
    except Exception as e:
        print(e)
        app.logger.error("error occured while creating the table, which is :" + ' ' + str(e))
        return "error occurred is:" + str(e)


@app.route('/cassandra_bulk_insert', methods=['POST'])  # for calling the API from Postman/SOAPUI
def bulk_cass_table():
    import csv
    try:
        app.logger.info("inserting the bulk data into the database")
        if request.method == 'POST':
            with open('Data_Train.csv', 'r') as data:
                next(data)
                data_csv = csv.reader(data, delimiter=',')
                # csv reader object
                print(data_csv)
                all_value = []
                for i in data_csv:
                    session.execute(
                        "insert into train_data.data (ID,Airline, Date_of_Journey, Source, Destination, Route,"
                        "Dep_Time, Arrival_Time, Duration, Total_Stops,Additional_Info, Price) values(%s,%s,%s,%s,%s,"
                        "%s,%s,%s,%s,%s,%s,%s)",
                        [int(i[0]), i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], int(i[11])])
            app.logger.info("data insertion is completed")
            return 'finished'

    except Exception as e:
        print(e)
        app.logger.error("error occurred while inserting the data into database, which is:" + ' ' + str(e))
        return f"error occur is:" + str(e)


@app.route('/cassandra_download', methods=['POST'])  # for calling the API from Postman/SOAPUI
def download():
    import csv
    try:
        app.logger.info("initializing the data for download into csv file")
        if request.method == 'POST':
            qurey = f"SELECT * from train_data.data;"
            online_data = pd.DataFrame(list(session.execute(qurey)))
            online_data.drop("id", axis=1, inplace=True)
            online_data.to_csv("Data_Train_Download.csv", index=False)
            app.logger.info("successfully downloaded the data into csv file")
            return "All data is downloaded in download.csv"

    except Exception as e:
        print(e)
        app.logger.error("error occurred while downloading the data:" + ' ' + str(e))
        return "error occur is:" + str(e)


if __name__ == '__main__':
    app.run()
