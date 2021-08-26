import pandas as pd
import logging as lg

lg.basicConfig(filename='data_transform.log', level=lg.INFO,
               format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M', filemode='w')


def conversion(a,b):
    """This function takes two inputs from user one for conversion of the excel file to csv file.
        The user must provide the path to excel file or just the fie name if the data file and program file are in the same path.
        the second input is for saving the data from the excel file into a csv file, the user must provide the file name where the data to be stored"""
    try:
        df = pd.read_excel(a)
        lg.info("the data is read from the file" + ' ' + str(a) + ' ' + "into pandas")
        df.to_csv(b + '.csv')
        lg.info("the data is downloaded into the file named:" + ' ' + str(b))
        print("data loaded successfully\nthe data was successfully downloaded into csv file")
        lg.info("displayed the first five rows of data from file:" + ' ' + str(b))
        return df.head()
    except Exception as e:
        return "exception occured is" + str(e)


a = input(str("enter the filename/path to be loaded:"))
lg.warning("input file given is:" + ' ' + str(a))
b = input(str("downloadable file name:"))
lg.warning("the downloaded file name is:" + ' ' + str(b))
conversion(a,b)