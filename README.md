# Flight Fare Prediction
## Problem Statement:

<p>This project is all about creating an machine learning algorithm, that helps the user to know their flight journey cost by providing the required input. </p>

<p>Although the data provided resembles the real time data, if we can add the advantages of real time data like flight and seats availability, weather, country wise transportation charges we can increase the accuracy of prediction upto 20%-22%. And can make a significant impact on flight bookings. </p>

## Approach
<p>As the main goal is to predict the fares of the flights based on different factors available in the dataset.</p>
<pre> 
<li> Data Exploration     : I started exploring dataset using pandas,numpy,matplotlib and seaborn. </li>
<li> Data visualization   : Plotted graphs to get insights about dependent and independent variables. </li>
<li> Feature Engineering  :  Removed missing values and created new features as per insights.</li>
<li> Model Selection I    :  1. Tested all base models to check the base accuracy.
                             2. Also plotted residual plot to check whether a model is a good fit or not.</li>
<li> Model Selection II   :  Performed Hyperparameter tuning using gridsearchCV and randomizedSearchCV.</li>
<li> Pickle File          :  Selected model as per best accuracy and created pickle file using joblib .</li>
<li> Webpage & deployment :  Created a webform that takes all the necessary inputs from user and shows output.
                                And deployed the project on heroku</li></pre>

## Deployment Link
<p> Link Heroku : https://flightfare10101.herokuapp.com/prediction </p>

## UserInterface
![plot](./images/interface.png)


## Technologies Used
<pre> 
1. Python 
2. Sklearn
3. Flask
4. Html
5. Css
6. Pandas, Numpy 
7. Database (Cassandra)

</pre>
