# Travel Recommendation System

## Summary:

I like most people love travel. At the end of any trip, I'm already thinking about my next adventure. However, I often have trouble finding inspiration for my next trip. Many of the great travel sites, such as Fodors or Lonely Planet, have great content online but their websites often lack a strong search engine. For example, I recently tried searching 'I want great food in Spain' into the search bar on Lonely Plant's website and the search results were related to London, England... not exactly what I was looking for. I wanted to see if I could improve Lonely Planets's search engine and develop an app that would identify the perfect articles for my search interests. The following is a python flask application that takes thousands of articles from Lonelyplanet.com and identifies articles based on a users search input. I hope you enjoy!

## Instructions:

**Website Scrape**: Start by running the python script 'lonelyScrap.py' to scrape lonelyplanet.com and get all the articles on the website. I went ahead and ran the script for you with the articles stored in the file 'data_lonelyPlanet.json' located in the 'Data' folder. Please run the python script if you want the most up-to-date articles.

**Data Cleaning and Modeling**: Once the articles are obtained, run the jupyter notebook 'DataCleaning/ipynb'. The script requires the three json files in the 'Data' folder, so make sure to have the files downloaded. The script helps clean and categorize the articles prior to the modeling. Next, the script builds the necessary model using a combination of keyword targeting and an NLP technique called Doc2Vec. You can experiment with the results direcly in the script if you would like to see results outside of the Flask app. Finally, the script outputs a series of pickled files that you will need for the Flask app. 

**Flask Application**: The Flask app helps put all the information together in an easy to use format. Download the entire folder 'FlaskApp' and be sure to place the pickled files from the jupyter notebook into the folder. Next, run the the python script 'app.py'. The script will output a URL. The URL will direct you to the application where a user can input their desired travel interests and the application will output the three articles that best match the search input. The user can click 'Learn More' and the application will redirect the user to the full article on lonelyplanet.com.
