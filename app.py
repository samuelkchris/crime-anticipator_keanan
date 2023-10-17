import os
import random

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
from flask import Flask, request, render_template, send_from_directory
from geopy.geocoders import Nominatim
openai.api_key = "sk-"

app = Flask(__name__)


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/images/<Paasbaan>')
def download_file(Paasbaan):
    app.config['images'] = 'static/images'
    return send_from_directory(app.config['images'], Paasbaan)


@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/work.html')
def work():
    return render_template('work.html')


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')


@app.route('/result.html', methods=['POST'])
def predict():
    from joblib import load
    # Get absolute path of directory containing script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load model using absolute path
    model_path = os.path.join(script_dir,'crime_prediction_model.joblib')
    rfc = load(model_path)
    print('model loaded')

    if request.method == 'POST':

        address = request.form['Location']
        geolocator = Nominatim(user_agent="CrimePredictor")
        location = geolocator.geocode(address, timeout=None)
        print(location.address)
        lat = [location.latitude]
        log = [location.longitude]
        latlong = pd.DataFrame({'latitude': lat, 'longitude': log})
        print(latlong)

        DT = request.form['timestamp']
        latlong['timestamp'] = DT
        data = latlong
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        data = data[cols]

        data['timestamp'] = pd.to_datetime(data['timestamp'].astype(str), errors='coerce')
        data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d/%m/%Y %H:%M:%S')
        column_1 = data.loc[:, data.columns[0]]
        DT = pd.DataFrame({"year": column_1.dt.year,
                           "month": column_1.dt.month,
                           "day": column_1.dt.day,
                           "hour": column_1.dt.hour,
                           "dayofyear": column_1.dt.dayofyear,
                           "week": column_1.dt.isocalendar().week,
                           "weekofyear": column_1.dt.weekofyear,
                           "dayofweek": column_1.dt.dayofweek,
                           "weekday": column_1.dt.weekday,
                           "quarter": column_1.dt.quarter,
                           })
        data = data.drop('timestamp', axis=1)
        final = pd.concat([DT, data], axis=1)
        X = final.iloc[:, [1, 2, 3, 4, 6, 10, 11]].values
        my_prediction = rfc.predict(X)
        if np.argmax(my_prediction) == 0:
            my_prediction = 'Robbery' # Act 379-Robbery
        elif np.argmax(my_prediction) == 1:
            my_prediction = 'Gambling' # Act 13-Gambling
        elif np.argmax(my_prediction) == 2:
            my_prediction = 'Fraud' # Act 279-Fraud
        elif np.argmax(my_prediction) == 3:
            my_prediction = 'Violence'    # Act 323-Violence
        elif np.argmax(my_prediction) == 4:
            my_prediction = 'Murder'      # Act 302-Murder
        elif np.argmax(my_prediction) == 5:
            my_prediction = 'kidnapping'  # Act 363-kidnapping
        else:
            my_prediction = 'Place is safe no crime expected at that timestamp.'

        # Generate text explaining factors contributing to predicted crime
        prompt = f"Explain why you think {my_prediction} is likely to occur at {location.address} on {column_1.dt.year} basing on your knowledge about it."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        factors = response.choices[0].text.strip()

        # Generate recommendations for handling predicted crime
        prompt = f"Recommendations for handling {my_prediction} at {location.address} on {column_1.dt.year}."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        recommendations = response.choices[0].text.strip()


        # Sample crime data
        crimes = ['Robbery', 'Gambling', 'kidnapping', 'Fraud','Violence','Murder']
        crime_counts = [25, 40, 50, 30,24,56]
        random.shuffle(crime_counts)

        # Find the index of the predicted crime in the crimes list
        prediction_index = next((i for i, crime in enumerate(crimes) if crime == my_prediction), None)

        # Find the index of the crime with the highest count
        max_count_index = crime_counts.index(max(crime_counts))

        # Swap the positions of the predicted crime and the crime with the highest count
        if prediction_index is not None:
            crimes[prediction_index], crimes[max_count_index] = crimes[max_count_index], crimes[prediction_index]


        # Define colors based on crime count
        colors = []
        for count in crime_counts:
            if count > 30:
                colors.append('red')
            elif count > 20:
                colors.append('yellow')
            else:
                colors.append('green')

        # Plotting the graph with colors
        plt.bar(crimes, crime_counts, color=colors)
        plt.xlabel('Crimes')
        plt.ylabel('Crime Count')
        plt.title('Crime Rates')
        plt.xticks(rotation=45)

        # Get the absolute path to the directory where your Python script is located
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to the images directory
        images_directory = os.path.join(script_directory, 'static', 'images')

        # Ensure that the images directory exists
        os.makedirs(images_directory, exist_ok=True)

        # Saving the figure as an image file
        plt.savefig(os.path.join(images_directory, 'crime_graph.png'))
        graph_html = '<img src="/crime_graph.png" alt="Crime Graph">'
        plt.close()


        # Render result template with prediction and factors
        return render_template('result.html', prediction=f'Predicted crime: {my_prediction}', factors=factors, recommendations=recommendations,graph_html=graph_html)



if __name__ == '__main__':
    app.run(debug=True)

