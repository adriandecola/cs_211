"""
Course:        DCS 211 Winter 2021 Module D
Assignment:    Project 3
Topic:         Analysis and Data Visualization
Purpose:       To fetch data, parse HTMLs, and create Choropleth plots of data
               that we find interesting.

Student Name: Adrian deCola

Other students outside my pair that I received help from ('N/A' if none):
N/A

Other students outside my pair that I gave help to ('N/A' if none):
N/A

Citations/links of external references used ('N/A' if none):
N/A
"""

import numpy as np
import pandas as pd
import requests
import json
import time
import os
import re
import plotly.express as px
import us
import plotly.figure_factory as ff

from urllib.request import urlopen
from bs4 import BeautifulSoup
from progress.bar import Bar # see https://pypi.org/project/progress/

import geopy.geocoders
from geopy.geocoders import Nominatim
geopy.geocoders.options.default_user_agent = "dcs211_adecola@bates.edu/3"
geopy.geocoders.options.default_timeout    = 10

############################
def getMostPopulousCities():
    '''
    Scrapes a Wikipedia page of the most populous cities per US state, returning
    a list of strings of the form "city, state".  To avoid multiple requests,
    this function saves a local copy of the html for subsequent parsing.

    Parameters:
    -----------
    None

    Returns:
    --------
    list
        a list of 50 strings, in the form "city, state"

    '''

    url = "https://en.wikipedia.org/wiki/List_of_largest_cities_of_U.S._states_and_territories_by_population"

    fname = "cities.html"
    if os.path.exists(fname):
        # if the local cities.html already exists, just read its HTML content
        with open(fname, "rb") as infile:
            content = infile.read()
    else:
        # if the local cities.html does not exist, fetch via requests.get and
        # then write a local copy of the HTML content to a file
        page = requests.get(url)
        content = page.content
        with open(fname, "wb") as outfile:
            outfile.write(content)

    cities = []

    # parse the HTML content
    soup  = BeautifulSoup(content, "html.parser")

    # upon inspection using Firefox's Web Developer Tools, we can locate the
    # cities of interest in a <table> by finding the <span>s with class
    # state/territory 'flagicon', and then grabbing the <tr> row that
    # encompasses the span
    spans = soup.find_all("span", class_="flagicon")
    for i in range(len(spans)):
        # omitting indices for Washington DC and non-state districts and
        # territories (i.e., that aren't on the 50-state US map for later)
        if i in [2, 9, 12, 37, 42, 50]:
            continue  # move on to next ith iteration
        span = spans[i]
        # work back two levels to get the <tr> row surrounding this state
        tr = span.parent.parent
        # then grab all <td> columns in that row, and the state and city text for
        # the most populous city will be in the 1st and 3nd entries respectively
        tds   = tr.find_all("td")
        state = tds[0].a.get("title")
        city  = tds[2].a.get("title")
        # in the table, some cities already have the state in their text
        if not "," in city:
            city = f"{city}, {state}"
        # a few have parenthetic information with "state" inside -- remove
        # that via regex
        city = re.sub(r"(.*?)\(.*state\).*?", r"\1", city)
        cities.append(city)

    return cities

########################
def getLatLongs(cities):
    '''
    Given a list of strings corresponding to the most populous city per each of
    the 50 US states, uses geocode's Nominatim to get the (lat,long) of each
    city, return a dictionary mapping city to (lat,long).  To avoid unnecessary
    repeated requests of Nominatim, this function saves a local copy of the
    dictionary for subsequent json loading.

    Parameters:
    -----------
    cities : list
        a list of str, each corresponding to "city, state" for the most populous
        city in each of the 50 US states

    Returns:
    --------
    dict
        a dictionary mapping each "city, state" to its (lat,long) tuple

    '''
    fname = "city_lat_long.json"
    if os.path.exists(fname):
        # if the local dictionary text file already exists, just read and then
        # load via json.loads
        with open(fname, "r") as infile:
            data = infile.read()
        city_lat_long_dict = json.loads(data)
    else:
        # if the local file does not exist, get each (lat,long) via geocode's
        # Nominatim, building a dict along the way
        nominatim = Nominatim()
        city_lat_long_dict = {}
        bar = Bar("Processing cities: ", max = 50)
        for city in cities:
            time.sleep(1)  # be polite!
            location  = nominatim.geocode(city)
            latitude  = location.latitude
            longitude = location.longitude
            city_lat_long_dict[city] = (latitude, longitude)
            bar.next()
        bar.finish()

        # dump the newly-built dictionary to a local file for subsequent loading
        with open(fname, "w") as outfile:
            json.dump(city_lat_long_dict, outfile)

    return city_lat_long_dict

def getAirQuality(city_lat_long_dict):
    '''
    This functions fetches the current air quality for each of the most populous
    cities per state. It therefore accepts the city_lat_long_dict as a parameter.
    It fetches the data from openweathermap.org. It politely waits between each
    request.

    Parameters:
    -----------
    city_lat_long_dict: dict
        a dictionary that maps the most populous cities of each state, in the
        form city, state, to a tuple of its latitude and longitude

    Returns:
    --------
    dict
        a dictionary that maps the most populous cities of each state, in the
        form city, state, to its current air quality on a scals of 1-5 from
        openweathermap.org
    '''

    airQual = {}
    # progress bar
    bar = Bar("Finding current air quality for each of the most populous cities: ",
    max = 50)
    for city in city_lat_long_dict:
        # requesting information
        url = "http://api.openweathermap.org/data/2.5/air_pollution"
        inputs = {
            "lat"   : city_lat_long_dict[city][0],
            "lon"  : city_lat_long_dict[city][1],
            # user will have to get their own API key from openwathermap.org to run this function
            "appid" : "######################"
        }
        response = requests.get(url, params = inputs)
        # making sure requested information came through
        if not response.ok:
            print(response.text)
            return
        else:
            # building minMaxDict with requested information
            info = response.json()
            airQual[city] = info["list"][0]["main"]["aqi"]
        # progress bar and being polite
        bar.next()
        time.sleep(1)
    bar.finish()

    return airQual

def drawUSMap(airQual):
    '''
    This functions plots the data inside of airQual, a dictionary that maps the
    most populous cities of each state, in the form city, state, to its current
    air quality on a scals of 1-5 from openweathermap.org. It makes a choropleth
    map that will be opened in your browser.

    Parameters:
    -----------
    airQual: dict
        a dictionary that maps the most populous cities of each state, in the
        form city, state, to its current air quality on a scals of 1-5 from
        openweathermap.org

    Returns:
    --------
    None
    '''

    state_codes = []
    AQI = []
    for location in airQual:
        # building up AQI list
        AQI.append(airQual[location])
        # building up state_codes list
        state = location.rsplit(", ")[1].strip() #could've also used regular expressions
        state = us.states.lookup(state).abbr
        state_codes.append(state)
    #creating figure
    fig = px.choropleth(locations = state_codes,
                        locationmode = "USA-states",
                        color = AQI,
                        scope = "usa")
    fig.update_layout(title_text = "Air Quality Index of Most Populous US Cities in each State " + \
                                   "(1 is Good, 5 is Very Poor)")
    fig.show()

###################
def drawCovidMap():
    '''
    This non-fruitul function uses pandas to read the Covid.csv file. This file
    therefore needs to be accessible. This file is from github: more specifically,
    it is a csv file acessible from github that tracks CoVid data from the NY
    Times. Since this project asked for a specific CSV file, this function does
    not request the csv file from github each time, though this certainly would
    have been possible. The csv attached in lyceum contains CoVid data up to May
    4. This is the url for the raw(csv) data from github:
    https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv
    This function then makes a choropleth map for all the counties in the US
    based on the number of CoVid cases the county has had up to May 4. The most
    cases considered is 80,000. This information is interesting because, depending
    how many cases a county has had, could be indicative, to some amount, through
    a proper robust regression, of how willing citizens are to get vaccinated.
    Of course there would be other factors, education, new outlets used, etc.
    Nonetheless this information would certainly be useful in one aspect of this
    calculation of willingness to get vaccinated.
    I've seen lots of plots for CoVid cases, but never a choropleth one. Also,
    the way the choropleth map looked on counties looked so cool to me; inspiring
    me to create this map.

    Parameters:
    -----------
    None

    Returns:
    --------
    None
    '''

    # using geoJson approach
    url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
    with urlopen(url) as response:
        counties = json.load(response)
    #reading the file as a pandas data frame
    filename = "Covid.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, dtype = {"fips": str})
    else:
        print("Need to download the Covid.csv file.")
        return
    #creating the choropleth map
    fig = px.choropleth(df, geojson=counties, locations='fips', color='cases',
                           color_continuous_scale="Viridis",
                           range_color=(0, 80000),
                           scope="usa",
                           labels={'cases':'CoVid-19 Cases up to May 4, 2021'}
                           )
    fig.update_layout(title_text='CoVid-19 cases by US county by May 4, 2021',
                      annotations = [dict(
                        x=0.55,
                        y=0.1,
                        xref='paper',
                        yref='paper',
                        text='Source: <a href="https://raw.githubusercontent.com/nytimes/covid-19-data/master/live/us-counties.csv">\
                            Github NYTimes Data</a>',
                        showarrow = False
                        )]
                        )
    fig.show()

###########
def main():

    city_state_list    = getMostPopulousCities()
    city_lat_long_dict = getLatLongs(city_state_list)

    # This function will work once the user gets an API key from openweathermap.org and inputs it in the getAirQuality function
    """
    try:
        # suggestions for function names
        airQual        = getAirQuality(city_lat_long_dict)
        drawUSMap(airQual)
        # Due to the original data being taken down this no longer works. 
        # drawCovidMap()
    except Exception as error:
        print("Missing required function definition:")
        print(f"{error}")
    """


if __name__ == "__main__":
    main()
