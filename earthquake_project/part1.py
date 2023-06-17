"""
Course:        DCS 211 Winter 2021 Module D
Assignment:    Project 2
Topic:         API/Web Scraping
Purpose:       Use web scraping to gather and manipulate information from
               various websites. This includes making a dictionary of all
               the states and their capitals via scraping from wikipedia,
               writing the dictionary to a file, then finding the latitude
               and longitude of each capital including that in the dictionary
               and then writing that to another file, and then presenting minimum
               and maximum earthquakes between a given radius of each state
               capital. 

Student Name: Amanda Mai Becker
Partner Name: Adrian deCola

Other students outside my pair that I received help from ('N/A' if none):
N/A

Other students outside my pair that I gave help to ('N/A' if none):
N/A

Citations/links of external references used ('N/A' if none):
N/A
"""

import os
import requests
import json
import time
import re
from datetime import datetime
from bs4 import BeautifulSoup
from progress.bar import Bar
import geopy.geocoders
from geopy.geocoders import Nominatim
geopy.geocoders.options.default_user_agent = "dcs211_001119935/1"
geopy.geocoders.options.default_timeout    = 10
nominatim = Nominatim()

def createCapDict():
    """This function created a dictionary of all the states in the US and their
       capitals, scrapping the data from Wikipedia. 
    """

    url = "https://en.wikipedia.org/wiki/List_of_capitals_in_the_United_States"
    filename = "capitals.html"
    if os.path.exists(filename):
        with open(filename, "rb") as infile:
            html = infile.read()
    else:
        response = requests.get(url)
        if not response.ok:
            print(f"Error fetching {url}: {response.reason}")
            return None
        else:
            with open(filename, "wb") as outfile:
                outfile.write(response.content)
            html = response.content
    bSoup = BeautifulSoup(html, "html.parser")

    # grabs all states
    paras = bSoup.find_all('p')
    for para in paras:
        if "States:" in str(para):
            unorderedList = para.next_sibling.next_sibling
            unorderedListItems = unorderedList.find_all('li')
    states = []
    for li_item in unorderedListItems:
        a = li_item.find('a')
        title = a['title']
        # This wont work, but if we dont leave it in it doesn't work--will ask
        # after clasee
        title = re.sub(r" (state)", r"", title)
        states.append(title)

    # grabs all capitals
    tableInfo = bSoup.find_all('th', scope='row')
    info = []
    capitals = []
    for rows in tableInfo:
        links = rows.find_all('a')
        for results in links:
            title = results.get('title')
            info.append(title)
    for statesAndCaps in info:  # accessing capitals only
        splitStateCap = statesAndCaps.strip().split(', ')
        capitals.append(splitStateCap[0])

    # adds capitals and states to dictionary
    stateCapDict = {key: value for key, value in zip (capitals, states)}
    #writing a file with the dictionary
    if not "stateCapDict.txt" in os.listdir(os.getcwd()):
        filename = "stateCapDict.txt"
        with open(filename, "w") as outfile:
            json.dump(stateCapDict, outfile)  # dump the dict to a file
    return stateCapDict

def createLatLongDict():
    """This function creates a dictionary of each capital, state mapping to a tuple
    that contains its latitude then longitude and writes it using the json library
    to a text file called "location.txt". If this file already exist, it doesn't
    do this. No matter, the function prints this dictionary.
    This function assumes that stateCapDict.txt, from createCapDict() exists.
    This function is non-fruitful.
    Lat-longs appear to make sense by comparison and some checking.
    """

    #writing stateCapDict to txt file
    if not "location.txt" in os.listdir(os.getcwd()):
        #getting capital/state dict
        with open("stateCapDict.txt", "r") as infile:
            data = infile.read()
        capStateDict = json.loads(data)
        locDict = {}
        #writing lat long(tuple value) of each capital, state(key) to a file(json)
        bar = Bar("Finding lat-long for each state capital: ", max = 50)
        for cap in capStateDict:
            location = f"{cap}, {capStateDict[cap]}"
            #location but with lat and long accesible
            loc = nominatim.geocode(location)
            locDict[location] = (loc.latitude, loc.longitude)
            bar.next()
            time.sleep(1)
        bar.finish()
        filename = "location.txt"
        with open(filename, "w") as outfile:
            json.dump(locDict, outfile)  # dump the dict to a file

    #reading from location.txt file
    else:
        filename = "location.txt"
        with open(filename, "r") as infile:
            data = infile.read()        # read the dictionary as a string
        locDict = json.loads(data)
    print("Printing the states dictionary that contains info on their capital and "
          "the latitude and longitude of their capital:")
    print(locDict)

def printMinMaxQuakes():
    """This function asks the user for a start date, end date, and radius in km.
    The function then creates a dictionary that maps each state capital to a tuple
    containing first the minimum magnitude earthquake, then maximum magnitude
    earthquake, within the specified dates and radius and prints it. It also
    returns this dictionary.
    This function works when testing some of the capitals ( useing the project2a
    printQuakesByLatLong() )

    Returns
    -------
        dict: a dictionary that maps each state capital to a tuple containing
              first the minimum magnitude earthquake, then maximum magnitude
              earthquake, within the specified dates and radius
    """

    # if the dates entered are not valid, returns error message; otherwise,
    # proceeds with rest of code
    # assumes location.txt exists
    start = input("Enter a starting date (YYYY-MM-DD): ")
    end = input("Enter an ending date (YYYY-MM-DD): ")
    rad = input("Enter a radius (km): ")
    if isValidDate(start, end) == False:
        print(f"start date {start} and end date {end} must be in YYYY-MM-DD" + \
              f" format with start date before or equal to end date.")
    else:
        # downloading
        filename = "location.txt"
        with open(filename, "r") as infile:
            data = infile.read()  # read the dictionary as a string
        dict_ = json.loads(data)  # dict_: capital, state(key) maps to (lat, long)

        # progress bar
        bar = Bar("Finding min and max magnitude earthquake for each state capital: ",
        max = 50)
        #maps capital to min max magnitude earthquake (tuple)
        minMaxDict = {}
        for loc in dict_:
            #requesting information
            url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
            lat = dict_[loc][0]
            long = dict_[loc][1]
            inputs = { \
                "format" : "geojson",
                "starttime" : start,
                "endtime" : end,
                "maxradiuskm" : rad,
                "latitude" : lat,
                "longitude" : long}
            response = requests.get(url, params = inputs)
            # making sure requested information came through
            if not response.ok:
                print(response.text)
                return
            else:
                # building minMaxDict with requested information
                quake_dict = response.json()
                quake_list = quake_dict['features']
                minMag = None
                maxMag = None
                for i in range(len(quake_list)):
                    prop_list = quake_list[i]["properties"]
                    # some do not have a specified magnitude
                    if prop_list["mag"] != None:
                        if minMag == None or minMag > prop_list["mag"]:
                            minMag = prop_list["mag"]
                        if maxMag == None or maxMag < prop_list["mag"]:
                            maxMag = prop_list["mag"]
                    minMaxDict[loc] = (minMag, maxMag)
            # progress bar and being polite
            bar.next()
            time.sleep(1)
        bar.finish()
        #printing information
        print(f"Each state capital followed by the minimum magnitude earthquake " + \
              f"then the maximum magnitude earthquake within a distance of " + \
              f"{rad} km.")
        print("-"*60)
        for loc in minMaxDict:
            print(f"{loc:<30} \t {minMaxDict[loc][0]:<3} \t {minMaxDict[loc][1]:<3}")
        return minMaxDict

def isValidDate(start, end):
    """
    Checks if dates are in valid yyyy-mm-dd format and compares start and end
    dates to ensure start date preceeds end date.

    Parameters
    -----------
    start: str
        a string including a year, month, and day
    end: str
        a string including a year, month, and day

    Returns
    -------
    true: boolean
        true if date is in yyyy-mm-dd format and start date occurs before end
        date
    false: boolean
        false if date is not in yyyy-mm-dd format and/or start date occurs after
        end date
    """
    try:
        datetime.strptime(start, "%Y-%m-%d")
        datetime.strptime(end, "%Y-%m-%d")
        return True
    except ValueError:
        return False
    if start <= end:
        return True
    else:
        return False

def isValidQuantity(quant):
    """
    Checks if a number is a valid quantity, judging based on whether or not it
    can be converted to a float and if it is greater than or less than zero.

    Parameters
    -----------
    quant: numeric quantity
        a number

    Returns
    -------
    true: boolean
        true if number can be converted to a float and is greater than zero
    false: boolean
        false if number cannot be converted to a float and/or is less than or
        equal to zero
    """
    try:
        validQuant = float(quant)
    except ValueError:
        return False
    if validQuant > 0:
        return True
    else:
        return False

def printQuakeInfo(allInfo):
    """
    Prints information about earthquakes including magnitude, place, and time.

    Parameters
    -----------
    allInfo: dict
        a USGS dictionary containing information about earthquake properties

    Returns
    -------
    N/A
    Prints an f-string including magnitude, place, and time of earthquakes taken
    from the given dictionary
    """
    mag = float(allInfo['mag'])
    place = allInfo['place']
    theTime = fixTime(allInfo['time'])
    print(f"magnitude {mag} earthquake on {place} @ {theTime}")

def isValidLatLong(lat, long):
    """
    Checks if latitude and longitude are valid, judging based on whether or not
    latitude and longitude values can be converted to float and fall within the
    appropriate ranges (-90 to 90, inclusive for latitude and -180 to 180,
    inclusive for longitude).

    Parameters
    -----------
    lat: int/float
        a numeric value corresponding to a latitude
    long: int/float
        a numeric value corresponding to a longitude

    Returns
    -------
    true: boolean
        true if latitude and longitude can be converted to floats and are within
        the valid ranges for each (-90 to 90, inclusive for latitude and -180 to
        180, inclusive for longitude).
    false: boolean
        false if latitude and longitude cannot be converted to floats and/or
        fall outside of the valid ranges for each (-90 to 90, inclusive for
        latitude and -180 to 180, inclusive for longitude).
    """
    try:
        validLat = float(lat)
        validLong = float(long)
    except ValueError:
        return False
    if validLat >= -90 and validLat <= 90 and validLong >= -180 and validLong <= 180:
        return True
    else:
        return False

'''
def writeHTMLFiles(urls):
    filenames = []
    for url in urls:
        filename = url.split('/')[-1] + ".html"
        filenames.append(filename)
        if not os.path.exists(filename):
            time.sleep(2)
            response = requests.get(url)
            if response.ok:
                with open(filename, "wb") as outfile:
                    outfile.write(response.content)
                print(f"Writing {filename} successful")
            else:
                print(f"File {filename} already exists")
    return filenames
'''

def main():
    print("Printing the states and capitals dictionary: ")
    print(createCapDict())
    print()
    createLatLongDict()
    print()
    print("Gathering information on the earthquakes around each state capital "
          "within a given timeframe and radius.")
    print("Note: Do not make the radius to large as we are only allowed to request so much "
          "data at once.")
    printMinMaxQuakes()

###############################################################################

if __name__ == "__main__":
    main()
