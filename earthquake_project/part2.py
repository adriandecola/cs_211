"""
Course:        DCS 211 Winter 2021 Module D
Assignment:    Project 2a
Topic:         API/Web Scraping
Purpose:       Use web scraping to gather and manipulate information about earthquake data.
               In particular, the the program can display information on earthquakes within
               a given time frame and above a certain magnitude, within a given time frame
               and radius of a particular latitude-longitude coordinate, and withing a given
               time frame and radius of a particular address. 

Student Name: Amanda Mai Becker
Partner Name: Adrian deCola

Other students outside my pair that I received help from ('N/A' if none):


Other students outside my pair that I gave help to ('N/A' if none):


Citations/links of external references used ('N/A' if none):

"""

import requests
import time
from datetime import datetime
import geopy.geocoders
from geopy.geocoders import Nominatim
geopy.geocoders.options.default_user_agent = "dcs211_001119935/1"
geopy.geocoders.options.default_timeout    = 10
geolocator = Nominatim()

'''
Helper functions
'''
def fixTime(timeCode):
    """
    Converts a USGS time code into a date and time in standard format.

    Parameters
    -----------
    timeCode: int
        a USGS time code as an integer

    Returns
    -------
    timeStamp: str
        a string including the date in yyyy-mm-dd format and the time in
        hour:min UTC format.
    """
    fixed = int(timeCode) / 1e3 ; gmt = time.gmtime(fixed)
    year = str(gmt[0]) ; month = str(gmt[1]) ; day = str(gmt[2])
    hour = str(gmt[3]) ; min = str(gmt[4])
    if len(month) == 1:
        month = "0" + month
    if len(day) == 1:
        day = "0" + day
    timeStamp = year + "-" + month + "-" + day + " @ " + hour + ":" + min + " UTC"
    return timeStamp

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
Major Functions
'''
def printQuakes(start, end, minMag):
    # if the dates entered are not valid, returns error message; otherwise,
    # proceeds with rest of code
    if isValidDate(start, end) == False:
        print(f"start date {start} and end date {end} must be in YYYY-MM-DD" + \
              f" format with start date before or equal to end date.")
    else:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        inputs = { \
            "format" : "geojson",
            "minmagnitude" : minMag,
            "starttime" : start,
            "endtime" : end}
        response = requests.get(url, params = inputs)
        if not response.ok:
            print(response.text)
        else:
            quake_dict = response.json()
            quake_list = quake_dict['features']
            print(f"All earthquakes of magnitude {minMag} or greater between " + \
                  f"{start} and {end}:")
            for i in range(len(quake_list)):
                prop_list = quake_list[i]["properties"]
                printQuakeInfo(prop_list)

def printQuakesByLatLong(start, end, lat, long, rad):
    # if lat N, positive; if lat S, negative
    # if long E, positive; if long W, negative
    if isValidDate(start, end) == False:
        print(f"start date {start} and end date {end} must be in YYYY-MM-DD" + \
              f" format with start date before or equal to end date.")
    if isValidLatLong(lat, long) == False:
        print(f"Latitude {lat} must be in -90 to 90 range inclusive, and" + \
              f" longitude must be in -180 to 180 range inclusive.")
    if isValidQuantity(rad) == False:
        print(f"Radius {rad} must be able to convert to a float and be" + \
              f" greater than 0.")
    else:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        inputs = { \
            "format" : "geojson",
            "starttime" : start,
            "endtime" : end,
            "latitude" : lat,
            "longitude" : long,
            "maxradiuskm" : rad}
        response = requests.get(url, params = inputs)
        if not response.ok:
            print(response.text)
        else:
            quake_dict = response.json()
            quake_list = quake_dict['features']
            for i in range(len(quake_list)):
                prop_list = quake_list[i]["properties"]
                printQuakeInfo(prop_list)

def printQuakesByAddress(start, end, addr, rad):
    location = geolocator.geocode(addr)
    lat = location.latitude
    long = location.longitude
    printQuakesByLatLong(start, end, lat, long, rad)



def main():
    """
    #printQuakes("2021-01-01", "2021-01-05", 5.0)
    #print("-" * 55)
    #time.sleep(2)   # sleep for 2s

    #printQuakesByLatLong("2021-01-01", "2021-05-05", 58.3019, -134.4197, 500)
    #print("-" * 55)
    #time.sleep(2)   # sleep for 2s
    printQuakesByLatLong("2016-01-01", "2021-01-01", 34.7464809, -92.2895948, 50)
    print("-" * 55)
    time.sleep(2)   # sleep for 2s
    #printQuakesByLatLong("2020-08-01", "2020-08-05", 34.3853, 132.4553, 500)
    #printQuakesByAddress("2020-08-01", "2021-04-01", "175 5th Avenue NYC", 500)


    # print(fixTime(1609762574292))

    #print(isValidDate("01-01-05", "2021-01-01"))
    #print("should be False")
    #print(isValidDate("2021-01-01", "2021-01-05"))
    #print("should be True")

    #printQuakes("2021-01-01", "2021-01-05", 5.0)
    #print("-" * 55)
    #time.sleep(2)   # sleep for 2s

    '''
    print(isValidQuantity(-0.2))
    print("should be False")
    print(isValidQuantity(0.2))
    print("should be True")
    print(isValidQuantity(1))
    print("should be True")
    print(isValidQuantity(0))
    print("should be false")

    print("-" * 55)

    print(isValidLatLong(-20, -180))
    print("should be True")
    print(isValidLatLong(-100, -100))
    print("should be False")
    print(isValidLatLong(90, 180))
    print("should be True")
    print(isValidLatLong(0, 0))
    print("should be True")
    '''
    #print("-" * 55)
    #time.sleep(2)   # sleep for 2s
    """

    # Interaction created for a presentation on github
    
    print("Printing all the earthquakes within the given timeframe and above a certain magnitude.")
    start = input("Enter a starting date (YYYY-MM-DD): ")
    end = input("Enter an ending date (YYYY-MM-DD): ")
    mag = input("Enter a minimum magnitude: ")
    printQuakes(start, end, mag)
    print()

    print("Printing all the earthquakes within a given time frame and radius of a particular "
          "latitude-longitude coordinate.")
    start = input("Enter a starting date (YYYY-MM-DD): ")
    end = input("Enter an ending date (YYYY-MM-DD): ")
    lat = input("Enter a latitude (positive for North and negative for South): ")
    long = input("Enter a longitude (positive for East and negative for West): ")
    rad = input("Enter a radius (km): ")
    printQuakesByLatLong(start, end, lat, long, rad)
    print()

    # Inputing an address of the correct form is so rare so I omitted this for now
    """    
    print("Printing all the earthquakes withing a given time frame and radius of a particular address")
    start = input("Enter a starting date (YYYY-MM-DD): ")
    end = input("Enter an ending date (YYYY-MM-DD): ")
    addr = input("Enter an address: ")
    rad = input("Enter a radius (km): ")
    printQuakesByAddress(start, end, addr, rad)
    """
    

if __name__ == "__main__":
    main()
