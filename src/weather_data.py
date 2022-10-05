# importing weather data 

from datetime import datetime 
from meteostat import Point, Daily, Monthly 
import geopy as gp 
from  geopy.geocoders import Nominatim
import os  
import sys 
import pandas as pd 

class WeatherData:

	def __init__(self, start_date="2021-01-01", end_date="2021-03-01", city_name="Boston", state_name="Massachusetts", country="United States"):

		self.start_date = start_date
		self.end_date = end_date
		self.city_name = city_name
		self.state_name = state_name
		self.country = country
		self.location = None 
		self.weather_data = None 

		self._convert_date() 

	def _convert_date(self):

		"""
		Converts string formatted dates into datetime objects
		"""

		self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
		self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d")

	def _get_lat_long(self) -> gp.location.Location:

		"""
		Finds the latitude and longitude of the city 
		"""
		
		geolocator = Nominatim(user_agent="weather_forecast")
		self.location = geolocator.geocode(self.city_name+','+self.state_name+','+self.country)

	def _get_weather_data(self) -> pd.DataFrame:

		self._get_lat_long()
		location = Point(location_data.latitude, location_data.longitude)

		data = Monthly(location, self.start_date, self.end_date)
		data = data.fetch() 

		return data 


if __name__ == "__main__":

	weather = WeatherData()
	print(weather._get_weather_data())