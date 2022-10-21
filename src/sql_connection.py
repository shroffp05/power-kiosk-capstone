#----------------------------------
# Power Kiosk - Demand Forecasting
# SQL Connection
# Date: 09-28-2022
#----------------------------------

import pyodbc 
import sqlalchemy as sal 
from sqlalchemy import create_engine 
from sqlalchemy.engine import URL 
import pandas as pd 
from dataclasses import dataclass 

@dataclass 
class connect_to_sql:

	server: str = "evprodsql01.powerkiosk.com"
	database: str = "PowerKiosk"
	driver: str = "ODBC Driver 18 for SQL Server"
	servercert: str = "Yes"
	username: str = "priyank.shroff"
	password: str = "UUFAIj3!L2Du"


	def _sql_connection(self):

		connection_string = URL.create(
				"mssql+pyodbc",
				username=self.username,
				password=self.password,
				host=self.server,
				port=1433,
				database=self.database,
				query={
					"driver": self.driver,
					"TrustServerCertificate": self.servercert
				},
			)

		#'DRIVER='+self.driver+';SERVER=tcp:'+self.server+';PORT=1433;DATABASE='+self.database+';UID='+self.username+';PWD='+self.password+';TrustServerCertificate='+self.servercert
		self.engine = create_engine(connection_string)

	def _execute_sql_statement(self, sql_statement):

		self.conn = self.engine.connect()
		df = pd.read_sql(sql_statement, self.engine)
		self.conn.close() 
		return df 

	