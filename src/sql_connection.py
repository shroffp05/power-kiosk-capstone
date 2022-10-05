#----------------------------------
# Power Kiosk - Demand Forecasting
# SQL Connection
# Date: 09-28-2022
#----------------------------------

import pyodbc 
import pandas as pd 
from dataclasses import dataclass 

@dataclass 
class connect_to_sql:

	server: str = "evprodsql01.powerkiosk.com"
	database: str = "PowerKiosk"
	driver: str = "{ODBC Driver 18 for SQL Server}"
	servercert: str = "Yes"
	username: str = "priyank.shroff"
	password: str = "UUFAIj3!L2Du"


	def _sql_connection(self):

		connection_string = 'DRIVER='+self.driver+';SERVER=tcp:'+self.server+';PORT=1433;DATABASE='+self.database+';UID='+self.username+';PWD='+self.password+';TrustServerCertificate='+self.servercert
		self.conn = pyodbc.connect(connection_string)
		self.cursor = self.conn.cursor()

	def _execute_sql_statement(self, sql_statement):

		#self.cursor.execute(sql_statement)
		df = pd.read_sql(sql_statement, self.conn)
		return df #self.cursor.fetchall()




if __name__=="__main__":

	sql_statement = "SELECT TOP 2 * FROM contractLocation"

	sql_con = connect_to_sql()
	sql_con._sql_connection()
	print(sql_con._execute_sql_statement(sql_statement))
	