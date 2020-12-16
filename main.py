# encoding: utf-8

import urllib.request
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from kneed import KneeLocator
# supress warnings
import warnings
warnings.filterwarnings('ignore')

# Importing all required packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import xticks
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio
import chart_studio.plotly as py
import chart_studio.tools as tls
import yaml
import math
import pandas as pd
import json

class Application:
	vallist = []
	def __init__(self):
		Username = 'VegaDavid'
		API_Key = '0tGMYx7lv3SpG3UMiLKO'

		# API Base URL
		main_api = 'https://cryptoudgcoin.com/api/v1/'
		# API Endpoints
		sections = ["users", "wallets", "history"]

		# Generamos nuestro primer DataFrame con los datos de las billeteras
		with urllib.request.urlopen(main_api+"wallets") as url:
		    data_Wallets = url.read()
		    df_Wallets = pd.read_json(data_Wallets)
		    df_Wallets = df_Wallets.drop(['id','description', 'created_at', 'updated_at', 'decimal_places', 'holder_type'], axis=1)
		    df_Wallets['balance_UDGC'] = df_Wallets.apply(self.udgcBalance, axis=1)
		    df_Wallets['balance_MXN'] = df_Wallets.apply(self.mxnBalance, axis=1)
		    df_Wallets = df_Wallets.replace('None','').groupby('holder_id',as_index=False).agg('sum')
		    df_Wallets = df_Wallets.rename(columns={"balance": "balance_Total"})
		    df_Wallets['balance_Total'] = df_Wallets['balance_Total'].astype(float)
		    pd.set_option("display.max_rows", None, "display.max_columns", None)

		#Generamos nuestro segundo DataFrame con los datos de los usuarios
		with urllib.request.urlopen(main_api+"users") as url:
		    data_Users = url.read()
		    df_Users = pd.read_json(data_Users)
		    df_Users = df_Users.drop(['name', 'last_name', 'email_verified_at', 'conekta_customer_id', 'created_at', 'updated_at','nip', 'phone', 'email', 'udg_code'], axis=1)
		    df_Users = df_Users.sort_values(by='id')
		    df_Users = df_Users.reset_index()
		    df_Users = df_Users.drop(['index'], axis=1)
		    df_Users["career"].fillna("No Estudiante", inplace = True)

		#Generamos nuestro tercer Dataframe con el historial de transacciones
		with urllib.request.urlopen(main_api+"history") as url:
		    data_History = url.read()
		    df_History = pd.read_json(data_History)
		    df_History = df_History.drop(['payable_type', 'meta', 'created_at', 'updated_at', 'confirmed', 'uuid'], axis=1)
		    df_History = df_History.dropna()

		#Prepracion de DataFrame para usar el modelo ML
		df_Total = pd.concat([df_Wallets, df_Users], axis = 1)
		df_Total = df_Total.drop(['id', 'holder_id'], axis=1)

		x = df_Total
		y = df_Total['career']

		le = LabelEncoder()
		x['career'] = le.fit_transform(x['career'])
		y = le.transform(y)

		cols = x.columns
		ms = MinMaxScaler()

		x = ms.fit_transform(x)
		x = pd.DataFrame(x, columns=[cols])

		kmeans = KMeans(n_clusters=3,random_state=0)

		kmeans.fit(x)

		labels = kmeans.labels_

		# check how many of the samples were correctly labeled
		correct_labels = sum(y == labels)

		cs = []
		for i in range(1, 11):
		    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
		    kmeans.fit(x)
		    cs.append(kmeans.inertia_)
		plt.plot(range(1, 11), cs, markersize=8, lw=2)
		plt.grid(True)
		plt.title('El metodo del codo')
		plt.xlabel('Numero de Clusters')
		plt.ylabel('Inercia')
		plt.savefig('codo.png')
		plt.show()
		df_History.plot(kind = 'scatter', x='id', y='amount', color='#FECC0D')
		plt.title('All the transactions')
		plt.xlabel('Transaction_id')
		plt.ylabel('Cantidad')
		plt.grid(True)
		plt.savefig('scatter.png')
		plt.show()

		clustering = KMeans(n_clusters = 3, max_iter = 300)
		clustering.fit(x)
		df_Total['KMeans_Clusters'] = clustering.labels_ #Los resultados del clustering se guardan en labels dentro del modelo
		print(df_Total)

		pca = PCA(n_components=3)
		pca_total = pca.fit_transform(df_Total)
		pca_total_df = pd.DataFrame(data = pca_total, columns = ['Componente_1', 'Componente_2', 'Componente_3'])
		pca_nombres_total = pd.concat([pca_total_df, df_Total[['KMeans_Clusters']]], axis = 1)

		fig = plt.figure(figsize = (10,10))

		ax = fig.add_subplot(1,1,1)
		ax.set_xlabel('componente 1', fontsize = 15)
		ax.set_ylabel('componente 2', fontsize = 15)
		ax.set_title('Componentes principales', fontsize = 15)

		color_theme = np.array (["blue", "green", "red"])
		ax.scatter(x=pca_nombres_total.Componente_1, y =pca_nombres_total.Componente_2, c=color_theme[pca_nombres_total.KMeans_Clusters], s=20, cmap='viridis')
		plt.show()

		chart_studio.tools.set_credentials_file(username=Username, api_key = API_Key)

		PLOT = go.Figure()

		for C in list(df_Total.KMeans_Clusters.unique()):

		    PLOT.add_trace(go.Scatter3d(x = df_Total[df_Total.KMeans_Clusters == C]['career'],
		                                y = df_Total[df_Total.KMeans_Clusters == C]['balance_MXN'],
		                                z = df_Total[df_Total.KMeans_Clusters == C]['balance_UDGC'],
		                                mode = 'markers', marker_size = 8, marker_line_width = 1,
		                                name = 'Cluster ' + str(C)))


		PLOT.update_layout( template="plotly_dark",width = 800, height = 800, autosize = True, showlegend = True,
		                   scene = dict(xaxis=dict(title = 'Carrera', titlefont_color = 'white'),
		                                yaxis=dict(title = 'Pesos Méxicanos', titlefont_color = 'white'),
		                                zaxis=dict(title = 'UDGC', titlefont_color = 'white')),
		                   font = dict(family = "Gilroy", color  = 'white', size = 12))
		PLOT.savefig('CryptoUDGCoin Clusters.png')
		PLOT.show()
		py.plot(PLOT, filename = 'CryptoUDGCoin Clusters', auto_open = True)

	def udgcBalance(self, row):
		if row['slug'] == 'udgc_wallet':
			val = row['balance']
		else:
			val = None

		return val

	def mxnBalance(self, row):
		if row['slug'] == 'mxn_wallet':
			val = row['balance']
		else:
			val = None

		return val

application=Application()
