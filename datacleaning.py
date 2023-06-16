import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
# Read the data

data = pd.read_csv('data.csv')

to_drop = ["Date"
,"ConcoursOuvertMois"
,"ConcoursOuvertAnnée"
,"PromotionCompétitionSemaine"
,"PromotionCompétitionAnnée"
,"Mois"
,"Trimestre"
,"Année"
,"Jour"
,"Semaine"
,"PromotionIntermédiaire"]

data = data.dropna()

data = data.drop(to_drop, axis=1)

# dummy variables for categorical data
data = pd.get_dummies(data, columns=['TypeDeMagasin', 'Saison'])


# plot repartition of Ventes
# data['Ventes'].plot(kind='hist')
# plt.show()

data["cat_Ventes"] = pd.cut(data["Ventes"], bins=[0, 5000, 10000,350000], labels=[0, 1, 2])
data["cat_Ventes"] = data["cat_Ventes"].fillna(2)

# categrorise JourFérié (0,a,b,c,...)
data["JourFérié"] = data["JourFérié"].astype('category')
data["JourFérié"] = data["JourFérié"].cat.codes

# categrorise ChoixArticles
data["ChoixArticles"] = data["ChoixArticles"].astype('category')
data["ChoixArticles"] = data["ChoixArticles"].cat.codes

print(data.iloc[0,:])
data.to_parquet('data_cleaned.parquet')

