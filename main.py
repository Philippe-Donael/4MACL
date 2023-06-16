import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

display_graph = False

# Data Exploration

# Read the data
data = pd.read_csv('data.csv')

# Print the first 5 rows of the dataframe.

print(data.head())
print(data.describe())
print(data.columns)

cols = ['Magasin', 'JourDeSemaine', 'Date', 'Ventes', 'Clients', 'Ouverture',
        'PromotionsClassiques', 'JourFérié', 'VacancesScolaire',
        'TypeDeMagasin', 'ChoixArticles', 'DistanceCompétition',
        'ConcoursOuvertMois', 'ConcoursOuvertAnnée', 'PromotionCompétition',
        'PromotionCompétitionSemaine', 'PromotionCompétitionAnnée',
        'PromotionIntermédiaire', 'Mois', 'Trimestre', 'Année', 'Jour',
        'Semaine', 'Saison']

if display_graph:
    # plot Ventes vs JourDeSemaine
    sns.barplot(x='JourDeSemaine', y='Ventes', data=data)
    # add title to the plot
    plt.title("Nombre de ventes par jour de semaine")
    # replace the x-axis labels by the names of the days
    plt.xticks([0, 1, 2, 3, 4, 5, 6], ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    plt.show()

    # correlation matrix

    # drop columns with month, year, day, week, season



    # select the numerical columns
    num_cols = data._get_numeric_data().columns

    # calculate the correlation matrix
    corr = data[num_cols].corr()

    # plot the heatmap
    sns.heatmap(corr, annot=False, cmap='plasma')

    plt.show()

# display nan propositions
print(data.isna().sum()/len(data)*100)

# plot histo of nans
# data.isna().sum().plot(kind='bar')
# plt.show()


print(data.groupby('Magasin')['DistanceCompétition'].agg(['mean', 'median', 'min', 'max']))


# One Hot Encoding
data = data.drop(['Mois', 'Année', 'Jour', 'Semaine', 'Saison'], axis=1)
data = data.drop(['PromotionCompétitionSemaine', 'PromotionCompétitionAnnée',"PromotionIntermédiaire"], axis=1)

# select the categorical columns
cat_cols = data.select_dtypes(include='object').columns


# apply One Hot Encoding on the categorical columns
data = pd.get_dummies(data, columns=cat_cols)

# correlation matrix

# select the numerical columns
num_cols = data._get_numeric_data().columns

# calculate the correlation matrix
corr = data[num_cols].corr()

# plot the heatmap
sns.heatmap(corr, annot=False, cmap='plasma')

plt.show()
