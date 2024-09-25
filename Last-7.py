#!/usr/bin/env python
# coding: utf-8

# In[145]:


import os
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# In[147]:


print("Current working directory: {0}".format(os.getcwd()))


# In[149]:


employee = pd.read_csv('Employee.csv', delimiter = ',')
performancerate = pd.read_csv('PerformanceRating.csv', delimiter = ',')
#combineren van de datasets 
performancerate['ReviewDate'] = pd.to_datetime(performancerate['ReviewDate'])  # Zorg dat de aanstellingsdatum in datetime-formaat is
recent_performance = performancerate.loc[performancerate.groupby('EmployeeID')['ReviewDate'].idxmax()]
combined_dataset = pd.merge(employee, recent_performance, on='EmployeeID', how='left')


# In[151]:


combined_dataset.describe()


# In[153]:


st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: #0A1172;  /* Dark blue color */
        font-family: 'Arial', sans-serif;  /* You can change the font here */
    }
    </style>
    <h1 class="title">Voorspelling & Analyse van werknemersattritie</h1>
    """, unsafe_allow_html=True)


# In[155]:


combined_dataset['ReviewDate'] = pd.to_datetime(combined_dataset['ReviewDate'], errors='coerce')

# 1. Basisinformatie van de dataset
st.write("""
    Deze applicatie onderzoekt verschillende factoren die van invloed zijn op de werknemersattritie binnen een bedrijf. We analyseren een uitgebreide dataset en gebruiken geavanceerde machine learning-modellen om te voorspellen welke werknemers het grootste risico lopen te vertrekken. Je kunt patronen en verbanden ontdekken tussen variabelen zoals werkervaring, salaris, werktevredenheid en promotiefrequentie, en zien hoe deze bijdragen aan de beslissing om te blijven of te vertrekken.
    """)

# 2. Eerste paar rijen van de dataset
st.subheader("Weergave van de gebruikte dataset")
st.dataframe(combined_dataset.head())

# 6. Extra: Distributie van numerieke kolommen
st.subheader("Distributie van numerieke kolommen")
numeric_columns = combined_dataset.select_dtypes(include=['float64', 'int64']).columns
selected_column = st.selectbox("Kies een numerieke kolom om te visualiseren:", numeric_columns)

# Voeg een slider toe waarmee de gebruiker het aantal bins kan instellen
bins = st.slider("Selecteer het aantal bins voor het histogram", min_value=5, max_value=50, value=20)

fig, ax = plt.subplots()
# Plot het histogram met de gekozen hoeveelheid bins
sns.histplot(combined_dataset[selected_column], bins=bins, kde=True, ax=ax)
ax.set_title(f"Histogram van {selected_column} (met {bins} bins)")
st.pyplot(fig)


# In[157]:


combined_dataset['HireDate']=combined_dataset['HireDate'].astype('datetime64[ns]')


# In[159]:


combined_dataset.insert(0, 'FullName', combined_dataset['FirstName'] + ' ' + combined_dataset['LastName'])  # Plaats op positie 0 (eerste kolom)

# Verwijder de originele kolommen 'FirstName', 'LastName' en andere onodige kolomen 
combined_dataset.drop(columns=['FirstName', 'LastName','SelfRating','ManagerRating'], inplace=True)


# In[161]:


# Bekijk de unieke waarden in de 'Gender' kolom
print(combined_dataset['Gender'].unique())
print(combined_dataset['BusinessTravel'].unique())
print(combined_dataset['Attrition'].unique())
print(combined_dataset['Department'].unique())

# Verwijder leidende en volgende spaties in 'BusinessTravel' en 'Gender'
combined_dataset['BusinessTravel'] = combined_dataset['BusinessTravel'].str.strip()
combined_dataset['Gender'] = combined_dataset['Gender'].str.strip()

# Omzetten van de volgende kolomen om in numerieke warden 
combined_dataset['Gender'] = combined_dataset['Gender'].map({"Prefer Not To Say" :0, "Male": 1, "Female": 2, "Non-Binary":3})
combined_dataset['BusinessTravel'] = combined_dataset['BusinessTravel'].map({"No Travel": 0, "Some Travel": 1, "Frequent Traveller": 2})
combined_dataset['Attrition'] = combined_dataset['Attrition'].map({'Yes': 1, 'No': 0})

#Een nieuwe variabele toevoegen aan dataset
combined_dataset['PromotionFrequency'] = (combined_dataset['YearsAtCompany'] / (combined_dataset['YearsSinceLastPromotion'] + 1)).round().astype(int)

st.subheader("Data filteren op basis van Attritie ")

attrition_yes = st.checkbox("Attrition: Ja", key="attrition_yes")
attrition_no = st.checkbox("Attrition: Nee", key="attrition_no", value=not attrition_yes)

# Zorg ervoor dat als de ene checkbox wordt geselecteerd, de andere wordt gedeselecteerd
if attrition_yes:
    attrition_no = False
elif attrition_no:
    attrition_yes = False

# Toepassen van filtering op basis van de checkboxes
if attrition_yes and not attrition_no:
    filtered_dataset = combined_dataset[combined_dataset['Attrition'] == 1]
elif attrition_no and not attrition_yes:
    filtered_dataset = combined_dataset[combined_dataset['Attrition'] == 0]
else:
    filtered_dataset = combined_dataset

# Toon de gefilterde data
st.write("Gefilterde data op basis van Attrition:")
st.write(filtered_dataset)



# In[163]:


import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd


# In[165]:


# Bereken het percentage werknemers die het bedrijf verlaten of blijven
attrition_rate = combined_dataset['Attrition'].value_counts(normalize=True) * 100

# Titel en beschrijving toevoegen
st.subheader("Percentage werknemersattritie binnen de organisatie")
st.write("""
Deze tabel toont het percentage werknemers die het bedrijf verlaten ('1') of blijven ('0').
""")

# Weergeven van de resultaten in Streamlit
st.table(attrition_rate)


# In[167]:


# **Heatmap van correlaties**
st.subheader('Correlatiematrix van Factoren die Werknemersattritie Beïnvloeden')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(combined_dataset[['Attrition', 'JobSatisfaction', 'WorkLifeBalance', 
                           'Age', 'YearsSinceLastPromotion', 'DistanceFromHome (KM)', 'Salary','PromotionFrequency']].corr(), 
            annot=True, cmap='Blues', ax=ax)
st.pyplot(fig)  # Heatmap tonen


# In[168]:


# Alleen correlaties met Attrition extraheren
corr_matrix = combined_dataset[['Attrition', 'JobSatisfaction', 'WorkLifeBalance', 
                                'Age', 'YearsSinceLastPromotion', 'DistanceFromHome (KM)', 
                                'Salary','PromotionFrequency']].corr()

attrition_corr = corr_matrix['Attrition'].drop('Attrition')

# Correlatiedrempel instellen
threshold = 0.1
strong_corr = attrition_corr[(attrition_corr >= threshold) | (attrition_corr <= -threshold)]

# Tabel weergeven met sterke correlaties
st.subheader("Sterke Verbanden met Attrition")
st.write("""
Uit de bovenstaande correlatiematrix kunnen we een aantal belangrijke verbanden met betrekking tot werknemersattritie afleiden. Hieronder een overzicht van de sterkste correlaties:
""")

# Lijst van sterke correlaties in tabelvorm weergeven
st.table(strong_corr)

# Kort overzicht van de verbanden als tekst
if not strong_corr.empty:
    st.write("We zien dat de volgende factoren een sterke correlatie hebben met werknemersattritie:")
    for factor, corr_value in strong_corr.items():
        st.write(f"- **{factor}** heeft een correlatie van **{corr_value:.2f}** met werknemersattritie.")
else:
    st.write("Er zijn geen sterke correlaties (> |0.1|) gevonden tussen de factoren en werknemersattritie.")


# In[210]:


st.subheader('Aantal werknemers die het bedrijf hebben verlaten vs. gebleven')

OverTime_yes = st.checkbox("OverTime: Ja", key="OverTime_yes")
OverTime_no = st.checkbox("OverTime: Nee", key="OverTime_no", value=not OverTime_yes)

# Zorg ervoor dat als de ene checkbox wordt geselecteerd, de andere wordt gedeselecteerd
if OverTime_yes:
    OverTime_no = False
elif OverTime_no:
    OverTime_yes = False


# Toepassen van filtering op basis van de checkboxes
if OverTime_yes and not OverTime_no:
    filtered_dataset = combined_dataset[combined_dataset['OverTime'] == "Yes"]
elif OverTime_no and not OverTime_yes:
    filtered_dataset = combined_dataset[combined_dataset['OverTime'] == "No"]
else:
    filtered_dataset = combined_dataset

# Visualisatie van attritie
fig, ax = plt.subplots()
sns.countplot(x='Attrition', hue='Attrition', data=filtered_dataset, ax=ax, palette={1: 'red', 0: 'green'})
plt.xticks(ticks=[0, 1], labels=['Gebleven', 'Verlaten'], rotation=0)
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.1, int(p.get_height()), ha='center')

# Toon de plot
st.pyplot(fig)


# In[212]:


# Visualiseer de relatie tussen YearsSinceLastPromotion  en attritie van werknemers
st.subheader('Effect van YearsSinceLastPromotion op werknemersattritie')
median_values = combined_dataset.groupby('Attrition')['YearsSinceLastPromotion'].median()
fig, ax = plt.subplots()
sns.boxplot(x='Attrition', y='YearsSinceLastPromotion', data=combined_dataset, ax=ax, 
            hue='Attrition', palette={0: 'green', 1: 'red'}, dodge=True, legend=False)

plt.xticks(ticks=[0, 1], labels=['Gebleven', 'Verlaten'], rotation=0)
# Voeg de medianen toe aan de grafiek
for i, median in enumerate(median_values):
    ax.annotate(f'Mediaan: {median}', xy=(i, median), xytext=(i, median + 0.5), 
                ha='center', color='black', fontsize=10, weight='bold')
st.pyplot(fig) 

if median_values[0] < median_values[1]:
    conclusion = f"De mediane jaren sinds de laatste promotie voor werknemers die zijn vertrokken is **{median_values[1]}** " \
                 f"terwijl de mediane jaren voor werknemers die gebleven zijn **{median_values[0]}** is. Dit suggereert " \
                 "dat werknemers die langer geen promotie hebben gehad, een grotere kans hebben om het bedrijf te verlaten."
else:
    conclusion = f"De mediane jaren sinds de laatste promotie voor werknemers die zijn vertrokken is **{median_values[1]}** " \
                 f"terwijl de mediane jaren voor werknemers die gebleven zijn **{median_values[0]}** is. Dit suggereert " \
                 "dat werknemers die een recente promotie hebben gehad, meer geneigd zijn om bij het bedrijf te blijven."

st.write(conclusion)


# In[214]:


# Visualiseer de relatie tussen leeftijd en attritie van werknemers

# Leeftijd categoriseren
bins = [18, 30, 40, 50]  # Leeftijdsgrenzen
labels = ['18-29', '30-39', '40-49']  # Labels voor de groepen
combined_dataset['Leeftijdsgroep'] = pd.cut(combined_dataset['Age'], bins=bins, labels=labels, right=False)

# Aantal werknemers per leeftijdsgroep en attritiestatus
age_attrition = combined_dataset.groupby(['Leeftijdsgroep', 'Attrition'], observed=False).size().unstack(fill_value=0)

# Bereken het percentage vertrokken werknemers per leeftijdsgroep
age_attrition['Percentage_Verlaten'] = age_attrition[1] / (age_attrition[1] + age_attrition[0]) * 100

st.subheader('Effect van Leeftijd op werknemersattritie')

fig, ax1 = plt.subplots(figsize=(10, 6))

# Staafgrafiek: aantal werknemers per leeftijdsgroep en attritiestatus
age_attrition[[1, 0]].plot(kind='bar', stacked=False, ax=ax1, color=['red', 'green'], alpha=0.7)

# Secundaire y-as voor het percentage vertrokken werknemers
ax2 = ax1.twinx()
ax2.plot(age_attrition.index, age_attrition['Percentage_Verlaten'], color='blue', marker='o', linestyle='-', linewidth=2)
ax2.set_ylabel('Percentage Vertrokken Werknemers', color='blue')

# Grafiek labels
ax1.set_xlabel('Leeftijdsgroep')
ax1.set_ylabel('Aantal Werknemers')
ax1.set_title('Aantal Werknemers per Leeftijdsgroep en Attritie')
ax1.legend(['Verlaten', 'Gebleven'], title='Attritie')
ax1.grid(axis='y')

# Toon de grafiek in Streamlit
st.pyplot(fig)
max_percentage = age_attrition['Percentage_Verlaten'].max()
max_index = age_attrition['Percentage_Verlaten'].idxmax()  # Geeft de index (leeftijdsgroep) met het hoogste percentage

conclusion = f"De leeftijdsgroep **{max_index}** heeft het hoogste percentage vertrokken werknemers, " \
             f"wat kan wijzen op een verhoogde kans op attritie in deze groep."

st.write(conclusion)


# In[215]:


# Zet voorbeeld dataset om in een DataFrame
combined_dataset = pd.DataFrame(combined_dataset)

# 1. Salary Slider voor filtering van data
st.write("### Selecteer Salarisbereik")
min_salary, max_salary = st.slider("Kies een salarisbereik", 
                                   int(combined_dataset['Salary'].min()), 
                                   int(combined_dataset['Salary'].max()), 
                                   (int(combined_dataset['Salary'].min()), int(combined_dataset['Salary'].max())))

# Filter data op basis van salaris
filtered_dataset = combined_dataset[(combined_dataset['Salary'] >= min_salary) & 
                                    (combined_dataset['Salary'] <= max_salary)]

# 2. Dropdown voor filtering van Department
st.write("### Selecteer Afdeling")
departments = combined_dataset['Department'].unique()
selected_department = st.selectbox("Selecteer Afdeling", departments)

# Filter data op basis van zowel salaris als afdeling
filtered_data = filtered_dataset[filtered_dataset['Department'] == selected_department]

# Toon gefilterde data in Streamlit
st.write(f"Gefilterde data op basis van Salaris: {min_salary} - {max_salary} en Afdeling: {selected_department}")
st.write(filtered_data) 

# Visualiseer de relatie tussen salaris en attritie
st.subheader('Effect van Salaris op werknemersattritie')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=combined_dataset, x='Salary', y='Attrition', hue='Attrition', palette={0: 'green', 1: 'red'}, s=100)
# Grafiek labels
plt.title('Salaris versus Werknemersattritie')
plt.xlabel('Salaris')
plt.ylabel('Attritie')
plt.yticks([0, 1], ['Gebleven', 'Verlaten'])
plt.grid()
st.pyplot(plt)

mean_salary_stayed = combined_dataset[combined_dataset['Attrition'] == 0]['Salary'].mean()
mean_salary_left = combined_dataset[combined_dataset['Attrition'] == 1]['Salary'].mean()

if mean_salary_stayed > mean_salary_left:
    conclusion = f"Werknemers die zijn gebleven hebben een gemiddeld salaris van **€{mean_salary_stayed:.2f}**, terwijl " \
                 f"werknemers die zijn vertrokken een gemiddeld salaris van **€{mean_salary_left:.2f}** hebben. Dit kan erop wijzen " \
                 "dat hogere salarissen mogelijk bijdragen aan een lagere attritie."
else:
    conclusion = f"Werknemers die zijn vertrokken hebben een gemiddeld salaris van **€{mean_salary_left:.2f}**, terwijl " \
                 f"werknemers die zijn gebleven een gemiddeld salaris van **€{mean_salary_stayed:.2f}** hebben. Dit kan erop wijzen " \
                 "dat lagere salarissen bijdragen aan een hogere attritie."

st.write(conclusion)


# In[217]:


# Visualiseer de relatie tussen werktevredenheid en attritie
st.subheader('Effect van werktevredenheid op werknemersattritie')
median_values = combined_dataset.groupby('Attrition')['JobSatisfaction'].median()

fig, ax = plt.subplots()
sns.violinplot(x='Attrition', y='JobSatisfaction', data=combined_dataset, ax=ax, hue='Attrition', palette={0: 'green', 1: 'red'}, dodge=True)

# Labels en titel instellen
ax.set_xlabel('Werknemersattritie')
ax.set_ylabel('Werktevredenheid')
plt.xticks([0, 1], ['Gebleven', 'Verlaten'])
# Voeg de medianen toe aan de grafiek
for i, median in enumerate(median_values):
    ax.annotate(f'Mediaan: {median}', xy=(i, median), xytext=(i, median + 0.5), 
                ha='center', color='black', fontsize=10, weight='bold')

st.pyplot(fig)

mean_job_satisfaction_stayed = combined_dataset[combined_dataset['Attrition'] == 0]['JobSatisfaction'].mean()
mean_job_satisfaction_left = combined_dataset[combined_dataset['Attrition'] == 1]['JobSatisfaction'].mean()

if mean_job_satisfaction_stayed > mean_job_satisfaction_left:
    conclusion = f"Werknemers die zijn gebleven hebben een gemiddelde werktevredenheid van **{mean_job_satisfaction_stayed:.2f}**, terwijl " \
                 f"werknemers die zijn vertrokken een gemiddelde werktevredenheid van **{mean_job_satisfaction_left:.2f}** hebben. Dit suggereert dat hogere werktevredenheid " \
                 "kan bijdragen aan een lagere attritie."
else:
    conclusion = f"Werknemers die zijn vertrokken hebben een gemiddelde werktevredenheid van **{mean_job_satisfaction_left:.2f}**, terwijl " \
                 f"werknemers die zijn gebleven een gemiddelde werktevredenheid van **{mean_job_satisfaction_stayed:.2f}** hebben. Dit kan erop wijzen dat lagere werktevredenheid " \
                 "bijdraagt aan een hogere attritie."

st.write(conclusion)


# In[218]:


# Visualiseer de relatie tussen DistanceFromHome (KM) en attritie van werknemers
st.subheader('Effect van DistanceFromHome (KM) op werknemersattritie')
median_values = combined_dataset.groupby('Attrition')['DistanceFromHome (KM)'].median()

fig, ax = plt.subplots()
sns.boxplot(x='Attrition', y='DistanceFromHome (KM)', data=combined_dataset, ax=ax, 
            hue='Attrition', palette={0: 'green', 1: 'red'}, dodge=True)
plt.xticks(ticks=[0, 1], labels=['Gebleven', 'Verlaten'], rotation=0)
# Voeg de medianen toe aan de grafiek
for i, median in enumerate(median_values):
    ax.annotate(f'Mediaan: {median}', xy=(i, median), xytext=(i, median + 0.5), 
                ha='center', color='black', fontsize=10, weight='bold')

st.pyplot(fig)


# In[220]:


st.subheader('Effect van Werk-privébalans op werknemersattritie')
# Slider voor filtering van DistanceFromHome
min_distance, max_distance = st.slider(
    'Selecteer Afstand van huis (in km)',
    min_value=int(combined_dataset['DistanceFromHome (KM)'].min()),
    max_value=int(combined_dataset['DistanceFromHome (KM)'].max()),
    value=(int(combined_dataset['DistanceFromHome (KM)'].min()), int(combined_dataset['DistanceFromHome (KM)'].max()))
)

# Filter de data op basis van geselecteerde afstand
filtered_data = combined_dataset[
    (combined_dataset['DistanceFromHome (KM)'] >= min_distance) &
    (combined_dataset['DistanceFromHome (KM)'] <= max_distance)
]

# Dropdown voor filtering van MaritalStatus
marital_status = combined_dataset['MaritalStatus'].unique()
selected_status = st.selectbox("Selecteer burgerlijke staat", marital_status)

# Filter de data op basis van burgerlijke staat
filtered_data = filtered_data[filtered_data['MaritalStatus'] == selected_status]
st.write(f"Gefilterde data op basis van Afstand: {min_distance} - {max_distance} km en Burgerlijke staat: {selected_status}")

# Visualiseer de relatie tussen werk-privébalans en attritie
fig, ax = plt.subplots()
sns.boxplot(x='Attrition', y='WorkLifeBalance', data=filtered_data, ax=ax,
            hue='Attrition', palette={0: 'green', 1: 'red'}, dodge=True)

plt.xticks(ticks=[0, 1], labels=['Gebleven', 'Verlaten'], rotation=0)

st.pyplot(fig)
mean_work_life_balance_stayed = filtered_data[filtered_data['Attrition'] == 0]['WorkLifeBalance'].mean()
mean_work_life_balance_left = filtered_data[filtered_data['Attrition'] == 1]['WorkLifeBalance'].mean()

if mean_work_life_balance_stayed > mean_work_life_balance_left:
    conclusion = f"Werknemers die zijn gebleven hebben een gemiddelde werk-privébalans van **{mean_work_life_balance_stayed:.2f}**, terwijl " \
                 f"werknemers die zijn vertrokken een gemiddelde werk-privébalans van **{mean_work_life_balance_left:.2f}** hebben. Dit suggereert dat een betere werk-privébalans " \
                 "kan bijdragen aan een lagere attritie."
else:
    conclusion = f"Werknemers die zijn vertrokken hebben een gemiddelde werk-privébalans van **{mean_work_life_balance_left:.2f}**, terwijl " \
                 f"werknemers die zijn gebleven een gemiddelde werk-privébalans van **{mean_work_life_balance_stayed:.2f}** hebben. Dit kan erop wijzen dat een slechtere werk-privébalans " \
                 "bijdraagt aan een hogere attritie."


# In[222]:


st.subheader('Effect van PromotionFrequency op werknemersattritie')

# Maak de checkboxes met standaardwaarde 'Alle Afdelingen' aangevinkt
alle_afdelingen = st.checkbox("Alle Afdelingen", value=True, key="alle_afdelingen")
sales = st.checkbox("Sales", value=False, key="sales")
Technology = st.checkbox("Technology", value=False, key="Technology")
hr = st.checkbox("Human Resources", value=False, key="hr")

# Zorg ervoor dat slechts één checkbox tegelijkertijd kan worden geselecteerd
if alle_afdelingen:
    sales, Technology, hr = False, False, False
elif sales:
    alle_afdelingen, Technology, hr = False, False, False
elif Technology:
    alle_afdelingen, sales, hr = False, False, False
elif hr:
    alle_afdelingen, sales, Technology = False, False, False

# Filter de dataset op basis van de geselecteerde checkbox
if alle_afdelingen:
    filtered_data = combined_dataset  # Geen filtering toepassen
elif sales:
    filtered_data = combined_dataset[combined_dataset['Department'] == 'Sales']
elif Technology:
    filtered_data = combined_dataset[combined_dataset['Department'] == 'Technology']
elif hr:
    filtered_data = combined_dataset[combined_dataset['Department'] == 'Human Resources']

# Controleer of er gefilterde data beschikbaar is
if filtered_data.empty:
    st.write("Geen data beschikbaar voor de geselecteerde afdeling.")
else:
    # Maak de barplot voor de geselecteerde afdeling of alle afdelingen
    st.write(f"Promotie frequentie binnen {'alle afdelingen' if alle_afdelingen else 'de geselecteerde afdeling'}")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='PromotionFrequency', hue='Attrition', data=filtered_data, ax=ax,
                 palette={0: 'green', 1: 'red'})
# Labels en titels voor de grafiek
    ax.set_title(f'Promotie Frequentie versus Werknemersattritie in {"alle afdelingen" if alle_afdelingen else "de geselecteerde afdeling"}')
    ax.set_xlabel('Promotion Frequency (Aantal promoties per werknemer)')
    ax.set_ylabel('Aantal werknemers')
    ax.legend(title="Attrition", labels=["Gebleven", "Vertrokken"])
    ax.grid(True)

    # Toon de grafiek in Streamlit
    st.pyplot(fig)


mean_promotion_frequency_stayed = filtered_data[filtered_data['Attrition'] == 0]['PromotionFrequency'].mean()
mean_promotion_frequency_left = filtered_data[filtered_data['Attrition'] == 1]['PromotionFrequency'].mean()

if mean_promotion_frequency_stayed > mean_promotion_frequency_left:
    conclusion = f"Werknemers die zijn gebleven hebben een gemiddelde promotiefrequentie van **{mean_promotion_frequency_stayed:.2f}**, terwijl " \
                 f"werknemers die zijn vertrokken een gemiddelde promotiefrequentie van **{mean_promotion_frequency_left:.2f}** hebben. Dit suggereert dat een hogere promotiefrequentie " \
                 "kan bijdragen aan een lagere attritie."
else:
    conclusion = f"Werknemers die zijn vertrokken hebben een gemiddelde promotiefrequentie van **{mean_promotion_frequency_left:.2f}**, terwijl " \
                 f"werknemers die zijn gebleven een gemiddelde promotiefrequentie van **{mean_promotion_frequency_stayed:.2f}** hebben. Dit kan erop wijzen dat lagere promotiefrequentie " \
                 "bijdraagt aan een hogere attritie."

st.write(conclusion)


# In[225]:


import streamlit as st

import streamlit as st

# Functie om de kans op promotie te berekenen
def calculate_promotion_chance(YearsAtCompany, YearsInMostRecentRole, YearsSinceLastPromotion, TrainingOpportunitiesWithinYear, SelfRating, ManagerRating):
    # Gewichten toekennen
    w1 = 0.2  # Jaren bij het bedrijf
    w2 = 0.3  # Jaren in de huidige rol
    w3 = -0.1 # Jaren sinds de laatste promotie (negatieve impact)
    w4 = 0.2  # Training kansen
    w5 = 0.1  # Zelfbeoordeling
    w6 = 0.1  # Beoordeling door manager
    
    # Bereken kans op promotie
    chance = (w1 * YearsAtCompany +
              w2 * YearsInMostRecentRole +
              w3 * YearsSinceLastPromotion +
              w4 * TrainingOpportunitiesWithinYear +
              w5 * SelfRating +
              w6 * ManagerRating) * 100  # Omzetten naar percentage
    
    return max(0.0, min(100.0, chance))  # Zorg ervoor dat de kans tussen 0.0 en 100.0 blijft

# Streamlit interface
st.subheader("Kans op Promotie Calculator")

# Plaats invoervelden in rijen van twee met behulp van columns
col1, col2 = st.columns(2)
with col1:
    YearsAtCompany = st.number_input("Years At Company", min_value=0, max_value=30, value=5)
with col2:
    YearsInMostRecentRole = st.number_input("Years In Most Recent Role", min_value=0, max_value=30, value=2)

col3, col4 = st.columns(2)
with col3:
    YearsSinceLastPromotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=30, value=1)
with col4:
    TrainingOpportunitiesWithinYear = st.number_input("Training Opportunities Within Year", min_value=0, max_value=5, value=2)

col5, col6 = st.columns(2)
with col5:
    SelfRating = st.number_input("Self Rating (1-10)", min_value=1, max_value=10, value=5)
with col6:
    ManagerRating = st.number_input("Manager Rating (1-10)", min_value=1, max_value=10, value=5)

# Bereken de kans
promotion_chance = calculate_promotion_chance(YearsAtCompany, YearsInMostRecentRole, YearsSinceLastPromotion, TrainingOpportunitiesWithinYear, SelfRating, ManagerRating)

# Toon het resultaat
st.write(f"Je kans op promotie is: {promotion_chance:.2f}%")


# In[228]:


import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Voorbeeld dataset laden (vervang dit met jouw eigen dataset)
# combined_dataset = pd.read_csv('jouw_dataset.csv')

# Titel en beschrijving van de voorspelling
st.title("Voorspelling van werknemersattritie")
st.write("""
In deze sectie gebruiken we een Random Forest Classifier om te voorspellen of een werknemer het bedrijf zal verlaten 
(werknemersattritie) op basis van verschillende factoren zoals leeftijd, werktevredenheid, balans tussen werk en privé, enzovoort.
""")

# Select relevant features for prediction
features = ['JobSatisfaction', 'WorkLifeBalance', 'Age', 'YearsAtCompany', 
            'YearsSinceLastPromotion', 'DistanceFromHome (KM)', 'Salary', 'PromotionFrequency']
X = combined_dataset[features]
y = combined_dataset['Attrition']

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and evaluate a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Random Forest metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_report = classification_report(y_test, y_pred_rf, output_dict=True)

# Weergeven van de accuracy score
st.subheader("Modelresultaten")
st.write(f"De nauwkeurigheid van het Random Forest-model is: **{rf_accuracy:.2f}**")

# Weergeven van het classification report
st.write("Hieronder vind je het classificatierapport, dat de prestaties van het model per categorie (verbleven of vertrokken) weergeeft:")

# Converteer het classification report naar een DataFrame voor overzichtelijke weergave
rf_report_df = pd.DataFrame(rf_report).transpose()
st.dataframe(rf_report_df)

# Optioneel: Extra uitleg van het classificatierapport
st.write("""
Het classificatierapport toont de prestaties van het model in termen van precision, recall en F1-score voor beide klassen (werknemers die blijven en werknemers die vertrekken). 
Een hogere F1-score betekent dat het model beter presteert in het voorspellen van die klasse.
""")


# In[ ]:




