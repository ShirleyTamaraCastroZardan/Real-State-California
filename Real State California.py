#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from geopy.geocoders import ArcGIS
import folium
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# In[2]:


data = pd.read_csv('housing.csv')
data.head()


# In[3]:


data.columns


# ### **About this file**

# 1. longitude: A measure of how far west a house is; a higher value is farther west
# 2. latitude: A measure of how far north a house is; a higher value is farther north
# 3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
# 4. totalRooms: Total number of rooms within a block
# 5. totalBedrooms: Total number of bedrooms within a block
# 6. population: Total number of people residing within a block
# 7. households: Total number of households, a group of people residing within a home unit, for a block
# 8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
# 9. medianHouseValue: Median house value for households within a block (measured in US Dollars)
# 10. oceanProximity: Location of the house w.r.t ocean/sea

# In[4]:


data.dropna(inplace=True)


# In[5]:


data.info()


# # STATISTICAL ANALYSIS

# In[6]:


data.describe()


# Based on the provided dataset detailing real estate characteristics in California. The dataset comprises 20,433 observations with attributes including geographical coordinates (longitude and latitude), housing median age, total rooms, total bedrooms, population, households, median income, and median house value.
# 
# The median house value in the dataset ranges from USD 14,999 to USD 500,001, with a mean value of approximately USD 206,854 and a standard deviation of USD 115,436. This wide range indicates substantial variability in housing prices across different regions. Notably, the median house value at the 25th percentile is USD 119,500, while at the 75th percentile, it reaches USD 264,700, illustrating a significant skew towards higher-valued properties.
# 
# In terms of demographics, the median income spans from 0.4999 to 15 (in tens of thousands of dollars), averaging at USD 3.871 (or USD 38,710), with a median value of USD 3.5365 (USD 35,365). This spread highlights diverse economic conditions across the sampled blocks. Housing characteristics show a median age of 29 years, with most buildings being relatively mature given that the mean housing age is approximately 28.63 years.
# 
# The population within blocks varies widely, from as low as 3 to as high as 35,682 individuals, indicating both densely and sparsely populated areas. Similarly, household counts range from 1 to 6,082, with a mean of 499.43 households per block, suggesting a broad spectrum of residential densities.
# 

# In[7]:


X=data.drop(['median_house_value'],axis=1)
Y=data['median_house_value']


# In[8]:


X_train, X_test, Y_train, Y_test =train_test_split(X,Y, test_size=0.2)


# In[9]:


train_data =X_train.join(Y_train)


# In[10]:


train_data.hist(figsize=(15,8))


# The histograms from the training dataset provide a visual representation of the distribution of various real estate attributes in California. These visualizations complement the statistical summary previously discussed and offer additional insights into the characteristics of the housing market.
# 
# 1. **Housing Median Age**: The histogram for housing median age shows a relatively uniform distribution with a slight peak around the 20 to 30-year range. There are also notable spikes at the higher end, indicating the presence of older housing blocks within the dataset.
# 
# 2. **Total Rooms and Total Bedrooms**: Both histograms for total rooms and total bedrooms show right-skewed distributions, indicating that most blocks have a lower number of rooms and bedrooms, with a few blocks having significantly higher counts. The majority of blocks have fewer than 10,000 rooms and 2,000 bedrooms.
# 
# 3. **Population**: The population histogram is highly right-skewed, with most blocks having fewer than 5,000 residents. A small number of blocks have populations exceeding 10,000, highlighting the presence of densely populated areas within the dataset.
# 
# 4. **Households**: Similar to population, the households histogram is right-skewed. Most blocks have fewer than 1,000 households, with a few blocks having upwards of 4,000 households, indicating variability in block sizes and housing densities.
# 
# 5. **Median Income**: The median income histogram shows a distribution with a peak around the $40,000 to $50,000 range. There is a significant drop-off in frequency as income increases, suggesting that higher-income blocks are less common.
# 
# 6. **Median House Value**: The histogram for median house value displays a somewhat bimodal distribution, with peaks around the $100,000 to $200,000 range and another around the $400,000 to $500,000 range. This indicates two dominant house value ranges within the dataset, with a substantial number of houses clustered around these values.
# 

# In[11]:


train_data.ocean_proximity.value_counts()


# In[12]:


pd.get_dummies(train_data.ocean_proximity)


# In[13]:


train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)


# In[14]:


train_data


# In[15]:


train_data.corr()


# In[16]:


train_data['total_rooms'] = np.log(train_data['total_rooms']+1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms']+1)
train_data['population'] = np.log(train_data['population']+1)
train_data['households'] = np.log(train_data['households']+1)


# ### Normalization of the data

# In[17]:


train_data.hist(figsize=(15,8))


# The histograms of the transformed (logarithmic) data show a more normalized distribution, which can be beneficial for various statistical analyses and machine learning algorithms. Here’s a detailed analysis of each attribute post-log transformation:
# 
# 1. **Housing Median Age**: The housing median age distribution remains similar to its original form. The data shows a relatively uniform distribution with peaks around the 20 to 30-year range and older housing blocks above 40 years.
# 
# 2. **Total Rooms and Total Bedrooms**: Logarithmic transformation significantly normalizes the previously right-skewed distributions of total rooms and total bedrooms. The histograms now show more bell-shaped distributions, indicating that the log transformation successfully reduced the skewness and brought the data closer to a normal distribution.
# 
# 3. **Population**: The log-transformed population histogram also exhibits a more normalized distribution compared to the original right-skewed histogram. Most blocks now fall within a more compact range, indicating that the transformation has reduced the effect of extreme values.
# 
# 4. **Households**: The households histogram post-log transformation shows a similar improvement, with a more symmetric distribution around the mean. This suggests that the majority of blocks have household numbers clustered around a central value, with fewer extreme outliers.
# 
# 5. **Median Income**: The median income histogram now shows a more symmetric distribution, indicating a successful normalization. The majority of the blocks have median incomes within a tighter range, which can facilitate more accurate modeling and analysis.
# 
# 6. **Median House Value**: The histogram for median house value, post-log transformation, displays a distribution that is closer to normal, with a more pronounced central peak. This normalization is crucial for studying the factors influencing house prices, as it allows for more reliable statistical inferences.
# 
# This normalization helps in mitigating the impact of outliers and makes the data more suitable for various analytical and predictive modeling techniques. By transforming the data, we enhance the robustness of our statistical analyses, enabling more accurate identification of trends and relationships within the California housing market.

# In[18]:


train_data['bedroom_ratio'] = train_data['total_bedrooms']/train_data['total_rooms']
train_data['household_rooms'] = train_data['total_rooms']/train_data['households']


# # CORRELATION ANALYSIS

# In[19]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')


# 1. **Housing Median Age**:
#    - Housing median age exhibits a moderate negative correlation with total rooms (\(-0.31\)), total bedrooms (\(-0.27\)), and households (\(-0.25\)), suggesting that newer buildings tend to have more rooms and bedrooms, likely reflecting modern construction trends.
# 
# 2. **Total Rooms and Total Bedrooms**:
#    - Total rooms and total bedrooms are highly correlated (\(0.95\)), indicating that blocks with more rooms generally have more bedrooms. This is expected as larger blocks with more rooms naturally accommodate more bedrooms.
#    - These features also show strong correlations with households (\(0.93\) and \(0.97\) respectively) and population (\(0.87\) and \(0.90\) respectively), highlighting that larger blocks tend to have more households and higher populations.
# 
# 3. **Population and Households**:
#    - Population and households have a near-perfect correlation (\(0.93\)), suggesting that these two metrics are closely linked, with more households corresponding to higher population counts within a block.
# 
# 4. **Median Income**:
#    - Median income has a significant positive correlation with median house value (\(0.69\)), indicating that higher household incomes are associated with higher house values. This relationship underscores the economic influence on housing prices.
#    - Median income also shows a positive correlation with total rooms (\(0.20\)), implying that wealthier areas might have larger houses with more rooms.
# 
# 5. **Median House Value**:
#    - Apart from its strong correlation with median income, median house value also has a positive correlation with total rooms (\(0.15\)) and households (\(0.18\)), suggesting that blocks with more extensive housing infrastructure tend to have higher house values.
#    - The negative correlation with bedroom ratio (\(-0.51\)) implies that blocks with a higher number of bedrooms per room tend to have lower house values.
# 
# 6. **Ocean Proximity**:
#    - Proximity to the ocean significantly influences house values. The correlation with median house value is positive for locations near the ocean (\(<1H OCEAN\) at \(0.26\), NEAR BAY at \(0.16\), NEAR OCEAN at \(0.14\)), and negative for INLAND locations (\(-0.48\)). This indicates that houses closer to the ocean are valued higher, while those further inland tend to be valued lower.
#    - Longitude has a moderate correlation with \(<1H OCEAN\) (\(0.32\)) and NEAR BAY (\(-0.47\)), reflecting the coastal geography of California.
# 
# 7. **Derived Ratios**:
#    - The bedroom ratio (total bedrooms/total rooms) shows a strong positive correlation with total bedrooms (\(0.68\)) and population (\(0.58\)), indicating that higher bedroom densities are found in more populated blocks.
#    - Household rooms (total rooms/households) have a moderate positive correlation with households (\(0.18\)) but a strong negative correlation with the bedroom ratio (\(-0.74\)), suggesting that blocks with a higher number of rooms per household tend to have a lower number of bedrooms relative to total rooms.
# 

# In[20]:


plt.figure(figsize=(15,8))
sns.scatterplot(x="latitude", y="longitude",data=train_data, hue="median_house_value",palette="coolwarm")


# In[21]:


nom=ArcGIS()
nom.geocode('california')


# In[22]:


list = data[['median_house_value', 'latitude','longitude']].values.tolist()


# In[36]:


data_map=folium.Map(location=[39.37,-121.24],zoom_start=6)
fg=folium.FeatureGroup(name='tg_reserve')

for i in list:
    fg.add_child(folium.Marker(location=[i[1],i[2]], popup=i[0],icon=folium.Icon(color='green')))
data_map.add_child(fg)


# In[24]:


data_map.save('Real_State.html')


# # Linear Regression

# In[25]:


scaler = StandardScaler()
X_train, Y_train =train_data.drop(['median_house_value'],axis=1), train_data['median_house_value']
X_train_s =scaler.fit_transform(X_train)
reg=LinearRegression()
reg.fit(X_train_s, Y_train)


# In[26]:


test_data =X_test.join(Y_test)
test_data['total_rooms'] = np.log(test_data['total_rooms']+1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms']+1)
test_data['population'] = np.log(test_data['population']+1)
test_data['households'] = np.log(test_data['households']+1)
test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
test_data['bedroom_ratio'] = test_data['total_bedrooms']/test_data['total_rooms']
test_data['household_rooms'] = test_data['total_rooms']/test_data['households']


# In[27]:


X_test, Y_test = test_data.drop(['median_house_value'],axis=1),test_data['median_house_value']


# In[28]:


X_test_s = scaler.transform(X_test)


# In[29]:


reg.score(X_test_s, Y_test)


# This indicates that approximately 68.75% of the variance in the median house value can be explained by the model using the provided features.

# # Random Forest Regressor

# In[30]:


forest = RandomForestRegressor()
forest.fit(X_train_s,Y_train)


# In[31]:


forest.score(X_test_s,Y_test)


# In[32]:


forest = RandomForestRegressor()
param_grid={
    "n_estimators":[3,10,30],
    "max_features":[2,4,6,8]
}
grid_search=GridSearchCV(forest,param_grid, cv=5,
                        scoring="neg_mean_squared_error",
                        return_train_score=True)
grid_search.fit(X_train_s,Y_train)


# In[33]:


best_forest= grid_search.best_estimator_


# In[34]:


best_forest.score(X_test_s, Y_test)


# This normalization helps in mitigating the impact of outliers and makes the data more suitable for various analytical and predictive modeling techniques. By transforming the data, we enhance the robustness of our statistical analyses, enabling more accurate identification of trends and relationships within the California housing market.
# 
# The model achieves an R² score of 0.819996, indicating a decent fit, but there is still room for improvement. Further refinement of the model or additional feature engineering might help improve its performance.
