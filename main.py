
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Electricity Demand Forecasting", layout="wide")
st.title("Electricity Demand Forecasting")

# Load dataset
st.subheader("Dataset Preview")
df = pd.read_csv("combined_avg.csv")
st.write(df.head())



#import standard packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore", message=".*This axis already has a converter set.*")


#import preprocessing/modeling/error metric packages
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor


combined_avg = pd.read_csv('combined_avg.csv', index_col='time', parse_dates=True)
combined_avg.head(2)
combined_avg.info()


#function to make train-test split for time-indexed data
def ts_train_test(data, target_col_name = 'total load actual', test_size=0.15, stdzd=False, cols_to_scale=None):
    df = data.copy()
    test_index = int(len(df)*(1-test_size)) #get index where test set begins
        
    X_train = df.drop([target_col_name], axis = 1).iloc[:test_index]
    y_train = df[target_col_name].iloc[:test_index]
    X_test = df.drop([target_col_name], axis = 1).iloc[test_index:]
    y_test = df[target_col_name].iloc[test_index:]
    
    # StandardScaler fit seperately on training and test sets
    if stdzd == True:
        scaler = StandardScaler()
        X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
        X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    return X_train, X_test, y_train, y_test        


#Let's engineer some categorical features for use in a regression model like weekend/weekday, winter/summer/spring-fall
df_features = combined_avg.iloc[:,16:].drop(['price day ahead', 'price actual','total load forecast'], axis=1)
features_min = df_features.copy()
df_features.head()


#function to calculate basic season label based on month
def season_determination(month):
    if month in [6,7,8,9]: #June-Sept = summer (highest need for cooling in Spain)
        return "summer"
    elif month in [1,2,12]: #Dec, Jan, Feb = winter (highest need for heating)
        return "winter"
    else:
        return "spring/fall" #all other months are spring or fall (similar lower needs for heating/cooling)



day_of_week = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3: 'Thursday', 4: 'Friday', 5:'Saturday', 6:'Sunday'}
df_features['hour'] = df_features.index.hour
df_features['weekday'] = df_features.index.weekday.map(day_of_week)
df_features['month'] = df_features.index.month #have to create month column because cannot apply() on datetimeindex
df_features['season'] = df_features.month.apply(season_determination)
df_features['nonwork-work_day'] = np.where(df_features.index.weekday > 5, 0, 1)
st.write(df_features.shape, df_features.head());



df_features.drop("month", axis=1, inplace=True)


#let's split data and then encode the new catergorical variables
X_train, X_test, y_train, y_test = ts_train_test(data = df_features, stdzd=True, 
                                                 cols_to_scale=['temp','pressure','humidity',
                                                                'wind_speed','rain_1h','snow_3h','clouds_all'])
cat_cols = ['hour','weekday','season','city']
X_train_cat = X_train[['hour','weekday','season','city']]
X_test_cat = X_test[['hour','weekday','season','city']]

encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

encoder.fit(X_train_cat)
X_train.drop(columns = cat_cols, inplace=True)
X_train_cat = pd.DataFrame(encoder.transform(X_train_cat), index=X_train.index, 
                           columns = encoder.get_feature_names_out()).astype(int)
X_train = X_train.join(X_train_cat, how ='outer')
#X_train.head()
encoder.fit(X_test_cat)
X_test.drop(columns = cat_cols, inplace=True)
X_test_cat = pd.DataFrame(encoder.transform(X_test_cat), index = X_test.index, 
                           columns = encoder.get_feature_names_out()).astype(int)
X_test = X_test.join(X_test_cat, how ='outer')

st.write(X_train.head(3), X_test.head(3), X_train.columns)




#function for calculating and presenting error metrics, and storing in a dict for comparison at the end
error_dict = {} #dict to hold model name and error metrics for various models that are investigated

def error_metrics(y_true, y_pred, model_name = None):
    
    #function will print RMSE, R2, MAE, MAPE. Assumes y_pred is np array
    
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    MAPE = (np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    print('\nError metrics for model: {}'.format(model_name))
    print("RMSE: %.2f" % RMSE)
    print('Variance/R^2: %.2f' % R2)
    print('MAE: %.2f' % MAE)
    print('Mean Absolute Percentage Error: %.2f %%' % MAPE)
    
    key = ['Model Name','RMSE', 'R2', 'MAE', 'MAPE']
    value = [model_name, RMSE, R2, MAE, MAPE]
    pair = list(zip(key, value))
    '''
    for error in pair:
        error_dict[error[0]]= [error[1]]
    '''
    for error in pair:
        if error[0] in error_dict:
            error_dict[error[0]].append(error[1])
        else:
            error_dict[error[0]]= [error[1]]
            

#function for plotting time series of predicted vs true values
def plot_ts_pred_true(y_pred, y_true, model_name=None):
    fig, ax = plt.subplots(figsize =(15,10))
    ax.plot(y_true.index, y_pred, linestyle='-', linewidth=1, label = 'Model Forecasted Total Load', color = 'blue',alpha = 0.4)
    y_true.plot(linestyle='-', linewidth=1, label = 'Actual Total Load', color = 'red',alpha = 0.4)

    plt.ylabel('Load/Demand (MW)')
    plt.xlabel("Time")
    plt.title("Observed vs model-predicted total load (MWH) using {}".format(model_name))
    plt.legend()
    st.pyplot(plt.gcf())




#instantiate and fit linear regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)


#calculate error metrics for train set
error_metrics(linreg.predict(X_train), y_train, model_name = 'simple linear regression (train)')

#calculate error metrics for test set
error_metrics(linreg.predict(X_test), y_test, model_name = 'simple linear regression (test)')



plot_ts_pred_true(y_pred = linreg.predict(X_test), y_true = y_test, model_name = "simple linear regression")


# In[17]:


#let's try using a df with reduced feature space, by eliminating some possibly extraneous variables:

X_train_red = X_train.drop(['hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
       'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday', 
       'weekday_Thursday', 'weekday_Tuesday','weekday_Wednesday','city_Kolkata', 'city_Mumbai',
       'city_NCR'], axis=1)
X_test_red = X_test.drop(['hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9',
       'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 
       'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'weekday_Monday', 'weekday_Saturday', 'weekday_Sunday', 
       'weekday_Thursday', 'weekday_Tuesday','weekday_Wednesday','city_Kolkata', 'city_Mumbai',
       'city_NCR'], axis=1)


# ### Linear regression on reduced feature space

# In[18]:


#instantiate and fit linear regression model for reduced feature space train set
linreg_red = LinearRegression()
linreg_red.fit(X_train_red, y_train)


# In[19]:


#calculate error metrics for linear regr with reduced feature train set
error_metrics(y_train, linreg_red.predict(X_train_red), model_name = 'simple linear regression on reduced features (train)')


# We can see that the simple linear regression model using the larger set of features predicts the weekly seasonality fairly well but is often over or underpredicting the max daily load. The model using the reduced feature space performs worse. I will also check out a baseline model that just predicts the same as the value for the same date/time from the previous year:
# ### Simple Year-over-Year Baseline 

# In[20]:


# errors metrics for a baseline forecast (that simply repeats the values from the previous year)
#error_metrics(y_true, y_pred, model_name = None)
error_metrics(y_test, df_features.loc[X_test.index.shift(-8760, freq='h'), 'total load actual'],
              model_name='Baseline forecast (repeat of previous year) (test)')


# In[21]:


# plot y_pred and y-true for baseline year over year
plot_ts_pred_true(y_pred = df_features.loc[X_test.index.shift(-8760, freq='h'), 'total load actual'], y_true = y_test, 
                  model_name = "baseline forecast that repeats the value from the previous year")


# Visually we can can see that the baseline model that just uses the value from the same time from the previous year does a decent job, but misses is off pretty badly sometimes. Which makes sense, because on average things are similar year to year but the peak demand will definitely be much higher or much lower on certain days. Let's try out a couple more regression models before trying some time series forecasting:

# ### Random Forest

# In[22]:


#set up a parameter grid of different values for GridSearchCV of number of trees and max depth
n_est = [int(n) for n in np.logspace(start=1, stop=2.5, num=12)]
max_depth = list(range(1,6))

param_grid = {
        'n_estimators': n_est,
        'max_depth': max_depth
}
param_grid


# In[23]:


#create instance of base model
rfreg = RandomForestRegressor()

#create TS splits for cross val (instead of random splits, this uses progressively larger sets starting from beginning)
tss = TimeSeriesSplit(n_splits=5)

#create instance of RandomSearchCV
rf_cv = RandomizedSearchCV(rfreg, param_distributions=param_grid, cv=tss, random_state=47)

# Fit the random search model using reduced feature space X set
rf_cv.fit(X_train_red, y_train)

rf_cv.best_params_


# The R^2 score:

# In[24]:


rf_cv.score(X_train_red, y_train)


# In[25]:


rf_cv.score(X_test_red, y_test)


# In[26]:


#RF error metrics
error_metrics(y_train, rf_cv.predict(X_train_red),  
                  model_name = 'Random Forest Regression on reduced feature space tuned with Random Search CV (train)')


# Let's also try Random Forest on the full feature space dataset:

# In[ ]:


#create instance of base model
rfreg = RandomForestRegressor()

#create TS splits for cross val (instead of random splits, this uses progressively larger sets starting from beginning)
tss = TimeSeriesSplit(n_splits=5)

#create instance of RandomSearchCV
rf_cv_full = RandomizedSearchCV(rfreg, param_distributions=param_grid, cv=tss, random_state=47)

# Fit the random search model using reduced feature space X set
rf_cv_full.fit(X_train, y_train)

rf_cv_full.best_params_


# In[ ]:


rf_cv_full.score(X_train, y_train)


# In[ ]:


rf_cv_full.score(X_test, y_test)


# In[ ]:


error_metrics(y_train, rf_cv_full.predict(X_train), 
                  model_name = 'Random Forest Regression tuned with Random Search CV (train)')


# In[ ]:


error_metrics(y_test, rf_cv_full.predict(X_test), 
                  model_name = 'Random Forest Regression tuned with Random Search CV (test)')


# In[ ]:


plot_ts_pred_true(y_pred = rf_cv_full.predict(X_test), y_true = y_test, model_name = "Random Forest Regression")


# So Random Forest performs slightly worse that linear regression. It seems to consistently underpredict the true values. Let's also try KNeighbors Regression:
# ###  KNN

# In[ ]:


#set up a parameter grid for KNeighbors Random CV search:

n_neigh = [int(n) for n in np.logspace(start=0.5, stop=1.8, num=10)]
weights = ['uniform', 'distance']
leaf_size = [int(n) for n in np.logspace(start=0.5, stop=1.9, num=6)]

kn_param_grid = {
        'n_neighbors': n_neigh,
        'weights': weights,
        'leaf_size':leaf_size
}
kn_param_grid


# In[ ]:


#create instance of base model
kn = KNeighborsRegressor()

#create TS splits for cross val (instead of random splits, this uses progressively larger sets starting from beginning)
tss = TimeSeriesSplit(n_splits=5)

#create instance of RandomSearchCV
kn_cv = RandomizedSearchCV(kn, param_distributions = kn_param_grid, cv=tss, random_state=47)

# Fit the random search model using reduced feature space X set
kn_cv.fit(X_train, y_train)

kn_cv.best_params_


# In[ ]:


kn_cv.score(X_train, y_train)


# In[ ]:


kn_cv.score(X_test, y_test)


# In[ ]:


error_metrics(y_test, kn_cv.predict(X_test), 
                  model_name = 'KNN tuned with Random Search CV (test)')


# In[ ]:


plot_ts_pred_true(y_pred = kn_cv.predict(X_test), y_true = y_test, model_name = "Tuned KNN on test set")


# As expected, this model wildly overfits the training data and performs similarly to Random Forest and linear regression on the test data. Now to try some time series forecasting.

# ## XGBoost
# 
# One other model to try would be Extreme Gradient Boosting (or XGBoost) which is very good at finding patterns in data and often provides better results than other ML algorithms. XGBoost can do a decent job with time series forecasting where there is seasonality but not much trend, as it cannot extrapolate. Since this dataset essentially has no trend, it may perform well.

# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


#set up a parameter grid for hyperparameter Random CV search:

n_est = [int(n) for n in np.logspace(start=1, stop=2.5, num=10)]
max_depth = [0,3,6,9]
learn_rate = [0.1,0.2,0.3,0.4,0.5]

xgb_param_grid = {
        'n_estimators': n_est,
        'max_depth': max_depth,
        'learning_rate':learn_rate
}
xgb_param_grid


# In[ ]:


#create XGB model instance
xgb_model = XGBRegressor()

#create TS splits for cross val (instead of random splits, this uses progressively larger sets starting from beginning)
tss = TimeSeriesSplit(n_splits=5)

#create instance of RandomSearchCV
xgb_cv = RandomizedSearchCV(xgb_model, param_distributions=xgb_param_grid, scoring='neg_root_mean_squared_error', cv=tss, random_state=47)

# Fit the random search model using reduced feature space X set
xgb_cv.fit(X_train, y_train)

xgb_cv.best_params_


# In[ ]:


# calculate the RMSE of the model on the train set using the tuned hyperparameters
# best_params = {'n_estimators': 215, 'max_depth': 3, 'learning_rate': 0.4}
best_RMSE = xgb_cv.best_score_
print(-best_RMSE)


# In[ ]:


# predict on the test set
xgb_pred = xgb_cv.predict(X_test)


# In[ ]:


#calculate error metrics on train data
error_metrics(y_true = y_train, y_pred = xgb_cv.predict(X_train), 
              model_name = "XGBoost with weather data and engineered time-based features (train)")


# In[ ]:


# plot the y_pred and y_true for train set
plot_ts_pred_true(y_pred = xgb_cv.predict(X_train), y_true = y_train, 
                  model_name = "XGBoost with weather data and engineered time-based features on train data")


# In[ ]:


#calculate error metrics on test data
error_metrics(y_true = y_test, y_pred = xgb_pred, 
              model_name = "XGBoost with weather data and engineered time-based features (test)")


# In[ ]:


# plot the y_pred and y_true for test set
plot_ts_pred_true(y_pred = xgb_pred, y_true = y_test, 
                  model_name = "XGBoost with weather data and engineered time-based features on test data")


# # Conclusions
# Multiple models and variations on them were used to forecast the total energy demand in Spain (in MWH) using hourly load and weather data. The demand is highly dependent on temperature as well as time of day and day of week. As a time series, it exhibits daily, weely, and yearly seasonalities, but does not display much of a trend. Below is a compilation of the various models and their error metrics on training and test data.

# In[ ]:


#after running the error metric function on each model, the error_dict is populated w/ metrics for each
error_df = pd.DataFrame.from_dict(error_dict)

# create table and sort based on lowest RMSE
sorted_errors = np.round(error_df.pivot_table(index='Model Name', aggfunc='min').sort_values('RMSE', ascending=True),2)
sorted_errors


# ### Take Home: 
# Based on RMSE, Prophet barely edges out XGBoost in terms of performance on the 7-month forecast of total energy demand, based on 41 months of training data. Prophet performs similarly to SARIMAX days into the forecast window, and with considerably lower training time. But since the SARIMA does so well on the training data, it may be useful on very short forecast windows, one to several hours into the future. Both have a place, as accurate long term forecasts are important for general planning and overall energy generation mix strategy, while hour-ahead (to several hours ahead) forecasts are critical for firing on fast-to-ramp-up systems for meeting peak demands.

# ### Further Investigation
# Future efforts with this dataset could compare Prophet and/or XGBoost to the Spanish system operator's prediction of the total load. It would also be interesting to predict the contribution of wind generated energy given the wind speed and total load.

# In[ ]:





# PEAK DEMAND PER CITY


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# Load data
df = pd.read_csv('combined_avg.csv', index_col='time', parse_dates=True)

# Ensure dataset has necessary columns
required_cols = ['city', 'total load actual']
if not all(col in df.columns for col in required_cols):
    raise ValueError("Dataset must contain 'city' and 'total load actual' columns.")

# Extract date-related features
df['date'] = df.index.date
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['month'] = df.index.month

# Compute daily peak demand per city
peak_demand = df.groupby(['city', 'date'])['total load actual'].max().reset_index()


# Encode city column
label_encoder = LabelEncoder()
peak_demand['city_encoded'] = label_encoder.fit_transform(peak_demand['city'])

# Drop original city column
peak_demand.drop(columns=['city'], inplace=True)

# Create lag features (past demand)
for lag in range(1, 15):  # Try 14-day history
    peak_demand[f'lag_{lag}'] = peak_demand['total load actual'].shift(lag)

# Rolling Mean Features (Smooth Fluctuations)
peak_demand['rolling_7'] = peak_demand['total load actual'].rolling(7).mean()
peak_demand['rolling_14'] = peak_demand['total load actual'].rolling(14).mean()

# Drop NaN values
peak_demand.dropna(inplace=True)


# Feature & Target Split
X = peak_demand.drop(columns=['total load actual', 'date'])
y = peak_demand['total load actual']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
# Hyperparameter Tuning for XGBoost
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
}

model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model from Grid Search
best_model = grid_search.best_estimator_

# Predict
y_pred = best_model.predict(X_test)

# Evaluate Model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Optimized RMSE: {rmse}")
print(f"Optimized R² Score: {r2}")


# Plot Actual vs Predicted Peak Demand
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual Peak Demand', color='red')
plt.plot(y_test.index, y_pred, label='Predicted Peak Demand', linestyle='dashed', color='blue')
plt.title("Optimized Peak Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Peak Demand (MW)")
plt.legend()
st.pyplot(plt.gcf())

