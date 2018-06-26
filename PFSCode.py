#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 16:46:32 2018

@author: tidyquantpc
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
plt.style.use('ggplot') 

test = pd.read_csv("test.csv")
train=pd.read_csv("sales_train_v2.csv")
items=pd.read_csv("items-translated.csv")
items_2=pd.read_csv("items.csv")
item_categories = pd.read_csv("item_categories-translated.csv")
shops = pd.read_csv("shops-translated.csv")
items= pd.merge(items, items_2, how='inner', on='item_id')
items.drop(columns=['item_name'], inplace=True)
del items_2
shops.columns =  ('shop_id','shop_name')
item_categories.columns =  ('item_category_id','item_category_name')
items.columns = ('item_id', ' item_name', 'item_category_id' )

                           ####Feature Engineering####
                           
#Creating more generic item categories for analysis#
l = list(item_categories.item_category_name)

for ind in range(1,8):
    l[ind] = 'Access'

for ind in range(10,18):
    l[ind] = 'Consoles'

for ind in range(18,25):
    l[ind] = 'Consoles Games'

for ind in range(26,28):
    l[ind] = 'Phone Games'

for ind in range(28,32):
    l[ind] = 'CD games'

for ind in range(32,37):
    l[ind] = 'Card'

for ind in range(37,43):
    l[ind] = 'Movie'

for ind in range(43,55):
    l[ind] = 'Books'

for ind in range(55,61):
    l[ind] = 'Music'

for ind in range(61,73):
    l[ind] = 'Gifts'

for ind in range(73,79):
    l[ind] = 'Software'

item_categories['type'] = l #Inserting category names into dataset

#Formatting Date 
train['date'] = pd.to_datetime(train.date,format="%d.%m.%Y")

#Inserting Boolean Holiday Values
train['December'] = train.date_block_num.apply(lambda x: 1 if x ==23 else 0)
train['Newyear_Xmas'] = train.date_block_num.apply(lambda x: 1 if x in [12,24] else 0)
train['Valentines'] = train.date_block_num.apply(lambda x: 1 if x in [13,25] else 0)
train['WomenDay'] = train.date_block_num.apply(lambda x: 1 if x in [14,26] else 0)
train['Easter'] = train.date_block_num.apply(lambda x: 1 if x in [15,27] else 0)

#Merge all datasets#
train = pd.merge(train, items, how='left', on=['item_id'])
train = pd.merge(train, item_categories, how='left', on=['item_category_id'])
train = pd.merge(train, shops, how='left', on=['shop_id'])

##Adding Aggregate Features for Items,Categories and Shops, including the sum and mean values
sales_shop = train.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'shop_block_target_sum':'sum','shop_block_target_mean':np.mean}})
sales_shop.columns = [col[0] if col[-1]=='' else col[-1] for col in sales_shop.columns.values]
sales_new = pd.merge(sales_shop, train, how='left', on=['shop_id', 'date_block_num']).fillna(0)

sales_items = train.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'item_block_target_sum':'sum','item_block_target_mean':np.mean}})
sales_items.columns = [col[0] if col[-1]=='' else col[-1] for col in sales_items.columns.values]
sales_new = pd.merge(sales_items, sales_new, how='left', on=['item_id', 'date_block_num']).fillna(0)

sales_cat = train.groupby(['item_category_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'item_cat_block_target_sum':'sum','item_cat_block_target_mean':np.mean}})
sales_cat.columns = [col[0] if col[-1]=='' else col[-1] for col in sales_cat.columns.values]
sales_new = pd.merge(sales_new, sales_cat, how='left', on=['item_category_id', 'date_block_num']).fillna(0)
sales_new = sales_new.rename(columns={' item_name':'item_name'})
sales_new.info()

                        ############Prophet Modelling: Overall Company Sales#########


from fbprophet import Prophet
#Creating Appropriate Dataframe #
proph = train.groupby(['date_block_num'])[ 'item_cnt_day'].sum()
proph.index=pd.date_range(start='2013-01-01', end='2015-10-01', freq='MS')
proph = proph.to_frame().reset_index()
proph.columns = ['ds', 'y']
proph.head()
#Modelling#
model=Prophet(yearly_seasonality=True)
model.fit(proph)
#Making future predictions
future_data = model.make_future_dataframe(periods=1, freq='MS')
forecast_data = model.predict(future_data)
forecast = model.predict(future_data)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#Plotting Results
model.plot(forecast_data)
model.plot_components(forecast_data)

                    ############Prophet Modelling: Individual Store Sales#########

# get the unique combinations of item-store from the sales data at monthly level
monthly_sales=sales.groupby(["shop_id","item_id","date_block_num"])["item_cnt_day"].sum()
# rearrange dataset for modelling
monthly_sales=monthly_sales.unstack(level=-1).fillna(0)
monthly_sales=monthly_sales.T
dates=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
monthly_sales.index=dates
monthly_sales=monthly_sales.reset_index()
monthly_sales.head()
import time
start_time=time.time()


# Create a for loop, to create the shop forecasts using prophet
forecastsDict = {}
for node in range(len(monthly_sales)):
    # take the date-column and the col to be forecasted
    nodeToForecast = pd.concat([monthly_sales.iloc[:,0], monthly_sales.iloc[:, node+1]], axis = 1)
# rename for prophet compatability
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = m.predict(future)
    
#predictions = np.zeros([len(forecastsDict[0].yhat),1]) 
nCols = len(list(forecastsDict.keys()))+1
for key in range(0, nCols-1):
    f1 = np.array(forecastsDict[key].yhat)
    f2 = f1[:, np.newaxis]
    if key==0:
        predictions=f2.copy()
       # print(predictions.shape)
    else:
       predictions = np.concatenate((predictions, f2), axis = 1)
     
predictions_unknown=predictions[-1]
predictions_unknown
#These predictions are for each individual store, for the next month.#


                           ####Item-Store Prediction using XGBoost Model####
                           
train_1 = train.pivot_table(index=['shop_id','item_id'], columns='date_block_num', values='item_cnt_day',aggfunc='sum').fillna(0.0)
train_2 = train_1.reset_index()
train_2['shop_id']= train_2.shop_id.astype('str')
train_2['item_id']= train_2.item_id.astype('str')


item_to_cat_df = items.merge(item_categories[['item_category_id','type']], how="inner", on="item_category_id")[['item_id','type']]
item_to_cat_df[['item_id']] = item_to_cat_df.item_id.astype('str')

train_3 = train_2.merge(item_to_cat_df, how="inner", on="item_id")
train_3.head()
# Encode Categories
from sklearn import preprocessing

number = preprocessing.LabelEncoder()
train_3[['type']] = number.fit_transform(train_3.type)
train_3 = train_3[['shop_id', 'item_id', 'type'] + list(range(34))]
train_3.head()

import xgboost as xgb
param = {'max_depth':10, 
         'subsample':1,
         'min_child_weight':0.5,
         'eta':0.3, 
         'num_round':1000, 
         'seed':1,
         'silent':0,
         'eval_metric':'rmse'}

progress = dict()
xgbtrain = xgb.DMatrix(train_cleaned_df.iloc[:,  (train_cleaned_df.columns != 33)].values, train_cleaned_df.iloc[:, train_cleaned_df.columns == 33].values)
watchlist  = [(xgbtrain,'train-rmse')]

bst = xgb.train(param, xgbtrain)
preds = bst.predict(xgb.DMatrix(train_cleaned_df.iloc[:,  (train_cleaned_df.columns != 33)].values))
from sklearn.metrics import mean_squared_error 
rmse = np.sqrt(mean_squared_error(preds,train_cleaned_df.iloc[:, train_cleaned_df.columns == 33].values))
print(rmse)
xgb.plot_importance(bst)
apply_df = test
apply_df['shop_id']= apply_df.shop_id.astype('str')
apply_df['item_id']= apply_df.item_id.astype('str')

apply_df = test.merge(train_cleaned_df, how = "left", on = ["shop_id", "item_id"]).fillna(0.0)
d = dict(zip(apply_df.columns[4:],list(np.array(list(apply_df.columns[4:])) - 1)))
