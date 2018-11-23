#-------------------------------------------------------
#
# Script to analyse Airbnb data
# Hayden Eastwood - 20-11-2018
# Last updated: 20-11-2018
# Version: 1.0
#
#
# -------------------------------------------------------

from sklearn import linear_model
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import json
import numpy as np
import pandas as pd
import calendar
import seaborn as sb
import time
from datetime import datetime, timedelta, date
from operator import itemgetter
import seaborn as sns
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import mlhayden

# -------------------- Parameters -------------------------
home_country = 'BR' # Brazil country code

# Parameters for Random Forest Classifier
parameters = {
	'RF':{
	  	'n_estimators': [2, 4, 6, 8], 
	  	'criterion': ['entropy', 'gini'],
	  	'max_depth': [1, 2, 3, 5, 6, 7, 10, 12], 
	  	'min_samples_split': [3, 4, 5, 6, 7],
	  	'min_samples_leaf': [1, 3, 5, 7]
	}
}

# -------------------- End Parameters -------------------------


print " -- Loading data"
contacts = pd.read_csv('contacts.csv')	
listings = pd.read_csv('listings.csv')
guest_users = pd.read_csv('users.csv').rename(columns={'id_user_anon': 'id_guest_anon'}).drop_duplicates(subset=['id_guest_anon'])
host_users = pd.read_csv('users.csv').rename(columns={'id_user_anon': 'id_host_anon'}).drop_duplicates(subset=['id_host_anon'])

print " -- Merging lists, users and contacts into single table"
data_guests = pd.merge(pd.merge(contacts, listings, on='id_listing_anon', how='inner'), guest_users, on='id_guest_anon', how='inner')
data_hosts = pd.merge(pd.merge(contacts, listings, on='id_listing_anon', how='inner'), host_users, on='id_host_anon', how='inner')

print " -- Separate out hosts and guests for words in profile"
data = data_guests
data['guest_words_in_user_profile'] = data_guests['words_in_user_profile']
data['host_words_in_user_profile'] = data_hosts['words_in_user_profile']
data['COUNTER'] = 1 # set counter to make counting records easy with "groupby" commands
data[(data.isnull())]

print " -- Calculating 'time spent' information"
not_null = data[data.ts_booking_at.notnull()].index.tolist()
data.loc[data.index.isin(not_null), 'booked'] = 1
data['booked'] = data['booked'].fillna(0).astype(int)
data.ts_interaction_first = pd.to_datetime(data.ts_interaction_first)
data.ts_reply_at_first = pd.to_datetime(data.ts_reply_at_first)
data['response_hours_to_first_interaction'] = (data.ts_reply_at_first - data.ts_interaction_first).astype('timedelta64[h]')

data.ts_accepted_at_first = pd.to_datetime(data.ts_accepted_at_first)
data.ts_booking_at = pd.to_datetime(data.ts_booking_at)
data.ds_checkin_first = pd.to_datetime(data.ds_checkin_first)
data.ds_checkout_first = pd.to_datetime(data.ds_checkout_first)
data['hours_stay'] = (data.ds_checkout_first - data.ds_checkin_first).astype('timedelta64[h]')

data.ds_checkin_first = pd.to_datetime(data.ds_checkin_first)
data.ts_interaction_first = pd.to_datetime(data.ts_interaction_first)
data['hours_between_contact_and_checkin'] = (data.ds_checkin_first - data.ts_interaction_first).astype('timedelta64[h]')


print " -- Generate one hot encoding for contact_channel_first, guest_user_stage_first and room_type"
data_pipe1 = pd.merge(data, pd.get_dummies(data['contact_channel_first']), left_index=True, right_index=True)
data_pipe2 = pd.merge(data, pd.get_dummies(data['guest_user_stage_first']), left_index=True, right_index=True)
data_pipe3 = pd.merge(data, pd.get_dummies(data['room_type']), left_index=True, right_index=True)


data_pipe3['guest_brazillian'] = 0
data_pipe3.guest_brazillian.iloc[data_pipe3[(data_pipe3.country == home_country)].index.tolist()] = 1
data_booking_time = data_pipe3[(data_pipe3.booked == 1)] # keep this for later to examine booking times
data_pipe3 = data_pipe3.drop(['ts_booking_at', 'ds_checkin_first', 'ts_accepted_at_first', 'ts_interaction_first', 'ts_reply_at_first', 'ds_checkin_first', 'ds_checkout_first', 'room_type', 'guest_user_stage_first', 'contact_channel_first', 'words_in_user_profile'], 1)


country_aggregate = data_pipe3.groupby('country').agg(np.sum).reset_index()
country_aggregate['booking_ratio'] = country_aggregate['booked'] / country_aggregate['COUNTER']
country_aggregate = country_aggregate[(country_aggregate['COUNTER'] > 10)][['country', 'COUNTER', 'booked', 'booking_ratio']]
country_aggregate = country_aggregate.reset_index()

print " -- Generate one hot encoding for contact_channel_first, guest_user_stage_first and room_type"



guest_aggregate = data_pipe3.groupby('id_guest_anon').agg(np.sum).reset_index()

nbd_aggregate = data_pipe3.groupby('listing_neighborhood').agg(np.sum).reset_index()
nbd_aggregate['booking_ratio'] = nbd_aggregate['booked'] / nbd_aggregate['COUNTER']
nbd_aggregate_reduced = nbd_aggregate[(nbd_aggregate['COUNTER'] > 10)][['listing_neighborhood', 'COUNTER', 'booked', 'booking_ratio']].reset_index()

#data_pipe3[(data_pipe1.listing_neighborhood.isin(nbd_aggregate_red['listing_neighborhood']))]
for count in range(0, len(nbd_aggregate_reduced)):
	total_bookings = len(data_pipe3[(data_pipe3.listing_neighborhood == nbd_aggregate_reduced['listing_neighborhood'][count])])
	brazil_bookings = len(data_pipe3[(data_pipe3.listing_neighborhood == nbd_aggregate_reduced['listing_neighborhood'][count]) & (data_pipe3.country == home_country)])
	
	nbd_aggregate_reduced.loc[count, 'brazil_foreign_ratio'] = float(brazil_bookings) / total_bookings



print " -- Final cleaning"
print "      -- clean m_guests - assume missing guest is single person"
avg_guest_number = int(round(data_pipe3.m_guests.mean(), 0))
missing_guest_indexes = data_pipe3[(data_pipe3.m_guests.isnull())].index.tolist()
data_pipe3.loc[missing_guest_indexes, 'm_guests'] = avg_guest_number

print "      -- clean response_hours_to_first_interaction - for some reason these are coming out as NAN - removing them"
data_pipe3 = data_pipe3[(data_pipe3.response_hours_to_first_interaction.notnull())]


print " -- Get variable significance with Random Forest"
print "      -- scale data"

sc = StandardScaler()
data_pipe4 = data_pipe3.drop(['country', 'id_listing_anon', 'listing_neighborhood', 'id_host_anon', 'id_guest_anon', 'id_host_anon', 'id_listing_anon', 'listing_neighborhood', 'booked', 'country', 'COUNTER'], 1)

print "      -- split data into training and test"
train, test, train_target, test_target = train_test_split(sc.fit_transform(data_pipe4), data_pipe3.booked, test_size=0.3, random_state=42)



print "      -- perform grid search"
classifier = RandomForestClassifier()  
grid_object = GridSearchCV(classifier, parameters['RF'], cv=5)
grid_object.fit(train, train_target.tolist())

print "      -- make predictions"
predictions = grid_object.best_estimator_.predict(test)

print "      -- extract features"
main_features = pd.DataFrame(grid_object.best_estimator_.feature_importances_,index=data_pipe4.columns,columns=['importance']).sort_values('importance',ascending=False)
print main_features
mlhayden.perf_measure(test_target.tolist(), predictions, type='full')


cumsum_main_features = pd.DataFrame(np.cumsum(main_features['importance'])).reset_index()
plt.plot(cumsum_main_features['importance'])
plt.ylabel('Proportional importance')
plt.xlabel('Variable')

plt.plot(data.ts_booking_at, data.groupby(pd.TimeGrouper(freq='M')).mean()


plt.scatter(data['m_interactions'], data['total_reviews'], alpha=0.2, s=100*data['m_guests'], c=data.booked, cmap='viridis')

plt.scatter(data['m_interactions'], data['response_hours_to_first_interaction'], c=data.booked)

country_count = pd.DataFrame(data.groupby(['country']).agg(np.sum).reset_index()[['country', 'COUNTER']])
country_count = country_count.sort_values(['COUNTER'], ascending=False).reset_index()

print " -- Merging lists, users and contacts into single table"
cumsum = pd.DataFrame(np.cumsum(country_count['COUNTER']))
cumsum['percent'] = cumsum['COUNTER']/(cumsum['COUNTER'][len(cumsum) - 1])



#plots
data_booking_time.set_index('ts_booking_at', inplace=True)
bookings_per_week = data_booking_time.resample('W',how='sum')['booked']
plt.plot(bookings_per_week)

data_booking_time.resample('W',how='mean')

plt.plot(data[(data.booked > 0)])

#braz_plot = plt.plot(nbd_aggregate_red['brazil_foreign_ratio'], label='% brazilians')
#booking_plot = plt.plot(nbd_aggregate_red['booking_ratio'], label='proportion of booked enquiries')
#plt.legend()
#plt.show()
#data[(data.country == 'US')].groupby(['']).agg(np.sum)).reset_index()[['customer_code']]


#labels = data['country'].unique().tolist()
#mapping = dict( zip(labels,range(len(labels))))
#data.replace({'labels': mapping},inplace=True)

#plt.xticks(np.arange(len(country_aggregate)), country_aggregate['country'])
#plt.plot(country_aggregate['booking_ratio'])
#plt.show()

