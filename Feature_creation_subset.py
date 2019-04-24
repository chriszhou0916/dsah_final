#!/usr/bin/env python
# coding: utf-8

# ### Goal: create feature table that includes
# - abnormal lab values
# - min max mean of select lab values
# - min max mean of select chart events
# - sum of urine output
# - demographic data

# In[1]:


import pandas as pd
import numpy as np
import sqlite3 # library for working with sqlite database
conn = sqlite3.connect("./data/MIMIC.db") # Create a connection to the on-disk database
admissions_df = pd.read_sql("""SELECT * FROM admissions""",conn)
admissions_df['DISCHTIME'] = pd.to_datetime(admissions_df['DISCHTIME'])
admissions_df['ADMITTIME'] = pd.to_datetime(admissions_df['ADMITTIME'])
admissions_df['length_stay'] = (admissions_df['DISCHTIME'] - admissions_df['ADMITTIME']).dt.days
admissions_length = admissions_df[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','length_stay','HOSPITAL_EXPIRE_FLAG']]
patient_dead = admissions_length[admissions_length.HOSPITAL_EXPIRE_FLAG ==1]
patient_alive = admissions_length[admissions_length.HOSPITAL_EXPIRE_FLAG ==0]
patient_alive_list=patient_alive.sample(20)['HADM_ID'].to_list()
patient_dead_list=patient_dead.sample(20)['HADM_ID'].to_list()
patient_total = patient_alive_list + patient_dead_list
patient_total = [str(x) for x in patient_total]
patient_total= ",".join(patient_total)
patient_total = "("+patient_total+")"

# In[2]:
query_string="SELECT * FROM labevents WHERE HADM_ID IN" + patient_total
labevents_subset = pd.read_sql(query_string, conn)
# labevents_subset = pd.read_csv('./subsets_tables_collection/labevents_subset.csv')
labevents_subset['HADM_ID'] = labevents_subset['HADM_ID'].astype(np.int64)
labevents_subset['CHARTTIME'] = pd.to_datetime(labevents_subset['CHARTTIME'])
labevents_subset['ROW_ID'] = 'lab' + labevents_subset['ROW_ID'].astype(str)
print('Finished lab events')
query_string="SELECT * FROM chartevents WHERE HADM_ID IN" + patient_total
chartevents_subset = pd.read_sql(query_string, conn)
# chartevents_subset = pd.read_csv('./subsets_tables_collection/chartevents_subset.csv')
chartevents_subset['HADM_ID'] = chartevents_subset['HADM_ID'].astype(np.int64)
chartevents_subset['CHARTTIME'] = pd.to_datetime(chartevents_subset['CHARTTIME'])
chartevents_subset['ROW_ID'] = 'chart' + chartevents_subset['ROW_ID'].astype(str)
print('Finished chart events')
query_string="SELECT * FROM outputevents WHERE HADM_ID IN" + patient_total
outputevents_subset = pd.read_sql(query_string, conn)
# outputevents_subset = pd.read_csv('./subsets_tables_collection/outputevents_subset.csv')
outputevents_subset['HADM_ID'] = outputevents_subset['HADM_ID'].astype(np.int64)
outputevents_subset['CHARTTIME'] = pd.to_datetime(outputevents_subset['CHARTTIME'])
outputevents_subset['ROW_ID'] = 'output' + outputevents_subset['ROW_ID'].astype(str)
print('Finished output events')
lab_times = labevents_subset[['ROW_ID','CHARTTIME','HADM_ID']]
chart_times = chartevents_subset[['ROW_ID','CHARTTIME','HADM_ID']]
out_times = outputevents_subset[['ROW_ID','CHARTTIME','HADM_ID']]
lab_chart_out_times = pd.concat([lab_times, chart_times, out_times], sort=False)
grouped = lab_chart_out_times.groupby(['HADM_ID', pd.Grouper(freq='48H', key='CHARTTIME')])
count = 0
past_hadm = ''
def counter(x):
    global count, past_hadm
    curr_hadm = x.iloc[0]
    if past_hadm != curr_hadm:
        count = 0
        past_hadm = curr_hadm
    y = count
    count += 1
    return str(curr_hadm) + "_" + str(count)
lab_chart_out_times['HADM_datebin_num'] = grouped['HADM_ID'].transform(counter)


# In[3]:


labevents_subset_binned = labevents_subset.merge(lab_chart_out_times, on='ROW_ID')
chartevents_subset_binned = chartevents_subset.merge(lab_chart_out_times, on='ROW_ID')
outputevents_subset_binned = outputevents_subset.merge(lab_chart_out_times, on='ROW_ID')


# In[4]:


labitems = pd.read_sql("SELECT * FROM d_labitems", conn)
cols_to_use = ['ITEMID']
labevents_subset_binned = labevents_subset_binned.merge(labitems, on=cols_to_use)
labevents_subset_binned = labevents_subset_binned.drop(columns=['ROW_ID_y', 'index_y'])


# ### Make features out of abnormal lab values

# In[5]:


labevents_subset_abnormal = labevents_subset_binned[labevents_subset_binned['FLAG'] == 'abnormal']
labevents_subset_normal = labevents_subset_binned[labevents_subset_binned['FLAG'] != 'abnormal']


# In[6]:

num_abnormal_labels = labevents_subset_abnormal.LABEL.value_counts()
# dropping lab categories detected abnormal less than 5 times
abnormal_labels = num_abnormal_labels[num_abnormal_labels > 5].index


# In[7]:


#only greater than 5 times abnormal
labevents_subset_abnormal = labevents_subset_abnormal[labevents_subset_abnormal['LABEL'].isin(abnormal_labels)]



# In[9]:


labevents_subset_abnormal = pd.get_dummies(labevents_subset_abnormal, prefix='abnormal_lab', columns=['LABEL'])


# In[10]:


labevents_subset_features = pd.concat([labevents_subset_abnormal, labevents_subset_normal], sort=False)


# In[11]:


labevents_subset_features = labevents_subset_features.drop('LABEL', axis=1)


# In[12]:


filter_col = [col for col in labevents_subset_features if col.startswith('abnormal')]
filter_col.append('HADM_datebin_num')
labevents_subset_abnormal_features = labevents_subset_features[filter_col].groupby('HADM_datebin_num').sum()
labevents_subset_abnormal_features.fillna(0, inplace=True)


# ### Extract numeric features out of chartevents

# In[14]:


# What ITEMID from chartevents do we care about?
chart_ids_we_care = {
    'heart_rate': [211, 220045],
    'oxygen': [646, 220277, 834],
    'respiratory rate': [618,220210,3603,224689],
    'systolic blood pressure': [51, 455, 220179, 220050, 3313, 225309],
    'diastolic blood pressure': [8368, 8441, 220180, 220051, 8502, 225310],
    'mean blood pressure': [52, 456, 220181, 220052, 3312, 225312],
    'Glascow': [184, 723, 454, 220739, 223900, 223901],
    'temp F': [678, 223761, 679, ]
}
outputevents_ids_we_care = {
    'urine': [40055, 226559, 43175, 40069, 40094, 40065, 40061, 40715, 226627, 40473]
}
lab_ids_we_care = {
    'Hematocrit': [51221],
    'potassium': [50971],
    'sodium': [50983],
    'creatinine': [50912],
    'chloride': [50902],
    'platelets': [51265],
    'white blood cell': [51301],
    'hemoglobin': [51222],
    'glucose': [50931],
    'RBC count': [51279]
}


# In[15]:


l = list(chart_ids_we_care.values())
flat_list = [item for sublist in l for item in sublist]
chartevents_subset_binned['ITEMID'] = pd.to_numeric(chartevents_subset_binned['ITEMID'])
chartevents_subset_binned['VALUENUM'] = pd.to_numeric(chartevents_subset_binned['VALUENUM'])
chartevents_subset_selected = chartevents_subset_binned[chartevents_subset_binned['ITEMID'].isin(flat_list)]
itemid_to_var_df = pd.read_csv('./resources/itemid_to_variable_map.csv')
chartevents_subset_selected = chartevents_subset_selected.merge(itemid_to_var_df, on='ITEMID')
cols_to_keep = ['SUBJECT_ID', 'HADM_ID_x', 'ITEMID', 'CHARTTIME_x', 'VALUE', 'VALUENUM', 'VALUEUOM', 'LEVEL2', 'HADM_datebin_num']
chartevents_subset_selected = chartevents_subset_selected[cols_to_keep]


# In[16]:


#get rid off empty values
chartevents_subset_selected = chartevents_subset_selected[chartevents_subset_selected['VALUENUM'].notnull()]


# In[17]:


chartevents_features_max = chartevents_subset_selected.groupby(['HADM_datebin_num','LEVEL2'])['VALUENUM'].max().unstack()
chartevents_features_min = chartevents_subset_selected.groupby(['HADM_datebin_num','LEVEL2'])['VALUENUM'].min().unstack()
chartevents_features_mean = chartevents_subset_selected.groupby(['HADM_datebin_num','LEVEL2'])['VALUENUM'].mean().unstack()


# In[18]:


chartevents_features_max = chartevents_features_max.add_suffix('_max')
chartevents_features_min = chartevents_features_min.add_suffix('_min')
chartevents_features_mean = chartevents_features_mean.add_suffix('_mean')
chartevents_features = pd.concat([chartevents_features_max, chartevents_features_min, chartevents_features_mean], axis=1)


# In[19]:



# ### Extract numeric features from lab events

# In[20]:


l = list(lab_ids_we_care.values())
flat_list = [item for sublist in l for item in sublist]
labevents_subset_selected = labevents_subset_binned[labevents_subset_binned['ITEMID'].isin(flat_list)]
labevents_subset_selected = labevents_subset_selected.merge(itemid_to_var_df, on='ITEMID')
cols_to_keep = ['SUBJECT_ID', 'HADM_ID_x', 'ITEMID', 'CHARTTIME_x', 'VALUE', 'VALUENUM', 'VALUEUOM', 'LEVEL2', 'HADM_datebin_num']
labevents_subset_selected = labevents_subset_selected[cols_to_keep]
labevents_subset_selected = labevents_subset_selected[labevents_subset_selected['VALUENUM'].notnull()]
labevents_features_max = labevents_subset_selected.groupby(['HADM_datebin_num','LEVEL2'])['VALUENUM'].max().unstack()
labevents_features_min = labevents_subset_selected.groupby(['HADM_datebin_num','LEVEL2'])['VALUENUM'].min().unstack()
labevents_features_mean = labevents_subset_selected.groupby(['HADM_datebin_num','LEVEL2'])['VALUENUM'].mean().unstack()
labevents_features_max = labevents_features_max.add_suffix('_max')
labevents_features_min = labevents_features_min.add_suffix('_min')
labevents_features_mean = labevents_features_mean.add_suffix('_mean')
labevents_features = pd.concat([labevents_features_max, labevents_features_min, labevents_features_mean], axis=1)


# ### Extract numeric features from outevents

# In[21]:


l = list(outputevents_ids_we_care.values())
flat_list = [item for sublist in l for item in sublist]
outputevents_subset_selected = outputevents_subset_binned[outputevents_subset_binned['ITEMID'].isin(flat_list)]
outputevents_subset_selected = outputevents_subset_selected.merge(itemid_to_var_df, on='ITEMID')
cols_to_keep = ['SUBJECT_ID', 'HADM_ID_x', 'ITEMID', 'CHARTTIME_x', 'VALUE', 'VALUEUOM', 'LEVEL2', 'HADM_datebin_num']
outputevents_subset_selected = outputevents_subset_selected[cols_to_keep]
outputevents_subset_selected = outputevents_subset_selected[outputevents_subset_selected['VALUE'].notnull()]
outputevents_features_sum = outputevents_subset_selected.groupby(['HADM_datebin_num','LEVEL2'])['VALUE'].sum().unstack()

outputevents_features_sum = outputevents_features_sum.add_suffix('_sum')
# labevents_features.reset_index(inplace=True)
# outputevents_features_sum.fillna(outputevents_features_sum.mean(),inplace=True)



# ### Merge Everything

# In[23]:


feature_table = labevents_features.merge(chartevents_features, left_index=True, right_index=True, how='outer')
feature_table = feature_table.merge(labevents_subset_abnormal_features, left_index=True, right_index=True, how='outer')
feature_table = feature_table.merge(outputevents_features_sum, left_index=True, right_index=True, how='outer')


# ### Add HADM_ID back into the feature table, also create bin_num to mark which bin it is for that patient. these will be useful for assigning death labels

# In[24]:


feature_table['HADM_ID'] = feature_table.index
feature_table[['HADM_ID', 'bin_num']] = feature_table['HADM_ID'].str.split('_', expand=True)
feature_table['HADM_ID'] = pd.to_numeric(feature_table['HADM_ID'])
feature_table['bin_num'] = pd.to_numeric(feature_table['bin_num'])


# In[25]:



# ### Obtain number of diagnoses for each patient

# In[26]:

query_string="SELECT * FROM diagnoses_icd WHERE HADM_ID IN" + patient_total
diagnoses_subset = pd.read_sql(query_string, conn)
# diagnoses_subset = pd.read_csv('./subsets_tables_collection/diagnoses_icd_subset.csv')


# In[27]:


diagnoses_count = diagnoses_subset.groupby('HADM_ID')['ICD9_CODE'].count()


# ### Obtain number of medications for each patient

# In[28]:

query_string="SELECT * FROM prescriptions WHERE HADM_ID IN" + patient_total
prescriptions_subset = pd.read_sql(query_string, conn)
# prescriptions_subset = pd.read_csv('./subsets_tables_collection/prescriptions_subset.csv')
prescriptions_count = prescriptions_subset.groupby(['HADM_ID', 'DRUG_TYPE'])['DRUG'].count().unstack()
prescriptions_count.fillna(0, inplace=True)
prescriptions_count = prescriptions_count.add_suffix('_drug')


# In[29]:


prescriptions_count.reset_index(level=0, inplace=True)


# In[30]:




# In[31]:


diagnoses_count= pd.DataFrame({'HADM_ID':diagnoses_count.index, 'diag_count':diagnoses_count.values})


# In[32]:




# In[33]:




# In[34]:


prescription_diag_merge = pd.merge(diagnoses_count, prescriptions_count, how='outer', on='HADM_ID')
prescription_diag_merge = prescription_diag_merge.fillna(0)


# In[35]:



# Merge diagnoses and medications

# ### Incorporate demographics info

# In[36]:

query_string="SELECT subject_id, gender FROM patients"
patients = pd.read_sql(query_string, conn)
patients.replace(['M','F'], [0,1], inplace=True)
patients.rename(columns={'GENDER': 'is_female'}, inplace=True)
admissions_with_gender = admissions_df.merge(patients,on='SUBJECT_ID')
admissions_with_gender = admissions_with_gender[['HADM_ID', 'INSURANCE', 'ETHNICITY', 'is_female']]
admissions_features = pd.get_dummies(admissions_with_gender, prefix=['insurance','ethnicity'], columns=['INSURANCE', 'ETHNICITY'])
patient_features = admissions_features.merge(prescription_diag_merge, on='HADM_ID')


# In[37]:


# In[38]:


feature_table = feature_table.merge(patient_features, on='HADM_ID')

# In[41]:




# In[42]:


idx=feature_table.groupby(['HADM_ID'],sort=False)['bin_num'].transform(max)==feature_table['bin_num']
feature_table=feature_table[-idx]


# In[47]:


dead_patient_HADM_ID = admissions_df[admissions_df['HOSPITAL_EXPIRE_FLAG'] == 1]['HADM_ID']


# In[45]:


bool_max=feature_table.groupby(['HADM_ID'], sort=False)['bin_num'].transform(max)
feature_table['label']=np.where((bool_max==feature_table['bin_num']) & (feature_table['HADM_ID'].isin(dead_patient_HADM_ID)),1,0)


# In[48]:


feature_table.to_csv("./data/feature_with_label40.csv",index=None, header=True)


# In[ ]:
