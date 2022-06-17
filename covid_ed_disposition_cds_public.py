
# Authors:          Johns Hopkins Center for Data Science in Emergency Medicine

# Version:          Version 3.6.10

# Last Updated:     2022-06-15

# Description:      The script executes retrospective model development and evaluation of a dataframe with
#                   predictors listed in the Supplementary Table 1 and described in the Methods Section (Outcome and Predictor Measures) and
#                   outcomes described in the Methods Section (Outcome and Predictor Measures).
    
# Content Blocks:   1. Import required packages
#                   2. Loading formatted data
#                   3. Training random forest modelS
#                   4. Create risk thresholds and setting risk-levels
#                   5. Evaluate models


#%% 1. IMPORT REQUIRED PACKAGES

# import packages
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import matplotlib.pyplot as plt
import shap

#%% 2. LOAD FORMATTED TRAINING DATA
#Note - these data are fake and were randomly generated

# setting directory path (adjust)
path = "C:\\[SET DIRECTORY]\\"

# loading outcomes data indexed
enc_set = pd.read_csv (path+'enc_set.csv')
outcomes = ['out_cc','out_ac'] # <- dataframe column names for the critical care 'out_cc' and acute care 'out_ac' binary outcomes
enc_set['arrdt'] = pd.to_datetime(enc_set['arrdt'])

# loading corresponding predictor data (indexed to enc_set) in dummy variable (binary) format (see summary Table 1)
pred_dummy = pd.read_csv (path+'pred_dummy.csv')


#%% 3. TRAINING RANDOM FOREST MODELS

##############################################################################
# @ param test_size = Proportion of patients in the test ste
# @ param leaf = The minimum number of samples required to be at a leaf node
# @ param tree_estimators = The number of trees created in the random forest
##############################################################################

# setting parameters
test_size = 0.33
leaf = 10
tree_estimators = 50 


# create training and test set index
idx_train, idx_test = sklearn.model_selection.train_test_split(enc_set.index, test_size = test_size, random_state = 33)
 
# creating predicted probability dataframe
y_pred = pd.DataFrame(np.zeros(shape=(len(enc_set),len(outcomes))), columns = outcomes) 

# training and random forest model and gathering predictive performance data on the independent test set
rf = {}
for outcome in outcomes:
    
    # train 
    rf[outcome] = RandomForestClassifier(n_estimators = tree_estimators, min_samples_leaf = leaf, n_jobs = 4)
    rf[outcome].fit(pred_dummy.loc[idx_train], enc_set[outcome].loc[idx_train])
    y_pred[outcome].loc[idx_train] = rf[outcome].predict_proba(pred_dummy.loc[idx_train])[:,1]
        
    # test evaluation
    y_pred[outcome].loc[idx_test] = rf[outcome].predict_proba(pred_dummy.loc[idx_test])[:,1]
    rf[outcome+'_auc_test'] = roc_auc_score(enc_set[outcome].loc[idx_test],y_pred[outcome].loc[idx_test])
    fpr_rf, tpr_rf, _ = sklearn.metrics.roc_curve(enc_set[outcome].loc[idx_test],y_pred[outcome].loc[idx_test])
    rf[outcome+'_rocdata_test'] = pd.DataFrame({'fpr':fpr_rf,'tpr':tpr_rf})        
    print('test '+outcome+' auc = '+str(rf[outcome+'_auc_test'])) 




#%% 4. CREATING RISK THRESHOLDS AND SETTING RISK-LEVELS

# creating arbitrary risk profile
riskprofile = pd.DataFrame({'level':list(reversed(range(1,11))),
                             'out_cc_thresh':0,
                             'out_ac_thresh':0})
riskprofile['out_cc_thresh'].loc[range(0,4)] = np.percentile(y_pred['out_cc'],[90,80,70,60]) # <- thresholds for the critical care outcome
riskprofile['out_ac_thresh'].loc[range(3,10)] = np.percentile(y_pred['out_ac'],[80,65,50,35,20,15,5]) # <- thresholds for the acute care outcome


# setting level risk level based on predicted probability and
def set_disp_level_agg (data,riskprofile):
    if 'level' in data.columns:
        del data['level']
    data['level'] = int(0)
    for level in riskprofile['level']:
        ridx = np.where(riskprofile['level'] == level)[0].item()
        if level == 10: 
            idx = data['prob_cc'] >= riskprofile['out_cc_thresh'].loc[ridx]
        # consider out_cc only
        elif (riskprofile['out_cc_thresh'].loc[ridx] > 0) & (riskprofile['out_ac_thresh'].loc[ridx] == 0) & (level!=10):
            idx = (data['prob_cc'] >= riskprofile['out_cc_thresh'].loc[ridx]) & \
                  (data['level'] == 0)
        # consider both
        elif (riskprofile['out_cc_thresh'].loc[ridx] > 0) & (riskprofile['out_ac_thresh'].loc[ridx] > 0):
            idx = ((data['prob_cc'] >= riskprofile['out_cc_thresh'].loc[ridx]) & \
                  (data['prob_cc'] < riskprofile['out_cc_thresh'].loc[ridx-1])) | \
                  (data['prob_ac'] >= riskprofile['out_ac_thresh'].loc[ridx]) & \
                  (data['level'] == 0)    
        # consider only out_ac
        elif (riskprofile['out_cc_thresh'].loc[ridx] == 0) & (riskprofile['out_ac_thresh'].loc[ridx] > 0) & (level!=1): 
            idx = (data['prob_ac'] >= riskprofile['out_ac_thresh'].loc[ridx]) & \
                  (data['prob_ac'] < riskprofile['out_ac_thresh'].loc[ridx-1]) & \
                  (data['level'] == 0)     
        elif level == 1:
            idx = (data['prob_ac'] < riskprofile['out_ac_thresh'].loc[ridx]) & \
                  (data['level'] == 0)
        data['level'].loc[idx] = level
    return data

# merging outcomes to predicted probabilities
data = pd.merge(enc_set.loc[idx_test],y_pred.loc[idx_test].rename(columns={'out_cc':'prob_cc','out_ac':'prob_ac'}),how='left',left_index=True,right_index=True,copy=False)
# setting levels
data = set_disp_level_agg(data,riskprofile)


#%% 5. EVALUATE MODELS


# plotting ROC curves ########################################################

def roc_plot(rocdata,label,auc,fignum,color,figtitle,linetype='-'):
    fs = 18
    fig_x = plt.figure(num=fignum, figsize=(10, 10), dpi=300, facecolor='w', edgecolor='w')
    #plt.subplot(211)    
    plt.plot(rocdata['fpr'],rocdata['tpr'],color = color,linestyle=linetype,linewidth=4,label=label+' | AUC = '+str(auc))
    plt.xticks(fontsize=fs)
    plt.xlabel('False positive rate',fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.ylabel('True positive rate',fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(loc='lower right',fontsize=fs,frameon=False)
    plt.title(figtitle,fontsize=fs+4)
    plt.tight_layout()
    return fig_x

fig_roc_cc = roc_plot(rf['out_cc_rocdata_test'],'Test Set',"{:.2f}".format(rf['out_cc_auc_test']),1,'darkgrey','Critical Care Outcome',linetype='-') # critical care outcome
fig_roc_ac = roc_plot(rf['out_ac_rocdata_test'],'Test Set',"{:.2f}".format(rf['out_ac_auc_test']),2,'darkgrey','Acute Care Outcome',linetype='-') # acute care outcome

# precision versus recall statistics #########################################

def precision_recall_stats(data,riskprofile,outcome):
    # setting up performance stats
    pstats = riskprofile.copy()
    cols = ['Total','Outcome','Precision','Recall','NPV','TP:FP','Sensitivity','Specificity']
    for col in cols:
        pstats[col] = 0
    # creating table
    for level in pstats['level']:
        ridx = np.where(pstats['level'] == level)[0].item()
        # total
        pstats['Total'].loc[ridx] = sum(data['level']>=level)
        # outcomes
        pstats['Outcome'].loc[ridx] = sum(data[outcome].loc[data['level']>=level])
        # tp:fp
        tp = sum(data[outcome].loc[data['level']>=level])
        fp = sum(data['level']>=level) - sum(data[outcome].loc[data['level']>=level])
        tn = sum((data[outcome] == 0) & (data['level'] < level))
        n = sum(data[outcome] == 0)
        pstats['TP:FP'].loc[ridx] = str(int(tp))+':'+str(int(fp))
        # precision
        pstats['Precision'].loc[ridx] = precision_score(data[outcome],(data['level']>=level))
        #recall
        pstats['Recall'].loc[ridx] = recall_score(data[outcome],(data['level']>=level))
        # npv
        pstats['NPV'].loc[ridx] = tn/float(n)
        # sensitivity
        pstats['Sensitivity'].loc[ridx] = tp/sum(data[outcome])
        # specificity
        pstats['Specificity'].loc[ridx] = sum((data['level']<level) & (data[outcome] == 0)) / (len(data)-sum(data[outcome]))
    return pstats    


pstats_cc = precision_recall_stats(data.loc[idx_test],riskprofile,'out_cc')
pstats_ac = precision_recall_stats(data.loc[idx_test],riskprofile,'out_ac')

# shap analyses ##############################################################
n = 20
# @param n = number of features to show by importance

# critical care ##############################################################
explainer_cc = shap.TreeExplainer(rf['out_cc'],pred_dummy.loc[idx_test])
shap_values_cc = explainer_cc.shap_values(pred_dummy.loc[idx_test])
# bar plot
fig_bar_cc = plt.figure(num=3, figsize=(8, round(n/3)), dpi=300, facecolor='w', edgecolor='w')
shap.summary_plot(shap_values_cc[1], pred_dummy.loc[idx_test], sort=True, max_display=20,color='#d62728',show=False,plot_type='bar')
plt.title('Critical Care Outcome',fontsize=14)
plt.xlabel('Mean(|SHAP Value|) - Average Impact on Model Output')
plt.tight_layout() 
# beeswarm plot
fig_bee_cc = plt.figure(num=4, figsize=(8, round(n/3)), dpi=300, facecolor='w', edgecolor='w')
shap.summary_plot(shap_values_cc[1], pred_dummy.loc[idx_test], sort=True, max_display=20,cmap='PiYG_r',show=False)
plt.title('Critical Care Outcome',fontsize=14)
plt.xlabel('SHAP Value - Impact on Model Output)')
plt.tight_layout() 

# acute care ##############################################################
explainer_ac = shap.TreeExplainer(rf['out_ac'],pred_dummy.loc[idx_test])
shap_values_ac = explainer_ac.shap_values(pred_dummy.loc[idx_test])
# bar plot
fig_bar_ac = plt.figure(num=5, figsize=(8, round(n/3)), dpi=300, facecolor='w', edgecolor='w')
shap.summary_plot(shap_values_ac[1], pred_dummy.loc[idx_test], sort=True, max_display=20,color='#1f77b4',show=False,plot_type='bar')
plt.title('Acute Care Outcome',fontsize=14)
plt.xlabel('Mean(|SHAP Value|) - Average Impact on Model Output')
plt.tight_layout() 
# beeswarm plot
fig_bee_ac = plt.figure(num=6, figsize=(8, round(n/3)), dpi=300, facecolor='w', edgecolor='w')
shap.summary_plot(shap_values_ac[1], pred_dummy.loc[idx_test], sort=True, max_display=20,cmap='PiYG_r',show=False)
plt.title('Critical Care Outcome',fontsize=14)
plt.xlabel('SHAP Value - Impact on Model Output)')
plt.tight_layout() 


#%%    
    
        
        
        
        