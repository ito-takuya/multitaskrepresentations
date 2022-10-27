import numpy as np
import nibabel as nib
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import tools
import pandas as pd
import matplotlib.image as img 
import statsmodels.sandbox.stats.multicomp as mc
import argparse
import os
import h5py
import sklearn
from sklearn.decomposition import PCA
import bct

basedir = '/home/ti236/taku/multiTaskHierarchy/'
outdir = basedir + 'derivatives/results/analysis3/'
if not os.path.exists(outdir): os.makedirs(outdir)

networkdef = np.loadtxt('/home/ti236/AnalysisTools/ColeAnticevicNetPartition/cortex_parcel_network_assignments.txt')
# need to subtract one to make it compatible for python indices
indsort = np.loadtxt('/home/ti236/AnalysisTools/ColeAnticevicNetPartition/cortex_community_order.txt',dtype=int) - 1 
indsort.shape = (len(indsort),1)
# network mappings for final partition set
networkmappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8, 'lan':6, 'dan':5, 'con':4, 'dmn':9, 
                   'pmulti':10, 'none1':11, 'none2':12}
networkkey2name = {1:'VIS1',2:'VIS2',3:'SMN',4:'CON',5:'DAN',6:'LAN',7:'FPN',8:'AUD',9:'DMN',10:'PMULTI',11:'VMM',12:'ORA'}

networks = networkmappings.keys()

## General parameters/variables
nParcels = 360
glasserfilename = '/home/ti236/AnalysisTools/Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser = np.squeeze(nib.load(glasserfilename).get_fdata())
schaeferfilename = '/home/ti236/AnalysisTools/Schaefer2018_400Parcels_7Networks_order.dlabel.nii'
schaefer = np.squeeze(nib.load(schaeferfilename).get_fdata())
gordonfilename = '/home/ti236/AnalysisTools/GordonParcels/Parcels_LR.dlabel.nii'
gordon = np.squeeze(nib.load(schaeferfilename).get_fdata())
 
# Set task ordering
unique_tasks = ['NoGo','Go','TheoryOfMind','VideoActions','VideoKnots','Math',
                'DigitJudgement','Objects','MotorImagery','FingerSimple','FingerSeq',
                'Verbal2Back','SpatialImagery','VerbGen','WordRead','Rest',
                'PermutedRules','SpatialMapEasy','SpatialMapMed','SpatialMapHard',
                'NatureMovie','AnimatedMovie','LandscapeMovie','UnpleasantScenes','PleasantScenes',
                'SadFaces','HappyFaces','Object2Back','IntervalTiming',
                'Prediction','PredictViol','PredictScram','VisualSearchEasy','VisualSearchMed','VisualSearchHard',
                'StroopIncon','StroopCon','MentalRotEasy','MentalRotMed','MentalRotHard',
                'BiologicalMotion','ScrambledMotion','RespAltEasy','RespAltMed','RespAltHard']

motor_mappings = {'NoGo': 'left', 'Go': 'left', 'TheoryOfMind': 'left', 'VideoActions': 'passive',
                'VideoKnots': 'passive', 'Math': 'right', 'DigitJudgement': 'right', 'Objects': 'passive',
                'MotorImagery': 'passive', 'FingerSimple': 'both', 'FingerSeq': 'both', 'Verbal2Back': 'left',
                'SpatialImagery': 'passive', 'VerbGen': 'passive', 'WordRead': 'passive', 'Rest': 'passive',
                'PermutedRules': 'both', 'SpatialMapEasy': 'both', 'SpatialMapMed': 'both', 'SpatialMapHard': 'both',
                'NatureMovie': 'passive', 'AnimatedMovie': 'passive', 'LandscapeMovie': 'passive', 'UnpleasantScenes': 'left',
                'PleasantScenes': 'left', 'SadFaces': 'right', 'HappyFaces': 'right', 'Object2Back': 'right',
                'IntervalTiming': 'right', 'Prediction': 'left', 'PredictViol': 'left', 'PredictScram': 'left',
                'VisualSearchEasy': 'left', 'VisualSearchMed': 'left', 'VisualSearchHard': 'left', 'StroopIncon': 'both',
                'StroopCon': 'both', 'MentalRotEasy': 'right', 'MentalRotMed': 'right', 'MentalRotHard': 'right',
                'BiologicalMotion': 'right', 'ScrambledMotion': 'right', 'RespAltEasy': 'both', 'RespAltMed': 'both',
                'RespAltHard': 'both'}


task_passivity = ['left','left','left','passive','passive','right',
                  'right','passive','passive','both','both',
                  'left','passive','passive','passive','passive',
                  'both','both','both','both',
                  'passive','passive','passive','left','left',
                  'right','right','right','right',
                  'left','left','left','left','left','left',
                  'both','both','right','right','right',
                  'right','right','both','both','both']

# sort tasks by passivity
unique_tasks = np.asarray(unique_tasks)
task_passivity = np.asarray(task_passivity)
unique_tasks2 = []
passivity_order = ['passive','left','right','both']
for i in passivity_order:
    ind = np.where(task_passivity==i)[0]
    unique_tasks2.extend(unique_tasks[ind])
unique_tasks = np.asarray(unique_tasks2)


subIDs=['02','03','04','06','08','09','10','12','14','15','18','20','22','25','27','29','31','17','19','21','24','26','28','30']
sessIDs = ['a1','a2','b1','b2']
runs = range(1,9)

parser = argparse.ArgumentParser('./main.py', description='Run RSM analysis for each parcel using vertex-wise activations')
parser.add_argument('--outfilename', type=str, default="analysis3", help='Prefix output filenames (Default: analysis3')
parser.add_argument('--decoder', type=str, default="distance", help='decoder type')
parser.add_argument('--distance', type=str, default="cosine", help='distance metric for computing representational distance (across RDMs)')
parser.add_argument('--atlas', type=str, default="glasser", help='parcellation atlas (Default: glasser)')
parser.add_argument('--groupavg', action='store_true', help='Compute decoding from group-averaged activations')
parser.add_argument('--ncomponents', type=int, default=3, help='number of principal components to compute (default: 5)')

def run(args):
    args 
    outfilename = args.outfilename
    decoder = args.decoder
    distance = args.distance
    ncomponents = args.ncomponents
    groupavg = args.groupavg
    atlas = args.atlas

    groupavg_str = '_groupavg' if groupavg else ''

    if atlas=='glasser': 
        nParcels=360
        networkdef = np.loadtxt('/home/ti236/AnalysisTools/ColeAnticevicNetPartition/cortex_parcel_network_assignments.txt')
    if atlas=='schaefer':
        outfilename = outfilename + '_' + atlas
        networkdef = np.loadtxt('/home/ti236/AnalysisTools/Schaefer2018_400Parcels_7Networks_info_short.txt',dtype=str)
        #networkdef = np.loadtxt('/home/ti236/AnalysisTools/Schaefer2018_400Parcels_7Networks_order_info.txt',dtype=str)
        nParcels = 400
        print('Running analyses on Schaefer parcellation with', nParcels, 'parcels')
    if atlas=='gordon':
        outfilename = outfilename + '_' + atlas
        df_networkdef = pd.read_excel('/home/ti236/AnalysisTools/GordonParcels/Parcels.xlsx')
        networkdef = df_networkdef.Community.values
        nParcels = 333
        print('Running analyses on Gordon parcellation with', nParcels, 'parcels')

    #### Set subject parameters
    triu_ind = np.triu_indices(nParcels,k=1)
    n_tasks = 45 #+ 4 # 4 motor mappings

    #### Load in decoding data
    print('LOADING DECODING DATA')
    outfile = outdir + outfilename + '_decoding_data_allsubjs' + groupavg_str + '.h5'
    if os.path.exists(outfile):
        print('\tData exists... skipping')
        h5f = h5py.File(outfile,'r')
        data_task_1 = h5f['data1'][:].copy()
        data_task_2 = h5f['data2'][:].copy()
        labels = h5f['labels'][:].copy()
        h5f.close()
    else:
        data_task_1 = []
        data_task_2 = []
        if not groupavg:
            labels = []
            for sub in subIDs:
                tmp1, tmp2 = tools.loadAveragedTaskActivations(sub, space='vertex',glm='betaseries',unique_tasks=unique_tasks)
                data_task_1.append(tmp1.T)
                data_task_2.append(tmp2.T)
                labels.append(np.arange(n_tasks))
            data_task_1 = np.asarray(data_task_1)
            data_task_2 = np.asarray(data_task_2)
        else:
            labels = np.arange(n_tasks)
            for sub in subIDs:
                tmp1, tmp2 = tools.loadAveragedTaskActivations(sub, space='vertex',glm='betaseries',unique_tasks=unique_tasks)
                data_task_1.append(tmp1.T)
                data_task_2.append(tmp2.T)
            data_task_1 = np.mean(np.asarray(data_task_1),axis=0)
            data_task_2 = np.mean(np.asarray(data_task_2),axis=0)
        #
        h5f = h5py.File(outfile,'a')
        try:
            h5f.create_dataset('data1',data=data_task_1)
            h5f.create_dataset('data2',data=data_task_2)
            h5f.create_dataset('labels',data=labels)
        except:
            del h5f['data2'], h5f['data1'], h5f['labels']
            h5f.create_dataset('data1',data=data_task_1)
            h5f.create_dataset('data2',data=data_task_2)
            h5f.create_dataset('labels',data=labels)
        h5f.close()
    
    #### Load in data
    print('RUNNING DECODING ANALYSIS ON', atlas, 'ATLAS')
    outfile = outdir + outfilename + '_decoding_' + decoder + '_allsubjs' + groupavg_str + '.h5'
    if os.path.exists(outfile):
        print('\tData exists... skipping')
        h5f = h5py.File(outfile,'r')
        parcel_classifications = h5f['data'][:].copy()
        parcel_confusionmats = h5f['confusion_mats'][:].copy()
        h5f.close()
    else:
        parcel_classifications = []
        parcel_confusionmats = []
        if not groupavg:
            for s in range(len(subIDs)):
                print('Running classifications on subject', s, '/', len(subIDs))
                tmp_classifications, tmp_confusionmats = tools.computeSplitHalfDecoding(data_task_1[s,:,:],data_task_2[s,:,:],labels[s,:],classifier=decoder,confusion=True,permutation=None,parcellation=atlas)
                parcel_classifications.append(tmp_classifications)
                parcel_confusionmats.append(tmp_confusionmats)
        else:
            print('Running classifications on group averaged task activations') 
            parcel_classifications, parcel_confusionmats = tools.computeSplitHalfDecoding(data_task_1[:,:],data_task_2[:,:],labels,classifier=decoder,confusion=True,permutation=None,parcellation=atlas)
        # Save out to h5f
        h5f = h5py.File(outfile,'a')
        try:
            h5f.create_dataset('data',data=parcel_classifications)
            h5f.create_dataset('confusion_mats',data=parcel_confusionmats)
        except:
            del h5f['data']
            h5f.create_dataset('data',data=parcel_classifications)
            h5f.create_dataset('confusion_mats',data=parcel_confusionmats)
        h5f.close()

    #### Compute region-to-region representational distances
    print("COMPUTING REPRESENTATIONAL DISTANCE OF EACH REGION'S RDM USING CONFUSION MATRIX")
    if os.path.exists(outdir + outfilename + '_' + distance + '_interregion_representational_distances_' + decoder + '_confusionmat.csv'):
        print('\tData exists... skipping')
        rep_dist_mat = np.loadtxt(outdir + outfilename + '_' + distance + '_interregion_representational_distances_' + decoder + '_confusionmat.csv')
        rep_dist_mat_unthresh = np.loadtxt(outdir + outfilename + '_' + distance + '_interregion_representational_distances_' + decoder + '_confusionmat_unthresholded.csv')
    else:
        rep_dist_mat = np.zeros((nParcels,nParcels))
        avg_rsms = np.mean(parcel_confusionmats,axis=0)
        #rsm_triu_ind = np.triu_indices(n_tasks,k=0)
        for i in range(nParcels): 
            for j in range(nParcels):
                if i>=j: continue
                if distance=='pearson':
                    r, p = stats.pearsonr(avg_rsms[i,:,:].reshape(-1),avg_rsms[j,:,:].reshape(-1))
                    if i!=j: r = np.arctanh(r)
                if distance=='spearman':
                    r, p = stats.spearmanr(avg_rsms[i,:,:].reshape(-1),avg_rsms[j,:,:].reshape(-1))
                    if i!=j: r = np.arctanh(r)
                if distance=='covariance':
                    r = np.mean(np.multiply(avg_rsms[i,:,:].reshape(-1), avg_rsms[j,:,:].reshape(-1)))
                if distance=='cosine':
                    r = np.dot(avg_rsms[i,:,:].reshape(-1), avg_rsms[j,:,:].reshape(-1))/(np.linalg.norm(avg_rsms[i,:,:].reshape(-1))*np.linalg.norm(avg_rsms[j,:,:].reshape(-1)))
                rep_dist_mat[i,j] = r

        # Fill out bottom triangle of matrix
        rep_dist_mat = rep_dist_mat + rep_dist_mat.T
        rep_dist_mat_unthresh = rep_dist_mat.copy()
        # Apply same threshold as rest fc matrix
        rep_dist_mat = bct.threshold_proportional(rep_dist_mat,0.2)
        if distance=='covariance':
            for i in range(nParcels): rep_dist_mat[i,i] = np.mean(np.multiply(avg_rsms[i,:,:][rsm_triu_ind], avg_rsms[i,:,:][rsm_triu_ind]))
        else:
            np.fill_diagonal(rep_dist_mat,0)
            #rep_dist_mat = np.arctanh(rep_dist_mat)
        np.savetxt(outdir + outfilename + '_' + distance + '_interregion_representational_distances_' + decoder + '_confusionmat.csv',rep_dist_mat)
        np.savetxt(outdir + outfilename + '_' + distance + '_interregion_representational_distances_' + decoder + '_confusionmat_unthresholded.csv',rep_dist_mat_unthresh)

    ####
    print("COMPUTE THE GRADIENTS OF REPRESENTATIONAL DISTANCE MATRIX") 
    if os.path.exists(outdir + outfilename + '_' + distance + '_' + str(ncomponents) + 'components_RSMs_representational_gradients' + '_' + decoder + '.csv'):
        print('\tData exists... skipping')
        gradients = np.loadtxt(outdir + outfilename + '_' + distance + '_' + str(ncomponents) + 'components_RSMs_representational_gradients' + '_' + decoder + '.csv')
        eigenvalues = np.loadtxt(outdir + outfilename + '_' + distance + '_' + str(ncomponents) + 'eigenvalues_RSMs_representational_gradients' + '_' + decoder + '.csv')
    else:
        pca = PCA(n_components=ncomponents,whiten=False)
        gradients = pca.fit_transform(rep_dist_mat)
        eigenvalues = pca.explained_variance_
        #gradients = pca.fit_transform(rep_dist_mat_unthresh)
        np.savetxt(outdir + outfilename + '_' + distance + '_' + str(ncomponents) + 'components_RSMs_representational_gradients' + '_' + decoder + '.csv',gradients)
        np.savetxt(outdir + outfilename + '_' + distance + '_' + str(ncomponents) + 'eigenvalues_RSMs_representational_gradients' + '_' + decoder + '.csv',eigenvalues)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
