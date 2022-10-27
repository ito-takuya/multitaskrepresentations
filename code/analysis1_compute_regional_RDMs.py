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
import cka

basedir = '/home/ti236/taku/multiTaskHierarchy/'
outdir = basedir + 'derivatives/results/analysis1/'
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
parser.add_argument('--outfilename', type=str, default="analysis1", help='Prefix output filenames (Default: analysis1')
parser.add_argument('--dimensionality', type=str, default="cosine", help='distance metric for computing representational distance (between conditions to create RSMs)')
parser.add_argument('--atlas', type=str, default="glasser", help='parcellation atlas (Default: glasser)')
parser.add_argument('--distance', type=str, default="cosine", help='distance metric for computing representational distance (between RSMs)')
parser.add_argument('--ncomponents', type=int, default=3, help='number of principal components to compute (default: 5)')
parser.add_argument('--motor_mapping', action='store_true', help="Include motor output activations")
parser.add_argument('--correct_geomean', action='store_true', help="Correct RSMs by the geometric mean of the diagonal terms")
parser.add_argument('--randomtasks', type=int, default=45, help='number of random tasks to sample from for analyses (default: 45, which is all)')
parser.add_argument('--randomtaskID', type=int, default=1, help='ID number of random task iteration (i.e., a unique ID identifier) (default: 1)')
parser.add_argument('--groupavg', action='store_true', help='Compute RSMs from group-averaged activations')
parser.add_argument('--zscore', action='store_true', help="Zscore brain maps")

def run(args):
    args 
    outfilename = args.outfilename
    distance = args.distance
    dimensionality = args.dimensionality
    ncomponents = args.ncomponents
    motor_mapping = args.motor_mapping
    correct_geomean = args.correct_geomean
    randomtasks = args.randomtasks
    randomtaskID = args.randomtaskID
    groupavg = args.groupavg
    atlas = args.atlas
    zscore = args.zscore
    if motor_mapping:
        outfilename = outfilename + '_withMotor'
        motormapping = motor_mappings
    else:
        motormapping = None
    if zscore:
        outfilename = outfilename + '_zscore'
    if groupavg:
        groupavg_str = '_groupavg'
    else:
        groupavg_str = ''
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
    if motor_mapping:
        n_tasks = 49
    else:
        n_tasks = 45 #+ 4 # 4 motor mappings

    #### Load in data
    print('LOADING REST DATA AND COMPUTING GROUP FC')
    if os.path.exists(outdir + outfilename + '_groupfc.csv'):
        print('\tData exists... skipping')
        groupfc = np.loadtxt(outdir + outfilename + '_groupfc.csv')
    else:
        fc = []
        for subj in subIDs:
            data = tools.loadrsfMRI(subj,space='parcellated',atlas=atlas)
            # If this subject has no data, skip
            if data.shape[0]==0: continue
            tmp = np.corrcoef(data)
            np.fill_diagonal(tmp,0)
            fc.append(np.arctanh(tmp))
            fc.append(tmp)
        fc = np.asarray(fc)
        groupfc = np.nanmean(fc,axis=0)
        np.savetxt(outdir + outfilename + '_groupfc_unthresh.csv',groupfc)
        groupfc = bct.threshold_proportional(groupfc,.2)
        np.savetxt(outdir + outfilename + '_groupfc.csv',groupfc)

    ####
    print("COMPUTE REST FC GRADIENTS") 
    if os.path.exists(outdir + outfilename + '_restFC_gradients.csv'):
        print('\tData exists... skipping')
        gradients = np.loadtxt(outdir + outfilename + '_restFC_gradients.csv')
    else:
        pca = PCA(n_components=ncomponents,whiten=False)
        gradients = pca.fit_transform(groupfc)
        np.savetxt(outdir + outfilename + '_restFC_gradients.csv',gradients)

    #### Load in data
    print('LOADING DATA AND COMPUTING REGIONAL RDMS')
    if os.path.exists(outdir + outfilename + '_regional_rdms_' + dimensionality + groupavg_str + '_allsubjs.h5'):
        print('\tData exists... skipping')
        h5f = h5py.File(outdir + outfilename + '_regional_rdms_' + dimensionality + groupavg_str + '_allsubjs.h5','r')
        rsms_parcels_allsubjs = h5f['data'][:].copy()
        h5f.close()
    else:
        if not groupavg:
            rsms_parcels_allsubjs = np.zeros((nParcels,len(subIDs),n_tasks,n_tasks))
            scount = 0
            for sub in subIDs:
                rsms, task_index = tools.computeSubjRSMCV(sub,space='vertex',glm='betaseries',measure=dimensionality,unique_tasks=unique_tasks,motor_mappings=motormapping, zscore=zscore, parcellation=atlas)
                for roi in range(nParcels):
                    rsms_parcels_allsubjs[roi,scount,:,:] = rsms[roi]
                scount += 1
            # Save out to h5f
            h5f = h5py.File(outdir + outfilename + '_regional_rdms_' + dimensionality + groupavg_str + '_allsubjs.h5','a')
            try:
                h5f.create_dataset('data',data=rsms_parcels_allsubjs)
            except:
                del h5f['data']
                h5f.create_dataset('data',data=rsms_parcels_allsubjs)
            h5f.close()

            #### Normalize all RSMs by the geometric mean of the diagonal
            if correct_geomean:
                for roi in range(rsms_parcels_allsubjs.shape[0]):
                    for subj in range(rsms_parcels_allsubjs.shape[1]):
                        tmp = rsms_parcels_allsubjs[roi,subj,:,:].copy()
                        diag = np.diag(tmp,k=0)
                        print(diag)
                        geomean = stats.gmean(diag)
                        print(geomean)
                        tmp = np.divide(tmp,geomean)
                        rsms_parcels_allsubjs[roi,subj,:,:] = tmp
        else:
            rsms_parcels_allsubjs = np.zeros((nParcels,n_tasks,n_tasks))
            rsms, task_index = tools.computeGroupRSMCV(space='vertex',glm='betaseries',measure=dimensionality,unique_tasks=unique_tasks, zscore=zscore, parcellation=atlas)
            for roi in range(nParcels):
                rsms_parcels_allsubjs[roi,:,:] = rsms[roi]
            # Save out to h5f
            h5f = h5py.File(outdir + outfilename + '_regional_rdms_' + dimensionality + groupavg_str + '_allsubjs.h5','a')
            try:
                h5f.create_dataset('data',data=rsms_parcels_allsubjs)
            except:
                del h5f['data']
                h5f.create_dataset('data',data=rsms_parcels_allsubjs)
            h5f.close()

            #### Normalize all RSMs by the geometric mean of the diagonal
            if correct_geomean:
                for roi in range(rsms_parcels_allsubjs.shape[0]):
                    tmp = rsms_parcels_allsubjs[roi,:,:].copy()
                    diag = np.diag(tmp,k=0)
                    print(diag)
                    geomean = stats.gmean(diag)
                    print(geomean)
                    tmp = np.divide(tmp,geomean)
                    rsms_parcels_allsubjs[roi,:,:] = tmp


    #### Compute dimensionality for each region's RDM
    print("COMPUTING DIMENSIONALITY OF EACH REGION'S RDM")
    geomean_str= '_geomean' if correct_geomean else ''
    task_str = '_randomtasks' + str(randomtasks) + '_ID' + str(randomtaskID) if randomtasks<45 else ''
    outfile = outdir + outfilename + '_parcel_' + dimensionality + '_dimensionality' + geomean_str + task_str + groupavg_str + '.csv'
    if os.path.exists(outfile):
        print('\tData exists... skipping')
        dim_rois = pd.read_csv(outfile)
    else:
        if randomtasks<45:
            ind = np.random.choice(np.arange(45),size=randomtasks,replace=False)
        else:
            ind = np.arange(45)
        ind.shape = (len(ind),1)

        if not groupavg:
            #
            dim_rois = {}
            dim_rois['Dimensionality'] = []
            dim_rois['Parcels'] = []
            dim_rois['Subject'] = []
            dim_rois['Network'] = []
            rsm_rois = np.zeros((nParcels,len(subIDs)))
            for roi in range(nParcels):
                if atlas=='glasser':
                    netkey = networkdef[roi]
                    netname = networkkey2name[netkey]
                elif atlas=='schaefer':
                    netname = networkdef[roi]
                elif atlas=='gordon':
                    netname = networkdef[roi]
                for scount in range(len(subIDs)):
                    if dimensionality=='correlation': rsms_parcels_allsubjs[roi,scount,:,:] = np.arctanh(rsms_parcels_allsubjs[roi,scount,:,:])
                    dim = tools.getDimensionality(np.squeeze(rsms_parcels_allsubjs[roi,scount,ind,ind.T]))
                    dim_rois['Dimensionality'].append(dim)
                    dim_rois['Parcels'].append(roi+1)
                    dim_rois['Subject'].append(subIDs[scount])
                    dim_rois['Network'].append(netname)
            dim_rois = pd.DataFrame(dim_rois)
            dim_rois.to_csv(outfile)
        else:
            #
            dim_rois = {}
            dim_rois['Dimensionality'] = []
            dim_rois['Parcels'] = []
            dim_rois['Network'] = []
            rsm_rois = np.zeros((nParcels,))
            for roi in range(nParcels):
                if atlas=='glasser':
                    netkey = networkdef[roi]
                    netname = networkkey2name[netkey]
                elif atlas=='schaefer':
                    netname = networkdef[roi]
                elif atlas=='gordon':
                    netname = networkdef[roi]
                if dimensionality=='correlation': rsms_parcels_allsubjs[roi,:,:] = np.arctanh(rsms_parcels_allsubjs[roi,:,:])
                dim = tools.getDimensionality(np.squeeze(rsms_parcels_allsubjs[roi,ind,ind.T]))
                dim_rois['Dimensionality'].append(dim)
                dim_rois['Parcels'].append(roi+1)
                dim_rois['Network'].append(netname)
            dim_rois = pd.DataFrame(dim_rois)
            dim_rois.to_csv(outfile)

    #### Compute region-to-region representational distances
    print("COMPUTING REPRESENTATIONAL DISTANCE OF EACH REGION'S RDM")
    geomean_str= '_geomean' if correct_geomean else ''
    outfile = outdir + outfilename + '_' + distance + '_interregion_representational_distances_' + dimensionality + geomean_str + groupavg_str + 'RSMs.csv'
    outfile_unthresh = outdir + outfilename + '_' + distance + '_interregion_representational_distances_' + dimensionality + geomean_str + groupavg_str + 'RSMs_unthresholded.csv'
    if os.path.exists(outfile):
        print('\tData exists... skipping')
        rep_dist_mat = np.loadtxt(outfile)
    else:
        rep_dist_mat = np.zeros((nParcels,nParcels))
        if dimensionality in ['correlation','cosine']: rsms_parcels_allsubjs = np.arctanh(rsms_parcels_allsubjs)
        #
        if not groupavg:
            avg_rsms = np.nanmean(rsms_parcels_allsubjs,axis=1)
        else:
            avg_rsms = rsms_parcels_allsubjs.copy()
        #
        rsm_triu_ind = np.triu_indices(n_tasks,k=0)
        for i in range(nParcels): 
            for j in range(nParcels):
                if i>=j: continue
                if distance=='pearson':
                    r, p = stats.pearsonr(avg_rsms[i,:,:][rsm_triu_ind],avg_rsms[j,:,:][rsm_triu_ind])
                    if i!=j: r = np.arctanh(r)
                if distance=='spearman':
                    r, p = stats.spearmanr(avg_rsms[i,:,:][rsm_triu_ind],avg_rsms[j,:,:][rsm_triu_ind])
                    if i!=j: r = np.arctanh(r)
                if distance=='covariance':
                    r = np.mean(np.multiply(avg_rsms[i,:,:][rsm_triu_ind], avg_rsms[j,:,:][rsm_triu_ind]))
                if distance=='cosine':
                    r = np.dot(avg_rsms[i,:,:][rsm_triu_ind], avg_rsms[j,:,:][rsm_triu_ind])/(np.linalg.norm(avg_rsms[i,:,:][rsm_triu_ind])*np.linalg.norm(avg_rsms[j,:,:][rsm_triu_ind]))
                if distance=='cka':
                    L_i = avg_rsms[i,:,:].copy()
                    L_j = avg_rsms[j,:,:].copy()
                    L_i = cka.centering(L_i)
                    L_j = cka.centering(L_j)
                    hsic = np.sum(L_i * L_j)
                    var1 = np.sqrt(np.sum(L_i * L_i))
                    var2 = np.sqrt(np.sum(L_j * L_j))
                    r = hsic / (var1 * var2)

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
        np.savetxt(outfile,rep_dist_mat)
        np.savetxt(outfile_unthresh,rep_dist_mat_unthresh)

    ####
    print("COMPUTE THE GRADIENTS OF THIS REPRESENTATIONAL DISTANCE MATRIX") 
    geomean_str= '_geomean' if correct_geomean else ''
    outfile = outdir + outfilename + '_' + distance + '_' + str(ncomponents) + 'components_' + dimensionality + geomean_str + groupavg_str + 'RSMs_representational_gradients.csv'
    outfile_eigenvalues= outdir + outfilename + '_' + distance + '_' + str(ncomponents) + 'components_' + dimensionality + geomean_str + groupavg_str + 'RSMs_representational_eigenvalues.csv'
    if os.path.exists(outfile):
        print('\tData exists... skipping')
        gradients = np.loadtxt(outfile)
       # eigenvalues = np.loadtxt(outfile_eigenvalues)
    else:
        pca = PCA(n_components=ncomponents,whiten=False)
        gradients = pca.fit_transform(rep_dist_mat)
        eigenvalues = pca.explained_variance_
        np.savetxt(outfile,gradients)
        np.savetxt(outfile_eigenvalues,eigenvalues)



if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
