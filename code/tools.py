# utility and tool functions

# Taku Ito
# 2/22/21

import numpy as np
import nibabel as nib
import scipy.stats as stats
import h5py
import sklearn
import sklearn.svm as svm
import os

datadir = '/gpfs/loomis/project/n3/Studies/MurrayLab/taku/mdtb_data/'
datadir = '/gpfs/gibbs/pi/n3/Studies/MurrayLab/taku/mdtb_data/'

try:
    networkdef = np.loadtxt('cortex_parcel_network_assignments.txt')
    # need to subtract one to make it compatible for python indices
    indsort = np.loadtxt('cortex_community_order.txt',dtype=int) - 1 
    indsort.shape = (len(indsort),1)
except: 
    networkdef = np.loadtxt('/Users/tito/data/multiTaskHierarchy/cortex_parcel_network_assignments.txt')
    # need to subtract one to make it compatible for python indices
    indsort = np.loadtxt('/Users/tito/data/multiTaskHierarchy/cortex_community_order.txt',dtype=int) - 1 
    indsort.shape = (len(indsort),1)

# network mappings for final partition set
networkmappings = {'fpn':7, 'vis1':1, 'vis2':2, 'smn':3, 'aud':8, 'lan':6, 'dan':5, 'con':4, 'dmn':9, 
                   'pmulti':10, 'none1':11, 'none2':12}
networks = networkmappings.keys()

xticks = {}
reorderednetworkaffil = networkdef[indsort]
for net in networks:
    netNum = networkmappings[net]
    netind = np.where(reorderednetworkaffil==netNum)[0]
    tick = np.max(netind)
    xticks[tick] = net

## General parameters/variables

sortednets = np.sort(list(xticks.keys()))
orderednetworks = []
for net in sortednets: orderednetworks.append(xticks[net])

networkpalette = ['royalblue','slateblue','paleturquoise','darkorchid','limegreen',
                  'lightseagreen','yellow','orchid','r','peru','orange','olivedrab']
networkpalette = np.asarray(networkpalette)

OrderedNetworks = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMM','VMM','ORA']

glasserfilename = 'Q1-Q6_RelatedParcellation210.LR.CorticalAreas_dil_Colors.32k_fs_RL.dlabel.nii'
glasser = np.squeeze(nib.load(glasserfilename).get_fdata())
#schaeferfilename = '/home/ti236/AnalysisTools/Schaefer2018_400Parcels_7Networks_order.dlabel.nii'
#schaefer = np.squeeze(nib.load(schaeferfilename).get_fdata())
#gordonfilename = '/home/ti236/AnalysisTools/GordonParcels/Parcels_LR.dlabel.nii'
#gordon = np.squeeze(nib.load(gordonfilename).get_fdata())

subIDs=['02','03','04','06','08','09','10','12','14','15','18','20','22','25','27','29','31','17','19','21','24','26','28','30']


def loadTaskActivations(sess, run, space='vertex', model='canonical'):
    """
    Load task activation maps (canonical HRF)

    return: 
    data        :       Activation vector
    task_index  :       Index array with labels of all task conditions
    """

    taskdatadir = datadir  + 'derivatives/postprocessing/'
    filename = taskdatadir + sess + '_tfMRI_' + space + '_' + model + '_qunex_bold' + str(run)
    h5f = h5py.File(filename + '.h5','r')
    data = h5f['betas'][:].copy()
    #task_index = np.loadtxt(filename + '_taskIndex.csv')

    task_index = []
    # open file and read the content in a list
    with open(filename + '_taskIndex.csv', 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            # add item to the list
            task_index.append(currentPlace)
            
    return data, task_index

def loadAveragedTaskActivations(sub, space='vertex',glm='betaseries',unique_tasks=None):
    """
    Computes the averaged activations (averaged across betas within task conditions for each subject
    Returns two averaged activations, activations for session 1 and 2
    """
    runs = range(1,9) # These are the run numbers for task data
    sess1 = ['a1', 'b1']
    sess2 = ['a2', 'b2']
    rsm_1 = []
    rsm_2 = []

    # Load in data for session a
    data_1 = []
    task_index_1 = []
    for sess in sess1:
        sess_id = sub + '_' + sess
        for run in runs:
            tmpdat, tmpind = loadTaskActivations(sess_id,run,space=space,model=glm)
            data_1.extend(tmpdat.T)
            task_index_1.extend(tmpind)
    data_1 = np.asarray(data_1)
    task_index_1 = np.asarray(task_index_1)

    # Load in data for session b
    data_2 = []
    task_index_2 = []
    for sess in sess2:
        sess_id = sub + '_' + sess
        for run in runs:
            tmpdat, tmpind = loadTaskActivations(sess_id,run,space=space)
            data_2.extend(tmpdat.T)
            task_index_2.extend(tmpind)
    data_2 = np.asarray(data_2)
    task_index_2 = np.asarray(task_index_2)
    
    if unique_tasks is None:
        unique_tasks = np.unique(task_index_1)
    # Ensure a and b have the same number of unique tasks
    if len(np.unique(task_index_1)) != len(np.unique(task_index_2)):
        raise Exception("Wait! Sessions 1 and 2 don't have the same number of tasks... Cannot generate cross-validated RSMs")

    
    # Now compute average activations for sessions 1 and 2 (separately)
    data_task_1 = []
    data_task_2 = []
    for task in unique_tasks:
        task_ind = np.where(task_index_1==task)[0]
        #data_task_1.append(stats.ttest_1samp(data_1[task_ind,:],0,axis=0)[0])
        data_task_1.append(np.mean(data_1[task_ind,:],axis=0))

        task_ind = np.where(task_index_2==task)[0]
        #data_task_2.append(stats.ttest_1samp(data_2[task_ind,:],0,axis=0)[0])
        data_task_2.append(np.mean(data_2[task_ind,:],axis=0))

    data_task_1 = np.asarray(data_task_1).T
    data_task_2 = np.asarray(data_task_2).T
    return data_task_1, data_task_2

def loadrsfMRI(subj,space='parcellated',atlas='glasser'):
    """
    Load in resting-state residuals
    """
    runs = ['bold9','bold10']
    if atlas=='glasser':
        atlas_str = ''
    elif atlas=='schaefer':
        atlas_str = '_' + atlas 
    elif atlas=='gordon':
        atlas_str = '_' + atlas 

    data = []
    for run in runs:
        try:
            h5f = h5py.File(datadir + 'derivatives/postprocessing/' +  subj + '_b2_rsfMRI' + atlas_str + '_' + space + '_qunex_' + run + '.h5','r')
            ts = h5f['residuals'][:].T
            data.extend(ts)
            h5f.close()
        except:
            print('Subject', subj, '| run', run, ' does not exist... skipping')

    try:
        data = np.asarray(data).T
    except:
        print('\tError')

    return data

def computeSubjRSM(sub,space='vertex',glm='canonical', wholebrain=False,unique_tasks=None,parcellation='glasser'):
    """
    Computes a cross-validated RSM - cross-validated on A sessions versus B sessions (diagonals are therefore not ~ 1)
    Returns: a cross-validated, subject-specific RSM with corresponding index (ordering)
    """
    if parcellation=='glasser':
        nParcels = 360
        atlas=glasser.copy()
    if parcellation=='schaefer':
        nParcels = int(np.max(schaefer))
        atlas=schaefer.copy()
    if parcellation=='gordon':
        nParcels = 333
        atlas=gordon.copy()

    runs = range(1,9) # These are the run numbers for task data
    sess1 = ['a1', 'b1']
    sess2 = ['a2', 'b2']
    rsm_1 = []
    rsm_2 = []

    # Load in data for session a
    data_1 = []
    task_index_1 = []
    for sess in sess1:
        sess_id = sub + '_' + sess
        for run in runs:
            tmpdat, tmpind = loadTaskActivations(sess_id,run,space=space,model=glm)
            data_1.extend(tmpdat.T)
            task_index_1.extend(tmpind)
    data_1 = np.asarray(data_1)
    task_index_1 = np.asarray(task_index_1)

    # Load in data for session b
    data_2 = []
    task_index_2 = []
    for sess in sess2:
        sess_id = sub + '_' + sess
        for run in runs:
            tmpdat, tmpind = loadTaskActivations(sess_id,run,space=space)
            data_2.extend(tmpdat.T)
            task_index_2.extend(tmpind)
    data_2 = np.asarray(data_2)
    task_index_2 = np.asarray(task_index_2)
    
    if unique_tasks is None:
        unique_tasks = np.unique(task_index_1)
    # Ensure a and b have the same number of unique tasks
    if len(np.unique(task_index_1)) != len(np.unique(task_index_2)):
        raise Exception("Wait! Sessions 1 and 2 don't have the same number of tasks... Cannot generate cross-validated RSMs")
    n_tasks = len(unique_tasks)

    data_task_1 = []
    data_task_2 = []
    for task in unique_tasks:
        task_ind = np.where(task_index_1==task)[0]
        #data_task_1.append(stats.ttest_1samp(data_1[task_ind,:],0,axis=0)[0])
        data_task_1.append(np.mean(data_1[task_ind,:],axis=0))

        task_ind = np.where(task_index_2==task)[0]
        #data_task_2.append(stats.ttest_1samp(data_2[task_ind,:],0,axis=0)[0])
        data_task_2.append(np.mean(data_2[task_ind,:],axis=0))


    data_task_1 = np.asarray(data_task_1).T
    data_task_2 = np.asarray(data_task_2).T

    if space=='vertex':
        # Compute whole-brain RSM
        if wholebrain:
            tmpmat = np.zeros((n_tasks,n_tasks))
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i>j: continue
                    tmpmat[i,j] = stats.pearsonr(data_task_1[:,i],data_task_2[:,j])[0]

            # Now make symmetric
            tmpmat = tmpmat + tmpmat.T
            # double counting diagonal so divide by 2
            np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
            rsms = tmpmat
        else:

            # compute RSM for each parcel
            rsms = []
            for roi in range(1,nParcels+1):
                roi_ind = np.where(atlas==roi)[0]
                roidat1 = data_task_1[roi_ind,:].T
                roidat2 = data_task_2[roi_ind,:].T
                tmpmat = np.zeros((n_tasks,n_tasks))
                for i in range(n_tasks):
                    for j in range(n_tasks):
                        if i>j: continue
                        tmpmat[i,j] = stats.pearsonr(roidat1[i,:],roidat2[j,:])[0]
                        #tmpmat[i,j] = np.mean(np.multiply(roidat1[i,:],roidat2[j,:]))

                # Now make symmetric
                tmpmat = tmpmat + tmpmat.T
                # double counting diagonal so divide by 2
                np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
                rsms.append(tmpmat)
            rsms = np.asarray(rsms)

    if space=='parcellated':
        # Compute whole-brain RSM
        if wholebrain:
            tmpmat = np.zeros((n_tasks,n_tasks))
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i>j: continue
                    tmpmat[i,j] = stats.pearsonr(data_task_1[:,i],data_task_2[:,j])[0]

            # Now make symmetric
            tmpmat = tmpmat + tmpmat.T
            # double counting diagonal so divide by 2
            np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
            rsms = tmpmat
        else:
            # compute rsm for each network
            rsms = {}
            for net in orderednetworks:
                net_ind = np.where(networkdef==networkmappings[net])[0]
                netdat1 = data_task_1[net_ind,:].T
                netdat2 = data_task_2[net_ind,:].T
                tmpmat = np.zeros((n_tasks,n_tasks))
                for i in range(n_tasks):
                    for j in range(n_tasks):
                        if i>j: continue
                        tmpmat[i,j] = stats.pearsonr(netdat1[i,:],netdat2[j,:])[0]
                        #tmpmat[i,j] = np.mean(np.multiply(netdat1[i,:],netdat2[j,:]))

                # Now make symmetric
                tmpmat = tmpmat + tmpmat.T
                # double counting diagonal so divide by 2
                np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
                rsms[net] = tmpmat

    return rsms, unique_tasks

    
def decoding(trainset,testset,trainlabels,testlabels,classifier='distance',confusion=False):
    unique_labels = np.unique(trainlabels)
    
    if classifier in ['distance','cosine']:
        #### Create prototypes from trainset
        prototypes = {}
        for label in unique_labels:
            ind = np.where(trainlabels==label)[0]
            prototypes[label] = np.mean(trainset[ind,:],axis=0)

        #### Now classifiy each sample n the testset
        predictions = []
        for i in range(testset.shape[0]):
            # Correlate sampple with each prototype
            rs = []
            for label in prototypes:
                if classifier == 'distance':
                    rs.append(stats.pearsonr(prototypes[label],testset[i,:])[0])
                if classifier == 'cosine':
                    rs.append(np.dot(prototypes[label],testset[i,:])/(np.linalg.norm(prototypes[label])*np.linalg.norm(testset[i,:])))
            
            # Find the closest prototype for sample
            max_ind = np.argmax(np.asarray(rs))
            predictions.append(unique_labels[max_ind])

        predictions = np.asarray(predictions)

    if classifier == 'logistic':

        #clf = sklearn.linear_model.LogisticRegression(solver='lbfgs',penalty='none',max_iter=1000)
        clf = sklearn.linear_model.LogisticRegression(solver='liblinear')
        clf.fit(trainset,trainlabels)
        predictions = clf.predict(testset)

    if classifier == 'ridge':

        clf = sklearn.linear_model.RidgeClassifier(solver='svd',max_iter=1000)
        clf.fit(trainset,trainlabels)
        predictions = clf.predict(testset)

    if classifier == 'svm':
        clf = svm.SVC(kernel='linear',probability=True)
        clf.fit(trainset,trainlabels)
        predictions = clf.predict(testset)

    accuracy = predictions == np.asarray(testlabels)
    confusion_mat = sklearn.metrics.confusion_matrix(testlabels, predictions, labels=unique_labels)

    #if classifier in ['svm','logistic']:
    #    decision_function = clf.predict_log_proba(testset)
    #    conditions = clf.classes_ 
    #    decision_on_test = []
    #    for i in range(len(testlabels)):
    #        ind = np.where(conditions==testlabels[i])[0]
    #        decision_on_test.append(decision_function[i,ind])
    #    return accuracy, decision_on_test 
    #elif classifier in ['ridge']:
    #    decision_function = clf.decision_function(testset)
    #    conditions = clf.classes_ 
    #    decision_on_test = []
    #    for i in range(len(testlabels)):
    #        ind = np.where(conditions==testlabels[i])[0]
    #        decision_on_test.append(decision_function[i,ind])
    #    return accuracy, decision_on_test 
    #else:
    if confusion:
        return accuracy, confusion_mat
    else:
        return accuracy

def getDimensionality(data):
    """
    data needs to be a square, symmetric matrix
    """
    corrmat = data
    eigenvalues, eigenvectors = np.linalg.eig(corrmat)
    dimensionality_nom = 0
    dimensionality_denom = 0
    for eig in eigenvalues:
        dimensionality_nom += np.real(eig)
        dimensionality_denom += np.real(eig)**2

    dimensionality = dimensionality_nom**2/dimensionality_denom

    return dimensionality

def computeSplitHalfDecoding(traindata,testdata,labels,classifier='distance',confusion=True,permutation=None,nproc=None,parcellation='glasser'):
    """
    Compute decoding on whole-brain across all conditions 
    Split half decoding, training on run 1, testing on run 2
    traindata
        samples x features (vertices)
    testdata 
        samples x features (vertices)
    """
    if parcellation=='glasser':
        nParcels = 360
        atlas=glasser.copy()
    if parcellation=='schaefer':
        nParcels = int(np.max(schaefer))
        atlas=schaefer.copy()
    if parcellation=='gordon':
        nParcels = 333
        atlas=gordon.copy()

    accuracies = []
    confusion_mats = []
    for roi in range(nParcels):
        #print('Decoding on ROI', roi+1, '| Decoder :', classifier, '| Permutation :', permutation)
        roi_ind = np.where(atlas==roi+1)[0]

        roidat1 = traindata[:,roi_ind]
        roidat2 = testdata[:,roi_ind]

        train_mean = np.mean(roidat1,axis=0)
        train_sd = np.std(roidat1,axis=0)
        train_sd.shape = (1,roidat1.shape[1])
        roidat1_zscore1to2 = np.divide(roidat1 - train_mean,train_sd)
        roidat2_zscore1to2 = np.divide(roidat2 - train_mean,train_sd)
        #
        train_mean = np.mean(roidat2,axis=0)
        train_sd = np.std(roidat2,axis=0)
        train_sd.shape = (1,roidat2.shape[1])
        roidat1_zscore2to1 = np.divide(roidat1 - train_mean,train_sd)
        roidat2_zscore2to1 = np.divide(roidat2 - train_mean,train_sd)

        if confusion:
            acc1, conf_mat1 = decoding(roidat1_zscore1to2,roidat2_zscore1to2,labels,labels,classifier=classifier,confusion=confusion)
            # split test
            acc2, conf_mat2 = decoding(roidat2_zscore2to1,roidat1_zscore2to1,labels,labels,classifier=classifier,confusion=confusion)
            accuracies.append((np.mean(acc1) + np.mean(acc2))/2)
            confusion_mats.append((conf_mat1+conf_mat2)/2.0)
        else:
            acc1 = decoding(roidat1_zscore1to2,roidat2_zscore1to2,labels,labels,classifier=classifier,confusion=confusion)
            # split test
            acc2 = decoding(roidat2_zscore2to1,roidat1_zscore2to1,labels,labels,classifier=classifier,confusion=confusion)
            accuracies.append((np.mean(acc1) + np.mean(acc2))/2)

    if confusion:
        return accuracies, confusion_mats
    else:
        return accuracies

def computeSubjRSMCV(sub, space='vertex',glm='betaseries', wholebrain=False, measure='correlation', unique_tasks=None,motor_mappings=None,zscore=False,parcellation='glasser'):
    """
    Computes a cross-validated RSM - cross-validated on session1 (a1, b1) versus session2 (a2, b2) sessions (diagonals are therefore not ~ 1)
    Includes motor output mappings per unique task (left-hand, right-hand, both, or passive responses)
    Returns: a cross-validated, subject-specific RSM with corresponding index (ordering)
    """

    data_task_1, data_task_2 = loadAveragedTaskActivations(sub, space=space,glm=glm,unique_tasks=unique_tasks)
    if parcellation=='glasser':
        nParcels = 360
        atlas=glasser.copy()
    if parcellation=='schaefer':
        nParcels = int(np.max(schaefer))
        atlas=schaefer.copy()
    if parcellation=='gordon':
        nParcels = 333
        atlas=gordon.copy()

    if zscore:
        data_task_1 = stats.zscore(data_task_1,0)
        data_task_2 = stats.zscore(data_task_2,0)

    # Include motor condition activations and append if parameter is included
    if motor_mappings is not None: 
        motor_conditions = []
        for task in unique_tasks:
            motor_conditions.append(motor_mappings[task])
        motor_conditions = np.asarray(motor_conditions)
        motor_activations_1 = []
        motor_activations_2 = []
        for motor in np.unique(motor_conditions):
            motor_ind = np.where(motor_conditions==motor)[0]
            motor_activations_1.append(np.mean(data_task_1[:,motor_ind],axis=1))
            motor_activations_2.append(np.mean(data_task_2[:,motor_ind],axis=1))
        motor_activations_1 = np.asarray(motor_activations_1)
        motor_activations_2 = np.asarray(motor_activations_2)

        # Now append/stack new motor activations onto existing task activations
        data_task_1 = np.hstack((data_task_1,motor_activations_1.T))
        data_task_2 = np.hstack((data_task_2,motor_activations_2.T))
        unique_tasks = np.hstack((unique_tasks,np.unique(motor_conditions)))

    n_tasks = len(unique_tasks)

    if space=='vertex':
        # Compute whole-brain RSM
        if wholebrain:
            tmpmat = np.zeros((n_tasks,n_tasks))
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i>j: continue
                    if measure=='correlation':
                        tmpmat[i,j] = stats.pearsonr(data_task_1[:,i],data_task_2[:,j])[0]
                    elif measure=='cosine':
                        tmpmat[i,j] = np.dot(data_task_1[:,i],data_task_2[:,j])/(np.linalg.norm(data_task_1[:,i])*np.linalg.norm(data_task_2[:,j]))
                    elif measure=='covariance':
                        tmpmat[i,j] = np.sum(np.multiply(data_task_1[:,i],data_task_2[:,j]))

            # Now make symmetric
            tmpmat = tmpmat + tmpmat.T
            # double counting diagonal so divide by 2
            np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
            rsms = tmpmat
        else:

            # compute RSM for each parcel
            rsms = []
            for roi in range(1,nParcels+1):
                roi_ind = np.where(atlas==roi)[0]
                roidat1 = data_task_1[roi_ind,:].T
                roidat2 = data_task_2[roi_ind,:].T
                tmpmat = np.zeros((n_tasks,n_tasks))
                if measure != 'inner':
                    for i in range(n_tasks):
                        for j in range(n_tasks):
                            if i>j: continue
                            if measure=='correlation':
                                tmpmat[i,j] = stats.pearsonr(roidat1[i,:],roidat2[j,:])[0]
                            elif measure=='cosine':
                                tmpmat[i,j] = np.dot(roidat1[i,:],roidat2[j,:])/(np.linalg.norm(roidat1[i,:])*np.linalg.norm(roidat2[j,:]))
                            elif measure=='covariance':
                                tmpmat[i,j] = np.mean(np.multiply(roidat1[i,:],roidat2[j,:]))
                            elif measure=='euclidean':
                                tmpmat[i,j] = np.linalg.norm(roidat1[i,:] - roidat2[j,:])
                    # Now make symmetric
                    tmpmat = tmpmat + tmpmat.T
                    # double counting diagonal so divide by 2
                    np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
                else:
                    # If inner product or 2nd moment correlation matrix
                    # Note the matrices are transposed so this looks like outer product instead, but it's actually inner product
                    tmpmat = np.matmul(roidat1,roidat2.T)/n_tasks # 2nd moment matrix

                rsms.append(tmpmat)
            rsms = np.asarray(rsms)

    if space=='parcellated':
        # Compute whole-brain RSM
        if wholebrain:
            tmpmat = np.zeros((n_tasks,n_tasks))
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i>j: continue
                    if measure=='correlation':
                        tmpmat[i,j] = stats.pearsonr(data_task_1[:,i],data_task_2[:,j])[0]
                    elif measure=='cosine':
                        tmpmat[i,j] = np.dot(data_task_1[:,i],data_task_2[:,j])/(np.linalg.norm(data_task_1[:,i])*np.linalg.norm(data_task_2[:,j]))
                    elif measure=='covariance':
                        tmpmat[i,j] = np.sum(np.multiply(data_task_1[:,i],data_task_2[:,j]))

            # Now make symmetric
            tmpmat = tmpmat + tmpmat.T
            # double counting diagonal so divide by 2
            np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
            rsms = tmpmat
        else:
            # compute rsm for each network
            rsms = {}
            for net in orderednetworks:
                net_ind = np.where(networkdef==networkmappings[net])[0]
                netdat1 = data_task_1[net_ind,:].T
                netdat2 = data_task_2[net_ind,:].T
                tmpmat = np.zeros((n_tasks,n_tasks))
                for i in range(n_tasks):
                    for j in range(n_tasks):
                        if i>j: continue
                        if measure=='correlation':
                            tmpmat[i,j] = stats.pearsonr(netdat1[i,:],netdat2[j,:])[0]
                        elif measure=='cosine':
                            tmpmat[i,j] = np.dot(netdat1[i,:],netdat2[j,:])/(np.linalg.norm(netdat1[i,:])*np.linalg.norm(netdat2[j,:]))
                        elif measure=='covariance':
                            tmpmat[i,j] = np.sum(np.multiply(netdat1[i,:],netdat2[j,:]))
                        #tmpmat[i,j] = np.mean(np.multiply(netdat1[i,:],netdat2[j,:]))

                # Now make symmetric
                tmpmat = tmpmat + tmpmat.T
                # double counting diagonal so divide by 2
                np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
                rsms[net] = tmpmat

    return rsms, unique_tasks

def computeGroupRSMCV(space='vertex',glm='betaseries', wholebrain=False, measure='correlation', unique_tasks=None, parcellation='glasser',zscore=False, motor_mappings=None):
    """
    Computes a cross-validated RSM - cross-validated on session1 (a1, b1) versus session2 (a2, b2) sessions (diagonals are therefore not ~ 1)
    Returns: a cross-validated, subject-specific RSM with corresponding index (ordering)
    """
    if parcellation=='glasser':
        nParcels = 360
        atlas=glasser.copy()
    if parcellation=='schaefer':
        nParcels = int(np.max(schaefer))
        atlas=schaefer.copy()
    if parcellation=='gordon':
        nParcels = 333
        atlas=gordon.copy()

    subIDs=['02','03','04','06','08','09','10','12','14','15','18','20','22','25','27','29','31','17','19','21','24','26','28','30']

    data_task_1 = []
    data_task_2 = []
    for sub in subIDs:
        tmp1, tmp2 = loadAveragedTaskActivations(sub, space=space,glm=glm,unique_tasks=unique_tasks)
        data_task_1.append(tmp1)
        data_task_2.append(tmp2)

    # Average across subjects
    data_task_1 = np.mean(data_task_1,axis=0)
    data_task_2 = np.mean(data_task_2,axis=0)

    if zscore:
        data_task_1 = stats.zscore(data_task_1,0)
        data_task_2 = stats.zscore(data_task_2,0)


    # Include motor condition activations and append if parameter is included
    if motor_mappings is not None: 
        motor_conditions = []
        for task in unique_tasks:
            motor_conditions.append(motor_mappings[task])
        motor_conditions = np.asarray(motor_conditions)
        motor_activations_1 = []
        motor_activations_2 = []
        for motor in np.unique(motor_conditions):
            motor_ind = np.where(motor_conditions==motor)[0]
            motor_activations_1.append(np.mean(data_task_1[:,motor_ind],axis=1))
            motor_activations_2.append(np.mean(data_task_2[:,motor_ind],axis=1))
        motor_activations_1 = np.asarray(motor_activations_1)
        motor_activations_2 = np.asarray(motor_activations_2)

        # Now append/stack new motor activations onto existing task activations
        data_task_1 = np.hstack((data_task_1,motor_activations_1.T))
        data_task_2 = np.hstack((data_task_2,motor_activations_2.T))
        unique_tasks = np.hstack((unique_tasks,np.unique(motor_conditions)))

    n_tasks = len(unique_tasks)

    if space=='vertex':
        # Compute whole-brain RSM
        if wholebrain:
            tmpmat = np.zeros((n_tasks,n_tasks))
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i>j: continue
                    if measure=='correlation':
                        tmpmat[i,j] = stats.pearsonr(data_task_1[:,i],data_task_2[:,j])[0]
                    elif measure=='cosine':
                        tmpmat[i,j] = np.dot(data_task_1[:,i],data_task_2[:,j])/(np.linalg.norm(data_task_1[:,i])*np.linalg.norm(data_task_2[:,j]))
                    elif measure=='covariance':
                        tmpmat[i,j] = np.sum(np.multiply(data_task_1[:,i],data_task_2[:,j]))

            # Now make symmetric
            tmpmat = tmpmat + tmpmat.T
            # double counting diagonal so divide by 2
            np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
            rsms = tmpmat
        else:

            # compute RSM for each parcel
            rsms = []
            for roi in range(1,nParcels+1):
                roi_ind = np.where(atlas==roi)[0]
                roidat1 = data_task_1[roi_ind,:].T
                roidat2 = data_task_2[roi_ind,:].T
                tmpmat = np.zeros((n_tasks,n_tasks))
                for i in range(n_tasks):
                    for j in range(n_tasks):
                        if i>j: continue
                        if measure=='correlation':
                            tmpmat[i,j] = stats.pearsonr(roidat1[i,:],roidat2[j,:])[0]
                        elif measure=='cosine':
                            tmpmat[i,j] = np.dot(roidat1[i,:],roidat2[j,:])/(np.linalg.norm(roidat1[i,:])*np.linalg.norm(roidat2[j,:]))
                        elif measure=='covariance':
                            tmpmat[i,j] = np.mean(np.multiply(roidat1[i,:],roidat2[j,:]))
                        elif measure=='euclidean':
                            tmpmat[i,j] = np.linalg.norm(roidat1[i,:] - roidat2[j,:])

                # Now make symmetric
                tmpmat = tmpmat + tmpmat.T
                # double counting diagonal so divide by 2
                np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
                rsms.append(tmpmat)
            rsms = np.asarray(rsms)

    if space=='parcellated':
        # Compute whole-brain RSM
        if wholebrain:
            tmpmat = np.zeros((n_tasks,n_tasks))
            for i in range(n_tasks):
                for j in range(n_tasks):
                    if i>j: continue
                    if measure=='correlation':
                        tmpmat[i,j] = stats.pearsonr(data_task_1[:,i],data_task_2[:,j])[0]
                    elif measure=='cosine':
                        tmpmat[i,j] = np.dot(data_task_1[:,i],data_task_2[:,j])/(np.linalg.norm(data_task_1[:,i])*np.linalg.norm(data_task_2[:,j]))
                    elif measure=='covariance':
                        tmpmat[i,j] = np.sum(np.multiply(data_task_1[:,i],data_task_2[:,j]))

            # Now make symmetric
            tmpmat = tmpmat + tmpmat.T
            # double counting diagonal so divide by 2
            np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
            rsms = tmpmat
        else:
            # compute rsm for each network
            rsms = {}
            for net in orderednetworks:
                net_ind = np.where(networkdef==networkmappings[net])[0]
                netdat1 = data_task_1[net_ind,:].T
                netdat2 = data_task_2[net_ind,:].T
                tmpmat = np.zeros((n_tasks,n_tasks))
                for i in range(n_tasks):
                    for j in range(n_tasks):
                        if i>j: continue
                        if measure=='correlation':
                            tmpmat[i,j] = stats.pearsonr(netdat1[i,:],netdat2[j,:])[0]
                        elif measure=='cosine':
                            tmpmat[i,j] = np.dot(netdat1[i,:],netdat2[j,:])/(np.linalg.norm(netdat1[i,:])*np.linalg.norm(netdat2[j,:]))
                        elif measure=='covariance':
                            tmpmat[i,j] = np.sum(np.multiply(netdat1[i,:],netdat2[j,:]))
                        #tmpmat[i,j] = np.mean(np.multiply(netdat1[i,:],netdat2[j,:]))

                # Now make symmetric
                tmpmat = tmpmat + tmpmat.T
                # double counting diagonal so divide by 2
                np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
                rsms[net] = tmpmat

    return rsms, unique_tasks


def mapToSurface(array,filename,space='parcellated',atlas='glasser'):
    """
    array : 1d or 2d array (vertex x cols)
    filename : outputfilename 
    space : is input array parcellated or vertex-wise?
    atlas: glasser atlas? or schaefer?
    """
    ## Create size of array
    ncols = len(array.shape)
    if atlas=='glasser':
        atlas_arr = glasser.copy()
        atlas_filename = glasserfilename
        nParcels = 360
        if ncols==1:
            outarray = np.zeros((len(atlas_arr),1))
        else:
            outarray = np.zeros((len(atlas_arr),array.shape[1]))

    if atlas=='schaefer':
        atlas_arr = schaefer.copy()
        atlas_filename = schaeferfilename
        nParcels = 400
        if ncols==1:
            outarray = np.zeros((len(atlas_arr),1))
        else:
            outarray = np.zeros((len(atlas_arr),array.shape[1]))

    if atlas=='gordon':
        atlas_arr = gordon.copy()
        atlas_filename = gordonfilename
        nParcels = 333
        if ncols==1:
            outarray = np.zeros((len(atlas_arr),1))
        else:
            outarray = np.zeros((len(atlas_arr),array.shape[1]))

    if space=='parcellated':
        for roi in range(nParcels):
            for col in range(ncols):
                vertex_ind = np.where(atlas_arr==roi+1)[0]
                if ncols==1:
                    outarray[vertex_ind,col] = array[roi]
                else:
                    outarray[vertex_ind,col] = array[roi,col]
    else:
        outarray = array

    ####
    # Write file to csv and run wb_command
    np.savetxt(filename + '.csv', outarray,fmt='%s')
    wb_file = filename + '.dscalar.nii'
    wb_command = 'wb_command -cifti-convert -from-text ' + filename + '.csv ' + atlas_filename + ' ' + wb_file + ' -reset-scalars'
    os.system(wb_command)
    os.remove(filename + '.csv')
