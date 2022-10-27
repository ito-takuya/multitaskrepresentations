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
outdir = basedir + 'derivatives/results/restfc/'
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
parser.add_argument('--outfilename', type=str, default="restfc", help='Prefix output filenames (Default: analysis1')

def run(args):
    args 
    outfilename = args.outfilename

    #### Load in data
    print('LOADING REST DATA AND COMPUTING GROUP FC')
    if os.path.exists(outdir + outfilename + '_groupfc.csv'):
        print('\tData exists... skipping')
        groupfc = np.loadtxt(outdir + outfilename + '_groupfc.csv')
    else:
        groupfc = []
        for subj in subIDs:
            data = tools.loadrsfMRI(subj,space='parcellated')
            # If this subject has no data, skip
            if data.shape[0]==0: continue
            tmp = np.corrcoef(data)
            np.fill_diagonal(tmp,0)
            fc = np.arctanh(tmp)
            groupfc.append(fc)
            np.savetxt(outdir + outfilename + '_' + str(subj) + '.csv', fc)
        groupfc = np.asarray(groupfc)
        groupfc = np.mean(groupfc,axis=0)
        np.savetxt(outdir + outfilename + '_groupfc.csv', groupfc)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
