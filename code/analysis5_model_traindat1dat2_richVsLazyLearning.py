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
import torch

basedir = '/home/ti236/taku/multiTaskHierarchy/'
outdir = basedir + 'derivatives/results/analysis5/'
#outdir = '/home/ti236/multiTaskHierarchy/derivatives/results/analysis5_tiedweights/'
#outdir = basedir + 'derivatives/results/analysis5_sgd_revision/'
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
parser.add_argument('--outfilename', type=str, default="analysis5", help='Prefix output filenames (Default: analysis5')
parser.add_argument('--weight_init', type=float, default=1.0, help="Weight initialization (default: 3.0 [lazy]), lazy large, rich low weight initializations")
parser.add_argument('--bias_init', type=float, default=1.0, help="SD of bias terms (default: 1.0)")
parser.add_argument('--weight_dist', type=str, default="normal", help='distribution for weight initialization [default: normal]')
parser.add_argument('--nhidden', type=int, default=500, help='number of units in each hidden layer (default: 10)')
parser.add_argument('--nlayers', type=int, default=10, help='number of hidden layers (default: 10)')
parser.add_argument('--relu', action='store_true', help="Include ReLU")
parser.add_argument('--untied', action='store_true', help="Include ReLU")
parser.add_argument('--optim', type=str, default="adam", help='learning optimizer [default: adam]')
parser.add_argument('--train_noise', type=float, default=None, help="Include noise in training data (int/float to determine SD of noise)")
parser.add_argument('--test_noise', type=float, default=None, help="Include noise in output/evaluation data (int/float to determine SD of noise)")
parser.add_argument('--dropout', action='store_true', help="Include dropout when training model")
parser.add_argument('--normalize', action='store_true', help="normalize features when training")
parser.add_argument('--lossfunc', type=str, default="mse", help='loss function')
parser.add_argument('--fixed', type=int, default=None, help='fixed number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate, default is ADAM with LR 0.0001")
parser.add_argument('--batch', type=int, default=None, help='number of samples in each training batch')
parser.add_argument('--cuda', action='store_true', help="Train model on GPU")

def run(args):
    args 
    outfilename = args.outfilename
    weight_init = args.weight_init
    bias_init = args.bias_init
    weight_dist = args.weight_dist
    nhidden = args.nhidden
    nlayers = args.nlayers
    relu = args.relu
    untied = args.untied
    tied = False if untied else True # weird logic, but untied weights train faster
    optim = args.optim
    linear = False if relu else True # set linear to true if relu is false
    train_noise = args.train_noise
    test_noise = args.test_noise
    dropout = args.dropout
    normalize = args.normalize
    lossfunc = args.lossfunc
    fixed = args.fixed
    batchsize = args.batch
    learning_rate = args.learning_rate
    cuda = args.cuda


    #### Set subject parameters
    triu_ind = np.triu_indices(nParcels,k=1)
    n_tasks = 45 

    #### Load in data
    print('LOADING TASK ACTIVATION DATA')
    tmpfilename = outdir + 'analysis5_task_activation_data_visXmotor_regions_allsubj.h5'
    if os.path.exists(tmpfilename):
        print('\tData exists... skipping')
        h5f = h5py.File(tmpfilename,'r')
        input_activation1 = h5f['input_activation1'][:].copy()
        input_activation2 = h5f['input_activation2'][:].copy()
        output_activation1 = h5f['output_activation1'][:].copy()
        output_activation2 = h5f['output_activation2'][:].copy()
        h5f.close()
    else:
        rsms_parcels_allsubjs = np.zeros((nParcels,len(subIDs),n_tasks,n_tasks))
        data1 = []
        data2 = []
        for sub in subIDs:
            tmp1, tmp2 = tools.loadAveragedTaskActivations(sub, space='vertex',glm='betaseries',unique_tasks=unique_tasks)
            data1.append(tmp1)
            data2.append(tmp2)
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)

        print('LOADING INPUT-OUTPUT GRADIENT AND IDENTIFY INPUT-OUTPUT REGIONS')
        parcellated_gradients = np.loadtxt(basedir + 'derivatives/results/analysis1/analysis1_restFC_gradients.csv')
        io_gradient = parcellated_gradients[:,1] # 2nd margulies gradient!
        # Identify 'input region'
        in_ind = np.where(parcellated_gradients[:,1]==np.min(parcellated_gradients[:,1]))[0][0]
        parcel_in_ind = np.where(glasser==in_ind+1)[0]
        # Identify 'output region'
        out_ind = np.where(parcellated_gradients[:,1]==np.max(parcellated_gradients[:,1]))[0][0]
        parcel_out_ind = np.where(glasser==out_ind+1)[0]

        # select input-output parcel vertices
        # data 1
        input_activation1 = data1[:,parcel_in_ind,:]
        output_activation1 = data1[:,parcel_out_ind,:]
        # data 2
        input_activation2 = data2[:,parcel_in_ind,:]
        output_activation2 = data2[:,parcel_out_ind,:]

        # Save out to h5f
        h5f = h5py.File(tmpfilename,'a')
        try:
            h5f.create_dataset('input_activation1',data=input_activation1)
            h5f.create_dataset('input_activation2',data=input_activation2)
            h5f.create_dataset('output_activation1',data=output_activation1)
            h5f.create_dataset('output_activation2',data=output_activation2)
        except:
            del h5f['input_activation1'], h5f['input_activation2'], h5f['output_activation1'], h5f['output_activation2']
            h5f.create_dataset('input_activation1',data=input_activation1)
            h5f.create_dataset('input_activation2',data=input_activation2)
            h5f.create_dataset('output_activation1',data=output_activation1)
            h5f.create_dataset('output_activation2',data=output_activation2)
        h5f.close()

    ntasks = data1.shape[2]


    print('LEARN TRANSFORMATIONS FOR DATA1 AND 2')
    tmp_outfilename = outdir + outfilename + '_data1and2_' + str(nhidden) + 'hidden_' + str(nlayers) + 'layers' 
    tmp_outfilename = tmp_outfilename + '_weightinit' + str(weight_init) + '_biasinit' + str(bias_init) + '_optim' + optim
    tmp_outfilename = tmp_outfilename + '_lr' + str(learning_rate)
    if relu: tmp_outfilename = tmp_outfilename + '_ReLU'
    if train_noise: tmp_outfilename = tmp_outfilename + '_trainNoise'
    if test_noise: tmp_outfilename = tmp_outfilename + '_testNoise'
    if dropout: tmp_outfilename = tmp_outfilename + '_dropout'
    if normalize: tmp_outfilename = tmp_outfilename + '_normalize'
    if lossfunc: tmp_outfilename = tmp_outfilename + '_' + lossfunc
    if fixed!=None: tmp_outfilename = tmp_outfilename + '_fixed' + str(fixed)
    if untied: tmp_outfilename = tmp_outfilename + '_untiedweights'
    if weight_dist == 'laplace':  tmp_outfilename = tmp_outfilename + '_laplace' 
    if not os.path.exists(tmp_outfilename + '.h5') and False:
        print('\tData exists... skipping')
        h5f = h5py.File(tmp_outfilename + '.h5','r')
        outputs1, hidden_activations1 = h5f['outputs1'][:].copy(), h5f['hidden1'][:].copy()
        outputs2, hidden_activations2 = h5f['outputs2'][:].copy(), h5f['hidden2'][:].copy()
        h5f.close()
    else:
        hidden_activations1 = []
        outputs1 = []
        hidden_activations2 = []
        outputs2 = []
        edge_mean, edge_norm, edge_sd, edge_kurtosis = [], [], [], []
        bias_mean, bias_norm, bias_sd, bias_kurtosis = [], [], [], []
        edge_mean_init, edge_norm_init, edge_sd_init, edge_kurtosis_init = [], [], [], []
        bias_mean_init, bias_norm_init, bias_sd_init, bias_kurtosis_init = [], [], [], []
        scount = 0
        for sub in subIDs:

            print("LEARNING TRANSFORMATIONS FOR SUBJECT", sub)
            if normalize: 
                # zscore across brain activation maps
                input_activation1[scount,:,:] = stats.zscore(input_activation1[scount,:,:],axis=0)
                output_activation1[scount,:,:] = stats.zscore(output_activation1[scount,:,:],axis=0)
                input_activation2[scount,:,:] = stats.zscore(input_activation2[scount,:,:],axis=0)
                output_activation2[scount,:,:] = stats.zscore(output_activation2[scount,:,:],axis=0)
            input_activations = np.hstack((input_activation1[scount,:,:],input_activation2[scount,:,:]))
            output_activations = np.hstack((output_activation1[scount,:,:],output_activation2[scount,:,:]))

            # Initialize ANN
            device='cuda' if cuda else 'cpu'
            num_inputs = input_activations.shape[0]
            num_outputs = output_activations.shape[0]
            # Instantiate model
            network = ANN(num_inputs=num_inputs,
                          num_hidden=nhidden,
                          num_outputs=num_outputs,
                          num_hidden_layers=nlayers,
                          learning_rate=learning_rate,
                          weight_init=weight_init,
                          bias_init=bias_init,
                          optim=optim,
                          linear=linear,
                          lossfunc=lossfunc,
                          tied=tied,
                          weight_dist=weight_dist,
                          device=device)

            e_mean, e_norm, e_sd, e_kurtosis, b_mean, b_norm, b_sd, b_kurtosis = computeWeightStatistics(network)
            edge_mean_init.append(e_mean)
            edge_norm_init.append(e_norm)
            edge_sd_init.append(e_sd)
            edge_kurtosis_init.append(e_kurtosis)
            bias_mean_init.append(b_mean)
            bias_norm_init.append(b_norm)
            bias_sd_init.append(b_sd)
            bias_kurtosis_init.append(b_kurtosis)



            out, hidden, network, nepochs = learnTransformations(input_activations, output_activations, network=network, weight_init=weight_init, bias_init=bias_init, optim=optim,
                                                                 nhidden=nhidden,nlayers=nlayers,linear=linear,
                                                                 training_noise=train_noise,testing_noise=test_noise,
                                                                 dropout=dropout,cuda=cuda,normalize=normalize,
                                                                 lossfunc=lossfunc,fixed=fixed,batchsize=batchsize)

            # After training revert network back to cpu
            if cuda:
                network = network.to('cpu')
            # Pass session 1 data through model
            indata = torch.from_numpy(input_activation1[scount,:,:].T).float()
            out1, hidden1 = network.forward(indata,noise=test_noise,dropout=dropout)
            out1 = out1.detach().numpy()
            detached_hidden = []
            for hidden in hidden1: 
                detached_hidden.append(hidden.detach().numpy())
            hidden1 = detached_hidden
            # Pass session 2 data through model
            indata = torch.from_numpy(input_activation2[scount,:,:].T).float()
            out2, hidden2 = network.forward(indata,noise=test_noise,dropout=dropout)
            out2 = out2.detach().numpy()
            detached_hidden = []
            for hidden in hidden2: 
                detached_hidden.append(hidden.detach().numpy())
            hidden2 = detached_hidden

            outputs1.append(out1)
            outputs2.append(out2)
            hidden_activations1.append(hidden1)
            hidden_activations2.append(hidden2)

            e_mean, e_norm, e_sd, e_kurtosis, b_mean, b_norm, b_sd, b_kurtosis = computeWeightStatistics(network)
            edge_mean.append(e_mean)
            edge_norm.append(e_norm)
            edge_sd.append(e_sd)
            edge_kurtosis.append(e_kurtosis)
            bias_mean.append(b_mean)
            bias_norm.append(b_norm)
            bias_sd.append(b_sd)
            bias_kurtosis.append(b_kurtosis)

            weights = network.get_weights()
            if network.tied:
                weights_1 = weights[0] # tied weights share across layers so only need one set
            else:
                weights_1 = weights

            scount += 1

        # Save out
        try:
            h5f = h5py.File(tmp_outfilename + '.h5','a')
            h5f.create_dataset('outputs1',data=outputs1)
            h5f.create_dataset('hidden1',data=hidden_activations1)
            h5f.create_dataset('outputs2',data=outputs2)
            h5f.create_dataset('hidden2',data=hidden_activations2)
            h5f.create_dataset('nepochs', data=nepochs)
            #
            h5f.create_dataset('edge_mean',data=edge_mean)
            h5f.create_dataset('edge_norm',data=edge_norm)
            h5f.create_dataset('edge_sd',data=edge_sd)
            h5f.create_dataset('edge_kurtosis',data=edge_kurtosis)
            h5f.create_dataset('bias_mean',data=bias_mean)
            h5f.create_dataset('bias_norm',data=bias_norm)
            h5f.create_dataset('bias_sd',data=bias_sd)
            h5f.create_dataset('bias_kurtosis',data=bias_kurtosis)
            # init values
            h5f.create_dataset('edge_mean_init',data=edge_mean_init)
            h5f.create_dataset('edge_norm_init',data=edge_norm_init)
            h5f.create_dataset('edge_sd_init',data=edge_sd_init)
            h5f.create_dataset('edge_kurtosis_init',data=edge_kurtosis_init)
            h5f.create_dataset('bias_mean_init',data=bias_mean_init)
            h5f.create_dataset('bias_norm_init',data=bias_norm_init)
            h5f.create_dataset('bias_sd_init',data=bias_sd_init)
            h5f.create_dataset('bias_kurtosis_init',data=bias_kurtosis_init)
            # Weights
            h5f.create_dataset('weights',data=weights_1)
        except:
            os.remove(tmp_outfilename + '.h5')
            h5f = h5py.File(tmp_outfilename + '.h5','a')
            h5f.create_dataset('outputs1',data=outputs1)
            h5f.create_dataset('hidden1',data=hidden_activations1)
            h5f.create_dataset('outputs2',data=outputs2)
            h5f.create_dataset('hidden2',data=hidden_activations2)
            h5f.create_dataset('nepochs', data=nepochs)
            #
            h5f.create_dataset('edge_mean',data=edge_mean)
            h5f.create_dataset('edge_norm',data=edge_norm)
            h5f.create_dataset('edge_sd',data=edge_sd)
            h5f.create_dataset('edge_kurtosis',data=edge_kurtosis)
            h5f.create_dataset('bias_mean',data=bias_mean)
            h5f.create_dataset('bias_norm',data=bias_norm)
            h5f.create_dataset('bias_sd',data=bias_sd)
            h5f.create_dataset('bias_kurtosis',data=bias_kurtosis)
            # init values
            h5f.create_dataset('edge_mean_init',data=edge_mean_init)
            h5f.create_dataset('edge_norm_init',data=edge_norm_init)
            h5f.create_dataset('edge_sd_init',data=edge_sd_init)
            h5f.create_dataset('edge_kurtosis_init',data=edge_kurtosis_init)
            h5f.create_dataset('bias_mean_init',data=bias_mean_init)
            h5f.create_dataset('bias_norm_init',data=bias_norm_init)
            h5f.create_dataset('bias_sd_init',data=bias_sd_init)
            h5f.create_dataset('bias_kurtosis_init',data=bias_kurtosis_init)
            # Weights
            h5f.create_dataset('weights',data=weights_1)
        h5f.close()


    print('COMPUTE RSM OF EACH LAYER')
    tmp_outfilename = outdir + outfilename + '_data1and2_' + str(nhidden) + 'hidden_' + str(nlayers) + 'layers' 
    tmp_outfilename = tmp_outfilename + '_weightinit' + str(weight_init) + '_biasinit' + str(bias_init) + '_optim' + optim
    tmp_outfilename = tmp_outfilename + '_lr' + str(learning_rate)
    if relu: tmp_outfilename = tmp_outfilename + '_ReLU'
    if train_noise: tmp_outfilename = tmp_outfilename + '_trainNoise'
    if test_noise: tmp_outfilename = tmp_outfilename + '_testNoise'
    if dropout: tmp_outfilename = tmp_outfilename + '_dropout'
    if normalize: tmp_outfilename = tmp_outfilename + '_normalize'
    if lossfunc: tmp_outfilename = tmp_outfilename + '_' + lossfunc
    if fixed!=None: tmp_outfilename = tmp_outfilename + '_fixed' + str(fixed)
    if untied: tmp_outfilename = tmp_outfilename + '_untiedweights'
    if weight_dist == 'laplace':  tmp_outfilename = tmp_outfilename + '_laplace' 
    if not os.path.exists(tmp_outfilename + '.h5') and False:
        print('\tData exists... skipping')
        h5f = h5py.File(tmp_outfilename + '.h5','r')
        rsms = h5f['data'][:].copy()
    else:
        rsms = np.zeros((len(subIDs),nlayers+2,ntasks,ntasks)) # nlayers + input and output layers
        for i in range(len(subIDs)):
            # Get RSM of input layer
            rsms[i,0,:,:] = computeRSM(input_activation1[i],input_activation2[i],distance='cosine') 
            # Get RSM of hidden layers
            for layer in range(nlayers):
                rsms[i,layer+1,:,:] = computeRSM(hidden_activations1[i][layer].T, hidden_activations2[i][layer].T, distance='cosine')
            # Get RSM of final output layer
            rsms[i,-1,:,:] = computeRSM(outputs1[i].T, outputs2[i].T, distance='cosine')
        h5f = h5py.File(tmp_outfilename + '.h5','a')
        try:
            h5f.create_dataset('data',data=rsms)
        except:
            del h5f['data']
            h5f.create_dataset('data',data=rsms)
        h5f.close()

    print("COMPUTE DIMENSIONALITY OF EACH LAYER'S RSM")
    tmp_outfilename = outdir + outfilename + '_data1and2_' + str(nhidden) + 'hidden_' + str(nlayers) + 'layers' 
    tmp_outfilename = tmp_outfilename + '_weightinit' + str(weight_init) + '_biasinit' + str(bias_init) + '_optim' + optim
    tmp_outfilename = tmp_outfilename + '_lr' + str(learning_rate)
    if relu: tmp_outfilename = tmp_outfilename + '_ReLU'
    if train_noise: tmp_outfilename = tmp_outfilename + '_trainNoise'
    if test_noise: tmp_outfilename = tmp_outfilename + '_testNoise'
    if dropout: tmp_outfilename = tmp_outfilename + '_dropout'
    if normalize: tmp_outfilename = tmp_outfilename + '_normalize'
    if lossfunc: tmp_outfilename = tmp_outfilename + '_' + lossfunc
    if fixed!=None: tmp_outfilename = tmp_outfilename + '_fixed' + str(fixed)
    if untied: tmp_outfilename = tmp_outfilename + '_untiedweights'
    if weight_dist == 'laplace':  tmp_outfilename = tmp_outfilename + '_laplace' 
    if os.path.exists(tmp_outfilename + '.csv') and False:
        print('\tData exists... skipping')
        dim_model = pd.read_csv(tmp_outfilename + '.csv')
    else:
        dim_model = {}
        dim_model['Dimensionality'] = []
        dim_model['Layer'] = []
        dim_model['Subject'] = []
        scount = 0
        for sub in subIDs: 
            # Get RSM of input layer
            for layer in range(rsms.shape[1]):
                dim = tools.getDimensionality(rsms[scount,layer,:,:])
                dim_model['Dimensionality'].append(dim)
                dim_model['Layer'].append(layer+1)
                dim_model['Subject'].append(sub)
            scount += 1
        dim_model = pd.DataFrame(dim_model)
        dim_model.to_csv(tmp_outfilename + '.csv')


def learnTransformations(data_in, data_out, network=None, learning_rate=0.001, weight_init=3.0, bias_init=1.0, optim='adam', nhidden=10,nlayers=10, linear=True,
                         training_noise=None,testing_noise=None,dropout=False,cuda=False,normalize=False,
                         self_supervised_learning=False,lossfunc='mse',fixed=None,batchsize=None):
    """
    Learn the transformation mappings between an input-output activation pair
    PARAMETERS
    data_in
        input activation matrix (space x samples/features)
    data_out
        output/target activation amtrisx (space x samples/features)
    nhidden
        number of hidden units (per layer)
    nlayers
        number of hidden layers
    linear
        use a linear activation function (if not, use ReLU)

    OUTPUT
    network (pytorch model)
    """
    device='cuda' if cuda else 'cpu'
    num_inputs = data_in.shape[0]
    num_outputs = data_out.shape[0]
    if batchsize==None: batchsize = data_in.shape[1] # num samples

    indata = torch.from_numpy(data_in.T).float()
    outdata = torch.from_numpy(data_out.T).float()

    if cuda:
        indata = indata.cuda()
        outdata = outdata.cuda()

    if fixed==None:
        loss = 200000
        if normalize:
            loss_thresh = 0.2
        else:
            loss_thresh = 100

        iterations = 0
        while loss > loss_thresh:

            outputs, targets, loss = train(network,indata,outdata,noise=training_noise,dropout=dropout)
            if iterations%5000==0:
                print('\tTraining iteration', iterations, '| loss =', loss)
            iterations += 1
        exit_iterations = iterations
        print('\tTraining exited after', iterations, 'iterations, with loss', loss)
    else:

        for iterations in range(fixed):
            indices = np.arange(indata.shape[0])
            batch_ind = np.random.choice(indices,batchsize,replace=False)
            loss_thresh = 0.2 # used for spatially z-scored vectors

            outputs, targets, loss = train(network,indata[batch_ind,:],outdata[batch_ind,:],noise=training_noise,dropout=dropout)
            if iterations%5000==0:
                print('\tTraining iteration', iterations, '| loss =', loss)
        print('\tTraining exited after', iterations, 'iterations with loss =', loss)
        exit_iterations = iterations

    network.eval()

    outputs, hidden_activations = network.forward(indata,noise=testing_noise,dropout=False)
    outputs = outputs.detach().cpu().numpy()
    detached_hidden = []
    for hidden in hidden_activations: 
        detached_hidden.append(hidden.detach().cpu().numpy())

    return outputs, detached_hidden, network, exit_iterations

def computeRSM(data1,data2,distance='cosine'):
    """
    Compute RSM of data given some distance metric
    data1
        space x condition/features matrix
    data2
        space x condition/features matrix
    distance
        distance metric - default is cosine distance
    """
    nconds = data1.shape[1]
    tmpmat = np.zeros((nconds,nconds))
    for i in range(nconds):
        for j in range(nconds):
            if i>j: continue
            if distance=='correlation':
                tmpmat[i,j] = stats.pearsonr(data1[:,i],data2[:,j])[0]
            elif distance=='cosine':
                tmpmat[i,j] = np.dot(data1[:,i],data2[:,j])/(np.linalg.norm(data1[:,i])*np.linalg.norm(data2[:,j]))
            elif distance=='covariance':
                tmpmat[i,j] = np.mean(np.multiply(data1[:,i],data2[:,j]))

    # Now make symmetric
    tmpmat = tmpmat + tmpmat.T
    # double counting diagonal so divide by 2
    np.fill_diagonal(tmpmat, tmpmat.diagonal()/2.0)
    if distance in ['correlation','cosine']: tmpmat = np.arctanh(tmpmat)
    return tmpmat

def computeWeightStatistics(network):
    """
    Compute statistics of ANN weights
    Currently computes: edge strength x layer, edge norm x layer, edge sd x layer, edge kurtosis x layer
    """
    weights = network.get_weights() # layers x weights x weights
    # Compute edge strength x layer
    edge_mean = []
    for layer in range(weights.shape[0]):
        edge_mean.append(np.mean(weights[layer],axis=1))

    edge_norm = []
    for layer in range(weights.shape[0]):
        edge_norm.append(np.linalg.norm(weights[layer],axis=1))

    edge_sd = []
    for layer in range(weights.shape[0]):
        edge_sd.append(np.std(weights[layer],axis=1))

    edge_kurtosis = []
    for layer in range(weights.shape[0]):
        edge_kurtosis.append(stats.kurtosis(weights[layer],axis=1))

    #### Biases
    biases = network.get_biases()
    # Compute bias strength x layer
    bias_mean = []
    for layer in range(weights.shape[0]):
        bias_mean.append(np.mean(biases[layer],axis=0))

    bias_norm = []
    for layer in range(weights.shape[0]):
        bias_norm.append(np.linalg.norm(biases[layer],axis=0))

    bias_sd = []
    for layer in range(weights.shape[0]):
        bias_sd.append(np.std(biases[layer],axis=0))

    bias_kurtosis = []
    for layer in range(weights.shape[0]):
        bias_kurtosis.append(stats.kurtosis(biases[layer],axis=0))

    return edge_mean, edge_norm, edge_sd, edge_kurtosis, bias_mean, bias_norm, bias_sd, bias_kurtosis







class ANN(torch.nn.Module):
    """
    Neural network object
    """

    def __init__(self,
                 num_inputs=10,
                 num_hidden=10,
                 num_outputs=10,
                 num_hidden_layers=2,
                 brain_layers=None,
                 learning_rate=0.0001,
                 weight_init=1.0,
                 bias_init=1.0,
                 optim='adam',
                 tied=True,
                 lossfunc='mse',
                 linear=True,
                 weight_dist='normal',
                 device='cpu'):

        # Define general parameters
        self.tied = tied
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.brain_layers = brain_layers
        self.linear = linear
        
        # Define entwork architectural parameters
        super(ANN,self).__init__()
        
        self.w_in = torch.nn.Linear(num_inputs,num_hidden,bias=True)
        if weight_dist=='normal':
            torch.nn.init.normal_(self.w_in.weight, 0, weight_init*(1.0/np.sqrt(num_inputs))) # re-initialize weights
        elif weight_dist=='laplace':
            beta = 1.25
            laplace = torch.from_numpy(stats.gennorm.rvs(beta,loc=0,scale=weight_init*(1.0/np.sqrt(num_inputs)),size=(num_inputs,num_hidden)))
            with torch.no_grad():
                self.w_in.weight[:] = laplace.T
        if bias_init==0:
            self.w_in.bias.data.fill_(0.0) # re-initialize bias to 0
        else:
            torch.nn.init.normal_(self.w_in.bias, 0, bias_init*(1.0/np.sqrt(num_inputs))) # re-initialize weights
        #
        self.w_out = torch.nn.Linear(num_hidden,num_outputs,bias=True)
        torch.nn.init.normal_(self.w_out.weight, 0, weight_init*(1.0/np.sqrt(num_hidden))) # re-initialize weights
        if bias_init==0:
            self.w_out.bias.data.fill_(0.0) # re-initialize bias to 0
        else:
            torch.nn.init.normal_(self.w_out.bias, 0, bias_init*(1.0/np.sqrt(num_hidden))) # re-initialize weights
        #
        self.func = torch.nn.ReLU()

        self.dropout_in = torch.nn.Dropout(p=0.1)
        self.dropout_hidden = torch.nn.Dropout(p=0.2)

        if num_hidden_layers>1:
            # use this to create an arbitrary number of hidden-hidden networks (e.g., excluding first and final hidden layerss)
            if linear:
                if tied:
                    ## Tied weights
                    w = torch.nn.Linear(num_hidden,num_hidden,bias=True)
                    self.w_hid = torch.nn.ModuleList([torch.nn.Sequential(w) for i in range(num_hidden_layers-1)])
                else:
                    ## Untied weights
                    self.w_hid = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(num_hidden,num_hidden,bias=True)) for i in range(num_hidden_layers-1)])
                # Set initial parameter values
                for w in self.w_hid:
                    w[0].bias.data.fill_(0.0)
                    if weight_dist=='normal':
                        torch.nn.init.normal_(w[0].weight, 0, weight_init*(1.0/np.sqrt(num_hidden)))
                    elif weight_dist=='laplace':
                        laplace = torch.distributions.laplace.Laplace(0,weight_init*(1.0/np.sqrt(num_hidden)))
                        beta = 1.25
                        laplace = torch.from_numpy(stats.gennorm.rvs(beta,loc=0,scale=weight_init*(1.0/np.sqrt(num_hidden)),size=(num_hidden,num_hidden)))
                        with torch.no_grad():
                            w[0].weight[:] = laplace.T
            else:
                if tied:
                    ## Tied weights
                    w = torch.nn.Linear(num_hidden,num_hidden,bias=True)
                    self.w_hid = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Sequential(w), torch.nn.ReLU()) for i in range(num_hidden_layers-1)])
                else:
                    ## Untied weights
                    self.w_hid = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(num_hidden,num_hidden,bias=True), torch.nn.ReLU()) for i in range(num_hidden_layers-1)])

                # Set initial parameter values
                for w in self.w_hid:
                    w[0].bias.data.fill_(0.0) # first module list, then access linear module, not relu
                    if weight_dist=='normal':
                        torch.nn.init.normal_(w[0].weight, 0, weight_init*(1.0/np.sqrt(num_hidden)))
                    elif weight_dist=='laplace':
                        beta = 1.25
                        laplace = torch.from_numpy(stats.gennorm.rvs(beta,loc=0,scale=weight_init*(1.0/np.sqrt(num_hidden)),size=(num_hidden,num_hidden)))
                        with torch.no_grad():
                            w[0].weight[:] = laplace.T

        else:
            raise Exception("This model can't have fewer than 2 hidden layers")

        # If brain layers not none, include mapping to intermediate brain surface vertices
        self.w2brain = torch.nn.ModuleList([])
        if brain_layers is not None:
            for blayer in brain_layers:
                self.w2brain.append(torch.nn.Linear(num_hidden,len(blayer)))

        # Define loss function
        if lossfunc=='mse':
            self.lossfunc = torch.nn.MSELoss(reduction='none')
        if lossfunc=='kl':
            self.lossfunc = torch.nn.KLDivLoss(reduction='none')
        if lossfunc=='l1':
            self.lossfunc = torch.nn.L1Loss(reduction='none')
        if lossfunc=='l1smooth':
            self.lossfunc = torch.nn.SmoothL1Loss(reduction='none')
        if lossfunc=='cosine':
            self.lossfunc = torch.nn.CosineEmbeddingLoss(reduction='none')

        # Construct optimizer
        self.learning_rate = learning_rate
        if optim=='sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
            self.optim = optim
        if optim=='adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            self.optim = optim
        
        self.device = device
        self.to(device)
    
    def initHidden(self):
        return torch.randn(1, self.num_hidden)

    def forward(self,inputs,brainlayer=None,noise=None,dropout=True):
        """
        Run a forward pass of a trial by input_elements matrix
        """
        #Add noise to inputs
        if noise is not None:
            inputs = inputs + torch.randn(inputs.shape, device=self.device, dtype=torch.float)*noise 

        hidden_activations = []

        hidden = self.w_in(inputs) 
        if dropout: hidden = self.dropout_in(hidden)
        if self.linear==False: hidden = self.func(hidden)

        hidden_activations.append(hidden)

        # Feedforward connections 
        layercount = 0
        for w in self.w_hid:
            # If brain layer is specified, return out using hidden to surface brain mapping
            if brainlayer==layercount:
                brain_activation = self.w2brain[brainlayer](hidden)
                layercount += 1
                return brain_activation, hidden_activations

            # if not, continue iterating forward pass
            hidden = w(hidden)
            if dropout: hidden = self.dropout_hidden(hidden)
                
            hidden_activations.append(hidden)
            layercount += 1

        if brainlayer==9:
            brain_activation = self.w2brain[brainlayer](hidden)
            return brain_activation, hidden_activations
        
        # Compute outputs
        outputs = self.w_out(hidden) # Generate linear outupts

        return outputs, hidden_activations

    def forward_selfsupervisedlearning(self,inputs,noise=None,dropout=True):
        """
        Run a forward pass of a trial by input_elements matrix
        """
        #Add noise to inputs
        if noise is not None:
            inputs = inputs + torch.randn(inputs.shape, device=self.device, dtype=torch.float)*noise 

        hidden_activations = []

        hidden = self.w_in(inputs) 
        if dropout: hidden = self.dropout_in(hidden)
        if self.linear==False: hidden = self.func(hidden)

        hidden_activations.append(hidden)

        # Feedforward connections 
        if self.num_hidden_layers>1:
            if type(self.w_hid)==torch.nn.modules.container.ModuleList:
                for w in self.w_hid:
                    hidden = w(hidden)
                    if dropout: hidden = self.dropout_hidden(hidden)
                    hidden_activations.append(hidden)
            else:
                hidden = self.w_hid(hidden)
                if self.linear==False: hidden = self.func(hidden)
                if dropout: hidden = self.dropout_hidden(hidden)
                hidden_activations.append(hidden)
        
        # Compute outputs
        outputs = self.w_selfsupervised_out(hidden) # Generate linear outupts

        return outputs, hidden_activations

    def get_biases(self):
        numpy_biases = []
        for w in self.w_hid:
            if self.linear:
                numpy_biases.append(w[0].bias.detach().cpu().numpy())
            else:
                numpy_biases.append(w[0].bias.detach().cpu().numpy())
        return np.asarray(numpy_biases)

    def get_weights(self):
        numpy_weights = []
        for w in self.w_hid:
            if self.linear:
                numpy_weights.append(w[0].weight.detach().cpu().numpy())
            else:
                numpy_weights.append(w[0].weight.detach().cpu().numpy())
        return np.asarray(numpy_weights)


def train(network, inputs, targets, brainlayer=None, noise=None, dropout=False):
    """Train network"""

    network.train()
    network.zero_grad()
    network.optimizer.zero_grad()


    outputs, hidden = network.forward(inputs,brainlayer=brainlayer,noise=noise,dropout=dropout)

    if isinstance(network.lossfunc,torch.nn.CosineEmbeddingLoss):
        loss = network.lossfunc(outputs,targets,torch.autograd.Variable(torch.Tensor(targets.size(0)).fill_(1.0)))
    else:
        loss = network.lossfunc(outputs,targets)
    loss = torch.mean(loss)

    ### PS regularization
    if False: #ps_optim is not None:
        ps_outputs, hidden = network.forward(ps_optim.inputs_ps,noise=False,dropout=False)
        ps = calculatePS(hidden,ps_optim.match_logic_ind)
        logicps = ps # Want to maximize
        # Sensory PS
        ps = calculatePS(hidden,ps_optim.match_sensory_ind)
        sensoryps =ps
        # Motor PS
        ps = calculatePS(hidden,ps_optim.match_motor_ind)
        motorps = ps
        #ps_reg = (logicps + sensoryps + motorps) * ps_optim.ps
        msefunc = torch.nn.MSELoss(reduction='mean')

        ps_reg = torch.tensor(0., requires_grad=True).to(network.device)
        ps_reg += (3.0-logicps+sensoryps+motorps) * ps_optim.ps
        loss += ps_reg

    # Backprop and update weights
    loss.backward()
    network.optimizer.step() # Update parameters using optimizer
    
    return outputs, targets, loss.item()





if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
