from __future__ import division
import numpy as np
import sys
import pandas as pd
import numbers
from mvn_inpdata_converter_cnt_dsc import conv_data_cnt_dsc


from utils import get_weights, check_args
# fix random seed for reproducibility
#np.random.seed(7)


def MLMVN(varargin):
    #savefile = ("/home/joshua/Documents/python_codes/complex_valued_neuralnetwork-master/Networks/Net1.npy")

    # FOR CORRECTION OF WEIGHTS VALUE
    LearningRate = 0
    
    
    # creatingvector 'bisectors' with theangle values of bisectors of all sectors that will
    # be a part of learning, indicated by the value #numberOfSectors
    bisectors = -1

    # creating a list of stopping criterias available as
    # possible checking methods for ending of function
    acceptableStoppingCriteriaValues = ["max", "rmse", "mse", "error rate", "test"]

    # the same for initial weights methods
    acceptableInitialWeightsValues = ["random"]

    # using variable 'givenInputs' to determine which values were passed into the function
    givenInputs = [False]*17

    # GATHERING INPUT FROM INPUT ARGUMENT 'varargin'

    #varargin = varargin.T
    length = len(varargin) - 1
    
    check = check_args(varargin)
    
    if not check:
        sys.exit()


    for c1 in xrange(0,length,2):

        if not varargin[c1].isalpha():#(ischar(varargin(c1))):    #should be a string
            print("error inputs should be ordered in property name,value pairs, i.e. str('\'sizeOfMLMVN\', [3,1]')]))")
            pass


            #converting string to lower case for string comparison and checking
            #cases
        __switch_0__ = varargin[c1].lower()
        if 0:
            pass
        if __switch_0__ == str('sizeofmlmvn'):
            sizeOfMlmvn = varargin[c1 + 1]
            givenInputs[0] = True
        elif __switch_0__ == str('inputs'):
            inputs = varargin[c1 + 1]
            givenInputs[1] = True
        elif __switch_0__ == str('stoppingcriteria'):
            stoppingCriteria = varargin[c1 + 1]
            givenInputs[2] = True
        elif __switch_0__ == str('discreteinput'):
            discreteInput = varargin[c1 + 1]
            givenInputs[3] = True
        elif __switch_0__ == str('discreteoutput'):
            discreteOutput = varargin[c1 + 1]
            givenInputs[4] = True           #(4).lvalue
        elif __switch_0__ == str('globalthresholdvalue'):
            globalThresholdValue = varargin[c1 + 1]
            givenInputs[5] = True
        elif __switch_0__ == str('localthresholdvalue'):
            localThresholdValue = varargin[c1 + 1]
            givenInputs[6] = True
        elif __switch_0__ == str('network'):
            network = varargin[c1 + 1]
            givenInputs[7] = True
        elif __switch_0__ == str('initialweights'):
            initialWeights = varargin[c1 + 1]
            givenInputs[8] = True
        elif __switch_0__ == str('numberofsectors'):
            numberOfSectors = varargin[c1 + 1]
            givenInputs[9] = True
        elif __switch_0__ == str('angularglobalthresholdvalue'):
            angularGlobalThresholdValue = varargin[c1 + 1]
            givenInputs[10] = True
        elif __switch_0__ == str('angularlocalthresholdvalue'):
            angularLocalThresholdValue = varargin[c1 + 1]
            givenInputs[11] = True
        elif __switch_0__ == str('softmargins'):
            SoftMargins = varargin[c1 + 1]
            givenInputs[12] = True
        elif __switch_0__ == str('figurehandle'):
            figureHandle = varargin[c1 + 1]
            givenInputs[13] = True
        elif __switch_0__ == str('save'):
            save = varargin[c1 + 1]
            givenInputs[14] = True
        elif __switch_0__ == str('maxiteration'):
            maxIter = varargin[c1 + 1]
            givenInputs[15] = True
        elif __switch_0__ == str('compinputs'):
            compInputs = varargin[c1 + 1]
            givenInputs[16] = True
            # of the switch
            # of for loop
            #clear("c1")

    
    if not givenInputs[12]:
        SoftMargins = False


    # *** END OF INPUT VALIDATION, BEGINNING NECESSARY
    # *** VARIABLE INITIALIZATION
    #storing the value 2pi instead of having to multiply to get it every time
    pi2 = 2 * np.pi

    # a variable, which is equal to a half of Number of sectors; floor is
    # needed in NumberOfSectors is odd

    numberOfSectorsHalf = np.floor(numberOfSectors/2)
    #print(givenInputs[12])
    # generation of the bisectors' angular values
    if SoftMargins:
        temp = np.arange(numberOfSectors)
        bisectors = pi2*(temp+.5)/numberOfSectors
        del temp


    # sectorsize is the angular size of one sector for a discrete output
    if discreteOutput:
        sectorSize = pi2/numberOfSectors

    # Generation of complex numbers - roots of unity on the sectors' borders
    # They'll be contained in the array Sectors
    Sector= np.zeros(numberOfSectors, dtype='complex128')
    for jj in range(numberOfSectors):#=1:numberOfSectors
        angSector = (pi2*jj/numberOfSectors)*1j
        Sector[jj]= np.exp(angSector)

    #print Sector
    # if network was given, then its size must be read
    if givenInputs[7]:
        # network should be a list of weight array
        network = np.load(network)
        sizeOfMlmvn = []
        for a in range(len(network)):
            sizeOfMlmvn.append(network[a].shape[0])  
        #[sizeOfMmlvn1, sizeOfMlmvn2] = [0,len(network)]
        #sizeOfMlmvn = sizeOfMmlvn1 # this should be the column of the last weight connecting the output
        #print(sizeOfMlmvn)
            
    # initializing variable 'numberOfOutputs'
    numberOfOutputs = sizeOfMlmvn[-1] 

    #initializing the variable 'inputsPerSample' which is the number of input
    #values given for each learning sample in the matrix 'inputs' and this
    #value is equal to 'columns(inputs) - numberOfOutputs' since each row 
    #first consists of the inputs for the sample followed by the outputs 
    #of the sample
    
    
    [rowsInputs, colsInputs] = np.array(inputs).shape
    inputsPerSample = colsInputs - numberOfOutputs

    ###grabbing the columns containing the desired outputs of the MLMVN
    desiredOutputs = np.array(inputs[:,inputsPerSample:])
    #print(desiredOutputs)

    ###ridding matrix 'inputs' of values now stored in 'desiredOutputs'
    inputs = inputs[:,0:inputsPerSample] #in otherwords, delete column of labels

    ###updating these values because of the change in size of outputs
    [rowsInputs, colsInputs] = inputs.shape

    ###storing number of layers
    numberOfLayers = len(sizeOfMlmvn)

    ### numberOfLayers_1 is used then in loops
    numberOfLayers_1 = numberOfLayers-1     #needed?

    ###storing value as this variable for ease of use
    numberOfInputSamples = rowsInputs

    ###preallocating two arrays, which will be used for calculation the errors in
    ###the case the desired and actual outputs are located close to each other,
    ###but accross the 0/2pi border
    jjj=range (numberOfOutputs)
    iii=range(numberOfInputSamples)

    #SoftMargins = True
    ###preallocating a matrix to hold temporary output of the network for each sample
    networkOutputs=np.zeros((numberOfInputSamples,numberOfOutputs), dtype=complex)
    #print networkOutputs.shape
    if SoftMargins:
        networkAngleOutputs = networkOutputs#np.zeros((numberOfInputSamples,numberOfOutputs), dtype='complex128')#networkOutputs

    # preallocating a matrix for correction of those errors, which  jump over
    # pi (or half of the number of sectors)
    mask = np.zeros((numberOfInputSamples,numberOfOutputs), dtype=complex)#networkOutputs
    [n1,n2] = mask.shape

    #initializing the variable that will hold the global error of the 
    # network for each input sample
    networkErrors = np.zeros((numberOfInputSamples,numberOfOutputs), dtype='complex128')#np.copy(networkOutputs)#[:,0]

    #print(networkErrors.shape)
    if SoftMargins:
        netAngErrors = networkErrors#np.zeros((numberOfInputSamples,numberOfOutputs), dtype='complex128')#networkErrors

          
    if not givenInputs[7]:
        network = get_weights(sizeOfMlmvn, inputsPerSample)
##        network = []
##        h = np.array([[0.23-0.38j,0.19-0.46j, 0.36-0.33j], [0.23-0.38j, 0.19-0.46j, 0.36-0.33j]])
##        h = np.array(h, ndmin=2, dtype='complex128')
##        network.append(h)
##        ho = np.array([[0.23-0.38j, 0.19-0.46j,0.36-0.33j ]])#.T        
##        ho = np.array(ho, ndmin=2, dtype='complex128')
##        network.append(ho)
    
    ##print(len(network))
    #print(network)
    
    #creating a variable to hold the outputs of all the neurons accross the network. It will be a cell
    #array of column vectors, because the layers aren't necessarily all the 
    # same sizes and having them as appropriately sized column vectors
    # will allow for optimization of the speed of the calculation of the
    # outputs of the layers of the network

    length = len( network )
    neuronOutputs = []#[None]*length#cell( 1, len );
    weightedSum = []
    #neuronErrors = []
    

    for ii in range(0,length):#= 1:len
        #print(np.zeros((sizeOfMlmvn[ii],1)))
        neuronOutputs.append(np.zeros((sizeOfMlmvn[ii],1)))
        weightedSum.append(np.zeros((sizeOfMlmvn[ii],1),dtype=complex))
        #neuronErrors.append(np.zeros((sizeOfMlmvn[ii],1)))
    #print(neuronOutputs)
    # creating a variable to hold the errors of the outputs of the neurons
    # ( has same size as neuronOutputs since each output will have 
    #   an associated error value )
    neuronErrors = neuronOutputs
    


    # initializing a variable (an array) to hold the weighted sums 
    # of all the neurons accross the network
    #weightedSum = neuronOutputs


    # a desired discrete output equals a root of unity corresponding to the
    # bisector of a desitred sector
    if discreteOutput:
        #d = desiredOutputs+0.5
        ComplexValuedDesiredOutputs = np.exp( (desiredOutputs+0.5)*1j*sectorSize )
        #print sectorSize

        # AngularDesiredOutputs - arguments of the discrete desired outputs
        AngularDesiredOutputs = np.mod(np.angle(ComplexValuedDesiredOutputs), pi2)
        #print (zip([ComplexValuedDesiredOutputs[0:11],AngularDesiredOutputs[0:11], desiredOutputs[0:11]]))
    else:
        # AngularDesiredOutputs - arguments of the continuous desired outputs
        #d = np.array(inputs[:,inputsPerSample:])
        AngularDesiredOutputs = desiredOutputs#np.array(inputs[:,inputsPerSample:])#
        
        ComplexValuedDesiredOutputs = np.exp( (desiredOutputs)*1j)
        #print(ComplexValuedDesiredOutputs)

    
    
    if discreteInput or compInputs:
        if not compInputs:
            # converting sector values (which are integers) into 
            # corresponding complex numbers located on the unit circle margumetnts of inputs
            theta = pi2 * (inputs) / numberOfSectors
            inputs =  np.exp(theta*1j) *np.ones( (rowsInputs, colsInputs) )
            #print inputs[0,:]
        else:
            pass
    else:
        if not compInputs:
            #continuous inputs: inputs are arguments of complex numbers in the range [0, 2pi]
            #converting angle values into complex numbers ( assumes angles are  given in radians )
            #[re, im] = pol2cart( inputs, ones( rowsInputs, colsInputs ) );
            inputs = np.exp(pi2* (inputs)* 1j)
            #print inputs[0,:]
        else:
            pass

    #CORRECT UP TILL THIS POINT
    #preallocating these since conversion between integer sector values and
    #complex numbers based on these values occur every iteration
    re = neuronOutputs
    im = neuronOutputs
    
    ##END OF VARIABLE INITIALIZATION
    #-------------------------------------------------------------

    ##DATA PROCESSING
    #-----------------------------------------------------------
    #ENTERING DATA PROCESSING STAGE 
    #-----------------------------------------------------------

    desiredAlgo = 0;
    #checking for desired algorithm based on stopping criteria
    if ( stoppingCriteria == 'mse' ):
        desiredAlgo = 1
    elif ( stoppingCriteria == 'rmse' ):
        desiredAlgo = 2
    elif ( stoppingCriteria == 'error rate' ):
        desiredAlgo = 3
    elif ( stoppingCriteria == 'max' ):
        desiredAlgo = 4
    elif ( stoppingCriteria, 'test' ):
        desiredAlgo = 5
    else: #not yet implemented stoppingCriteria algorithm
        pass

    #if somehow desiredAlgo stayed at 0
    if (desiredAlgo < 1):
        s = ['\n\nerror: a not yet implemented algorithm was requested\n\n']
        print('MLMVN:NotImplementedAlgorithm', s)
    # or if it accidentally has some weird value
    if (desiredAlgo > 5):
        s = ['\n\nerror: something unexpected has occurred\n\n']
        print( 'MLMVN:ExecutionError', s )


    iterations = 0;

    # ANGULAR RMSE 
    if SoftMargins:
        networkAngularErrors =np.zeros((numberOfInputSamples,numberOfOutputs))#[None]*numberOfInputSamples   #for RMSE
        maxOutputError = np.zeros((numberOfInputSamples,numberOfOutputs))# [None]*numberOfInputSamples
        outputErrors = np.zeros((numberOfInputSamples,numberOfOutputs))
        ArgActualOutputs =  neuronOutputs
        #ANGULAR RMSE ALGORITHM
        finishedLearning = False
        while not finishedLearning:
            # CALCULATING SAMPLE OUTPUT
            iterations=iterations+1 # iterations counter
            ErrorCounter=0 # initialization of the error counter on the iteration 
            # NET OUTPUT CALCULATION
            #******************************
            # CALCULATING THE OUTPUTS OF THE NETWORK FOR EACH OF THE
            # SAMPLES
            #looping through all samples
    #---------------------------------------------------------------------------------------
    ##     SAME IN ALL CASES, FEED FOWARD
    ##........................................................................................        
            for aa in range(numberOfInputSamples):				
                # *** PROCESSING FIRST LAYER ***
                ii = 0;# ( ii holds current layer index)
                # calculating weighted sums for the 1st hidden layer                           
                #neuronOutputs[ii] = (network[ii][:,1:].dot( inputs[aa,:]) + network[ii][:,0])[:,None]
                weightedSum[ii] = (network[ii][:,1:].dot(np.atleast_2d(inputs[aa,:]).T)) + np.atleast_2d(network[ii][:,0]).T
                #print(network[ii][:,1:].shape)
            
                #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                ## CONTINUOUS OUTPUT CALCULATION
                #output=weighted sum/abs(weighted sum)
                neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])
                

                ## END OF CONTINUOUS OUTPUT CALCULATION			
                # *** PROCESSING FOLLOWING LAYERS until the 2nd to the last***
                #ii holds current layer
                for ii in range(1,numberOfLayers_1):
    ##                print(neuronOutputs[ii-1].shape)
    ##                print(network[ii][:,1:].shape)
                    #neuronOutputs[ii] = (network[ii][:,1:].dot( neuronOutputs[ii-1])) + network[ii][:,0][:,None]
                    weightedSum[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                    #print(neuronOutputs[ii].shape)
                    
                    #APPLYING CONTINUOUS ACTIVATION FUNCTION
                    #TO THE FIRST HIDDEN LAYER NEURONS
                    ## CONTINUOUS OUTPUT CALCULATION
                    #output=weighted sum/abs(weighted sum)
                    neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])
                #end of ii for loop

                ## END OF CONTINUOUS OUTPUT CALCULATION
                # *** PROCESSING THE OUTPUT LAYER***
                #ii holds current layer (numberOfLayers is the output layer index)

                ii = numberOfLayers_1
                #CALCULATING WEIGHTED SUMS OF THE OUTPUT LAYER NEURONS Wx+b

                #neuronOutputs[ii] = (network[ii][:,1:].dot(neuronOutputs[ii-1])) + network[ii][:,0][:,None]
                weightedSum[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                #print(neuronOutputs[ii].shape)

    #---------------------------------------------------------------------------------------
    ##     NETWORK OUTPUT CALCULATION, SAME fOR
    ##........................................................................................
                
                ## NETWORK OUTPUT CALCULATION
                #applying the activation function
                #for angular rmse only discrete activation functionis used for the output neurons
                                                        
                # --- FOR DISCRETE OUTPUT ---
                # this will be the argument of output in the range [0, 2pi]
                neuronOutputs[ii] = np.mod( np.angle( weightedSum[ii][:] ), pi2 )
                
                # Actual arguments of the weighted sums of the
                # output neurons
                ArgActualOutputs = neuronOutputs[ii]
                # this will be the discrete output (the number of sector)
                neuronOutputs[ii] = np.atleast_2d(np.floor (neuronOutputs[ii]/sectorSize)).T
                
                
               
                ## END OF OUTPUT CALCULATION
                #above loop just calculated the output of the network for asample 

                #after calculation of the outputs of the neurons for the
                #sample copying the network output for for the sample and storing in 'networkOutputs'
                networkOutputs[aa,:] = neuronOutputs[ii]
                #if aa==1: print ArgActualOutputs[:].T# networkOutputs[aa,:]
            

                
    ##--------------------------------------------------------------------
    ##          ERROR CALCULATION
    ## ------------------------------------------------------------------           
                ## CALCULATION OF ANGULAR RMSE ERROR               
                #calculating angular error for the aa-th learning sample
                networkAngularErrors[aa,:] = np.abs(AngularDesiredOutputs[aa,:]-ArgActualOutputs)#[:].T)
                networkAngularErrors[aa,:] = np.mod (networkAngularErrors[aa,:],pi2)

##                networkAngularErrors[aa,:] = np.abs( bisectors(desiredOutputs[aa,:]+1) \
##                	- np.mod(np.angle(weightedSum[ii].T),pi2) )
                # calculation of the mean angular error for the aa-th learning sample over all output neurons
                if numberOfOutputs >1:
                    
                    netAngErrors[aa] = np.mean( networkAngularErrors[aa,:] )

                    # calculation of the absolute error in terms of the sector numbers for all output neurons
                    outputErrors[aa] = np.abs( networkOutputs[aa,:]-desiredOutputs[aa,:] )
                
                    # maximal error over all output neurons
                    maxOutputError[aa,:]= np.max( outputErrors[aa] )
                    #print(maxOutputError[aa])
                else:
                    
                    netAngErrors[aa] = networkAngularErrors[aa,:]
                    outputErrors[aa] = np.abs( networkOutputs[aa,:]-desiredOutputs[aa,:] )
                    maxOutputError[aa,:] = np.atleast_2d(outputErrors[aa]).T
            print maxOutputError
            
    ##-----------------------------------------------------------------------
    ##          RMSE OF NETWORK OVER ALL SAMPLES
    ##------------------------------------------------------------------------
            #above loop just calculated the ouputs of the network for all samples
            #Calculation of angular RMSE over all learning samples
            AngularRMSE = np.sqrt(np.sum(netAngErrors**2)/numberOfInputSamples)
            # if AngularRMSE<globalThresholdValue, then learning has finished
            check=(np.max(maxOutputError)==0)
    ##-----------------------------------------------------------------
    ##          CHECK IF TRAINING IS NOT REQUIRED
    ##-----------------------------------------------------------------------
            if (AngularRMSE<=angularGlobalThresholdValue) and check:
                finishedLearning = True

            finishedLearning= finishedLearning and check

            # the number of nonzero elements in maxOutputError - the number of errors
            ErrorCounter=np.count_nonzero(maxOutputError)
            #print('Iter %.5f  Errors %.5f  Angular RMSE %6.4f\n', iterations, ErrorCounter, AngularRMSE)
            print('Iter', iterations,  'Errors' , ErrorCounter, 'Angular RMSE\n',  AngularRMSE,)

            ErrorCounter=0 # reset of counter of the samples required learning on the current iteration
            ## END OF NET ERROR CALCULATION
    ##----------------------------------------------------------------------
    ##          MODIFICATION OF WEIGHTS IF LEARNING REQUIRED
    ##-----------------------------------------------------------------------

            ## LEARNING / MODIFICATION OF WEIGHTS
            # if the algorithm has not finished learning then output of the
            # network needs to be calculated again to start correction of errors
            if not finishedLearning:				
                #calculating the output of the network for each sample and
                #correcting weights if output is > localThresholdValue
    #---------------------------------------------------------------------------------------
    ##     FEED FOWARD - CHECK
    ##........................................................................................        
                for aa in range (numberOfInputSamples):			

                    ii = 0;# ( ii holds current layer index)
                    # calculating weighted sums for the 1st hidden layer
    ##                print(network[ii][:,1:].shape)
    ##                print(inputs[aa,:].shape)
    ##                print(network[ii][:,0].shape)
                    
                    #weightedSum[ii] = (network[ii][:,1:].dot( (inputs[aa,:])) + network[ii][:,0])[:,None]
                    weightedSum[ii] = (network[ii][:,1:].dot(np.atleast_2d(inputs[aa,:]).T)) + np.atleast_2d(network[ii][:,0]).T
                    
                    #print(weightedSum[ii].shape)

                    #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                    ## CONTINUOUS OUTPUT CALCULATION output=weighted sum/abs(weighted sum)
                    neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii]) 

                    ## END OF CONTINUOUS OUTPUT CALCULATION
                    # *** PROCESSING FOLLOWING LAYERS until the 2nd to the last***
                    #ii holds current layer

                    #ii holds current layer
                    for ii in range (1,numberOfLayers_1):
                        #CALCULATING WEIGHTED SUMS OF REMAINING LAYERS OF NEURONS until the 2nd to the last
                        #inputs of the neurons in the layer ii are the outputs of neurons in the layer ii-1
                        #weightedSum[ii] = (network[ii][:,1:].dot(neuronOutputs[ii-1])) + network[ii][:,0][:,None]
                        weightedSum[ii] =network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                        #print(weightedSum[ii].shape)

                        #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                        ## CONTINUOUS OUTPUT CALCULATION output=weighted sum/abs(weighted sum)
                        neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii]) 
    ##
                        ## END OF CONTINUOUS OUTPUT CALCULATION

                    # *** PROCESSING THE OUTPUT LAYER***
                    #ii holds current layer (numberOfLayers is the output layer index)
                    ii = numberOfLayers_1;
                    #CALCULATING WEIGHTED SUMS OF THE OUTPUT LAYER NEURONS 

                    #weightedSum[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + network[ii][:,0][:,None]
                    weightedSum[ii] =network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                    #print(weightedSum[ii].shape)
    #---------------------------------------------------------------------------------------
    ##              NETWORK OUTPUT, SAME FOR
    ##........................................................................................
                    
                    ## NETWORK OUTPUT CALCULATION
                    #applying the activation function for angular rmse only discrete activation function
                    #is used for the output neurons                                                   
                    # --- FOR DISCRETE OUTPUT ---
                    # this will be the argument of output in the range [0, 2pi]
                    neuronOutputs[ii] = np.mod( np.angle( weightedSum[ii][:] ), pi2 )
                    
                    # Actual arguments of the weighted sums of the output neurons
                    ArgActualOutputs = neuronOutputs[ii]                        
                    # this will be the discrete output (the number of sector)
                    neuronOutputs[ii] = np.floor (neuronOutputs[ii]/sectorSize)
                    #print(neuronOutputs[ii].shape)
                    ## END OF OUTPUT CALCULATION

                    #we just calculated the output of the network for a sample                

                    #after calculation of the outputs of the neurons for the
                    #sample copying the network output for for the sample
                    #and storing in 'networkOutputs'
                    networkOutputs[aa,:] =  neuronOutputs[-1]
                    #print(networkOutputs[aa,:].shape)
                    
                    #previous loop just calculated outputs of all neurons for the current sample
                    #now checking to see if the network output for that
                    #sample is <= localThresholdValue and if it isn't, then
                    #correction of the weights of the network begins
    #---------------------------------------------------------------------------------------
    ##              ERROR CALCULATION
    ##........................................................................................                                                                
                    ## CALCULATION OF ERROR
                    #calculating angular error for the aa-th learning sample
                    #print(ArgActualOutputs.shape)
                    networkAngularErrors[aa,:] = np.abs(AngularDesiredOutputs[aa,:]-ArgActualOutputs[:].T)
                    networkAngularErrors[aa,:] = np.mod (networkAngularErrors[aa,:],pi2)

                    # calculation of the mean angular error for the aa-th
                    # learning sample over all output neurons
                    SampleAngError = np.mean( networkAngularErrors[aa,:] )
                    #print(SampleAngError)

                    # calculation of the absolute error in terms of the
                    # sector numbers for all output neurons
                    outputErrors = np.abs( networkOutputs[aa,:]-desiredOutputs[aa,:] )
                    #print(outputErrors)

                    # this indicator is needed to take care of those errors
                    # whose "formal" actual value exceeds a half of the number
                    # of sectors (thus of those, which jump over pi)
                    indicator = (outputErrors>numberOfSectorsHalf)

                    if (np.count_nonzero(indicator)>0):
                        [i1] = np.where(indicator==1)
                        outputErrors[i1] = -(outputErrors[i1]-numberOfSectors)
                                       
                    # maximal error over all output neurons
                    maxOutputError= np.max( outputErrors )
                    # if there is a non-zero error, then it is necessary to
                    # start the actual learning process
                    ## END OF CALCULATION OF ERROR
    #---------------------------------------------------------------------------------------
    ##              WEIGHT CORRECTION BEGINING FROM OUTPUT LAYER
    ##........................................................................................        
                    #checking against the local threshold
                    check=(SampleAngError > angularLocalThresholdValue) or (maxOutputError>0)
                    if check:
                        ErrorCounter=ErrorCounter+1 # increment of the counter for the samples required learning on the current iteration
                                                    
                        # if greater than, then weights are corrected, else nothing happens
                        #**************************************************
                        #*** NOW CALCULATING THE ERRORS OF THE NEURONS ***					
                        #calculation of errors of neurons starts at last layer and moves to first layer
                        # ** handling special case, the output layer ***
                        ii = numberOfLayers_1

                        # neuronOutputs will now contain normalized 
                        # weighted sums for all output neurons 
                        neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])
                        # jj is a vector with all output neurons' indexes

                        # the global error for the jjj-th output neuron equals a root of unity corresponding to the
                        # desired output - normalized wightet sum for the corresponding output neuron
                        # jjj contains indexes 1:NumberOfOutputs                                                         
                        neuronErrors[ii] [jjj] = (ComplexValuedDesiredOutputs[aa,jjj]- \
                                                 np.array(neuronOutputs[ii][jjj], dtype=complex))#[:,None]                   
                        # finally we obtain the output neurons' errors
                        # normalizing the global errors (dividing them
                        # by the (number of neurons in the preceding layer+1)
                        neuronErrors[ii] = neuronErrors[ii] / (sizeOfMlmvn[ii-1]+1)
                    #----------------------------------------------
                    ##     OTHER LAYERS - BACK PROPAGATION
                    ##..........................................        
                        # handling the rest of the layers - ERROR BACKPROPAGATION
                        for ii in range (numberOfLayers_1-1,-1,-1): #check
                            #print(ii)
                            
                            #print(network[ii+1].shape)				
                            # calculation of the reciprocal weights for the
                            # layer ii and putting them in a vector-row
                            temp =  (1 / network[ii+1] ).T# .' is used to avoid transjugation
                            #print(temp.shape)
                            # extraction resiprocal weights corresponding only to the inputs
                            #(the 1st weight w0 will be dropped, since it is not needed for backpropagation
                            temp = temp[1:,:]
                            #print(temp.shape)
                            # backpropagation of the weights
                            if ii > 0:
                                neuronErrors[ii] = np.array(temp.dot((neuronErrors[ii+1])) / (sizeOfMlmvn[ii-1]+1))#[:,None]
                                #print(ii)
                                #print(neuronErrors[ii].shape)
                                    
                                    #print((sizeOfMlmvn[ii-1]+1))
                            else:    # to the 1st hidden layer
                                #print(neuronErrors[ii+1].shape)
                                neuronErrors[ii] = np.array(temp.dot((neuronErrors[ii+1])) / (inputsPerSample+1))#[:,None]
                                #print(neuronErrors[ii].shape)
                                
                            #end % end of the if statement
                        #end % e
    #---------------------------------------------------------------------------------------
    ##                  WEIGHT CORRECTION - FIRST HIDDEN LAYER
    ##........................................................................................                                
                        #**************************************************
                        # *** NOW CORRECTING THE WEIGHTS OF THE NETWORK ***
                        #handling the 1st hidden layer learning rate is a reciprocal absolute value
                        # of the weighted sum
                        learningRate = ( 1 / np.abs( weightedSum[0] ) )
                        # all weights except bias (w0 = w(1) in Matlab)
    ##                    print(network[0][:,1:].shape)
    ##                    print(neuronErrors[0].shape)
    ##                    print(np.conj(inputs[aa,:]).shape)

                        network[0][:,1:] = network[0][:,1:] +\
                                           (learningRate*neuronErrors[0]).dot(np.conj(inputs[aa,:])[None,:])

                        #print(network[0][:,1:].shape)
                        # bias (w0 = w(1) in Matlab)
                        #print(network[0][:,0].shape)
                        #print(neuronErrors[0].shape)
                        
                        network[0][:,0][:,None] = network[0][:,0][:,None] + learningRate * neuronErrors[0]
                        #print(network[0][:,0].shape)
                        #correcting following layers
                #-----------------------------------------------------
                ##      OTHER PRECEEDING LAYERS
                ##----------------------------------------------------
                        for ii in range(1,numberOfLayers):					
                            #**********************************************
                            #calculating new output of preceding layer
                            if (ii==1): # if a preceding layer is the 1st one
                                #print(network[0][:,1:].shape)
                                #print((inputs[aa,:]).shape)
                                weightedSum[0] = network[0][:,1:].dot((inputs[aa,:])) + network[0][:,0]
                                #print(weightedSum[0].shape)
                                
                            else: # if a preceding layer is not the 1st one
                                weightedSum[ii-1] = network[ii-1][:,1:].dot(neuronOutputs[ii-2]) + network[ii-1][:,0]


                            #APPLYING CONTINUOUS ACTIVATION FUNCTION
                            #TO THE FIRST HIDDEN LAYER NEURONS
                            ## CONTINUOUS OUTPUT CALCULATION
                            #output=weighted sum/abs(weighted sum)
                            neuronOutputs[ii-1] = weightedSum[ii-1] / np.abs(weightedSum[ii-1])							
                            #**********************************************
                            # learning rate is a reciprocal absolute value of the weighted sum					
                            learningRate = ( 1 / np.abs( weightedSum[ii] ) )
                            #learningRate not used for the output layer neurons

                            if ii < numberOfLayers_1:
                                print('here first')
            
                                #all weights except bias (w0=w(1) in Matlab)
                                a1=network[ii][:,1:]
                                b1=neuronErrors[ii]
                                #print(b1.shape)
                                b1=(learningRate * b1) /(sizeOfMlmvn[ii-1]+1)
                                c1=np.atleast_2d(neuronOutputs[ii-1]).T
                                #print(c1.shape)
                                c1=np.conj(c1)
                                c1=c1.T
                                #print(c1.shape)
                                e1=a1
                                for i1 in range (0,sizeOfMlmvn[ii]):
                                    d1=b1[i1]
                                    e1[i1,:]= d1*c1
                                    #end
                                f1=a1+e1
                                network[ii][:,1:] = f1
##                                    if aa==0:
##                                        print network[ii][:,1:]

                                # bias (w0 = w(1) in Matlab)
                                #print(network[ii][:,0].shape)
                                #print(neuronErrors[ii].shape)
                                network[ii][:,0][:,None] = np.atleast_2d(network[ii][:,0]).T + b1#learningRate * neuronErrors[ii]
                            else:  # correction of the output layer neurons' weights
                                
                                #all weights except bias (w0=w(1) in Matlab)
                                a1=network[ii][:,1:]
                                #if aa==0:print a1
                                
                                b1=neuronErrors[ii] /(sizeOfMlmvn[ii-1]+1)
                                c1=np.atleast_2d(neuronOutputs[ii-1]).T
                                c1=np.conj(c1)
                                c1=c1.T
                                e1=np.zeros((a1.shape), dtype=complex)#a1#np.copy(a1)*0

                                for i1 in range(sizeOfMlmvn[ii]):
                                    d1=b1[i1]
                                    e1[i1,:]=d1*c1
                                    
                                #e1=e1/ (sizeOfMlmvn[ii-1]+1)            #modification

                                f1=a1+e1
##                                    if aa == 0:
##                                        print e1
##                                        print f1
                                    
                                network[ii][:,1:] = f1
                                # bias (w0 = w(1) in Matlab)
                                #network[ii][:,0][:,None] = np.atleast_2d(network[ii][:,0]).T + (neuronErrors[ii]/(sizeOfMlmvn[ii-1]+1))
                                network[ii][:,0] = np.atleast_2d(network[ii][:,0]).T + b1
##                                    if aa==1:
##                                        print network[ii][:,1:]
##                                        print network[ii][:,0][:,None]

                if givenInputs[15]:
                    if iterations == maxIter:
                                finishedLearning= True
        print ("learning iterations: ", iterations)    
        if save == 1:
            np.save(savefile, network)
##            finishedLearning= True
    else:
        if desiredAlgo < 3: #%mse/rmse algo if ends line 2740
            #print inputs
                    
            ## RMSE ALGORITHM
            #--------------------------------------------------------------------------
            #converting desiredAlgo to different value based on whether
            #stopping criteria is rmse or mse
            if ( stoppingCriteria == 'rmse' ):
                desiredAlgo = 11
                #initializing variable rmse to hold root mean square error
                #of the network over all samples
                rmse = 0
            else:
                desiredAlgo = 12
            #end		
            #initializing variable mse to hold mean square error of the
            #network over all samples
            mse = 0;
            

            #value telling whether learning has finished or not
            finishedLearning = False

            #***************************************
            #BEGINNING THE LEARNING LOOP
            iterations = 0
            #learning loop continues until learning has finished
            while  not finishedLearning:
                if iterations % 10000 ==0:
                    print("iteration: ", iterations)
                iterations = iterations + 1;
                #******************************
                # CALCULATING THE OUTPUTS OF THE NETWORK FOR EACH OF THE SAMPLES
                #looping through all samples
    #---------------------------------------------------------------------------------------
    ##     SAME IN ALL CASES, FEED FOWARD
    ##........................................................................................        
                for aa in range(numberOfInputSamples ): #ends 1355
                    # *** PROCESSING FIRST LAYER ***
                    ii = 0# ( ii holds current layer )
                    #neuronOutputs[ii] = (network[ii][:,1:].dot( inputs[aa,:]) + network[ii][:,0])[:,None]
                    neuronOutputs[ii] = (network[ii][:,1:].dot(np.atleast_2d(inputs[aa,:]).T)) + np.atleast_2d(network[ii][:,0]).T
                    #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                    neuronOutputs[ii] = neuronOutputs[ii] / np.abs(neuronOutputs[ii])
                    #if aa==0: print("int output", neuronOutputs[ii])                
                                  
                    # *** PROCESSING FOLLOWING LAYERS ***
                    for ii in range(1,numberOfLayers_1):
                        #neuronOutputs[ii] = network[ii][:,1:].dot( neuronOutputs[ii-1]) + network[ii][:,0][:,None]
                        neuronOutputs[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                        #print(neuronOutputs[ii])
                        
                        #APPLYING CONTINUOUS ACTIVATION FUNCTIONTO THE FIRST HIDDEN LAYER NEURONS
                        neuronOutputs[ii] = neuronOutputs[ii] / np.abs(neuronOutputs[ii])
                        
                    # *** PROCESSING THE OUTPUT LAYER***
                    #ii holds current layer (numberOfLayers is the output layer index)
                    ii = numberOfLayers_1
                    #CALCULATING WEIGHTED SUMS OF THE OUTPUT LAYER NEURONS Wx+b
                    #neuronOutputs[ii] = (network[ii][:,1:].dot(neuronOutputs[ii-1]) + network[ii][:,0])[:,None]
                    neuronOutputs[ii] = np.array(network[ii][:,1:]).dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T      
    #---------------------------------------------------------------------------------------
    ##              OUTPUT CALCULATION
    ##........................................................................................
                    ## NETWORK OUTPUT CALCULATION - applying the activation function
                    if discreteOutput:
                        # --- FOR DISCRETE OUTPUT -- the argument of output in the range [0, 2pi]
                        neuronOutputs[ii] = np.mod( np.angle( neuronOutputs[ii][:] ), pi2 )
                        # this will be the discrete output (the number of sector)
                        neuronOutputs[ii] = np.floor (neuronOutputs[ii]/sectorSize)
                    else:
                        # --- FOR CONTINUOUS OUTPUT ---
                        neuronOutputs[ii] = neuronOutputs[ii] / np.abs(neuronOutputs[ii])
                        #if aa==0: print(neuronOutputs[ii])
                    #copying the ]etwork output for for the sample and storing in 'networkOutputs'
                    networkOutputs[aa,:] = neuronOutputs[ii]
                    #if aa==0: print "fin output", networkOutputs[aa,:]
                # END OF SECTION CALCULATING OUTPUTS OF THE NETWORK FOR EACH OF THE SAMPLES
    #---------------------------------------------------------------------------------------
    ##          ERROR CALCULATION AND COMPARISON
    ##........................................................................................
                
                #**************************************************************
                #CALCULATING GLOBAL ERROR AND COMPARING VALUE TO THRESHOLD
                # calculation of the squared error for each sample
                if discreteOutput==1:  # discrete outputs
                    errors = np.abs( networkOutputs-desiredOutputs )
                    # indicator needed to handle those errors whose "formal" actual value
                    # exceeds a hald of the number of sectors (thus of those, which jump over pi)
                    indicator = (errors>numberOfSectorsHalf)
                    if (np.count_nonzero(indicator)>0):
                        i3 = (indicator==1)
                        mask[i3] = numberOfSectors
                        errors[i3] = (mask[i3]-errors[i3])#
                else:  # continuous outputs
                    #errors is all the difference between angles of predicted - target
                    errors = np.abs(np.mod(np.angle(networkOutputs), pi2)-AngularDesiredOutputs)
                    # indicator is needed for errors whose "formal" actual value jump over pi                
                    indicator = (errors>np.pi)
                    if (np.count_nonzero(indicator)>0):
                        i3 = (indicator==1)
                        mask[i3] = pi2
                        errors[i3] = (mask[i3]-errors[i3])#.astype(np.float32)
                    #if (indicator(iii,jjj)==1)
                    #  errors(iii,jjj)=pi2-errors(iii,jjj);
                    #end                 
                    #mask=zeros(n1,n2); % resetting of mask
                    mask=np.zeros((n1,n2)) # resetting of mask
                networkErrors = np.sum(errors[iii,:]**2)/numberOfOutputs
                #print(networkErrors)
                #calculating combined mse of all samples
                mse = np.sum( networkErrors ) / numberOfInputSamples
                #print mse
                #calculating rmse if that was what was called for
                if desiredAlgo == 10:
                    rmse = np.sqrt( mse )
                    #print rmse
                    print('Iter', iterations, ', RMSE\n', rmse)
                    #now checking if value is <= globalThresholdValue to
                    #determine if algo has reached the end of learning
                    if rmse <= globalThresholdValue:
                        #if it has, then learning is completed
                        finishedLearning = True
                   # and if it hasn't then learning will go on
                else:
                    #performing same check but comparison is between mse and the globalThresholdValue
                    if mse <= globalThresholdValue:
                        finishedLearning = True
    ##-----------------------------------------------------------------------------------------------
    ## THIS WHOLE SECTION MAY BE REDUNDANT AFTER ALL, IT SEEMS THE OUTPUTS WERE ALREADY CALCULATED ABOVE
    ##----------------------------------------------------------------------------------------------
                ## LEARNING / MODIFICATION OF WEIGHTS
                # if not finished learning, network output to be calculated again to start correction of errors
                if  not finishedLearning: #end 1705
                    #calculating network output per sample, correcting weights if output is > localThresholdValue
                    for aa in range (numberOfInputSamples): #end 1704
                        # *** PROCESSING FIRST LAYER ***
                        ii = 0 # ( ii holds current layer )
                        #neuronOutputs[ii] = (network[ii][:,1:].dot(inputs[aa,:]) + network[ii][:,0])[:,None]
                        weightedSum[ii] = (network[ii][:,1:].dot(np.atleast_2d(inputs[aa,:]).T)) + np.atleast_2d(network[ii][:,0]).T
                        #if aa == 0: print(weightedSum[ii])
                        #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                        neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])
##                        if aa==0: print(neuronOutputs[ii])
                        
                        # *** PROCESSING FOLLOWING LAYERS ***
                        #ii holds current layer
                        for ii in range( 1,numberOfLayers_1):
                            #neuronOutputs[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + network[ii][:,0][:,None]
                            weightedSum[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                            #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                            neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])

                        # *** PROCESSING THE OUTPUT LAYER**
                        #ii holds current layer (numberOfLayers is the output layer index)
                        ii = numberOfLayers_1
                        #CALCULATING WEIGHTED SUMS OF THE OUTPUT LAYER NEURONS 
                        #neuronOutputs[ii] = network[ii][:,1:].dot( neuronOutputs[ii-1]) + network[ii][:,0][:,None]
                        weightedSum[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
##                        if aa==1: print(weightedSum[ii])
    #---------------------------------------------------------------------------------------
    ##          NETWORK OUTPUT CALCULATION
    ##........................................................................................                           

                        ## NETWORK OUTPUT CALCULATION - applying the activation function
                        if discreteOutput:
                            # --- FOR DISCRETE OUTPUT ---this is the argument of output in [0, 2pi]
                            neuronOutputs[ii] = np.mod( np.angle( weightedSum[ii][:] ), pi2 )
                            # this will be the discrete output (the number of sector)
                            neuronOutputs[ii] = np.floor (neuronOutputs[ii]/sectorSize) 
                        else:
                            # --- FOR CONTINUOUS OUTPUT ---
                            neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])
                            #if aa ==0: print(neuronOutputs[ii])
                        ## END OF OUTPUT CALCULATION
                        networkOutputs[aa,:] = neuronOutputs[ii]
                        #print(networkOutputs[aa,:])

                        #check if the network output for sample <= localThresholdValue, if not
                        #correction of the weights of the network begins
    #---------------------------------------------------------------------------------------
    ##                  ERROR CALCULATION
    ##........................................................................................        
                        ## CALCULATION OF ERROR
                        if discreteOutput==1:  # discrete outputs
                            errors = np.abs( networkOutputs-desiredOutputs )
                            #print(errors)
                            # indicator handles errors whose "formal" actual value which jump over pi)
                            indicator = (errors>numberOfSectorsHalf)
                            if (np.count_nonzero(indicator)>0):
                                i3 = (indicator==1)
                                mask[i3] = numberOfSectors
                                errors[i3] = (mask[i3]-errors[i3])#.astype(np.float32)
                        else:  # continuous outputs
                            errors = np.abs(np.mod(np.angle(networkOutputs), pi2)-AngularDesiredOutputs)
                            #if aa==0: print errors
                            # this indicator is needed to take care of those errors whose "formal" actual jump over pi                
                            indicator = (errors>np.pi)
                            if (np.count_nonzero(indicator)>0):
                                i3 = (indicator==1)
                                mask[i3] = pi2
                                errors[i3] = (mask[i3]-errors[i3])#.astype(np.float32)
                            mask=mask*0#np.zeros((n1,n2)) # resetting of mask
                        ## END OF CALCULATION OF ERROR

    ##-----------------------------------------------------------------------------------------------
    ##                  ERROR CORRECTION FROM OUTPUT LAYER
    ##----------------------------------------------------------------------------------------------
                        
                        networkError = np.max(errors[aa,:])
##                        if aa==0: print networkError
                        #checking against the local
                        if networkError > localThresholdValue:
                            # if greater than, then weights are corrected, else nothing happens
                            #**************************************************
                            #*** NOW CALCULATING THE ERRORS OF THE NEURONS ***
                            #starts at last layer and moves to first layer
                            # *** handling special case, the output layer ***
                            ii = numberOfLayers_1
                            # neuronOutputs will now contain normalized  weighted sums for all output neurons 
                            neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])
                            # jj is a vector with all output neurons' indexes
                            # global error for the jjj-th output neuron
                            # = a root of unity corresponding to the desired output - normalized wightet sum for
                            # the corresponding output neuron
                            neuronErrors[ii] [jjj] = ComplexValuedDesiredOutputs[aa,:]\
                                                     - np.array((neuronOutputs[ii]),dtype=complex)# d_12
                            #if aa==0: print("neuron errors= ",neuronErrors[ii][jjj])                        
                            # normalizing the global errors (dividing by number of neurons in the preceding layer+1)
                            neuronErrors[ii] = neuronErrors[ii] / (sizeOfMlmvn[ii-1]+ 1)    #this is split output error
##                            if aa==0: print(neuronErrors[ii])
    ##-----------------------------------------------------------------------------------------------
    ##                  BACK PROPAGATING THE ERRORS
    ##----------------------------------------------------------------------------------------------
                            # *** handling the rest of the layers ***
                            # ERROR BACKPROPAGATION
                            for ii in range (numberOfLayers_1-1,-1,-1):
                                temp = ( 1/ network[ii+1] ).T#.';
##                                if aa==0: print(temp)
                                temp = temp[1:,:]
                                # weight inverse *output error / # of neurons b4 = back propagation minus bias layer                                
                                if ii > 0:
                                    neuronErrors[ii] = np.array(temp.dot(neuronErrors[ii+1])) / (sizeOfMlmvn[ii-1]+1)                                    
                                else:    # to the 1st hidden layer
                                    neuronErrors[ii] = np.array(temp .dot (neuronErrors[ii+1])) / (inputsPerSample+1)
##                                if aa==0:
##                                    print(neuronErrors[ii])
    ##-------------------------------------------------------------------------------
    ##                  WEIGHT CORRECTION
    ##------------------------------------------------------------------------------
                            #print("n",neuronErrors[ii].shape)
                            #**************************************************
                            # *** NOW CORRECTING THE WEIGHTS OF THE NETWORK ***
                            #handling the 1st hidden layer learning rate is a reciprocal absolute value
                            # of the weighted sum
                            
                            learningRate = ( 1 / np.abs( weightedSum[0] ) )
##                            if aa==0: print learningRate
                            # all weights except bias (w0 = w(1) in Matlab)
                            network[0][:,1:] = network[0][:,1:] +(learningRate*neuronErrors[0]).dot(np.conj(inputs[aa,:])[None,:])
##                            if aa==0: network[0][:,1:] print network[0][:,1:]
                            # bias (w0 = w(1) in Matlab)
                            network[0][:,0][:,None] = np.array(network[0][:,0])[:,None] + learningRate * neuronErrors[0]
##                            if aa==0: print network[0][:,0][:,None]

                            #handling the following layers
                            for ii in range (1,numberOfLayers):
                                #**********************************************
                                #calculating new output of preceding layer
                                if (ii==1): # if a preceding layer is the 1st one
                                    weightedSum[0] = network[0][:,1:] .dot( (inputs[aa,:])) + network[0][:,0]
                                    #weightedSum[0]= network[0][:,1:].dot(inputs[aa,:]) + np.atleast_2d(network[0][:,0])
                                else: # if a preceding layer is not the 1st one
                                    weightedSum[ii-1] = network[ii-1][:,1:].dot(neuronOutputs[ii-2]) + network[ii-1][:,0]
                                    #weightedSum[ii-1] =network[ii][:,1:].dot(neuronOutputs[ii-2]) + np.atleast_2d(network[ii-1][:,0]).T
##                                if aa ==0: print weightedSum[ii-1]
                                
                                #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                                neuronOutputs[ii-1] = weightedSum[ii-1] / np.abs(weightedSum[ii-1])
##                                if aa==0: print neuronOutputs[ii-1]


                                #**********************************************
                                # learning rate is a reciprocal absolute value of the weighted sum					
                                learningRate = 1 / np.abs(weightedSum[ii])
##                                if aa==0: print weightedSum[ii]
                                #learningRate not used for the output layer neurons
                                if ii < numberOfLayers_1:         
                                    #all weights except bias (w0=w(1) in Matlab)
                                    a1=network[ii][:,1:]
                                    if aa== 0: print a1
                                    b1=neuronErrors[ii]
                                    b1=(learningRate * b1) /(sizeOfMlmvn[ii-1]+1)
                                    c1=np.atleast_2d(neuronOutputs[ii-1]).T
                                    c1=np.conj(c1)
                                    c1=c1.T
                                    #print(c1.shape)
                                    e1=a1
                                    for i1 in range (0,sizeOfMlmvn[ii]):
                                        d1=b1[i1]
                                        e1[i1,:]= d1*c1
                                    f1=a1+e1
                                    network[ii][:,1:] = f1
##                                    if aa==0: print network[ii][:,1:]
                                    # bias (w0 = w(1) in Matlab)
                                    network[ii][:,0][:,None] = np.atleast_2d(network[ii][:,0]).T + b1#learningRate * neuronErrors[ii]
                                else:  # correction of the output layer neurons' weights     
                                    #all weights except bias (w0=w(1) in Matlab)
                                    a1=network[ii][:,1:] # weights exept bias                                    
                                    b1=neuronErrors[ii] /(sizeOfMlmvn[ii-1]+1) # output error/#of neurons before
                                    c1=np.atleast_2d(neuronOutputs[ii-1]).T
                                    c1=np.conj(c1)
                                    c1=c1.T #conjugate of modified weighted sum feeding to output
                                    e1=np.zeros((a1.shape), dtype=complex)#a1#np.copy(a1)*0

                                    for i1 in range(sizeOfMlmvn[ii]):
                                        d1=b1[i1]
                                        #if aa == 0: print d1
                                        e1[i1,:]=d1*c1
                                        #if aa==0: print e1[i1,:]
                                    f1=a1+e1
##                                    if aa == 0:
##                                        print e1
##                                        print f1   
                                    network[ii][:,1:] = f1
                                    # bias (w0 = w(1) in Matlab)
                                    #network[ii][:,0][:,None] = np.atleast_2d(network[ii][:,0]).T + (neuronErrors[ii]/(sizeOfMlmvn[ii-1]+1))
                                    network[ii][:,0] = np.atleast_2d(network[ii][:,0]).T + b1
                                    #if aa==0: print network[ii][:,0]#[:,None]
                    if givenInputs[15]:
                        if iterations == maxIter:
                            finishedLearning= True
            print ("learning iterations: ", iterations)
            if save == 1:
                np.save(savefile, network)
                
                #print network[-1]
    #**************************************************
    # END OF RMSE ALGORITHM
        elif desiredAlgo == 3:#error rate algo:
            ## ERROR RATE ALGORITHMi
            finishedLearning = False
            iterations=0

            while not finishedLearning:
                iterations=iterations+1 # iterations counter
                ## NET OUTPUT CALCULATION
                #******************************
                # CALCULATING THE OUTPUTS OF THE NETWORK FOR EACH OF THE SAMPLES     
                for aa in range(numberOfInputSamples ): #ends 1355
                    ii = 0# ( ii holds current layer )

                    #neuronOutputs[ii] = (network[ii][:,1:].dot( inputs[aa,:]) + network[ii][:,0])[:,None]
                    neuronOutputs[ii] = (network[ii][:,1:].dot(np.atleast_2d(inputs[aa,:]).T)) + np.atleast_2d(network[ii][:,0]).T
                    #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                    neuronOutputs[ii] = neuronOutputs[ii] / np.abs(neuronOutputs[ii])
                    #print(neuronOutputs[ii])
                    # *** PROCESSING FOLLOWING LAYERS ***
                    for ii in range(1,numberOfLayers_1):
                        #neuronOutputs[ii] = network[ii][:,1:].dot( neuronOutputs[ii-1]) + network[ii][:,0][:,None]
                        neuronOutputs[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                        #print(neuronOutputs[ii])
                        
                        #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                        neuronOutputs[ii] = neuronOutputs[ii] / np.abs(neuronOutputs[ii])

                    # *** PROCESSING THE OUTPUT LAYER***
                    ii = numberOfLayers_1
                    #neuronOutputs[ii] = (network[ii][:,1:].dot(neuronOutputs[ii-1]) + network[ii][:,0])[:,None]
                    neuronOutputs[ii] = np.array(network[ii][:,1:]).dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
    #---------------------------------------------------------------------------------------
    ##     NETWORK OUTPUT CALCULATION
    ##........................................................................................                
                    
                    ## NETWORK OUTPUT CALCULATION
                    #applying the activation function
                    if discreteOutput:
                        # --- FOR DISCRETE OUTPUT -- the argument of output in the range [0, 2pi]
                        neuronOutputs[ii] = np.mod( np.angle( neuronOutputs[ii][:] ), pi2 )
                        # this will be the discrete output (the number of sector)
                        neuronOutputs[ii] = np.floor (neuronOutputs[ii]/sectorSize)
                    else:   # --- FOR CONTINUOUS OUTPUT ---
                        neuronOutputs[ii] = neuronOutputs[ii] / np.abs(neuronOutputs[ii])
                        #print(neuronOutputs[ii])

                    # copying the ]etwork output for for the sample and storing in 'networkOutputs'
                    networkOutputs[aa,:] = neuronOutputs[ii]
##                    if aa==0: print weightedSum

                ##END OF SECTION CALCULATING OUTPUTS OF THE NETWORK FOR EACH OF THE SAMPLES
    #---------------------------------------------------------------------------------------
    ##     CALCULATING NET ERROR - this part same
    ##........................................................................................
                ## NET ERROR CALCULATION
                ## CALCULATING GLOBAL 'ERROR RATE' ERROR BASED ON OUTPUT
                # absolute network errors calculated for every output neuron
                # and for every learning sample (in terms of sectors' numbers
                # for discrete outputs and "as they are" for continuous outputs
                # this section is needed if desired and actual outputs are
                # located close to each other, but accross the 0/2pi borders  
                
                if discreteOutput==1:  # discrete outputs
                    errors = np.abs( networkOutputs-desiredOutputs )
                    # indicator handles those errors which jump over pi)
                    indicator = (errors>numberOfSectorsHalf)

                    if (np.count_nonzero(indicator)>0):
                        i3 = (indicator==1)
                        mask[i3] = numberOfSectors
                        errors[i3] = mask[i3]-errors[i3]
                else:  # continuous outputs
                    errors = np.abs(np.mod(np.angle(networkOutputs), pi2)-AngularDesiredOutputs)
                    # indicator is needed to take care of those errors that jump over pi                
                    indicator = (errors>np.pi)
                    if (np.count_nonzero(indicator)>0):
                        i3 = (indicator==1)
                        mask[i3] = pi2
                        errors[i3] = mask[i3]-errors[i3]             
                    mask=np.zeros((n1,n2)) # resetting of mask

    #---------------------------------------------------------------------------------------
    ##     CHECKING FOR ERRORS THAT EXCEED THRESHOLD
    ##........................................................................................
                # evaluation of those errors, which exceed localThresholdValue
                errors = (errors > localThresholdValue).astype(int)
                # a vector containing the number of errors per every input
                # sample (this number does not exceed 1 for a single output neuron
                networkErrors = (errors!=0)
                networkErrors = np.sum(networkErrors)
                networkErrors = np.max(networkErrors)
                #check if learning is done by calculating percent of samples with incorrect output value
                ErrorRate = (100 * networkErrors / numberOfInputSamples)
    #---------------------------------------------------------------------------------------
    ##     DETERMINING IF LEARNING IS DONE
    ##........................................................................................
                check = ( ErrorRate <= globalThresholdValue)
                # if check = true then the learning process is complete
                if check:
                    finishedLearning = True
                    if iterations % 1000 == 0:
                        print('Iter:', iterations, 'Error Rate: ',  ErrorRate, '%')
                ## END OF CALCULATION OF ERROR	
        
    #---------------------------------------------------------------------------------------
    ##     LEARNING / MODIFYING WEIGHTS
    ##......................................................................................
                ## LEARNING/MODIFICATION OF WEIGHTS
                if  not finishedLearning: #end 1705
                    #calculating the output of the network per sample. correcting weights if output is > localThresholdValue
                    for aa in range (numberOfInputSamples): #end 1704
                        # *** PROCESSING FIRST LAYER ***
                        ii = 0 # ( ii holds current layer )
                        #neuronOutputs[ii] = (network[ii][:,1:].dot(inputs[aa,:]) + network[ii][:,0])[:,None]
                        weightedSum[ii] = (network[ii][:,1:].dot(np.atleast_2d(inputs[aa,:]).T)) + np.atleast_2d(network[ii][:,0]).T
##                        if aa == 0: print(weightedSum[ii])
                        #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                        neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])
##                        if aa==1: print(neuronOutputs[ii])
                        # *** PROCESSING FOLLOWING LAYERS ***
                        for ii in range( 1,numberOfLayers_1):
                            #neuronOutputs[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + network[ii][:,0][:,None]
                            weightedSum[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                            #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                            neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])
                            
                        # *** PROCESSING THE OUTPUT LAYER**
                        ii = numberOfLayers_1
                        #CALCULATING WEIGHTED SUMS OF THE OUTPUT LAYER NEURONS 
                        #neuronOutputs[ii] = network[ii][:,1:].dot( neuronOutputs[ii-1]) + network[ii][:,0][:,None]
                        weightedSum[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
##                        if aa==1:  print(weightedSum[ii])
    #---------------------------------------------------------------------------------------
    ##              CALCULATING NETWORK OUTPUTS
    ##........................................................................................
                        ## NETWORK OUTPUT CALCULATION
                        #applying the activation function
                        if discreteOutput:
                            # --- FOR DISCRETE OUTPUT -- be the argument of output in the range [0, 2pi]
                            neuronOutputs[ii] = np.mod( np.angle( weightedSum[ii][:] ), pi2 )
                            # this will be the discrete output (the number of sector)
                            neuronOutputs[ii] = np.floor (neuronOutputs[ii]/sectorSize)    
                        else: # --- FOR CONTINUOUS OUTPUT ---
                            neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])
##                            if aa ==0: print(neuronOutputs[ii])
                        ## END OF OUTPUT CALCULATION
                        networkOutputs[aa,:] = neuronOutputs[ii]
                        #print(networkOutputs[aa,:])

                        #now checking if the network output for sample is <= localThresholdValue 
                        #if not, correction of the weights of the network begins
    #---------------------------------------------------------------------------------------
    ##             SPECIFIC CALCULATION FOR ERROR RATE - this part different
    ##........................................................................................
                        ## ERROR RATE SPECIFIC ERROR CALCULATION SECTION
                        if discreteOutput==1:
                            errorCurrent = np.abs( networkOutputs[aa,:]-desiredOutputs[aa,:] )
                            # indicator needed for errors which jump over pi)
                            indicator = (errorCurrent>numberOfSectorsHalf)
                            if (np.count_nonzero(indicator)>0):
                                [i1] = np.where(indicator==1)
                                errorCurrent[i1] = numberOfSectors-errorCurrent[i1]
                        else:
                            errorCurrent = np.abs(np.mod(np.angle(networkOutputs[aa,:]), pi2)-AngularDesiredOutputs[aa,:])
                            # this indicator is needed to take care of those errors whose "formal" actual jump over pi                
                            indicator = (errorCurrent>np.pi)
                            if (np.count_nonzero(indicator)>0):
                                [i1] = numpy.where(indicator==1)
                                errorCurrent[i1] = pi2-errorCurrent[i1]
                        # maximal error over all output neurons for the aa-th learning sample and
                        maxError= np.max(errorCurrent)

                        check = (maxError > localThresholdValue)
                        # if check is true, that is at least one of the output
                        # neurons' errors exceeds localThresholdValue, then the weights must be adjusted
                                                                        
                        ## END OF ERROR RATE ERROR CALCULATION SECTION

    #---------------------------------------------------------------------------------------
    ##              ERROR CHECK =>, CORRECT WEIGHTS from output layer
    ##........................................................................................
                        #checking against the local threshold, the weights must be corrected if check = true

                        if check:
                            #**************************************************
                            #*** NOW CALCULATING THE ERRORS OF THE NEURONS ***
                            #starts at last layer and moves to first layer
                            # ** handling special case, the output layer ***
                            ii = numberOfLayers_1

                            # neuronOutputs will now contain normalized  weighted sums for all output neurons 
                            neuronOutputs[ii] = np.array(weightedSum[ii] / np.abs(weightedSum[ii]))#, dtype=complex)
                            ##print(neuronOutputs[ii])
                            # jj is a vector with all output neurons' indexes
                            # the global error for the jjj-th output neuron equals a root of unity corresponding to the
                            # desired output - normalized wightet sum for the corresponding output neuron
                            # jjj contains indexes 1:NumberOfOutputs
                                                                                     
                            neuronErrors[ii] [jjj] = ComplexValuedDesiredOutputs[aa,jjj].T - \
                                                    np.array(neuronOutputs[ii][jjj], dtype=complex)
                            
     
                            # output neurons' errors= normalizing the global errors (dividing them
                            # by the (number of neurons in the preceding layer+1)
                            neuronErrors[ii] = neuronErrors[ii] / (sizeOfMlmvn[ii-1]+1)
                            #print(neuronErrors[ii].shape)

    #---------------------------------------------------------------------------------------
    ##                  BACK PROPAGATE ERROR
    ##........................................................................................
                            
                            # handling the rest of the layers - ERROR BACKPROPAGATION
                            for ii in range (numberOfLayers_1-1,-1,-1):
                                temp = ( 1/ network[ii+1] ).T
                                temp = temp[1:,:]
                                if ii > 0:
                                    neuronErrors[ii] = np.array(temp.dot(neuronErrors[ii+1])) / (sizeOfMlmvn[ii-1]+1)
                                else:    # to the 1st hidden layer
                                    neuronErrors[ii] = np.array(temp .dot (neuronErrors[ii+1])) / (inputsPerSample+1)
##                                    if aa==0: print(neuronErrors[ii])
    #---------------------------------------------------------------------------------------
    ##              CORRECTING WEIGHTS
    ##........................................................................................
                            #**************************************************
                            # *** NOW CORRECTING THE WEIGHTS OF THE NETWORK ***
                            #handling the 1st hidden layer
                            # learning rate is a reciprocal absolute value of the weighted sum
                            learningRate = ( 1 / np.abs( weightedSum[0] ) )
##                            if aa==0: print learningRate
                            network[0][:,1:] = network[0][:,1:] +(learningRate*neuronErrors[0]).dot(np.conj(inputs[aa,:])[None,:])
##                            if aa==0: print network[0][:,1:]
                            # bias (w0 = w(1) in Matlab)
                            network[0][:,0][:,None] = np.array(network[0][:,0])[:,None] + learningRate * neuronErrors[0]
##                            if aa==0: print network[0][:,0][:,None]

                            #handling the following layers
                            for ii in range (1,numberOfLayers):
                                #**********************************************
                                #calculating new output of preceding layer
                                if (ii==1): # if a preceding layer is the 1st one
                                    weightedSum[0] = network[0][:,1:] .dot( (inputs[aa,:])) + network[0][:,0]
##                                    if aa ==0: print weightedSum[ii-1]
                                else: # if a preceding layer is not the 1st one
                                    weightedSum[ii-1] = network[ii-1][:,1:].dot(neuronOutputs[ii-2]) + network[ii-1][:,0]
##                                    if aa ==0:  print weightedSum[ii-1]
                                #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                                neuronOutputs[ii-1] = weightedSum[ii-1] / np.abs(weightedSum[ii-1])
##                                if aa==0:  print neuronOutputs[ii-1]							

                                #**********************************************
                                # learning rate is a reciprocal absolute value of the weighted sum					
                                learningRate = 1 / np.abs(weightedSum[ii-1])
##                                if aa==0: print weightedSum[ii-1]
                                #learningRate not used for the output layer neurons
                                if ii < numberOfLayers_1:
                                    print('here first')
                                    #all weights except bias (w0=w(1) in Matlab)
                                    a1=network[ii][:,1:]
                                    b1=neuronErrors[ii]
                                    b1=(learningRate * b1) /(sizeOfMlmvn[ii-1]+1)
                                    c1=np.atleast_2d(neuronOutputs[ii-1]).T
                                    c1=np.conj(c1)
                                    c1=c1.T
                                    e1=a1
                                    for i1 in range (0,sizeOfMlmvn[ii]):
                                        d1=b1[i1]
                                        e1[i1,:]= d1*c1
                                    f1=a1+e1
                                    network[ii][:,1:] = f1
##                                    if aa==0: print network[ii][:,1:]
                                    # bias (w0 = w(1) in Matlab)
                                    network[ii][:,0][:,None] = np.atleast_2d(network[ii][:,0]).T + b1#learningRate * neuronErrors[ii]
                                else:  # correction of the output layer neurons' weights
                                    #all weights except bias (w0=w(1) in Matlab)
                                    a1=network[ii][:,1:]
                                    b1=neuronErrors[ii] /(sizeOfMlmvn[ii-1]+1)
                                    c1=np.atleast_2d(neuronOutputs[ii-1]).T
                                    c1=np.conj(c1)
                                    c1=c1.T
                                    e1=np.zeros((a1.shape), dtype=complex)#a1#np.copy(a1)*0

                                    for i1 in range(sizeOfMlmvn[ii]):
                                        d1=b1[i1]
                                        e1[i1,:]=d1*c1
                                    f1=a1+e1
##                                    if aa == 0:
##                                        print e1
##                                        print f1
                                    network[ii][:,1:] = f1
                                    # bias (w0 = w(1) in Matlab)
                                    #network[ii][:,0][:,None] = np.atleast_2d(network[ii][:,0]).T + (neuronErrors[ii]/(sizeOfMlmvn[ii-1]+1))
                                    network[ii][:,0] = np.atleast_2d(network[ii][:,0]).T + b1
##                                    if aa==1:
##                                        print network[ii][:,1:]
##                                        print network[ii][:,0][:,None]
                    if givenInputs[15]:
                        if iterations == maxIter:
                            finishedLearning= True
            print ("learning iterations: ", iterations)
            if save == 1:
                np.save(savefile, network)
                
                
        elif desiredAlgo ==4:#max algorithm
            finishedLearning = False;
            iterations=0; 
            while not finishedLearning:
                iterations=iterations+1;
                if iterations % 500 == 0:
                    print("epoch: ", iterations) 
                # number of errors counter
                Errors=0;

                ## NET OUTPUT CALCULATION AND LEARNING
                # There is no "global" error preliminary estimated here 
                #******************************
                # CALCULATING THE OUTPUTS OF THE NETWORK FOR EACH OF THE SAMPLES
                #calculating the output of the network for each sample and
                #correcting weights if output is > localThresholdValue
    ##-----------------------------------------------------------------------------------------
    ##              FEEDFOWARD
    ##------------------------------------------------------------------------------------------
                #looping through all samples
                for aa in range(numberOfInputSamples):
                    #print(inputs[aa,:])
                    # *** PROCESSING FIRST LAYER ***
                    ii = 0# ( ii holds current layer index)
                    # calculating weighted sums for the 1st hidden layer
                    weightedSum[ii] = (network[ii][:,1:].dot(np.atleast_2d(inputs[aa,:]).T)) + np.atleast_2d(network[ii][:,0]).T
                    #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                    neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii]);
                    #print(neuronOutputs[ii])
                    
                    # *** PROCESSING FOLLOWING LAYERS ***
                    #ii holds current layer
                    for ii in range( 1,numberOfLayers_1):
                        #weightedSum[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                        weightedSum[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                        #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS                   
                        neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii]) 

                    # *** PROCESSING THE OUTPUT LAYER**
                    ii = numberOfLayers_1
                    #CALCULATING WEIGHTED SUMS OF THE OUTPUT LAYER NEURONS
                    weightedSum[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                    #print(weightedSum[ii])                
    ##---------------------------------------------------------------------------------------
    ##              OUTPUT CALCULATION - DISCRETE ONLY
    ##---------------------------------------------------------------------------------------
                    ## NETWORK OUTPUT CALCULATION
                    # --- FOR DISCRETE OUTPUT --- the argument of output in the range [0, 2pi]
                    neuronOutputs[ii] = np.mod( np.angle( weightedSum[ii][:] ), pi2 )
                    # this will be the discrete output (the number of sector)
                    neuronOutputs[ii] = np.floor (neuronOutputs[ii]/sectorSize)                    
                    ## END OF OUTPUT CALCULATION
                    networkOutputs[aa,:] = neuronOutputs[ii]
                    #print(networkOutputs[aa,:])

    ##---------------------------------------------------------------------------------------
    ##              SPECIFIC ERROR RATE CALCULATION
    ##---------------------------------------------------------------------------------------
                    
                    ## ERROR RATE SPECIFIC ERROR CALCULATION SECTION

                    errorCurrent = np.abs( networkOutputs[aa,:]-desiredOutputs[aa,:] )
                    #this indicator is needed to take care of those errors
                    # whose "formal" actual value exceeds a hald of the number of sectors (thus of those, which jump over pi)
                    indicator = (errorCurrent>numberOfSectorsHalf)
                    if (np.count_nonzero(indicator)>0):
                        [i1] = np.where(indicator==1)
                        errorCurrent[i1] = numberOfSectors-errorCurrent[i1]
                    # maximal error over all output neurons for the aa-th learning sample and
                    maxError= np.max(errorCurrent)

                    check = (maxError > localThresholdValue)
                    # if check is true, that is at least one of the output
                    # neurons' errors exceeds localThresholdValue, then the weights must be adjusted
                                                                    
                    ## END OF ERROR RATE ERROR CALCULATION SECTION
    ##---------------------------------------------------------------------------------------
    ##              CHECKING AGAINST LOCAL THRESHOLD FOR ERROR CORRECTION FROM OUTPUT LAYER
    ##---------------------------------------------------------------------------------------
                    #checking against the local threshold, the weights must  be corrected if check = true
                    if check:	                    
                        # increment the number of errors on the current iteration
                        Errors = Errors+1
                        #**************************************************
                        #*** NOW CALCULATING THE ERRORS OF THE NEURONS ***
                        #calculation of errors of neurons starts at last layer and moves to first layer
                        # ** handling special case, the output layer ***
                        ii = numberOfLayers_1
                        # neuronOutputs will now contain normalized  weighted sums for all output neurons 
                        neuronOutputs[ii] = weightedSum[ii] / np.abs(weightedSum[ii])#, dtype=complex)
                        # jj is a vector with all output neurons' indexes
                        # the global error for the jjj-th output neuron equals a root of unity corresponding to the
                        # desired output - normalized wightet sum for the corresponding output neuron
                        # jjj contains indexes 1:NumberOfOutputs                                                         
                        neuronErrors[ii] [jjj] = ComplexValuedDesiredOutputs[aa,jjj].T - \
                                                    np.array(neuronOutputs[ii][jjj],dtype=complex)
                        #print(np.array(neuronOutputs[ii][jjj],dtype=complex))

                        # finally we obtain the output neurons' errors normalizing the global errors (dividing them
                        # by the (number of neurons in the preceding layer+1)
                        neuronErrors[ii] = neuronErrors[ii] / (sizeOfMlmvn[ii-1]+1)
    ##---------------------------------------------------------------------------------------
    ##              BACKPROPAGATING ERROR FROM OUTPUT LAYER
    ##---------------------------------------------------------------------------------------
                        # handling the rest of the layers - ERROR BACKPROPAGATION
                        for ii in range (numberOfLayers_1,-1,1):
                            # calculation of the reciprocal weights for the
                            # layer ii and putting them in a vector-row
                            temp = ( 1 / network[ii+1] ).T#'; % .' is used to avoid transjugation
                            # extraction resiprocal weights corresponding only to the inputs (the 1st weight w0 will be
                            # dropped, since it is not needed for backpropagation
                            temp = temp[1:,:]
                            # backpropagation of the weights
                            if ii > 0: # to all hidden layers except the 1st
                                neuronrrors[ii] = (temp * neuronErrors[ii+1]) / (sizeOfMlmvn[ii-1]+1)
                            else: # to the 1st hidden layer
                                neuronErrors[ii] = (temp * neuronErrors[ii+1]) / (inputsPerSample+1)

    ##---------------------------------------------------------------------------------------
    ##              WEIGHT CORRECTION
    ##---------------------------------------------------------------------------------------
    ##
                        #**************************************************
                        # *** NOW CORRECTING THE WEIGHTS OF THE NETWORK *** handling the 1st hidden layer
                        # learning rate is a reciprocal absolute value of the weighted sum
                        learningRate = ( 1 / np.abs( weightedSum[0] ) )
                        # all weights except bias (w0 = w(1) in Matlab)
                        network[0][:,1:] = network[0][:,1:] +(learningRate*neuronErrors[0]).dot(np.conj(inputs[aa,:])[None,:])
                        # bias (w0 = w(1) in Matlab)
                        network[0][:,0][:,None] = np.array(network[0][:,0])[:,None] + learningRate * neuronErrors[0]

                        #correcting following layers
                        for ii in range (1,numberOfLayers):
                            #**********************************************
                            #calculating new output of preceding layer
                            if (ii==1): # if a preceding layer is the 1st one
                                 weightedSum[0] = network[0][:,1:] .dot( (inputs[aa,:])) + network[0][:,0]
                            else:  #if a preceding layer is not the 1st one
                                weightedSum[ii-1] = network[ii-1][:,1:].dot(neuronOutputs[ii-2]) + network[ii-1][:,0]
                            #end

    ##---------------------------------------------------------------------------------------
    ##              APPLYING CONTUNUOUS ACTIVATION FUNCTION FROM FIRST HIDDEN LAYER
    ##---------------------------------------------------------------------------------------
                            #APPLYING CONTINUOUS ACTIVATION FUNCTION
                            #TO THE FIRST HIDDEN LAYER NEURONS - CONTINUOUS OUTPUT CALCULATION
                            neuronOutputs[ii-1] = weightedSum[ii-1] / np.abs(weightedSum[ii-1])							

                            #**********************************************
                            # learning rate is a reciprocal absolute value of the weighted sum					
                            learningRate = 1 / np.abs(weightedSum[ii-1])                         
                            #learningRate not used for the output layer neurons
                            if ii < numberOfLayers_1:
                                #all weights except bias (w0=w(1) in Matlab)
                                a1=network[ii][:,1:]
                                b1=neuronErrors[ii]
                                b1=(learningRate * b1) /(sizeOfMlmvn[ii-1]+1)
                                c1=np.atleast_2d(neuronOutputs[ii-1]).T
                                c1=np.conj(c1)
                                c1=c1.T
                                e1=a1
                                for i1 in range (0,sizeOfMlmvn[ii]):
                                    d1=b1[i1]
                                    e1[i1,:]= d1*c1
                                f1=a1+e1
                                network[ii][:,1:] = f1

                                network[ii][:,0][:,None] = np.atleast_2d(network[ii][:,0]).T + b1#learningRate * neuronErrors[ii]
                            else:  # correction of the output layer neurons' weights    
                                #all weights except bias (w0=w(1) in Matlab)
                                a1=network[ii][:,1:]
                                b1=neuronErrors[ii] /(sizeOfMlmvn[ii-1]+1)
                                c1=np.atleast_2d(neuronOutputs[ii-1]).T
                                c1=np.conj(c1)
                                c1=c1.T
                                e1=np.zeros((a1.shape), dtype=complex)#a1#np.copy(a1)*0

                                for i1 in range(sizeOfMlmvn[ii]):
                                    d1=b1[i1]
                                    e1[i1,:]=d1*c1
                                f1=a1+e1
##                                    if aa == 0:
##                                        print e1
##                                        print f1
                                network[ii][:,1:] = f1
                                # bias (w0 = w(1) in Matlab)
                                #network[ii][:,0][:,None] = np.atleast_2d(network[ii][:,0]).T + (neuronErrors[ii]/(sizeOfMlmvn[ii-1]+1))
                                network[ii][:,0] = np.atleast_2d(network[ii][:,0]).T + b1
##                                    if aa==1:
##                                        print network[ii][:,1:]
##                                        print network[ii][:,0][:,None]
                    if givenInputs[15]:
                        if iterations == maxIter:
                            finishedLearning= True
            print ("learning iterations: ", iterations)
            if save == 1:
                np.save(savefile, network)        

                
        elif desiredAlgo == 5: #TESTING - In this branch only testing is implemented
            
            ## TESTING ALGORITHM
            #--------------------------------------------------------------------------

            ## NET OUTPUT CALCULATION
            #******************************
            # CALCULATING THE OUTPUTS OF THE NETWORK FOR EACH OF THE SAMPLES

            #looping through all samples
            #print network[-1]
            for aa in range(numberOfInputSamples ): #ends 1355
                # *** PROCESSING FIRST LAYER ***
                ii = 0# ( ii holds current layer )
                neuronOutputs[ii] = (network[ii][:,1:].dot(np.atleast_2d(inputs[aa,:]).T)) + np.atleast_2d(network[ii][:,0]).T
                #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                neuronOutputs[ii] = neuronOutputs[ii] / np.abs(neuronOutputs[ii])
                #print(neuronOutputs[ii])         
                # *** PROCESSING FOLLOWING LAYERS ***

                for ii in range(1,numberOfLayers_1):
                    #neuronOutputs[ii] = network[ii][:,1:].dot( neuronOutputs[ii-1]) + network[ii][:,0][:,None]
                    neuronOutputs[ii] = network[ii][:,1:].dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T
                    #print(neuronOutputs[ii])
                    #APPLYING CONTINUOUS ACTIVATION FUNCTION
                    neuronOutputs[ii] = neuronOutputs[ii] / np.abs(neuronOutputs[ii])
                # *** PROCESSING THE OUTPUT LAYER***

                ii = numberOfLayers_1
                neuronOutputs[ii] = np.array(network[ii][:,1:]).dot(neuronOutputs[ii-1]) + np.atleast_2d(network[ii][:,0]).T                
#---------------------------------------------------------------------------------------
##              OUTPUT CALCULATION
##........................................................................................
                ## NETWORK OUTPUT CALCULATION
                #applying the activation function
                if discreteOutput:
                    # --- FOR DISCRETE OUTPUT -- the argument of output in the range [0, 2pi]
                    neuronOutputs[ii] = np.mod( np.angle( neuronOutputs[ii][:] ), pi2 )
                    # this will be the discrete output (the number of sector)
                    neuronOutputs[ii] = np.floor (neuronOutputs[ii]/sectorSize)
                else: # --- FOR CONTINUOUS OUTPUT ---
                    neuronOutputs[ii] = neuronOutputs[ii] / np.abs(neuronOutputs[ii])
                    #print(neuronOutputs[ii])
                networkOutputs[aa,:] = neuronOutputs[ii]
                #if aa==2: print networkOutputs[aa,:]
            #print networkOutputs
    ##--------------------------------------------------------------------------------------
    ##              ANALYSING RESULTS
    ##-------------------------------------------------------------------------------------------
            ## ANALYSYS OF THE RESULTS - NET ERROR CALCULATION
            #**************************************************************
            #CALCULATING GLOBAL ERROR AND COMPARING VALUE TO THRESHOLD
            if givenInputs[10]:
                a = angularGlobalThresholdValue
                if isinstance(a, numbers.Number) and np.isscalar(a):
                    if a < 0:
                        s = 'angularGlobalThresholdValue must be greater than 0'	
                        print('MLMVN:BadInput',s)
                        sys.exit()
                    #end
                else:
                    s = '\n\nangularGlobalThresholdValue must be a scalar numeric value\n\n'
                    print('MLMVN:BadInput',s)
                    sys.exit()
            else:
                if SoftMargins:
                    s = '\n\angularGlobalThresholdValue needed for angular rmse learning\n\n'
                    print( 'MLMVN:BadInput', s )
                    sys.exit()

##            for aa in range (numberOfInputSamples):
    ##--------------------------------------------------------------------------------------
    ##              GLOBAL ERROR CALCULATIONS
    ##-------------------------------------------------------------------------------------------
            # calculation of the: squared error for each sample
            if discreteOutput==1:  # discrete outputs
                errors = np.abs( networkOutputs-desiredOutputs );
                
                # this indicator is needed for  errors which jump over pi)
                indicator = (errors>numberOfSectorsHalf)
                if (np.count_nonzero(indicator)>0):
                    i3 = (indicator==1)
                    mask[i3] = numberOfSectors;
                    errors[i3] =(mask[i3]-errors[i3])#.astype(np.float32)
            else:  #continuous outputs
                #print networkOutputs
                errors = np.abs(np.mod(np.angle(networkOutputs), pi2)-AngularDesiredOutputs)
                # this indicator needed for errors whose "formal" actual jump over pi                
                indicator = (errors>np.pi)
                if (np.count_nonzero(indicator)>0):
                    i3 = (indicator==1)
                    mask[i3] = pi2
                    errors[i3] = (mask[i3]-errors[i3])#.astype(np.float32)
                    #print(errors[i3])
##                    if (indicator(iii,jjj)==1):
##                        errors(iii,jjj)=pi2-errors(iii,jjj);                
                mask=np.zeros((n1,n2))#  resetting of mask
                #if aa==2: print errors
                # converts continuous network outputs from the complex numbers located on the unit circle to their
                # arguments in order to create a meaningful result, which can then be used in output.NetworkOutputs
                networkOutputs = np.mod(np.angle(networkOutputs), pi2)
            #print (networkOutputs)

            ## END OF NET ERRORS CALCULATION
            ## ERRORS EVALUATION
    ##--------------------------------------------------------------------------------------
    ##              IF MORE THAN 1 OUTPUT NODE ANALYSING RESULTS
    ##-------------------------------------------------------------------------------------------
            # Absolute Errors
            if numberOfOutputs>1: #if more than 1 output neuron then we determine max error per output layer
                networkError = np.max(errors[iii,:])#.T # max errors over output neurons
            else:
                networkError = errors # for 1 output neuron networkError = errors
                #print networkError
            
            indicator = (networkError>localThresholdValue)
            AbsErrors = np.sum(indicator)
            Accuracy = 100-(AbsErrors/numberOfInputSamples)*100
            

            
            #RMSE
            if numberOfOutputs>1:
                networkErrors = np.sum(errors[iii,:].T**2)/numberOfOutputs  
            else:
                #print networkErrors
                networkErrors = errors**2
            #calculating combined mse of all samples
            mse = np.sum( networkErrors ) / numberOfInputSamples
            
            rmse = np.sqrt( mse )
            # Output of the testing (evaluation) results

            print('Errors',AbsErrors, 'Accuracy = \n',  Accuracy)
            print('RMSE = \n', rmse)
            #**************************************************************
            #*************************************************************

        ## END OF TESTING ALGORITHM
        else: #something weird happened, control is not supposed to reach here
            s = 'desiredAlgo has somehow acquired a strange value'
            print('\n\n', s, '\n\n')
            print('MLMVN:ControlPathError', s)
            sys.exit()
    #end if softmargins

    ## RETURNING OF RESULTS

    ###copying the weights of the network to the output variable
    output = {}
    output['network'] = network;
    if (stoppingCriteria=='test'): # output the evaluation results for testing
        variables = ["DesiredOutputs", "NetworkOutputs", "AbsoluteError", "Accuracy", "RMSE"]
        results = [desiredOutputs, networkOutputs, AbsErrors, Accuracy, rmse]
        for i in range(len(variables)):
            output[variables[i]] = results[i]
    else:
        output['iterations'] = iterations # output: the number of iterations for learning

    if desiredAlgo == 5:
        return output
    else:
        return network

##
##savefile = ("/home/joshua/Documents/python_codes/complex_valued_neuralnetwork-master/Networks/Net2.npy")
###data = np.loadtxt("mnist_dataset/tested.txt")
##testfile = "/home/joshua/Documents/python_codes/complex_valued_neuralnetwork-master/iris_dataset/iris_test_conv_20.csv"
##trainfile = "/home/joshua/Documents/python_codes/complex_valued_neuralnetwork-master/iris_dataset/iris_train_conv_80.csv"
##
##
##
##data = np.array([[-0.4639-0.8859j, -0.5048+0.8632j, 0.76],
##    [ 0.5872-0.8094j, 0.3248+0.9458j, 2.56],
##    [-0.5057+0.8627j, 1+0j , 5.35]])
##varargin = ('sizeOfMlmvn', [2,1], 'inputs', data, 'stoppingCriteria', 'mse', 'discreteInput', 0, \
##'discreteOutput', 1, 'globalthresholdvalue', 0.00025, 'localThresholdValue', 0.05, 'initialWeights','random', \
##            'SoftMargins', 1, 'angularGlobalThesholdValue', 0.5, 'angularLocalThresholdValue', 0,\
##            'numberOfSectors', 1, 'save', 1 ,'maxIteration',20, "save",1, "compInputs", 1)
##
##Weights = MLMVN(varargin)
##
##
##varargin = ('network', savefile, 'inputs', data, 'stoppingCriteria', 'test', 'discreteInput', 0, \
##'discreteOutput', 1, 'globalthresholdvalue', 0.5, 'localThresholdValue', 0.05, \
##            'numberOfSectors', 1, 'save', 0 , "compInputs", 1)
##Results = MLMVN(varargin)


inpfile = 'iris_dataset/iris.csv'
epsi= .284
_, data, tedata= conv_data_cnt_dsc (inpfile, epsi, True, True, 0)

##print data.shape
##print tedata.shape
#data = pd.read_csv(trainfile,  sep=",", header =-1)
data = np.array(data)
varargin = ('sizeOfMlmvn', [3,1], 'inputs', data, 'stoppingCriteria', 'mse', 'discreteInput', 0, \
'discreteOutput', 1, 'globalthresholdvalue', 0, 'localThresholdValue', 0,  \
    'angularGlobalThresholdValue', 0.00001, 'angularLocalThresholdValue', 0.01, 'initialWeights','random', \
            'SoftMargins', 1, 'angularGlobalThesholdValue', 0.5, 'angularLocalThresholdValue', 0,\
            'numberOfSectors', 9, 'save', 1 ,'maxIteration', 20000, "save",1, "compInputs", 0)

Weights = MLMVN(varargin)

#tedata = pd.read_csv(testfile,  sep=",", header =-1)
tedata = np.array(tedata)

vararginT = ('network', savefile, 'inputs', tedata, 'stoppingCriteria', 'test', 'discreteInput', 0, \
'discreteOutput', 1, 'globalthresholdvalue', 0.4, 'localThresholdValue', 0.25, \
            'numberOfSectors', 6, 'save', 1 , "compInputs", 0)
Results = MLMVN(vararginT)

