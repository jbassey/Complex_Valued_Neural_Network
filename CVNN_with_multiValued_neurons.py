from __future__ import division
import numpy as np
import cmath
import math
#from itertools import imap
import sympy
from sympy import symbols, diff
from scipy.fftpack import fft, ifft
import timeit
#np.random.seed(60)

import sys
import pandas as pd
import numbers
from mvn_inpdata_converter_cnt_dsc import conv_data_cnt_dsc
from converterTest import conv_data
from utils import get_weights, check_args

class neuralNetwork:
    
    def __init__(self, inputnodes, layers, outputnodes, cats, periods, discrete, stop, gThresh, lThresh):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes  # increament for bias node
        #self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.layers=layers
        self.discreteOutput = discrete
        self.globalThresholdValue = gThresh
        self.localThresholdValue = lThresh
        
        
        self.numberOfLayers=len(layers)
        #numberOfLayers_1= self.numberOfLayers-1
        
        self.weights = []
        weightedSum=[]
        weightedBias=[]
        self.bias=[]
        self.dW = []
        self.dB = []

        r = np.random.uniform(-0.5, 0.5,(self.layers[0] , self.inodes))
        #r = np.random.normal(0.0, pow(1.0, -0.5),(self.layers[0] , self.inodes))
        r = np.array(r, ndmin=2, dtype='complex128')
        r += 1j * np.random.uniform(-0.5, 0.5, (self.layers[0], self.inodes))
        #r += 1j * np.random.normal(0.0, pow(1.0, -0.5),(self.layers[0] , self.inodes))

        r1 = np.random.uniform(-0.5, 0.5,(self.layers[0],1))
        #r = np.random.normal(0.0, pow(1.0, -0.5),(self.layers[0] , 1))
        r1 = np.array(r1,  dtype='complex128')
        r1 += 1j * np.random.uniform(-0.5, 0.5, (self.layers[0],1))
        #r += 1j * np.random.normal(0.0, pow(1.0, -0.5),(self.layers[0] , 1))

        #r1 = np.zeros((layers[0], inodes))
        self.weights.append(np.round(r, 5))
        self.bias.append(np.round(r1, 5))
        self.dW.append(np.round(r*0, 5))
        self.dB.append(np.round(r1*0, 5))
        if len(layers) > 1:  
            for i in range(1, len(layers) - 1):
                #r = np.random.normal(0.0, pow(1.0, -0.5),(self.layers[i], self.layers[i-1]))
                r = np.random.uniform(-0.5, 0.5,(self.layers[i], self.layers[i-1]))
                r = np.array(r, ndmin=2, dtype='complex128')
                #r += 1j * np.random.normal(0.0, pow(1.0, -0.5), (self.layers[i], self.layers[i-1]))
                r += 1j * np.random.uniform(-0.5,0.5, (self.layers[i], self.layers[i-1]))

                r1 = np.random.uniform(-0.5, 0.5,(self.layers[i] , 1))
                #r1 = np.random.normal(0.0, pow(1.0, -0.5), (self.layers[i], 1))
                r1 = np.array(r1, ndmin=2, dtype='complex128')
                r1 += 1j * np.random.uniform(-0.5, 0.5, (self.layers[i], 1))
                #r1 += 1j * np.random.normal(0.0, pow(1.0, -0.5), (self.layers[i], 1))
                
                self.weights.append(np.round(r, 5))
                self.bias.append(np.round(r1, 5))
                self.dW.append(np.round(r*0, 5))
                self.dB.append(np.round(r1*0, 5))
        i=len(layers)-1
        #r = np.random.normal(0.0, pow(1.0, -0.5), (self.layers[i], self.layers[i-1]))
        r = np.random.uniform(-0.5,0.5, (self.layers[i], self.layers[i-1]))
        r = np.array(r, ndmin=2, dtype='complex128')
        #r += 1j * np.random.normal(0.0, pow(1.0, -0.5), (self.layers[i], self.layers[i-1]))
        r = 1j * np.random.uniform(-0.5,0.5, (self.layers[i], self.layers[i-1]))

        r1 = np.random.uniform(-0.5, 0.5,(self.layers[i] , 1))
        #r1 = np.random.normal(0.0, pow(1.0, -0.5), (self.layers[i], 1))
        r1 = np.array(r1, ndmin=2, dtype='complex128')
        r1 += 1j * np.random.uniform(-0.5, 0.5, (self.layers[i], 1))
        #r1 += 1j * np.random.normal(0.0, pow(1.0, -0.5), (self.layers[i], 1))


        #r1 = np.zeros((layers[i], layers[i-1]))
        self.weights.append(np.round(r, 5))
        self.bias.append(np.round(r1, 5))
        self.dW.append(np.round(r*0, 5))
        self.dB.append(np.round(r1*0, 5))
        

##        self.weights = []
##        h = np.array([[0.19-0.46j, 0.36-0.33j], [0.19-0.46j, 0.36-0.33j]])
##        h = np.array(h, ndmin=2, dtype='complex128')
##        self.weights.append(np.round(h, 5))
##        ho = np.array([[0.19-0.46j,0.36-0.33j ]])#.T        
##        ho = np.array(ho, ndmin=2, dtype='complex128')
##        self.weights.append(np.round(ho, 5))
##
##        self.bias=[]
##        bb = np.array([[0.23-0.38j],[0.23-0.38j]])
##        bb = np.array(bb, ndmin=2, dtype='complex128')
##        self.bias.append(np.round(bb, 5))
##        bb1= np.array([[0.23-0.38j]])
##        bb1 = np.array(bb1, ndmin=2, dtype='complex128')
##        self.bias.append(np.round(bb1, 5))
        #print self.weights[0].shape
            
        # number of output class categories
        self.categories = cats
        
        # todo periodicity
        self.periodicity = periods

        self.weightedSum=[]
        self.neuronOutputs=[]
        self.neuronErrors =[]
        
        for j in range(self.numberOfLayers):
            self.weightedSum.append(np.zeros((self.layers[j],1), dtype=complex))
            self.neuronOutputs.append(np.zeros((self.layers[j],1), dtype=complex))
            self.neuronErrors.append(np.zeros((self.layers[j],1), dtype=complex))
##        networkOutputs = np.zeros((numberOfInputSamples, self.layers[-1]), dtype=complex)
##        networkErrors = np.zeros((numberOfInputSamples, self.layers[-1]), dtype=complex)

        if stop == "mse":
            self.criteria =0
        elif stop == "rmse":
            self.criteria =1
        
        pass
    
    def z_to_class(self, z):       
        # first work out the angle, but shift angle from [-pi/2, +pi.2] to [0,2pi]
        angle = np.mod(np.angle(z) + 2*np.pi, 2*np.pi)
        # from angle to category
        #p = int(np.floor (self.categories * angle / (2*np.pi)))
        p = int(np.floor (self.categories * self.periodicity * angle / (2*np.pi)))
        p = np.mod(p, self.categories)
        return p#result

    def class_to_angle(self, c):
        angle = (c + 0.5 + (self.categories * np.arange(self.periodicity))) / (self.categories * self.periodicity) * 2 * np.pi
        #print angle
        return angle
    
    def status(self):
##        print ("self.wih = ", self.wih)
##        print ("self.who = ", self.who)##        
        pass
    def angle(self,z):
        angle = np.mod(np.angle(z) + 2*np.pi, 2*np.pi)
        return angle

    def query(self, inputs_list):
        numberOfInputSamples=len(inputs_list)
        numberOfLayers_1= self.numberOfLayers-1
        self.networkOutputs = np.zeros((numberOfInputSamples, self.layers[-1]), dtype=complex)
        self.networkErrors = np.zeros((numberOfInputSamples, self.layers[-1]), dtype=complex)
          
        ## NETWORK OUTPUT CALCULATION
        a= np.exp(1j*np.atleast_2d(inputs_list).T)
        ## NETWORK OUTPUT CALCULATION
        ii = 0      # - begining from first layer
        self.weightedSum[ii] = np.dot(self.weights[ii],a) + self.bias[ii]
        self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / np.abs(self.weightedSum[ii]), 5)
        #if aa==0: print self.neuronOutputs[ii].shape
        for ii in range( 1,numberOfLayers_1): #Other layers exept for final layer
            self.weightedSum[ii] = np.dot(self.weights[ii], self.neuronOutputs[ii-1]) + self.bias[ii]
            self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / np.abs(self.weightedSum[ii]), 5)

        ii = numberOfLayers_1   #final layer
        self.weightedSum[ii] = np.dot(self.weights[ii], self.neuronOutputs[ii-1]) + self.bias[ii]
        ## NETWORK OUTPUT CALCULATION - applying the activation function
        if discreteOutput:
            self.neuronOutputs[ii]= np.round(np.exp(1j * np.angle(self.weightedSum[ii])), 5)

        else:
            self.neuronOutputs[ii]= np.round(np.exp(1j * np.angle(self.weightedSum[ii])), 5)
            #self.neuronOutputs[ii] =np.round( self.weightedSum[ii] / np.abs(self.weightedSum[ii]), 5)

        if discreteOutput:
            output_classes = self.z_to_class(self.neuronOutputs[ii])
        else:
            output_classes = self.angle(self.neuronOutputs[ii])
        
        #print output_classes
        return output_classes

    def activation(self, value):
        #u_xy = numpy.sinh(value.real)
        #v_xy = numpy.sinh(value.imag)
        act_fxn = numpy.tan(value)#u_xy + v_xy
        return act_fxn
    
    def mat_rep(self, f):
        g = numpy.array([[f.real, -f.imag], [f.imag, f.real]])
        return g
    
    def train(self, inputs_list, target,maxEpochs):
        lamb=0.001 +0.001j
        numberOfInputSamples=len(inputs_list)
        numberOfLayers_1= self.numberOfLayers-1
        numberOfOutputs = self.layers[-1]
        jjj=range (self.weights[-1].shape[0])
        iii=range(numberOfInputSamples)
        self.networkOutputs = np.zeros((numberOfInputSamples, self.layers[-1]), dtype=complex)
        self.networkErrors = np.zeros((numberOfInputSamples, self.layers[-1]), dtype=complex)
         
        ## Obtain targets
        desired_angles = self.class_to_angle(np.atleast_2d(target).T)
##        print desired_angles
##        print self.z_to_class(np.atleast_2d(np.exp(1j*desired_angles[4,0])).T)
        if discreteOutput:
            targets = np.round(np.exp(1j * desired_angles), 4)
        else:
            targets =  np.round(np.exp(1j * np.atleast_2d(target).T), 4)

    
        finishedLearning = False
        iterations = 0
        while not finishedLearning:
            iterations = iterations + 1
            if iterations % 1000 ==0:
                print("iteration: ", iterations)
            for aa in range (numberOfInputSamples):
                             
                a= np.exp(1j*np.atleast_2d(inputs_list[aa]).T)
                #if aa==0: print a.shape
                ## NETWORK OUTPUT CALCULATION
                ii = 0      # - begining from first layer
                self.weightedSum[ii] = np.dot(self.weights[ii],a) + self.bias[ii]
                self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / np.abs(self.weightedSum[ii]),5)
                #self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / 1+np.abs(self.weightedSum[ii]), 5)
                for ii in range( 1,numberOfLayers_1): #Other layers exept for final layer
                    self.weightedSum[ii] = np.dot(self.weights[ii], self.neuronOutputs[ii-1]) + self.bias[ii]
                    self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / np.abs(self.weightedSum[ii]), 5)
                    #self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / 1+np.abs(self.weightedSum[ii]), 5)
                    
                ii = numberOfLayers_1   #final layer
                self.weightedSum[ii] = np.dot(self.weights[ii], self.neuronOutputs[ii-1]) + self.bias[ii]

                ## NETWORK OUTPUT CALCULATION - applying the activation function
                if discreteOutput:
                    self.neuronOutputs[ii]=np.exp(1j * np.angle(self.weightedSum[ii]))#[0]
                else:
                    self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / np.abs(self.weightedSum[ii]),5)#[0],5)
                    #self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / 1+np.abs(self.weightedSum[ii]), 5)

                self.networkOutputs[aa,:] = self.neuronOutputs[ii]#[0]
 
            if discreteOutput==1:  # discrete output5
                errors = np.round(np.abs(self.angle(self.networkOutputs) - desired_angles ), 5)
                errors = np.round(errors.min(axis=1),5)#[:,np.argmin(np.abs(errors),axis=1) ], 5)
            else:  # continuous outputs               
                errors = np.round(np.abs(self.angle(self.networkOutputs)- np.atleast_2d(target).T), 5)

            
            self.networkErrors = np.sum(errors**2)/numberOfOutputs
            mse = np.sum( self.networkErrors ) / numberOfInputSamples

            if iterations % 100 ==0: print ("mse at iteration", iterations, " =  ", mse)
            
            if self.criteria == 1:#"rmse":
                rmse = np.sqrt(mse)
                if rmse <= self.globalThresholdValue:
                    print ("Learning Completed")
                    print('Iter', iterations, ', RMSE\n', rmse)
                    finishedLearning = True
            elif self.criteria == 0:      
                if mse <= self.globalThresholdValue:
                    print ("Learning Completed")
                    print('Iter', iterations, ', MSE\n', mse)
                    finishedLearning = True
      

            if  not finishedLearning:           
                for aa in range (numberOfInputSamples): #end 1704
                    #i =np.random.randint(inputs_list.shape[0])
                    #a =inputs_list[i]
                    a= np.exp(1j*np.atleast_2d(inputs_list[aa]).T)
                    m = a.shape[1]
                    #if aa==0: print a
                    
                    desired_angle = self.class_to_angle(np.atleast_2d(target[aa]).T)
                    
                    if discreteOutput:
                        targ = np.exp(1j * desired_angle)
                    else:
                        targ =  np.round(np.exp(1j * np.atleast_2d(target[aa]).T), 6)
                        
                    ## NETWORK OUTPUT CALCULATION
                    ii = 0      # - begining from first layer
                    self.weightedSum[ii] = np.dot(self.weights[ii],a) + self.bias[ii]
                    self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / np.abs(self.weightedSum[ii]), 6)
                    #self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / 1+np.abs(self.weightedSum[ii]), 5)
                    for ii in range( 1,numberOfLayers_1): #Other layers exept for final layer
                        self.weightedSum[ii] = np.dot(self.weights[ii], self.neuronOutputs[ii-1]) + self.bias[ii]
                        self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / np.abs(self.weightedSum[ii]), 6)
                        #self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / 1+np.abs(self.weightedSum[ii]), 5)
                    ii = numberOfLayers_1   #final layer
                    self.weightedSum[ii] = np.dot(self.weights[ii], self.neuronOutputs[ii-1]) + self.bias[ii]
                    ## NETWORK OUTPUT CALCULATION - applying the activation function
                    if discreteOutput:
                        self.neuronOutputs[ii]= np.round(np.exp(1j * np.angle(self.weightedSum[ii])), 6)

                    else:
                        self.neuronOutputs[ii] =np.round( self.weightedSum[ii] / np.abs(self.weightedSum[ii]), 6)
                        #self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / 1+np.abs(self.weightedSum[ii]), 5)


                    self.networkOutputs[aa,:] = self.neuronOutputs[ii]#[0]
                    #if aa==0: print ("out1: ", self.networkOutputs[aa,:])

                    if discreteOutput:  # discrete outputs
                        #error = np.round(np.atleast_2d(target[aa]).T -  self.networkOutputs[aa,:], 5)
                        error = np.round(np.abs(self.angle(self.networkOutputs[aa,:]) - desired_angle), 6)
##                        if aa==0:
##                            print ("out, error1: ", self.angle(self.networkOutputs[aa,:]), desired_angle)
                                   
                        error = np.round(error.min(),6)
##                        if aa==50:
##                            print ("error2: ", error)
                        #np.round(np.abs( self.networkOutputs[aa,:] - desired_angles[aa] ), 5)
                    else:  # continuous outputs
                        error = np.round(np.abs(np.mod(np.angle(self.networkOutputs[aa,:]), np.pi*2)- np.atleast_2d(target[aa]).T), 6)
                
                    # select smallest error
                        
                    networkError = error#np.abs(e)#np.max(e[aa,:])
                    if (aa ==17) and (iterations % 50 ==0):
                        print ("error: ",self.angle(self.networkOutputs[aa,:]), networkError, self.z_to_class(self.networkOutputs[aa,:])) 
                        #print (self.z_to_class(np.atleast_2d(targ[0,2]).T))
                    if (aa==31) and (iterations % 50 ==0):
                        print ("error: ",self.angle(self.networkOutputs[aa,:]), networkError, self.z_to_class(self.networkOutputs[aa,:]))
                        #print (self.z_to_class(np.atleast_2d(targ[0,2]).T))
                    if (aa==41) and (iterations % 50 ==0):
                        print ("error: ",self.angle(self.networkOutputs[aa,:]), networkError, self.z_to_class(self.networkOutputs[aa,:]))
                        #print (self.z_to_class(np.atleast_2d(targ[0,2]).T))
                    if (aa==46) and (iterations % 50 ==0):
                        print ("error: ",self.angle(self.networkOutputs[aa,:]), networkError, self.z_to_class(self.networkOutputs[aa,:]))
                        #print (self.z_to_class(np.atleast_2d(targ[0,2]).T))
                    if (aa==55) and (iterations % 50 ==0):
                        print ("error: ",self.angle(self.networkOutputs[aa,:]), networkError, self.z_to_class(self.networkOutputs[aa,:]))
                    if (aa==97) and (iterations % 50 ==0):
                        print ("error: ",self.angle(self.networkOutputs[aa,:]), networkError, self.z_to_class(self.networkOutputs[aa,:]))
                        print (self.z_to_class(np.atleast_2d(targ[0,0]).T))

                    if (networkError) > self.localThresholdValue or \
                        self.networkOutputs[aa,:] != targ[:,np.argmin(targ - self.networkOutputs[aa,:])]:
                        self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / np.abs(self.weightedSum[ii]), 6)#[0]
                        #self.neuronOutputs[ii] = np.round(self.weightedSum[ii] / 1+np.abs(self.weightedSum[ii]), 5)
                        err = np.round(targ - self.neuronOutputs[ii],5)#self.networkOutputs[aa,:], 5)
                        self.neuronErrors[ii][jjj]= np.round(err.min(axis=1),5)#**2
                        self.neuronErrors[ii] = np.round(self.neuronErrors[ii] / (self.layers[ii-1]+1), 6)

                        # ERROR BACKPROPAGATION#
                        for ii in range (numberOfLayers_1-1,-1,-1):
                            #temp = np.round(( 1/ self.weights[ii+1] ).T, 6)#
                            temp = (np.conj(self.weights[ii+1])/(np.abs(self.weights[ii+1])**2)).T
                            if ii > 0:
                                self.neuronErrors[ii] =np.round( np.dot(temp, self.neuronErrors[ii+1]) / (self.layers[ii-1]+1), 6)                                   
                            else:    # to the 1st hidden layer
                                self.neuronErrors[ii] = np.round(np.dot(temp, self.neuronErrors[ii+1]),6)# / (self.inodes+1), 6)

                        ##WEIGHT CORRECTION
                        learningRate = (1 / np.abs(self.weightedSum[0]))#np.round(, 5)
                        self.dW[0] = np.round(learningRate*np.dot(self.neuronErrors[0], (np.conj(a)).T),6)
                        self.weights[0] += np.round(self.dW[0], 6)
                        self.dB[0] = np.round(learningRate*self.neuronErrors[0], 6)
                        self.bias[0] += np.round(self.dB[0], 6)
                        #handling the following layers
                        for ii in range (1,self.numberOfLayers):
                            if (ii==1): # if a preceding layer is the 1st one
                                self.weightedSum[0] = np.round(np.dot(self.weights[0],a) + self.bias[0], 6)
                            else: # if a preceding layer is not the 1st one
                                self.weightedSum[ii-1] = np.round(np.dot(self.weights[ii-1], self.neuronOutputs[ii-2]) + self.bias[ii-1], 6)
                            
                            #APPLYING CONTINUOUS ACTIVATION FUNCTION TO THE FIRST HIDDEN LAYER NEURONS
                            self.neuronOutputs[ii-1] = np.round(self.weightedSum[ii-1] / np.abs(self.weightedSum[ii-1]), 6)

                            learningRate = np.round(1 / np.abs(self.weightedSum[ii]), 5)
                            if ii < numberOfLayers_1:
                                reg = lamb*self.weights[ii]
                                b = np.round(learningRate*self.neuronErrors[ii]/(self.layers[ii-1]+1), 6)
                                #)#(learningRate * self.neuronErrors[ii]) /(self.layers[ii-1]+1)
                                self.dW[ii] = np.round(np.dot(b,np.conj(self.neuronOutputs[ii-1]).T) , 6)
                                self.dB[ii] = b
                                self.weights[ii] += self.dW[ii]
                                self.bias[ii] += self.dB[ii]
                            else:
                                reg = lamb*self.weights[ii]
                                b = np.round(self.neuronErrors[ii]/(self.layers[ii-1]+1), 6)
                                self.dW[ii] = np.round(np.dot(b,np.conj(self.neuronOutputs[ii-1]).T) , 6)#+ 
                                self.dB[ii] = b
                                self.weights[ii] += self.dW[ii]
                                self.bias[ii] += self.dB[ii]
                        #print self.weights[-1]
                        
                if iterations == maxEpochs:
                    finishedLearning= True
            #finishedLearning= True            
    pass



# number of input, hidden and output nodes
input_nodes = 784#59#68#315#
#hidden_nodes = 4#2
output_nodes = 1

# categories, periodicity
categories = 10#10#3#
periodicity = 3 #1
discreteOutput = True#False#
stopCriteria= "mse"
globalThreshold = 0.0001#0.001
localthreshold = 0.05#0.02#0.05

# create instance of neural network
n = neuralNetwork(input_nodes,[1000,1],output_nodes, categories, periodicity,discreteOutput,\
                  stopCriteria, globalThreshold, localthreshold)
n.status()
##main = "/home/joshua/Documents/python_codes/complex_valued_neuralnetwork-master/mnist_dataset/mnist_mini.csv"
##testSize=0.0909
##header = -1
##batchsize=22
##full,train,test = conv_data (main, 0.284, True, True, header, testSize, batchsize)
##print test[:,-1]
##
##print train.shape

epochs = 100#00
##data1 = np.array([[-0.4639-0.8859j, -0.5048+0.8632j, 0.76],
##    [ 0.5872-0.8094j, 0.3248+0.9458j, 2.56],
##    [-0.5057+0.8627j, 1+0j , 5.35]])
##print data1[:,-1].real.shape
##n.train (data1[:,0:-1], data1[:,-1].real, epochs)
##data = np.array([[-1.0, -1.0, 0],[-1.0, 1.0, 1],[1.0, -1.0, 2],[1.0, 1.0, 0]])


epsi= 0.284#014
##inpfile = 'iris_dataset/iris.csv'
##### FOR IRIS
##_, train, test= conv_data_cnt_dsc (inpfile, epsi, True, True, 0, 0.2)
##data1 = np.array(train)
##n.train (data1[:,0:-1], data1[:,-1], epochs)


# FOR MNIST
mini = "/home/joshua/Documents/python codes/complex_valued_neuralnetwork-master/mnist_dataset/mnist_train_100.csv"
main = "/home/joshua/Documents/python_codes/complex_valued_neuralnetwork-master/mnist_dataset/mnist_main.csv"
testSize=(1/11)#0909
header = -1
batch = 22#0
#_,train,test = conv_data_cnt_dsc (main, epsi, True, True, header, testSize)
_,train,test = conv_data (mini, epsi, True, True, header, testSize, batch, False)
data1 = np.array(train)
test = train[31:51,:]



start_time = timeit.default_timer()
n.train (data1[:,0:-1], data1[:,-1], epochs)
elapsed = timeit.default_timer() - start_time
print ("training time: ", elapsed)

##
#print np.array(test)
scorecard = []
#for idx, data in data:#iris_test.iterrows():
for data in np.array(test):
    data_list = data.tolist()
    correct_species = data_list[-1]#int(data_list[-1])
    lengths = data_list[0:-1]
    answer = n.query(lengths)
    print(correct_species, answer)
    #print np.abs(float(answer) - float(correct_species))
    if (np.abs(float(answer) - float(correct_species))) <= localthreshold :
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass

# calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)


