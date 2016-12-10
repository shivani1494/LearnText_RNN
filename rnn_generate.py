import cPickle, gzip
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import random

class RNN:
	 
	def __init__(sf):

		#what should take more time over the same text -> longer or shorter sequences?
		sf.learningRate = 0.01 #best learning rate

		#why does increase in hidden unit affect the accuracy badly? 
		#and create repitions?
		sf.input_dim = 256
		sf.hidden_dim = 120
		sf.output_dim = 256
		sf.temp = 0.3

		sf.allData = []
		sf.label = []

		sf.sequenceLen = 10 #best sq len if we change the alpha and this then it starts repeating

		sf.buildInputData()
		sf.buildLabelData()

		sf.numEpochs = 100
		sf.datasetLen = 20000
		#sf.testDataLen = 100

		#adagrad
		sf.fudge_fac = 1e-6
		#keeps a running squares of gradients
		sf.grad_I_Output = []
		sf.grad_I_Input = []
		sf.grad_I_HH = []

		sf.yIJ = np.array([])
		sf.yJK = []
		
		sf.gradDescOutput = []
		sf.gradDescInput = []
		sf.gradDescHH = []

		sf.weightsJK = []
		sf.weightsIJ = []
		sf.weightsHH = []

		sf.hiddenActivation = []

		sf.target = []
		sf.charset = []
		sf.asciiLen = 256
		
		for i in range(sf.asciiLen):
			sf.charset.append(0)

		sf.charset = np.array(sf.charset)
		sf.buildTarget()
		sf.populateWeights()

		sf.training_loss_I = []

	def buildInputData(sf):
		f = open("world_war_data.txt", 'r')

		sf.allData = []     
		#print "*********************************************************"
		while(True):
			chunk = f.read(sf.sequenceLen)
			if( not chunk ):
				break
			sf.allData.append( chunk )
		#print "*********************************************************"
		
		f.close()

	def buildLabelData(sf):
		f = open("world_war_test.txt", 'r')
		
		#f.read(1)
		sf.label = []
		#print "*********************************************************"
		while(True):
			chunk = f.read(sf.sequenceLen)
			if( not chunk ):
				break
			sf.label.append( chunk )
		#print "*********************************************************"
		f.close()

	def buildTarget(sf):
		target_one_hot = []
		for i in range(sf.output_dim):
			target_one_hot.append(0)

		for i in range(sf.output_dim):
			sf.target.append(list(target_one_hot))

		for i in range(sf.output_dim):
			sf.target[i][i] = 1

		sf.target = np.array(sf.target)

	def populateWeights(sf):

		#input->hidden layer
		weightsPerClassIJ = []

		for i in range(sf.hidden_dim):
			mu, sigma = 0, 1.0/math.sqrt(256.0) #mean and standard deviation
			weightsPerClassIJ = np.random.normal(mu, sigma, sf.input_dim)
			sf.weightsIJ.append(list(weightsPerClassIJ))

		sf.weightsIJ = np.array(sf.weightsIJ)

		#hidden - output layer
		weightsPerClassJK = []

		for i in range(sf.output_dim):
			mu, sigma = 0, 1.0/math.sqrt(120.0) # mean and standard deviation
			weightsPerClassJK = np.random.normal(mu, sigma, sf.hidden_dim)
			sf.weightsJK.append(list(weightsPerClassJK))
		#converting into numpy array for multiplication
		sf.weightsJK = np.array(sf.weightsJK)

		#hidden - output layer
		weightsPerClassHH = []

		for i in range(sf.hidden_dim):
			mu, sigma = 0, 1.0/math.sqrt(120.0) # mean and standard deviation
			weightsPerClassHH = np.random.normal(mu, sigma, sf.hidden_dim)
			sf.weightsHH.append(list(weightsPerClassHH))
		#converting into numpy array for multiplication
		sf.weightsHH = np.array(sf.weightsHH)

	def setHiddenActivation(sf, start):    
		
		#append zeros for all timesteps
		hh_act_timestep = []
		for h in range(sf.hidden_dim):
			hh_act_timestep.append(0.0)

		prev_act = []
		if not start:
			prev_act = list(sf.hiddenActivation[sf.sequenceLen-1])
		else:
			#first time, no activations yet
			prev_act = list(hh_act_timestep)

		hh_act_temp = []
		for t in range(sf.sequenceLen):
			hh_act_temp.append( list(hh_act_timestep) )
		hh_act_temp.append(list(prev_act))

		sf.hiddenActivation = np.array( list(hh_act_temp) )
			   
	def generate(sf):
		
		calc_type = 1

		print "*** testing- generating sequences ***"

		str1 = ""		
		for a in range(50):

			#asciiChar = str1[-1]
			asciiChar = chr( random.randint(0,255) )
			sf.setHiddenActivation(False)
			
			for t in range( sf.sequenceLen ):
								
				sf.calc_y_hidden_layer(asciiChar, t, calc_type)
				
				p = sf.calc_y_softmax_output_layer(0, t, calc_type)
				asciiInt = np.random.choice(sf.asciiLen, p=np.ravel(p) )
				asciiChar = chr(asciiInt)
				str1 += asciiChar

		print str1

		print "*** testing- generating sequences ***"		

	def calc_y_hidden_layer(sf, currData, timestep, calc_type):
	
		if calc_type == 0:
			asciiChar = ord(sf.allData[currData][timestep])
			sf.charset[asciiChar] = 1
		if calc_type == 1:          
			asciiChar = ord(currData)
			sf.charset[asciiChar] = 1

		#print "charset: ", sf.charset
		sf.yIJ = []    

		sf.yIJ = np.tanh( np.dot( sf.weightsHH, sf.hiddenActivation[timestep - 1] ) + np.dot( sf.weightsIJ, sf.charset) )   
		
		sf.hiddenActivation[timestep] = sf.yIJ
		
		sf.charset[asciiChar] = 0

		#if timestep - 1 == -1:
		#print "calc_y_hidden_layer: ", sf.hiddenActivation[timestep-1][1]

		#print "hiddenActivation[timestep]: ", sf.hiddenActivation[timestep]
		#print sf.yIJ
		#print "calc_y_hidden_layer: ", currData
		#print "yIJ", sf.yIJ.shape
		#print "calc_y_hidden_layer: yIJ", yIJ
		
	def calc_y_softmax_output_layer(sf, targetIndx, timestep, calc_type):
		
		sf.yJK = []
		netJK = []

		#weighted sum
		netJK = np.dot(sf.weightsJK, sf.hiddenActivation[timestep])
		netSum = 0.0  
		for i in range(sf.output_dim):
			try:
				if calc_type == 1:
					netSum += math.exp(netJK[i]/sf.temp)
				else:	
					netSum += math.exp(netJK[i])
			except OverflowError:
				netSum = float('inf')
		
		#fix softmax
		if calc_type == 1:
			for i in range(sf.output_dim):        
				sf.yJK.append( float(math.exp(netJK[i]/sf.temp) ) / ( float(netSum) ) )

		if calc_type == 0:
			for i in range(sf.output_dim):        
				sf.yJK.append( float(math.exp(netJK[i])) /( float(netSum) ) ) #adding temperature

		sf.yJK = np.array(sf.yJK)

		#print sf.yJK
		#print "calc_y_softmax_output_layer: yJK", yJK
		#print len(yJK)
		'''
		if calc_type == 0:
			print "expctd res: ", targetIndx
			print "res prob-softmax: ", sf.yJK[targetIndx]

		print "recvd char- ascii value: ", np.argmax(sf.yJK)
		print "chr: ",chr(np.argmax(sf.yJK))
		print "recvd prob: ", np.max(sf.yJK)
		'''
		return sf.yJK

	def forward_back_propogation(sf):

		training_acc_epochs = []
		for j in range(sf.numEpochs):

			print "------------------- start of epoch: ", j+1, "-------------------"

			sf.grad_I_Output = []
			sf.grad_I_Input = []
			sf.grad_I_HH = []

			sf.forward_back_prop_single_epoch(j+1)
			sf.generate()

			print "------------------- end of epoch: ", j+1, "-------------------"
			#training_acc_epochs.append(training_acc)
			#print "training_acc_epochs: ", training_acc_epochs    
			#return training_acc_epochs 

	def resetParameters(sf, i):
		
		sf.gradDescOutput = []
		sf.gradDescInput = []
		sf.gradDescHH = []
		
		if i == 0:
			sf.setHiddenActivation(True)
		else:
			sf.setHiddenActivation(False)
		

	def forward_prop(sf, i, j, calc_type, targetIndx):
		sf.calc_y_hidden_layer(i, j, calc_type)
		sf.calc_y_softmax_output_layer(targetIndx, j, calc_type)
	
	def backward_prop(sf, currData, timestep, targetIndx):
		delta_K = sf.calc_deltaK_gradient_descent_output_layer(targetIndx, timestep)
		sf.bptt(delta_K, timestep, currData)     
	
	#sgd-weight update, does not work as well
	def weight_update(sf):
		sf.weightsIJ += np.dot( sf.learningRate, sf.gradDescInput )
		sf.weightsJK += np.dot( sf.learningRate, sf.gradDescOutput )
		sf.weightsHH += np.dot( sf.learningRate, sf.gradDescHH )

	def forward_back_prop_single_epoch(sf, epoch): 
		training_acc = []
		calc_type = 0
		accuracyCounter = 0

		#over all training examples
		for i in range(sf.datasetLen):          
			#print "*** begin data ex: ", i, "***"

			sf.resetParameters(i) #resets gradient matrixes

			for t in range( sf.sequenceLen ):

				#print "---time/sq ex: ", t, "---"
				#print "input & expected char", sf.allData[i][t], sf.label[i][t]		
				targetIndx = ord( sf.label[i][t] )  
				#print "expected char",sf.label[i][t]

				sf.forward_prop(i, t, calc_type, targetIndx)
				
				sf.backward_prop(i, t, targetIndx)

				'''
				if np.argmax(sf.yJK) != np.argmax(sf.target[targetIndx]):
					errorCounter += 1
				'''

				if np.argmax(sf.yJK) == np.argmax(sf.target[targetIndx]):
					accuracyCounter += 1
				
			#print "hiddenActivation: ", sf.hiddenActivation
			sf.adagrad_weight_update()
			
			#print "*** end data ex: ", i, "***"

		acc = ( float(accuracyCounter)/float(sf.datasetLen*sf.sequenceLen) )

		print "Training Accuracy: dataset len: ", sf.datasetLen, ", ", epoch, "th Epoch: ", acc*100
		
		#print "sf.weightsHH[0]", sf.weightsHH[0][2]

		#print "loss: ", sf.loss()

		return acc*100


	def bptt(sf, delta_K, timestep, currData):

		delta_t = np.dot( sf.weightsJK.T, delta_K )*( 1 - ( sf.hiddenActivation[timestep] ** 2) ) 
		
		for t in range(timestep+1)[::-1]:           
			#print "Backprop: timestep=%d & step t=%d " % (timestep, t)

			asciiChar = ord( sf.allData[currData][t] )
			#print "bptt: asciiChar", asciiChar
			sf.charset[asciiChar] = 1

			if sf.gradDescHH != [] and sf.gradDescInput != []:
				sf.gradDescHH += np.outer( delta_t, sf.hiddenActivation[t-1] ) #what happens when hiddenActivation at t = 0, is 0
				sf.gradDescInput += np.outer(delta_t, sf.charset )
				
			else:
				sf.gradDescHH = np.outer( delta_t, sf.hiddenActivation[t-1] )
				sf.gradDescInput = np.outer(delta_t, sf.charset )
				

			delta_t = np.dot( sf.weightsHH.T, delta_t ) * (1 - ( sf.hiddenActivation[t - 1] ** 2 ) )

			sf.charset[asciiChar] = 0

		sf.charset[asciiChar] = 0
			

	def calc_deltaK_gradient_descent_output_layer(sf, targetIndx, timestep):

		delta_K = np.array( sf.target[targetIndx] - sf.yJK )
		#print "yJK", sf.yJK
		delta_K = np.array(delta_K)

		if sf.gradDescOutput == []:
			sf.gradDescOutput = np.outer( delta_K, sf.hiddenActivation[timestep] )
			
		else:
			sf.gradDescOutput += np.outer( delta_K, sf.hiddenActivation[timestep] )
			
		#print "sf.gradDescOutput: ", sf.gradDescOutput[0]
		return delta_K
		#print "target expected", target[targetIndx]
		#print "yJK->received", yJK

	def loss(sf):
		L = 0
		# For each example
		for i in np.arange(len(sf.label)):
			# For each timestep
			output = np.zeros([len(sf.label[i]),256])
			target = np.zeros(len(sf.label[i]))
			for t in range(len(sf.label[i])):
				asciiChar_input = ord(sf.allData[i][t])
				sf.calc_y_hidden_layer(i, t, calc_type=0)
				asciiChar_predict = ord(sf.calc_y_softmax_output_layer(0, t, calc_type=0))
				asciiChar_target = ord(sf.label[i][t])
				target[t] = asciiChar_target
				output[t,asciiChar_predict] = 1

			correct_word_predictions = output[:, target.astype(int)]
			
			L += -1 * np.sum(np.log(correct_word_predictions))
		return L

	def adagrad_weight_update(sf):

		if sf.grad_I_Input != []:
			sf.grad_I_Input += sf.gradDescInput**2
		else:
			sf.grad_I_Input = sf.gradDescInput**2

		adagrad = sf.fudge_fac + np.sqrt(sf.grad_I_Input)
		adagrad_grad = sf.gradDescInput/adagrad
		sf.weightsIJ += np.dot(sf.learningRate, adagrad_grad)
		
		if sf.grad_I_Output != []:
			sf.grad_I_Output += sf.gradDescOutput**2
		else:
			sf.grad_I_Output = sf.gradDescOutput**2
		adagrad = sf.fudge_fac + np.sqrt(sf.grad_I_Output)
		adagrad_grad = sf.gradDescOutput/adagrad
		sf.weightsJK += np.dot(sf.learningRate, adagrad_grad)
		
		if sf.grad_I_HH != []:
			sf.grad_I_HH += sf.gradDescHH**2
		else:
			sf.grad_I_HH = sf.gradDescHH**2
		adagrad = sf.fudge_fac + np.sqrt(sf.grad_I_HH)
		adagrad_grad = sf.gradDescHH/adagrad
		sf.weightsHH += np.dot(sf.learningRate, adagrad_grad)

if __name__ == '__main__':
	
	RNN = RNN()
	RNN.forward_back_propogation()


