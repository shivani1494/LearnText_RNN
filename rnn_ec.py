import cPickle, gzip
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import random

class RNN:
	 
	def __init__(sf):

		#why do we get repeatitions?
		#what should take more time over the same text -> longer or shorter sequences?
		sf.learningRate = 0.01

		#how will increasing the number of hidden units help?
		#increase hidden units to 
		sf.input_dim = 256
		sf.hidden_dim = 120
		sf.output_dim = 256

		#why does a lower temperature help?
		sf.temp = 0.3

		sf.allData = []
		sf.label = []

		#training loss
		sf.training_loss_I = []
		sf.training_loss = []
		sf.loss = 0.0

		#best sq len if we change the alpha and this then it starts repeating
		sf.sequenceLen = 10

		sf.buildInputData()
		sf.buildLabelData()

		sf.numEpochs = 50
		sf.datasetLen = 10000
		#sf.testDataLen = 100

		#why did adagrad work great and not sgd
		#adagrad
		sf.fudge_fac = 1e-6
		#keeps the running squares of gradients
		sf.grad_I_Output = []
		sf.grad_I_Input = []
		sf.grad_I_HH = []
		sf.grad_I_GG = []

		sf.yIJ = np.array([])
		sf.yJJ = np.array([])
		sf.yJK = []
		
		sf.gradDescOutput = []
		sf.gradDescInput = []
		sf.gradDescHH = []
		sf.gradDescGG = []

		sf.weightsJK = []
		sf.weightsIJ = []
		sf.weightsJJ = []
		sf.weightsHH = []
		sf.weightsGG = []

		sf.hiddenActivation = []
		sf.hiddenActivation2 = []

		sf.target = []
		sf.charset = []
		sf.asciiLen = 256
		
		for i in range(sf.asciiLen):
			sf.charset.append(0)

		sf.charset = np.array(sf.charset)
		sf.buildTarget()
		sf.populateWeights()

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

		#hidden - hidden2
		weightsPerClassHH = []

		for i in range(sf.hidden_dim):
			mu, sigma = 0, 1.0/math.sqrt(120.0) # mean and standard deviation
			weightsPerClassHH = np.random.normal(mu, sigma, sf.hidden_dim)
			sf.weightsHH.append(list(weightsPerClassHH))
		#converting into numpy array for multiplication
		sf.weightsHH = np.array(sf.weightsHH)

		weightsPerClassGG = []

		#hidden - output layer
		for i in range(sf.hidden_dim):
			mu, sigma = 0, 1.0/math.sqrt(120.0) # mean and standard deviation
			weightsPerClassGG = np.random.normal(mu, sigma, sf.hidden_dim)
			sf.weightsGG.append(list(weightsPerClassGG))
		#converting into numpy array for multiplication
		sf.weightsGG = np.array(sf.weightsGG)

		weightsPerClassJJ = []

		#hidden - output layer
		for i in range(sf.hidden_dim):
			mu, sigma = 0, 1.0/math.sqrt(120.0) # mean and standard deviation
			weightsPerClassJJ = np.random.normal(mu, sigma, sf.hidden_dim)
			sf.weightsJJ.append(list(weightsPerClassJJ))
		#converting into numpy array for multiplication
		sf.weightsJJ = np.array(sf.weightsJJ)

	def setHiddenActivation(sf, start):    
		
		#append zeros for all timesteps
		hh_act_timestep = []
		gg_act_timestep = []
		for h in range(sf.hidden_dim):
			hh_act_timestep.append(0.0)
			gg_act_timestep.append(0.0)

		prev_actHH = []
		prev_actGG = []
		if not start:
			prev_actHH = list(sf.hiddenActivation[sf.sequenceLen-1])
			prev_actGG = list(sf.hiddenActivation2[sf.sequenceLen-1])
		else:
			#first time, no activations yet
			prev_actHH = list(hh_act_timestep)
			prev_actGG = list(gg_act_timestep)

		hh_act_temp = []
		gg_act_temp = []
		for t in range(sf.sequenceLen):
			hh_act_temp.append( list(hh_act_timestep) )
			gg_act_temp.append( list(gg_act_timestep) )
		hh_act_temp.append(list(prev_actHH))
		gg_act_temp.append(list(prev_actGG))

		sf.hiddenActivation = np.array( list(hh_act_temp) )
		sf.hiddenActivation2 = np.array( list(gg_act_temp) )
			   
	def generate(sf):
		
		calc_type = 1

		print "*** testing- generating sequences ***"
		str1 = ""		
		for a in range(100 / sf.sequenceLen):

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

	def generate_from(sf, chunk):

		calc_type = 1
		matches = 0.0
		chunk_test = chunk[1::]
		n = sf.sequenceLen
		split_chunk_data = [ chunk[i:i+n] for i in range(0, len(chunk), n) ]
		split_chunk_test = [ chunk_test[i:i+n] for i in range(0, len(chunk_test), n) ]

		str_out = ""	

		for c in range( len(split_chunk_data) - 1):

			if c == 0:
				sf.setHiddenActivation(False)
			else:
				sf.setHiddenActivation(True)

			curr_seq_d = split_chunk_data[c]
			curr_seq_t = split_chunk_test[c]
			str_out += curr_seq_d[0] #append the first char

			for i in range( len(curr_seq_d) ):
				sf.calc_y_hidden_layer(curr_seq_d[i], 1, calc_type)
				p = sf.calc_y_softmax_output_layer(0, 1, calc_type)
				asciiInt = np.argmax(p)
				asciiChar = chr(asciiInt)
				if curr_seq_t[i] == asciiChar:
					matches += 1.0
				str_out += asciiChar

		print "Output text: ", str_out
		return (matches / len(chunk) )*100

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

		sf.yJJ = np.tanh(  np.dot( sf.weightsGG, sf.hiddenActivation2[timestep - 1] ) + np.dot( sf.weightsJJ, sf.hiddenActivation[timestep]) )

		sf.hiddenActivation2[timestep] = sf.yJJ
		
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
		netJK = np.dot(sf.weightsJK, sf.hiddenActivation2[timestep])
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
				sf.yJK.append( float(math.exp(netJK[i]/sf.temp) )/( float(netSum) ) )

		if calc_type == 0:
			for i in range(sf.output_dim):        
				sf.yJK.append( float(math.exp(netJK[i]))/( float(netSum) ) ) #adding temperature

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
			sf.grad_i_GG = []

			sf.loss = 0.0
			sf.training_loss_I = []

			sf.forward_back_prop_single_epoch(j+1)
			len_chunk = 10
			if (j + 1) % 20 == 0:
				print sf.generate_from("".join(sf.allData[:len_chunk]))

			print "------------------- end of epoch: ", j+1, "-------------------"

		sf.createTrainingLossPlot()
		for temp in [0.1, 0.3, 0.5, 1, 2]:
			sf.temp = temp
			print
			sf.generate()

		#training_acc_epochs.append(training_acc)
		#print "training_acc_epochs: ", training_acc_epochs    
		#return training_acc_epochs 

	def resetParameters(sf, i):
		
		sf.gradDescOutput = []
		sf.gradDescInput = []
		sf.gradDescHH = []
		sf.gradDescGG = []
		#training_loss_I = []
		#sf.loss = 0
		
		if i == 0:
			sf.setHiddenActivation(True)
		else:
			sf.setHiddenActivation(False)

	def forward_prop(sf, i, j, calc_type, targetIndx):
		sf.calc_y_hidden_layer(i, j, calc_type)
		p = sf.calc_y_softmax_output_layer(targetIndx, j, calc_type)
		predict = p[targetIndx]
		sf.training_loss_I.append(predict)
	
	def backward_prop(sf, currData, timestep, targetIndx):
		delta_K = sf.calc_deltaK_gradient_descent_output_layer(targetIndx, timestep)
		sf.bptt(delta_K, timestep, currData)     
	
	#do not use this! -> SGD does not work well for lots of epochs
	def weight_update(sf):
		sf.weightsIJ += np.dot( sf.learningRate, sf.gradDescInput )
		sf.weightsJJ += np.dot( sf.learningRate, sf.gradDescJJ )
		sf.weightsJK += np.dot( sf.learningRate, sf.gradDescOutput )
		sf.weightsHH += np.dot( sf.learningRate, sf.gradDescHH )
		sf.weightsGG += np.dot( sf.learningRate, sf.gradDescGG )

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
		
		#compute the loss for one example
		sf.loss += -1.0 * np.sum( np.log(sf.training_loss_I) )
			#reset for next data example

		loss_epoch = sf.loss/float(sf.datasetLen)
		
		sf.training_loss.append( loss_epoch )			
		#print "*** end data ex: ", i, "***"
		acc = ( float(accuracyCounter)/float(sf.datasetLen*sf.sequenceLen) )

		print "Training Accuracy: dataset len: ", sf.datasetLen, ", ", epoch, "th Epoch: ", acc*100
		print "Training Loss: ", loss_epoch
		#print "sf.weightsHH[0]", sf.weightsHH[0][2]

		#print "loss: ", sf.loss()

		return acc*100


	def bptt(sf, delta_K, timestep, currData):

		delta_t = np.dot( sf.weightsJK.T, delta_K )*( 1 - ( sf.hiddenActivation2[timestep] ** 2) ) 
		
		for t in range(timestep+1)[::-1]:           
			#print "Backprop: timestep=%d & step t=%d " % (timestep, t)

			asciiChar = ord( sf.allData[currData][t] )
			#print "bptt: asciiChar", asciiChar
			sf.charset[asciiChar] = 1

			if sf.gradDescGG != [] and sf.gradDescHH != []:
				sf.gradDescGG += np.outer( delta_t, sf.hiddenActivation2[t-1] ) #what happens when hiddenActivation2 at t = 0, is 0
				sf.gradDescHH += np.outer(delta_t, sf.hiddenActivation[t] )
				
			else:
				sf.gradDescGG = np.outer( delta_t, sf.hiddenActivation2[t-1] )
				sf.gradDescHH = np.outer(delta_t, sf.hiddenActivation[t] )

			sf.charset[asciiChar] = 0

			delta_u = np.dot( sf.weightsJJ.T, delta_t )*( 1 - ( sf.hiddenActivation[t] ** 2) )
			for t2 in range(t+1)[::-1]:           
				#print "Backprop: timestep=%d & step t=%d " % (timestep, t)

				asciiChar = ord( sf.allData[currData][t2] )
				#print "bptt: asciiChar", asciiChar
				sf.charset[asciiChar] = 1

				if sf.gradDescHH != [] and sf.gradDescInput != []:
					sf.gradDescHH += np.outer( delta_u, sf.hiddenActivation[t2-1] ) #what happens when hiddenActivation2 at t = 0, is 0
					sf.gradDescInput += np.outer(delta_u, sf.charset )
				
				else:
					sf.gradDescHH = np.outer( delta_u, sf.hiddenActivation[t2-1] )
					sf.gradDescInput = np.outer(delta_u, sf.charset )
				
				sf.charset[asciiChar] = 0

				delta_u = np.dot( sf.weightsHH.T, delta_u ) * (1 - ( sf.hiddenActivation[t2 - 1] ** 2 ) )

			delta_t = np.dot( sf.weightsGG.T, delta_t ) * (1 - ( sf.hiddenActivation2[t - 1] ** 2 ) )

		sf.charset[asciiChar] = 0
			

	def calc_deltaK_gradient_descent_output_layer(sf, targetIndx, timestep):

		delta_K = np.array( sf.target[targetIndx] - sf.yJK )
		#print "yJK", sf.yJK
		delta_K = np.array(delta_K)

		if sf.gradDescOutput == []:
			sf.gradDescOutput = np.outer( delta_K, sf.hiddenActivation2[timestep] )
			
		else:
			sf.gradDescOutput += np.outer( delta_K, sf.hiddenActivation2[timestep] )
			
		#print "sf.gradDescOutput: ", sf.gradDescOutput[0]
		return delta_K
		#print "target expected", target[targetIndx]
		#print "yJK->received", yJK

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

		if sf.grad_I_GG != []:
			sf.grad_I_GG += sf.gradDescGG**2
		else:
			sf.grad_I_GG = sf.gradDescGG**2
		adagrad = sf.fudge_fac + np.sqrt(sf.grad_I_GG)
		adagrad_grad = sf.gradDescGG/adagrad
		sf.weightsGG += np.dot(sf.learningRate, adagrad_grad)

	def createTrainingLossPlot(sf):

		epochs = np.arange( sf.numEpochs )
		plt.plot(epochs, sf.training_loss, '-r')
		#axis boundary 0 to max flower feature value
		plt.axis([0, len(epochs), 0, max(sf.training_loss) + 1])

		#make labels
		plt.xlabel("Number of Epochs")        
		plt.ylabel("Training Loss")
		plt.title("Learning Rate for Training Dataset")
		plt.savefig("Epochs_vs_Training_Loss")


if __name__ == '__main__':
	
	rnn = RNN()
	rnn.forward_back_propogation()
