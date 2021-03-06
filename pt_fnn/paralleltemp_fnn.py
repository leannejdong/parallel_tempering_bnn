import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math


# An example of a class
class Network:
    def __init__(self, Topo, Train, Test):
        self.Top = Topo  # NN topology [input, hidden, output]
        self.TrainData = Train
        self.TestData = Test
        np.random.seed()

        self.W1 = np.random.randn(self.Top[0], self.Top[1]) / np.sqrt(self.Top[0])
        self.B1 = np.random.randn(1, self.Top[1]) / np.sqrt(self.Top[1])  # bias first layer
        self.W2 = np.random.randn(self.Top[1], self.Top[2]) / np.sqrt(self.Top[1])
        self.B2 = np.random.randn(1, self.Top[2]) / np.sqrt(self.Top[1])  # bias second layer

        self.hidout = np.zeros((1, self.Top[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.Top[2]))  # output last layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.Top[2]
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def BackwardPass(self, Input, desired, vanilla):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))

        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        w_layer1size = self.Top[0] * self.Top[1]
        w_layer2size = self.Top[1] * self.Top[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.Top[0], self.Top[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.Top[1], self.Top[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.Top[1]]
        self.B2 = w[w_layer1size + w_layer2size + self.Top[1]:w_layer1size + w_layer2size + self.Top[1] + self.Top[2]]

    def evaluate_proposal(self, data, w):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.

        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for pat in xrange(0, size):
            Input[:] = data[pat, 0:self.Top[0]]
            Desired[:] = data[pat, self.Top[0]:]

            self.ForwardPass(Input)
            fx[pat] = self.out

        return fx


# --------------------------------------------------------------------------

class MCMC:
    def __init__(self, samples, traindata, testdata, topology, tempr):
        self.samples = samples  # NN topology [input, hidden, output]
        self.topology = topology  # max epocs
        self.traindata = traindata  #
        self.testdata = testdata
		self.temprature = tempr
		w_size = (self.topology[0] * self.topology[1]) + (self.topology[1] * self.topology[2]) + self.topology[1] + self.topology[2]
        self.pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
        self.pos_tau = np.ones((samples, 1))
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        self.fxtrain_samples = np.ones((samples, trainsize))  # fx of train data over all samples
        self.fxtest_samples = np.ones((samples, testsize))  # fx of test data over all samples
        self.rmse_train = np.zeros(samples)
        self.rmse_test = np.zeros(samples)
        # ----------------

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def likelihood_func(self, neuralnet, data, w, tausq):
        y = data[:, self.topology[0]]
        fx = neuralnet.evaluate_proposal(data, w)
        rmse = self.rmse(fx, y)
        loss = -0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq
        return [(np.sum(loss))* (1.0/self.temprature), fx, rmse]

    def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
        h = self.topology[1]  # number hidden neurons
        d = self.topology[0]  # number input neurons
        part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss
		
    def sampler(self):

        # ------------------- initialize MCMC
        testsize = self.testdata.shape[0]
        trainsize = self.traindata.shape[0]
        samples = self.samples

        x_test = np.linspace(0, 1, num=testsize)
        x_train = np.linspace(0, 1, num=trainsize)

        netw = self.topology  # [input, hidden, output]
        y_test = self.testdata[:, netw[0]]
        y_train = self.traindata[:, netw[0]]
        print y_train.size
        print y_test.size

        w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

        pos_w = self.pos_w  # posterior of all weights and bias over all samples
        pos_tau = self.pos_tau

        fxtrain_samples = self.fxtrain_samples  # fx of train data over all samples
        fxtest_samples = self.fxtest_samples  # fx of test data over all samples
        rmse_train = self.rmse_train
        rmse_test = self.rmse_test
        w = np.random.randn(w_size)
        w_proposal = np.random.randn(w_size)

        step_w = 0.02;  # defines how much variation you need in changes to w
        step_eta = 0.01;
        # --------------------- Declare FNN and initialize

        neuralnet = Network(self.topology, self.traindata, self.testdata)
        print 'evaluate Initial w'

        pred_train = neuralnet.evaluate_proposal(self.traindata, w)
        pred_test = neuralnet.evaluate_proposal(self.testdata, w)

        eta = np.log(np.var(pred_train - y_train))
        tau_pro = np.exp(eta)

        sigma_squared = 25
        nu_1 = 0
        nu_2 = 0

        prior_likelihood = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

        [likelihood, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w, tau_pro)
        [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w, tau_pro)

        print likelihood

        naccept = 0
        print 'begin sampling using mcmc random walk'
        plt.plot(x_train, y_train)
        plt.plot(x_train, pred_train)
        plt.title("Plot of Data vs Initial Fx")
        plt.savefig('mcmcresults/begin.png')
        plt.clf()

        plt.plot(x_train, y_train)

        for i in range(samples - 1):

            w_proposal = w + np.random.normal(0, step_w, w_size)

            eta_pro = eta + np.random.normal(0, step_eta, 1)
            tau_pro = math.exp(eta_pro)

            [likelihood_proposal, pred_train, rmsetrain] = self.likelihood_func(neuralnet, self.traindata, w_proposal,
                                                                                tau_pro)
            [likelihood_ignore, pred_test, rmsetest] = self.likelihood_func(neuralnet, self.testdata, w_proposal,
                                                                            tau_pro)

            # likelihood_ignore  refers to parameter that will not be used in the alg.

            prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                               tau_pro)  # takes care of the gradients

            diff_likelihood = likelihood_proposal - likelihood
            diff_priorliklihood = prior_prop - prior_likelihood

            mh_prob = min(1, math.exp(diff_likelihood + diff_priorliklihood))

            u = random.uniform(0, 1)

            if u < mh_prob:
                # Update position
                print    i, ' is accepted sample'
                naccept += 1
                likelihood = likelihood_proposal
                prior_likelihood = prior_prop
                w = w_proposal
                eta = eta_pro

                print  likelihood, prior_likelihood, rmsetrain, rmsetest, w, 'accepted'

                pos_w[i + 1,] = w_proposal
                pos_tau[i + 1,] = tau_pro
                fxtrain_samples[i + 1,] = pred_train
                fxtest_samples[i + 1,] = pred_test
                rmse_train[i + 1,] = rmsetrain
                rmse_test[i + 1,] = rmsetest
				lhood = likelihood
                plt.plot(x_train, pred_train)


            else:
                pos_w[i + 1,] = pos_w[i,]
                pos_tau[i + 1,] = pos_tau[i,]
                fxtrain_samples[i + 1,] = fxtrain_samples[i,]
                fxtest_samples[i + 1,] = fxtest_samples[i,]
                rmse_train[i + 1,] = rmse_train[i,]
                rmse_test[i + 1,] = rmse_test[i,]

                # print i, 'rejected and retained'

        print naccept, ' num accepted'
        print naccept / (samples * 1.0), '% was accepted'
        accept_ratio = naccept / (samples * 1.0) * 100

        plt.title("Plot of Accepted Proposals")
        plt.savefig('mcmcresults/proposals.png')
        plt.savefig('mcmcresults/proposals.svg', format='svg', dpi=600)
        plt.clf()

        return (pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, accept_ratio, lhood)

class ParallelTempering:

    def __init__(self, num_chains, maxtemp,NumSample,traindata,testdata,topology):
        
        self.maxtemp = maxtemp
        self.num_chains = num_chains
        self.chains = []
        self.tempratures = []
        self.NumSamples = int(NumSample/self.num_chains)
        self.sub_sample_size = int( 0.1* self.NumSamples)
		self.traindata = traindata
		self.testdata = testdata
		self.topology = topology
        #self.sub_sample_size =  100
        
              
        self.fx_samples = np.ones((num_chains,self.NumSamples, ydata.size))
        self.fxtrain_samples = np.ones((num_chains,samples, trainsize))  # fx of train data over all samples
        self.fxtest_samples = np.ones((num_chains,samples, testsize))  # fx of test data over all samples
        self.rmse_train = np.zeros((num_chains,samples))
        self.rmse_test = np.zeros((num_chains,samples))
        self.pos_w = np.ones((num_chains,self.NumSamples, (nModels)))
        self.pos_tau = np.ones((num_chains,self.NumSamples,  1))
          
    
    # assigin tempratures dynamically   
    def assign_temptarures(self):
        tmpr_rate = (self.maxtemp /self.num_chains)
        temp = 1
        for i in xrange(0, self.num_chains):            
            self.tempratures.append(temp)
            temp += temp #tmpr_rate
            print(self.tempratures[i])
            
    
    # Create the chains.. Each chain gets its own temprature
    def initialize_chains (self):
        self.assign_temptarures()
        for i in xrange(0, self.num_chains):
            self.chains.append(MCMC(self.NumSamples,self.traindata,self.testdata,self.topology, self.tempratures[i]))
            
    # Propose swapping between adajacent chains        
    def propose_swap (self, swap_proposal):
         for l in range( self.num_chains-1, 0, -1):            
                u = random.uniform(0, 1) 
                swap_prob = min(1, swap_proposal[l-1])
                if u < swap_prob : 
                    self.swap_info(self.chains[l],self.chains[-1])
                    print('chains swapped')     
            
            
    # Swap configuration of two chains    
    def swap_info(self, chain_cooler, chain_warmer):  
        
        temp_chain = chain_cooler;
		
        chain_cooler.fxtrain_samples = chain_warmer.fxtrain_samples
        chain_cooler.fxtest_samples = chain_warmer.fxtest_samples
        chain_cooler.x_train = chain_warmer.x_train
		chain_cooler.x_test = chain_warmer.x_test
		chain_cooler.rmse_train = chain_warmer.rmse_train
		chain_cooler.rmse_test = chain_warmer.rmse_test
        chain_cooler.pos_w = chain_warmer.pos_w
        chain_cooler.pos_tau = chain_warmer.pos_tau
        
        chain_warmer.fxtrain_samples = temp_chain.fxtrain_samples
        chain_warmer.fxtest_samples = temp_chain.fxtest_samples
        chain_cooler.x_train = temp_chain.x_train
		chain_cooler.x_test = temp_chain.x_test
		chain_cooler.rmse_train = temp_chain.rmse_train
		chain_cooler.rmse_test = temp_chain.rmse_test
        chain_cooler.pos_w = temp_chain.pos_w
        chain_cooler.pos_tau = temp_chain.pos_tau
        
    # Merge different MCMC chains y stacking them on top of each other       
    def merge_chain (self, chain):
        comb_chain = []
        for i in xrange(0, self.num_chains):
            for j in xrange(0, self.NumSamples):
                comb_chain.append(chain[i][j].tolist())		
        return np.asarray(comb_chain)
		

    def run_chains (self):
        self.initialize_chains()
        swap_proposal = np.ones(self.num_chains-1) # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        
        print (self.NumSamples,self.sub_sample_size,self.NumSamples/self.sub_sample_size)
        #input ()
        start = 0
        end =  start + self.sub_sample_size
        #for i in range(0, int(self.NumSamples/self.sub_sample_size)):
        while (end < self.NumSamples):
            
            
            print (start, end)
            print ('--------------------------------------\n\n')

            lhood = np.zeros(self.num_chains)
            
            #run each chain for a fixed number of SAMPLING Period along the MCMC Chain
            for j in range(0,self.num_chains):        
                pos_w[j,], pos_tau[j,], fxtrain_samples[j,], fxtest_samples[j,], x_train[j,], x_test[j,], rmse_train[j,], rmse_test[j,], accept_ratio, lhood[j] = self.chains[j].sampler()
                print (j, lhood[j])
                
            
            #calculate the swap acceptance rate for parallel chains    
            for k in range(0, self.num_chains-1): 
                 swap_proposal[k]=  (lhood[k]/lhood[k+1])*(1/self.tempratures[k] * 1/self.tempratures[k+1])  
                
            
            #propose swapping
            self.propose_swap(swap_proposal)
            
            #update the starting and ending positon within one chain
            start =  end
            end =  start + self.sub_sample_size
        
        
        
        
        
        end =  self.NumSamples-1   
        for j in range(0,self.num_chains):        
            pos_w[j,], pos_tau[j,], fxtrain_samples[j,], fxtest_samples[j,], x_train[j,], x_test[j,], rmse_train[j,], rmse_test[j,], accept_ratio, lhood[j] = self.chains[j].sampler()
            print (j, lhood[j])    
  
        #concatenate all chains into one complete chain by stacking them on each other 
        chain_fxtrain = self.merge_chain(fxtrain_samples)
        chain_fxtest = self.merge_chain(fxtest_samples)
        chain_w = self.merge_chain(pos_w)
        chain_tau = self.merge_chain(pos_tau)
        chain_rmse_train = self.merge_chain(rmse_train)
		chain_rmse_test = self.merger_chain(rmse_test)   
             
            
        return chain_fxtrain,chain_fxtest,chain_w,chain_tau,chain_rmse_train, chain_rmse_test
# -------------------------------------------------------------------
def main():
        # load univariate data in same format as given
    modeldata = np.loadtxt('simdata.txt')
	topology = None
    traindata = None
	testdata = None
    x = np.linspace(1 / ydata.size, 1, num=ydata.size)  # (input x for ydata)

    NumSample = 50000  # need to pick yourself
    
    #Number of chains of MCMC required to be run
    num_chains = 6

    #Maximum tempreature of hottest chain  
    maxtemp = 100
    
    #Create A a Patratellel Tempring object instance 
    pt = ParallelTempering(num_chains, maxtemp,NumSample,traindata,testdata,topology)

    #run the chains in a sequence in ascending order
    chain_fxtrain,chain_fxtest,chain_w,chain_tau,chain_rmse_train, chain_rmse_test = pt.run_chains( nModels, x, ydata)
    
    print('sucessfully sampled')
	
if __name__ == "__main__": main()
