import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time


def geo_train(device,x_in,y_in,xd,yd,ud,vd,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho ):
    if (Flag_batch):
     x = torch.Tensor(x_in).to(device)
     y = torch.Tensor(y_in).to(device)
     xd = torch.Tensor(xd).to(device)
     yd = torch.Tensor(yd).to(device)
     ud = torch.Tensor(ud).to(device)
     vd = torch.Tensor(vd).to(device)
     if(1): #Cuda slower in double? 
         x = x.type(torch.cuda.FloatTensor)
         y = y.type(torch.cuda.FloatTensor)
         xd = xd.type(torch.cuda.FloatTensor)
         yd = yd.type(torch.cuda.FloatTensor)
         ud = ud.type(torch.cuda.FloatTensor)
         vd = vd.type(torch.cuda.FloatTensor)
         
     
     
     dataset = TensorDataset(x,y)
     dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
    else:
     x = torch.Tensor(x_in).to(device)
     y = torch.Tensor(y_in).to(device) 
       
    
    h_n = 150 #no of neurons
    input_n = 2 # this is what our answer is a function of. 

    class Swish(nn.Module):
        def __init__(self, inplace=True):
            super(Swish, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            if self.inplace:
                x.mul_(torch.sigmoid(x))
                return x
            else:
                return x * torch.sigmoid(x)
    
    class Net2_u(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_u, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
            
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                
                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            return  output


    class Net2_v(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_v, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
            
            return  output 

    class Net2_p(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_p, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                nn.Linear(h_n,h_n),
                
                Swish(),
                

                nn.Linear(h_n,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
        
            return  output

    
    ################################################################
    
    net2_u = Net2_u().to(device)
    net2_v = Net2_v().to(device)
    net2_p = Net2_p().to(device)
    

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)

    # use the modules apply function to recursively apply the initialization
    net2_u.apply(init_normal)
    net2_v.apply(init_normal)
    net2_p.apply(init_normal)
    
    ############################################################################
    optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    
    ##############################################
    def Loss_data(xd,yd,ud,vd):
    

        net_in1 = torch.cat((xd, yd), 1)
        out1_u = net2_u(net_in1)
        out1_u = out1_u.view(len(out1_u), -1)
        out1_v = net2_v(net_in1)
        out1_v = out1_v.view(len(out1_v), -1)
        
       
        loss_f = nn.MSELoss()
        loss_d = loss_f(out1_u, ud) + loss_f(out1_v, vd)  


        return loss_d   
    
    ###############################################

    # Main loop
    tic = time.time()

    if(Flag_batch):# This one uses dataloader

            
            for epoch in range(epochs):
                
                loss_data_n = 0
                n = 0
                for batch_idx, (x_in,y_in) in enumerate(dataloader):
                    net2_u.zero_grad()
                    net2_v.zero_grad()
                    net2_p.zero_grad()
                    
                    loss_data = Loss_data(xd,yd,ud,vd)
                    loss = loss_data 
                    loss.backward()

                    optimizer_u.step() 
                    optimizer_v.step() 
                    optimizer_p.step()
                    
                    loss_data_a= loss_data.detach().cpu().numpy()
                    loss_data_n += loss_data_a
                    n += 1         
                      
                    if batch_idx % 40 ==0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f} Loss data {:.10f}'.format(
                            epoch, batch_idx * len(x_in), len(dataloader.dataset),
                            100. * batch_idx / len(dataloader), loss.item(), loss_data.item()))
                    
                
                mean_data = loss_data_n/n
                
                print('***Total avg Loss : Loss eqn {:.10f}'.format(mean_data) )
                print('****Epoch:', epoch,'learning rate is: ', optimizer_u.param_groups[0]['lr'])
                
        

                if epoch % 100 == 0:#save network
                 torch.save(net2_u.state_dict(),path+"fwd_step_u_velocity_from_fluent_low_fidelity"+str(epoch)+".pt")
                 torch.save(net2_v.state_dict(),path+"fwd_step_v_velocity_from_fluent_low_fidelity_"+str(epoch)+".pt")
                 torch.save(net2_p.state_dict(),path+"fwd_step_p_velocity_from_fluent_low_fidelity_"+str(epoch)+".pt")
          
                
    else:
        for epoch in range(epochs):
            net2_u.zero_grad()
            net2_v.zero_grad()
            net2_p.zero_grad()
                    
            loss_data = Loss_data(xd,yd,ud,vd)
            loss = loss_data 
            loss.backward()
            
            optimizer_u.step() 
            optimizer_v.step() 
            optimizer_p.step()
            
            if epoch % 10 ==0:
                print('Train Epoch: {} \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
                    epoch, loss.item(),loss_bc.item()))

    toc = time.time()
    elapseTime = toc - tic
    print ("elapse time in parallel = ", elapseTime)
    ###################
    net2_u.eval()
    net2_v.eval()
    net2_p.eval()
    
    net_in = torch.cat((x.requires_grad_(),y.requires_grad_()),1)
    output_u = net2_u(net_in)  #evaluate model
    output_v = net2_v(net_in)  #evaluate model
    output_u = output_u.cpu().data.numpy() #need to convert to cpu before converting to numpy
    output_v = output_v.cpu().data.numpy()
    
    x = x.cpu()
    y = y.cpu()


    


    return


    
#######################################################
#Main code:

device = torch.device("cuda")

Flag_batch = True #False #USe batch or not  

batchsize = 512 #Total number of batches 
learning_rate = 3e-3 
epochs  = 101
Flag_schedule = False #If true change the learning rate at 3 levels
if (Flag_schedule):
    learning_rate = 3e-4
    step_epoch = 3000 
    decay_rate = 0.1

x_scale = 2.8 #The length of the  domain 

Diff =0.01
rho = 1.0 

nPt = 10
nPt3 = 6 
nPt2= 8
nPt1 = 4
xStart_out = 0.4
xEnd_out = 1.0
xStart_up = 0.3
xEnd_up = 0.4
xStart_in = 0
xEnd_in = 0.3
yStart = 0.
yEnd = 1.0
yStart_up = 0.5



x = np.linspace(xStart_in, xEnd_in, nPt)    #inlet
y = np.linspace(yStart, yEnd, nPt2)
x, y = np.meshgrid(x, y)
x = np.reshape(x, (np.size(x[:]),1))
y = np.reshape(y, (np.size(y[:]),1))

x2 = np.linspace(xStart_up  , xEnd_up, nPt3)      #upper
y2 = np.linspace(yStart_up , yEnd, nPt1,endpoint=False)
x2, y2 = np.meshgrid(x2, y2)
x2 = np.reshape(x2, (np.size(x2[:]),1))
y2 = np.reshape(y2, (np.size(y2[:]),1))
    
x3 = np.linspace(xStart_out  , xEnd_out, nPt)     #outlet
y3 = np.linspace(yStart , yEnd, nPt2,endpoint=False)
x3, y3 = np.meshgrid(x3, y3)
x3 = np.reshape(x3, (np.size(x3[:]),1))
y3 = np.reshape(y3, (np.size(y3[:]),1))

x = np.concatenate((x,x2,x3), axis=0)
y = np.concatenate((y,y2,y3), axis=0)


print('shape of x',x.shape)
print('shape of y',y.shape)


with open("low_fidelity_cfd_results.csv", "r") as file_source: #read low fidelity cfd results
         file_plot = csv.reader(file_source)
         with open("result.txt","w") as result:
            wtr = csv.writer(result)
            for row in file_plot:
                wtr.writerow( (row[0],row[1],row[3],row[4]))
    
data = np.genfromtxt("result.txt", delimiter= ',');

 
#data storage:
x_data = data[:,2]
y_data = data[:,3] 
u_data = data[:,0]
v_data = data[:,1]
 
print("reading and saving cfd done!") 
x_data = np.asarray(x_data)  #convert to numpy 
y_data = np.asarray(y_data) #convert to numpy 
u_data = np.asarray(u_data) #convert to numpy
v_data = np.asarray(v_data) #convert to numpy
x_scale = 2.8
x_data = x_data/x_scale
xd= x_data.reshape(-1, 1) #need to reshape to get 2D array
yd= y_data.reshape(-1, 1) #need to reshape to get 2D array
ud= u_data.reshape(-1, 1) #need to reshape to get 2D array
vd= v_data.reshape(-1, 1) #need to reshape to get 2D array

print("x_data", xd.shape)
print("y_data", yd.shape)
print("u_data", ud.shape)
print("v_data", vd.shape)


path = "Results/"


geo_train(device,x,y,xd,yd,ud,vd,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho )
