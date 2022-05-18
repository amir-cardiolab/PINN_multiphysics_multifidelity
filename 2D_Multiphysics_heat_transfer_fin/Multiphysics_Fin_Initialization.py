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


def geo_train(device,x_influid,y_influid,x_infin,y_infin,xd,yd,xTd,yTd,xTcd,yTcd,ud,vd,Td,Tcd,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho):
    if (Flag_batch):
     xfluid = torch.Tensor(x_influid).to(device)
     yfluid = torch.Tensor(y_influid).to(device)
     xfin = torch.Tensor(x_infin).to(device)
     yfin = torch.Tensor(y_infin).to(device)
     xd = torch.Tensor(xd).to(device)
     yd = torch.Tensor(yd).to(device)
     xTd = torch.Tensor(xTd).to(device)
     yTd = torch.Tensor(yTd).to(device)
     xTcd = torch.Tensor(xTcd).to(device)
     yTcd = torch.Tensor(yTcd).to(device)
     Td = torch.Tensor(Td).to(device)
     ud = torch.Tensor(ud).to(device)
     vd = torch.Tensor(vd).to(device)
     Tcd = torch.Tensor(Tcd).to(device)
     if(1): #Cuda slower in double? 
         xfluid = xfluid.type(torch.cuda.FloatTensor)
         yfluid = yfluid.type(torch.cuda.FloatTensor)
         xfin = xfin.type(torch.cuda.FloatTensor)
         yfin = yfin.type(torch.cuda.FloatTensor)
         xd = xd.type(torch.cuda.FloatTensor)
         yd = yd.type(torch.cuda.FloatTensor)
         xTd = xTd.type(torch.cuda.FloatTensor)
         yTd = yTd.type(torch.cuda.FloatTensor)
         xTcd = xTcd.type(torch.cuda.FloatTensor)
         yTcd = yTcd.type(torch.cuda.FloatTensor)
         Td = Td.type(torch.cuda.FloatTensor)
         ud = ud.type(torch.cuda.FloatTensor)
         vd = vd.type(torch.cuda.FloatTensor)
         Tcd = Tcd.type(torch.cuda.FloatTensor)
       
     dataset = TensorDataset(xfluid,yfluid,xfin,yfin)
     dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False)
    
    else:
     xfluid = torch.Tensor(x_influid).to(device)
     yfluid = torch.Tensor(y_influid).to(device) 
     xfin = torch.Tensor(x_infin).to(device)
     yfin = torch.Tensor(y_infin).to(device) 
     
    h_nd = 140  #no of neurons in net T
    h_n = 150 #no of neurons in net V
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

    class Net2_T(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_T, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                

                nn.Linear(h_nd,1),
            )
        #This function defines the forward rule of
        #output respect to input.
        def forward(self,x):
            output = self.main(x)
    
            return  output  

    class Net2_Tc(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_Tc, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(input_n,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                nn.Linear(h_nd,h_nd),
                
                Swish(),
                

                nn.Linear(h_nd,1),
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
    net2_T = Net2_T().to(device)
    net2_Tc = Net2_Tc().to(device)
    

    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
            

    # use the modules apply function to recursively apply the initialization
    net2_u.apply(init_normal)
    net2_v.apply(init_normal)
    net2_p.apply(init_normal)
    net2_T.apply(init_normal)
    net2_Tc.apply(init_normal)
    
    ############################################################################

    optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_T = optim.Adam(net2_T.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_Tc = optim.Adam(net2_Tc.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    
    ###############################################
    def Loss_data(xd,yd,xTd,yTd,xTcd,yTcd,ud,vd,Td,Tcd):
    

        
        net_in1 = torch.cat((xTd, yTd), 1)
        net_in2 = torch.cat((xTcd, yTcd), 1)
        net_in3 = torch.cat((xd, yd), 1)

        out1_T = net2_T(net_in1)
        out1_T = out1_T.view(len(out1_T), -1)

        out1_Tc = net2_Tc(net_in2)
        out1_Tc = out1_Tc.view(len(out1_Tc), -1)
        
        out1_u = net2_u(net_in3)
        out1_u = out1_u.view(len(out1_u), -1)
        out1_v = net2_v(net_in3)
        out1_v = out1_v.view(len(out1_v), -1)
        

        loss_f = nn.MSELoss()
        loss_d = loss_f(out1_T, Td) + loss_f(out1_Tc, Tcd) + loss_f(out1_u, ud) + loss_f(out1_v, vd)


        return loss_d    
        ##############################################################
    # Main loop
    tic = time.time()


    if(Flag_batch):# This one uses dataloader

            
            for epoch in range(epochs):
                
                loss_data_n = 0
                n = 0
                for batch_idx, (x_influid,y_influid, x_infin, y_infin) in enumerate(dataloader):
                    net2_u.zero_grad()
                    net2_v.zero_grad()
                    net2_p.zero_grad()
                    net2_T.zero_grad()
                    net2_Tc.zero_grad()
                    
                    loss_data = Loss_data(xd,yd,xTd,yTd,xTcd,yTcd,ud,vd,Td,Tcd)
                    loss =  loss_data
                    
                    loss.backward()
                    optimizer_u.step() 
                    optimizer_v.step() 
                    optimizer_p.step()
                    optimizer_T.step()
                    optimizer_Tc.step()
                    
                    loss_data_a= loss_data.detach().cpu().numpy()
                    loss_data_n += loss_data_a
                    n += 1         
                     
                    if batch_idx % 40 ==0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f} Loss Data {:.10f}'.format(
                            epoch, batch_idx * len(x_influid), len(dataloader.dataset),
                            100. * batch_idx / len(dataloader), loss.item(), loss_data.item()))
                    
                
                mean_data = loss_data_n/n
                
                print('***Total avg Loss : Loss Data {:.10f}'.format(mean_data) )
                print('****Epoch:', epoch,'learning rate is: ', optimizer_T.param_groups[0]['lr'])
                
                
                if epoch % 100 == 0:#save network
                 torch.save(net2_u.state_dict(),path+"fwd_step_u_heat_conduction_from_fluent_"+str(epoch)+".pt")
                 torch.save(net2_v.state_dict(),path+"fwd_step_v_heat_conduction_from_fluent_"+str(epoch)+".pt")
                 torch.save(net2_p.state_dict(),path+"fwd_step_p_heat_conduction_from_fluent_"+str(epoch)+".pt")
                 torch.save(net2_T.state_dict(),path+"fwd_step_T_heat_conduction_from_fluent_"+str(epoch)+".pt")
                 torch.save(net2_Tc.state_dict(),path+"fwd_step_Tc_heat_conduction_from_fluent_"+str(epoch)+".pt")
                
    else:
        for epoch in range(epochs):
            net2_u.zero_grad()
            net2_v.zero_grad()
            net2_p.zero_grad()
            net2_T.zero_grad()
            net2_Tc.zero_grad()
            
            loss_data = Loss_data(xd,yd,xTd,yTd,xTcd,yTcd,ud,vd,Td,Tcd)
            loss =  loss_data
            
            loss.backward()
            
            optimizer_u.step() 
            optimizer_v.step() 
            optimizer_p.step()
            optimizer_T.step()
            optimizer_Tc.step()  
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
    net2_T.eval()
    net2_Tc.eval()
    
    
    net_in = torch.cat((xfluid.requires_grad_(),yfluid.requires_grad_()),1)
    net_in2 = torch.cat((xfin.requires_grad_(),yfin.requires_grad_()),1)
    output_u = net2_u(net_in)  #evaluate model
    output_u = output_u.cpu().data.numpy()
    output_v = net2_v(net_in)  #evaluate model
    output_v = output_v.cpu().data.numpy()
    output_T = net2_T(net_in)  #evaluate model
    output_Tc = net2_Tc(net_in2)  #evaluate model
    output_T = output_T.cpu().data.numpy()
    output_Tc = output_Tc.cpu().data.numpy()
    xfluid = xfluid.cpu()
    yfluid = yfluid.cpu()
    xfin = xfin.cpu()
    yfin = yfin.cpu()

    

    return



#######################################################
#Main code:

device = torch.device("cuda")

Flag_batch = True #False #USe batch or not  
batchsize = 256  #Total number of batches 
learning_rate = 3e-3 
epochs  = 201
Flag_schedule = False #If true change the learning rate at 3 levels
if (Flag_schedule):
    learning_rate = 3e-4
    step_epoch = 2500 #100
    decay_rate = 0.1


x_scale = 2.8 #The length of the  domain 
K = 0.02  # heat conductivity of Fluid
Cp =1.0
Kc = 0.08     #heat conductivity of Solid
Diff = 0.01
rho = 1 


nPt = 7 
nPt1= 10
nPt3= 15
nPt4 = 18
xStart_out = 0.4
xEnd_out = 1.0
xStart_up = 0.3
xEnd_up = 0.4
xStart_in = 0
xEnd_in = 0.3
yStart = 0.
yEnd = 1.0
yStart_up = 0.5



x = np.linspace(xStart_in, xEnd_in, nPt1)    #inlet
y = np.linspace(yStart, yEnd, nPt1)
x, y = np.meshgrid(x, y)
x = np.reshape(x, (np.size(x[:]),1))
y = np.reshape(y, (np.size(y[:]),1))

x2 = np.linspace(xStart_up  , xEnd_up, nPt)      #upper
y2 = np.linspace(yStart_up , yEnd, nPt1,endpoint=False)
x2, y2 = np.meshgrid(x2, y2)
x2 = np.reshape(x2, (np.size(x2[:]),1))
y2 = np.reshape(y2, (np.size(y2[:]),1))
    
x3 = np.linspace(xStart_out  , xEnd_out, nPt1)     #outlet
y3 = np.linspace(yStart , yEnd, nPt1,endpoint=False)
x3, y3 = np.meshgrid(x3, y3)
x3 = np.reshape(x3, (np.size(x3[:]),1))
y3 = np.reshape(y3, (np.size(y3[:]),1))

xfin = np.linspace(xEnd_in  , xStart_out, nPt3)     #Solid
yfin = np.linspace(yStart , yStart_up, nPt4,endpoint=False)
xfin, yfin = np.meshgrid(xfin, yfin)
xfin = np.reshape(xfin, (np.size(xfin[:]),1))
yfin = np.reshape(yfin, (np.size(yfin[:]),1))

xfluid = np.concatenate((x,x2,x3), axis=0)
yfluid = np.concatenate((y,y2,y3), axis=0)


print('shape of xfluid',xfluid.shape)
print('shape of yfluid',yfluid.shape)
print('shape of xfin',xfin.shape)
print('shape of yfin',yfin.shape)



with open("low_fidelity_heat_conduction.csv", "r") as file_source: #read velocity
         file_plot = csv.reader(file_source)
         with open("result.txt","w") as result:
            wtr = csv.writer(result)
            for row in file_plot:
                wtr.writerow( (row[1],row[2],row[3],row[4]))
data = np.genfromtxt("result.txt", delimiter= ',');

#data storage:
x_data = data[:,2]
y_data = data[:,3] 
u_data = data[:,0]
v_data = data[:,1]

with open("low_fidelity_heat_conduction_T.csv", "r") as file_source: # read temperature in fluid domain
         file_plot = csv.reader(file_source)
         with open("result_T.txt","w") as result:
            wtr = csv.writer(result)
            for row in file_plot:
                wtr.writerow( (row[0],row[3],row[4]))
data1 = np.genfromtxt("result_T.txt", delimiter= ',');   

xT_data = data1[:,1]
yT_data = data1[:,2] 
T_data = data1[:,0]

with open("low_fidelity_heat_conduction_Tc.csv", "r") as file_source:  # read temperature in fluid domain
         file_plot = csv.reader(file_source)
         with open("result_Tc.txt","w") as result:
            wtr = csv.writer(result)
            for row in file_plot:
                wtr.writerow( (row[0],row[3],row[4]))
data2 = np.genfromtxt("result_Tc.txt", delimiter= ',');   

xTc_data = data2[:,1]
yTc_data = data2[:,2] 
Tc_data = data2[:,0]

print("reading and saving cfd done!") 
x_data = np.asarray(x_data)  #convert to numpy 
y_data = np.asarray(y_data) #convert to numpy 
xT_data = np.asarray(xT_data)  #convert to numpy 
yT_data = np.asarray(yT_data) #convert to numpy 
xTc_data = np.asarray(xTc_data)  #convert to numpy 
yTc_data = np.asarray(yTc_data) #convert to numpy 
u_data = np.asarray(u_data) #convert to numpy
v_data = np.asarray(v_data) #convert to numpy
T_data = np.asarray(T_data) #convert to numpy
Tc_data = np.asarray(Tc_data) #convert to numpy
x_data = x_data/x_scale
xT_data = xT_data/x_scale
xTc_data = xTc_data/x_scale
xd= x_data.reshape(-1, 1) #need to reshape to get 2D array
yd= y_data.reshape(-1, 1) #need to reshape to get 2D array
xTd= xT_data.reshape(-1, 1) #need to reshape to get 2D array
yTd= yT_data.reshape(-1, 1) #need to reshape to get 2D array
xTcd= xTc_data.reshape(-1, 1) #need to reshape to get 2D array
yTcd= yTc_data.reshape(-1, 1) #need to reshape to get 2D array
ud= u_data.reshape(-1, 1) #need to reshape to get 2D array
vd= v_data.reshape(-1, 1) #need to reshape to get 2D array
Td= T_data.reshape(-1, 1) #need to reshape to get 2D array
Tcd= Tc_data.reshape(-1, 1) #need to reshape to get 2D array

print("x_data", xd.shape)
print("y_data", yd.shape)
print("xT_data", xTd.shape)
print("yT_data", yTd.shape)
print("xTc_data", xTcd.shape)
print("yTc_data", yTcd.shape)
print("u_data", ud.shape)
print("v_data", vd.shape) 
print("T_data", Td.shape)
print("Tc_data", Tcd.shape)

path = "Results/"


geo_train(device,xfluid,yfluid,xfin,yfin,xd,yd,xTd,yTd,xTcd,yTcd,ud,vd,Td,Tcd,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho )






