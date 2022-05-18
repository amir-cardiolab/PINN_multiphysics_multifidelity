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

#Solve steady 2D N.S for a rectangular fin


def geo_train(device,x_in,y_in,xb,yb,xb_wall,yb_wall,ub,vb,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC,nPt ):
    if (Flag_batch):
     x = torch.Tensor(x_in).to(device)
     y = torch.Tensor(y_in).to(device)
     xb = torch.Tensor(xb).to(device)
     yb = torch.Tensor(yb).to(device)
     ub = torch.Tensor(ub).to(device)
     vb = torch.Tensor(vb).to(device)
     xb_wall = torch.Tensor(xb_wall).to(device)
     yb_wall = torch.Tensor(yb_wall).to(device)
     if(1): #Cuda slower in double? 
         x = x.type(torch.cuda.FloatTensor)
         y = y.type(torch.cuda.FloatTensor)
         xb = xb.type(torch.cuda.FloatTensor)
         yb = yb.type(torch.cuda.FloatTensor)
         ub = ub.type(torch.cuda.FloatTensor)
         vb = vb.type(torch.cuda.FloatTensor)
         xb_wall = xb_wall.type(torch.cuda.FloatTensor)
         yb_wall = yb_wall.type(torch.cuda.FloatTensor)
         
     
     dataset = TensorDataset(x,y)
     dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
    else:
     x = torch.Tensor(x_in).to(device)
     y = torch.Tensor(y_in).to(device) 
       
    h_n = 150 # no of neurons
    input_n = 2 # this is what our answer is a function of (x,y). 

    class Swish(nn.Module): #Activation Function
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
    if (Flag_initialization):
        net2_u.apply(init_normal) #initialize by transfer learning
        net2_v.apply(init_normal)
    
    net2_p.apply(init_normal) # network p is randomly initialized
    

    ############################################################################
# optimizer
    optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
###################################################
# PDE loss    
    def criterion(x,y):


        x.requires_grad = True
        y.requires_grad = True
        

        net_in = torch.cat((x,y),1)
        u = net2_u(net_in)
        u = u.view(len(u),-1)
        v = net2_v(net_in)
        v = v.view(len(v),-1)
        P = net2_p(net_in)
        P = P.view(len(P),-1)
     
        
        u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

        P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

       
        
        loss_1 = u*(v_x/x_scale)+v*v_y - Diff*(v_xx/(x_scale**2)+v_yy)+1/rho*P_y #Y-dir
        loss_2 = u*(u_x/x_scale)+v*u_y - Diff*(u_xx/(x_scale**2)+u_yy)+1/rho*(P_x/x_scale) #X-dir
        loss_3 = (u_x/x_scale + v_y) #continuity
        

        # MSE LOSS
        loss_f = nn.MSELoss()
        

        #Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss_1,torch.zeros_like(loss_1))+ loss_f(loss_2,torch.zeros_like(loss_2))+loss_f(loss_3,torch.zeros_like(loss_3))
        
        return loss
    ############################################################
    ###################################################################
    # Boundary conditions loss
    def Loss_BC(xb,yb,xb_wall,yb_wall,ub,vb,x,y):
        if(0):
            xb = torch.FloatTensor(xb).to(device)
            yb = torch.FloatTensor(yb).to(device)
            ub = torch.FloatTensor(ub).to(device)
            vb = torch.FloatTensor(vb).to(device)
            
        
        yb_wall.requires_grad = True
        
        net_in1 = torch.cat((xb, yb), 1)
        out1_u = net2_u(net_in1 )
        out1_v = net2_v(net_in1 )
        out1_u = out1_u.view(len(out1_u), -1)
        out1_v = out1_v.view(len(out1_v), -1)

       

        loss_f = nn.MSELoss()
        loss_bc = loss_f(out1_u, ub)+loss_f(out1_v,vb)
        return loss_bc
    ##############################################
    ###############################################

    # Main loop
    tic = time.time()

    # load low-fidelity data for initialization
    if (Flag_pretrain):
        print('Reading previous results')
        net2_u.load_state_dict(torch.load(path+"fwd_step_u_velocity_from_fluent_low_fidelity_100"+".pt"))
        net2_v.load_state_dict(torch.load(path+"fwd_step_v_velocity_from_fluent_low_fidelity_100"+".pt"))
   
    if (Flag_pretrain):    
        net2_u.eval()
        net2_v.eval()
    
    # INSTANTIATE STEP LEARNING SCHEDULER CLASS
    if (Flag_schedule):
        scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
        scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
        scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)
       

    if(Flag_batch):# This one uses dataloader

            
            loss_bc_array = np.zeros((epochs,1))
            loss_eqn_array = np.zeros((epochs,1))
            
            for epoch in range(epochs):
                loss_bc_n = 0
                loss_eqn_n = 0
                n = 0
                for batch_idx, (x_in,y_in) in enumerate(dataloader):
                    net2_u.zero_grad()
                    net2_v.zero_grad()
                    net2_p.zero_grad()
                    
                    loss_eqn = criterion(x_in,y_in)
                    loss_bc = Loss_BC(xb,yb,xb_wall,yb_wall,ub,vb,x,y)
                    loss = loss_eqn + Lambda_BC* loss_bc
                    loss.backward()
                    
                    optimizer_u.step() 
                    optimizer_v.step() 
                    optimizer_p.step()
                    
                    loss_eqn_a =loss_eqn.detach().cpu().numpy()
                    loss_eqn_n += loss_eqn_a
                    loss_bc_a= loss_bc.detach().cpu().numpy()
                    loss_bc_n += loss_bc_a  
                    n += 1         
                      
                    if batch_idx % 40 ==0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f} Loss Eqn {:.10f} Loss BC {:.6f}'.format(
                            epoch, batch_idx * len(x_in), len(dataloader.dataset),
                            100. * batch_idx / len(dataloader), loss.item(), loss_eqn.item(), loss_bc.item()))
                    
                if (Flag_schedule):
                        scheduler_u.step()
                        scheduler_v.step()
                        scheduler_p.step()
                        
    
                mean_eqn = loss_eqn_n/n
                mean_bc = loss_bc_n/n
                
                print('***Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f}'.format(mean_eqn, mean_bc) )
                print('****Epoch:', epoch,'learning rate is: ', optimizer_u.param_groups[0]['lr'])
                

                if epoch % 1000 == 0:#save network
                 torch.save(net2_u.state_dict(),path+"fwd_step_u_low_fidelity_initialization_"+str(epoch)+".pt")
                 torch.save(net2_v.state_dict(),path+"fwd_step_v_v_low_fidelity_initialization_"+str(epoch)+".pt")
                 torch.save(net2_p.state_dict(),path+"fwd_step_p_low_fidelity_initialization_"+str(epoch)+".pt")
       
              
            
    else:
        for epoch in range(epochs):
            
            net2_u.zero_grad()
            net2_v.zero_grad()
            net2_p.zero_grad()
            loss_eqn = criterion(x,y)
            loss_bc = Loss_BC(xb,yb,xb_wall,yb_wall,ub,vb,x,yl)
            if (Flag_BC_exact):
                loss = loss_eqn #+ loss_bc
            else:
                loss = loss_eqn + Lambda_BC * loss_bc
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
Flag_BC_exact = False #If True enforces BC exactly 
Flag_pretrain = True   # False for random initialization
Flag_initialization = False # True for random initialization
Lambda_BC  = 30 # weigh boundary condition loss


batchsize = 256  #Total number of batches 

epochs  = 10001
Flag_schedule = True #If true change the learning rate at 3 levels
if (Flag_schedule):
    learning_rate = 3e-4
    step_epoch = 3000 
    decay_rate = 0.1


x_scale = 2.8 #The length of the  domain 

Diff =0.01
rho = 1.0

nPt = 100
nPt1 = 50
xStart_out = 0.4
xEnd_out = 1.0
xStart_up = 0.3
xEnd_up = 0.4
xStart_in = 0
xEnd_in = 0.3
yStart = 0.
yEnd = 1.0
yStart_up = 0.5

# Geometry

x1 = np.linspace(xStart_in, xEnd_in, nPt)    #inlet
y1 = np.linspace(yStart, yEnd, nPt)
x1, y1 = np.meshgrid(x1, y1)
x1 = np.reshape(x1, (np.size(x1[:]),1))
y1 = np.reshape(y1, (np.size(y1[:]),1))

x2 = np.linspace(xStart_up  , xEnd_up, nPt)      #upper
y2 = np.linspace(yStart_up , yEnd, nPt1,endpoint=False)
x2, y2 = np.meshgrid(x2, y2)
x2 = np.reshape(x2, (np.size(x2[:]),1))
y2 = np.reshape(y2, (np.size(y2[:]),1))

x3 = np.linspace(xStart_out  , xEnd_out, nPt)     #outlet
y3 = np.linspace(yStart , yEnd, nPt1,endpoint=False)
x3, y3 = np.meshgrid(x3, y3)
x3 = np.reshape(x3, (np.size(x3[:]),1))
y3 = np.reshape(y3, (np.size(y3[:]),1))

x = np.concatenate((x1,x2,x3), axis=0)
y = np.concatenate((y1,y2,y3), axis=0)


print('shape of x',x.shape)
print('shape of y',y.shape)



U_BC_in = 0.5
T_BC_in = 0.0
T_wall = 1.0


#boundary conditions
nPt_BC = 2 *nPt
xleft = np.linspace(xStart_in, xStart_in, nPt_BC)
yleft = np.linspace(yStart, yEnd, nPt_BC)
xright = np.linspace(xEnd_out, xEnd_out, nPt_BC)
yright = np.linspace(yStart, yEnd, nPt_BC)
xup = np.linspace(xStart_in, xEnd_out, nPt_BC)
yup = np.linspace(yEnd, yEnd, nPt_BC)
xdown_1 = np.linspace(xStart_in, xEnd_in, nPt_BC)
ydown_1 = np.linspace(yStart, yStart, nPt_BC)
xdown_2 = np.linspace(xStart_out, xEnd_out, nPt_BC)
ydown_2 = np.linspace(yStart, yStart, nPt_BC)
xleft_fin = np.linspace(xEnd_in, xEnd_in, nPt_BC)
yleft_fin = np.linspace(yStart, yStart_up, nPt_BC)
xright_fin = np.linspace(xStart_out, xStart_out, nPt_BC)
yright_fin = np.linspace(yStart, yStart_up, nPt_BC)
xup_fin = np.linspace(xEnd_in, xStart_out, nPt_BC)
yup_fin = np.linspace(yStart_up, yStart_up, nPt_BC)


T_in_BC = np.linspace(T_BC_in, T_BC_in, nPt_BC)
T_wall_BC = np.linspace(T_wall, T_wall, nPt_BC)
u_in_BC = U_BC_in*(yleft[:])*(1-yleft[:])/0.25
v_in_BC = np.linspace(0., 0., nPt_BC)
u_wall_BC = np.linspace(0., 0., nPt_BC)
v_wall_BC = np.linspace(0., 0., nPt_BC)

xb = np.concatenate((xleft, xup, xdown_1,xdown_2,xleft_fin, xup_fin, xright_fin), 0)
yb = np.concatenate((yleft, yup, ydown_1,ydown_2,yleft_fin, yup_fin, yright_fin), 0)
ub = np.concatenate((u_in_BC, u_wall_BC, u_wall_BC,u_wall_BC,u_wall_BC, u_wall_BC,u_wall_BC), 0)
vb = np.concatenate((v_in_BC, v_wall_BC, v_wall_BC,v_wall_BC,v_wall_BC, v_wall_BC,v_wall_BC), 0)

xb_wall = np.concatenate((xdown_1,xdown_2,xup),0)
yb_wall = np.concatenate((ydown_1,ydown_2,yup),0)


xb= xb.reshape(-1, 1) #need to reshape to get 2D array
yb= yb.reshape(-1, 1) #need to reshape to get 2D array
ub= ub.reshape(-1, 1) #need to reshape to get 2D array
vb= vb.reshape(-1, 1) #need to reshape to get 2D array
xb_wall= xb_wall.reshape(-1, 1) #need to reshape to get 2D array
yb_wall= yb_wall.reshape(-1, 1)

print('shape of xb',xb.shape)
print('shape of yb',yb.shape)
print('shape of ub',ub.shape) 
print('shape of vb',vb.shape)
print('shape of xb_wall',xb_wall.shape)
print('shape of yb_wall',yb_wall.shape)


path = "Results/"


geo_train(device,x,y,xb,yb,xb_wall,yb_wall,ub,vb,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC,nPt )




 










 









