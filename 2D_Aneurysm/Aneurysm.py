import torch
import numpy as np
#import foamFileOperation
#from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
import vtk
from vtk.util import numpy_support as VN
#Solve steady 2D N.S.

#Maziar's 2D paper: 10 layer, 256 neurons per layer, Tanh or Swish, Relu cannot do second derivative. 

def geo_train(device,x_in,y_in,xb,yb,ub,vb,xb_in,yb_in,u_in_BC,v_in_BC,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC):
    if (Flag_batch):
     x = torch.Tensor(x_in).to(device)
     y = torch.Tensor(y_in).to(device)
     xb = torch.Tensor(xb).to(device)
     yb = torch.Tensor(yb).to(device)
     ub = torch.Tensor(ub).to(device)
     vb = torch.Tensor(vb).to(device)
     xb_in = torch.Tensor(xb_in).to(device)
     yb_in = torch.Tensor(yb_in).to(device)
     u_in_BC = torch.Tensor(u_in_BC).to(device)
     v_in_BC = torch.Tensor(v_in_BC).to(device)
     if(1): #Cuda slower in double? 
         x = x.type(torch.cuda.FloatTensor)
         y = y.type(torch.cuda.FloatTensor)
         xb = xb.type(torch.cuda.FloatTensor)
         yb = yb.type(torch.cuda.FloatTensor)
         ub = ub.type(torch.cuda.FloatTensor)
         vb = vb.type(torch.cuda.FloatTensor)
         xb_in = xb_in.type(torch.cuda.FloatTensor)
         yb_in = yb_in.type(torch.cuda.FloatTensor)
         u_in_BC = u_in_BC.type(torch.cuda.FloatTensor)
         v_in_BC = v_in_BC.type(torch.cuda.FloatTensor)
     
     dataset = TensorDataset(x,y)
     dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
    else:
     x = torch.Tensor(x_in).to(device)
     y = torch.Tensor(y_in).to(device) 
     

    h_n = 170 #80 #50  #20 #128
    input_n = 2 # this is what our answer is a function of. In the original example 3 : x,y,scale 

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
#   net2_u.apply(init_normal) #initialize by TL
#   net2_v.apply(init_normal)
    net2_p.apply(init_normal)
    

    ############################################################################

    optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
    optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)


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

    
        XX_scale = (X_scale**2)
        YY_scale = (Y_scale**2)
        UU_scale  = U_scale **2
    
        loss_2 = u*u_x*UU_scale / X_scale + v*u_y*U_scale/ Y_scale - Diff*( u_xx*U_scale/XX_scale  + u_yy*U_scale /YY_scale  )+ 1/rho* (P_x / X_scale   )  #X-dir
        loss_1 = u*v_x*U_scale / X_scale + v*v_y / Y_scale - Diff*( v_xx/ XX_scale + v_yy / YY_scale )+ 1/rho*(P_y / Y_scale   ) #Y-dir
        loss_3 = (u_x*U_scale / X_scale + v_y / Y_scale) #continuity
        

        # MSE LOSS
        loss_f = nn.MSELoss()
        

        #Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss_1,torch.zeros_like(loss_1))+ loss_f(loss_2,torch.zeros_like(loss_2))+loss_f(loss_3,torch.zeros_like(loss_3))
        
        return loss
    ############################################################
    
    def Loss_BC(xb,yb,ub,vb,xb_in,yb_in,u_in_BC,v_in_BC,x,y):
        if(0):
            xb = torch.FloatTensor(xb).to(device)
            yb = torch.FloatTensor(yb).to(device)
            ub = torch.FloatTensor(ub).to(device)
            vb = torch.FloatTensor(vb).to(device)
            
    
        net_in = torch.cat((xb_in, yb_in), 1)
        out1_u_in = net2_u(net_in )
        out1_v_in = net2_v(net_in )
        out1_u_in = out1_u_in.view(len(out1_u_in), -1)
        out1_v_in = out1_v_in.view(len(out1_v_in), -1)
        
    
        net_in1 = torch.cat((xb, yb), 1)
        out1_u = net2_u(net_in1 )
        out1_v = net2_v(net_in1 )
        out1_u = out1_u.view(len(out1_u), -1)
        out1_v = out1_v.view(len(out1_v), -1)

        
        loss_f = nn.MSELoss()
        loss_bc = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v,torch.zeros_like(out1_v)) + loss_f(out1_u_in, u_in_BC) + loss_f(out1_v_in, v_in_BC) 
        return loss_bc
    ##############################################


    # Main loop
    tic = time.time()

    #load cfd results
    if (Flag_pretrain):
        print('Reading previous results')
        net2_u.load_state_dict(torch.load(path+"fwd_step_u_anurysm_from_fluent_6layers_100"+".pt"))
        net2_v.load_state_dict(torch.load(path+"fwd_step_v_anurysm_from_fluent_6layers_100"+".pt"))

    

    # INSTANTIATE STEP LEARNING SCHEDULER CLASS
    if (Flag_schedule):
        scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
        scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
        scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)
    
    net2_u.eval()
    net2_v.eval()   
    

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
                    loss_bc = Loss_BC(xb,yb,ub,vb,xb_in,yb_in,u_in_BC,v_in_BC,x,y)
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
                 torch.save(net2_u.state_dict(),path+"fwd_step_u_anurysm_low_fidelity_initialization_"+str(epoch)+".pt")
                 torch.save(net2_v.state_dict(),path+"fwd_step_v_anurysm_low_fidelity_initialization_"+str(epoch)+".pt")
                 torch.save(net2_p.state_dict(),path+"fwd_step_p_anurysm_low_fidelity_initialization_"+str(epoch)+".pt")
                
                 
        
    else:
        for epoch in range(epochs):
            
            net2_u.zero_grad()
            net2_v.zero_grad()
            net2_T.zero_grad()
            loss_eqn = criterion(x,y)
            loss_bc = Loss_BC(xb,yb,ub,vb,xb_in,yb_in,u_in_BC,v_in_BC,x,y)
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

Flag_batch = True #False #USe batch or not  #With batch getting error...
Flag_BC_exact = False #If True enforces BC exactly HELPS ALOT!!! Not implemented in 2D
Flag_pretrain = True
Lambda_BC  = 70 

batchsize = 256  #Total number of batches 
epochs  = 10001
Flag_schedule = True #If true change the learning rate at 3 levels
if (Flag_schedule):
    learning_rate = 3e-4
    step_epoch = 3000 #100
    decay_rate = 0.1


Diff = 0.001
rho = 1 

X_scale = 2.5 #The length of the  domain 
Y_scale = 1.0 
U_scale = 2.0
U_BC_in = 2.0



Directory = "/scratch/ma3367/Files/Anurysm/"
mesh_file = Directory + "anurysm_mesh.vtk"
bc_file_in = Directory + "anurysm_inlet_boundary.vtk"
bc_file_wall = Directory + "anurysm_wall_boundary.vtk"

print ('Loading', mesh_file)
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(mesh_file)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of the mesh:' ,n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
    pt_iso  =  data_vtk.GetPoint(i)
    x_vtk_mesh[i] = pt_iso[0]   
    y_vtk_mesh[i] = pt_iso[1]
    VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

x  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
y  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))



print('shape of x',x.shape)
print('shape of y',y.shape)

print ('Loading', bc_file_in)
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(bc_file_in)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of at inlet' ,n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
    pt_iso  =  data_vtk.GetPoint(i)
    x_vtk_mesh[i] = pt_iso[0]   
    y_vtk_mesh[i] = pt_iso[1]
    VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb_in  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh[:]),1)) 
yb_in  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh[:]),1))

print ('Loading', bc_file_wall)
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(bc_file_wall)
reader.Update()
data_vtk = reader.GetOutput()
n_pointsw = data_vtk.GetNumberOfPoints()
print ('n_points of at wall' ,n_pointsw)
x_vtk_mesh = np.zeros((n_pointsw,1))
y_vtk_mesh = np.zeros((n_pointsw,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_pointsw):
    pt_iso  =  data_vtk.GetPoint(i)
    x_vtk_mesh[i] = pt_iso[0]   
    y_vtk_mesh[i] = pt_iso[1]
    VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
yb  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))



u_in_BC = (yb_in[:]) * ( 0.3 - yb_in[:] )  / 0.0225 * U_BC_in #parabolic


v_in_BC = np.linspace(0., 0., n_points)
ub = np.linspace(0., 0., n_pointsw)
vb = np.linspace(0., 0., n_pointsw)



xb= xb.reshape(-1, 1) #need to reshape to get 2D array
yb= yb.reshape(-1, 1) #need to reshape to get 2D array
ub= ub.reshape(-1, 1) #need to reshape to get 2D array
vb= vb.reshape(-1, 1) #need to reshape to get 2D array
xb_in= xb_in.reshape(-1, 1) #need to reshape to get 2D array
yb_in= yb_in.reshape(-1, 1) #need to reshape to get 2D array
u_in_BC= u_in_BC.reshape(-1, 1) #need to reshape to get 2D array
v_in_BC= v_in_BC.reshape(-1, 1) #need to reshape to get 2D array

print('shape of xb',xb.shape)
print('shape of yb',yb.shape)
print('shape of ub',ub.shape)
print('shape of vb',vb.shape)


path = "Results/"


geo_train(device,x,y,xb,yb,ub,vb,xb_in,yb_in,u_in_BC,v_in_BC,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC )










