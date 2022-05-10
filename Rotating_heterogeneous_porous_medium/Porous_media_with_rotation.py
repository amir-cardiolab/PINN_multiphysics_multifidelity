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


def geo_train(device,x_in,y_in,xb_in,yb_in,xb_out,yb_out,xb_up,yb_up,xb_down,yb_down,batchsize,learning_rate,epochs,path,Flag_batch,Flag_BC_exact,Lambda_BC,nPt ):
    if (Flag_batch):
     x = torch.Tensor(x_in).to(device)
     y = torch.Tensor(y_in).to(device)
     xb_in = torch.Tensor(xb_in).to(device)
     yb_in = torch.Tensor(yb_in).to(device)
     xb_out = torch.Tensor(xb_out).to(device)
     yb_out = torch.Tensor(yb_out).to(device)
     xb_up = torch.Tensor(xb_up).to(device)
     yb_up = torch.Tensor(yb_up).to(device)
     xb_down = torch.Tensor(xb_down).to(device)
     yb_down = torch.Tensor(yb_down).to(device)
     
     
     if(1): #Cuda slower in double? 
         x = x.type(torch.cuda.FloatTensor)
         y = y.type(torch.cuda.FloatTensor)
         xb_in = xb_in.type(torch.cuda.FloatTensor)
         yb_in = yb_in.type(torch.cuda.FloatTensor)
         xb_out = xb_out.type(torch.cuda.FloatTensor)
         yb_out = yb_out.type(torch.cuda.FloatTensor)
         xb_up = xb_up.type(torch.cuda.FloatTensor)
         yb_up = yb_up.type(torch.cuda.FloatTensor)
         xb_down = xb_down.type(torch.cuda.FloatTensor)
         yb_down = yb_down.type(torch.cuda.FloatTensor)
        
     dataset = TensorDataset(x,y)
     dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
    else:
     x = torch.Tensor(x_in).to(device)
     y = torch.Tensor(y_in).to(device) 
     
   
    h_n = 70 #no. of neurons
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
 

    class Net2_psi(nn.Module):

        #The __init__ function stack the layers of the 
        #network Sequentially 
        def __init__(self):
            super(Net2_psi, self).__init__()
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
        #def forward(self,x):
        def forward(self,x):    
            output = self.main(x)
            return output 
            
    
    ################################################################

    net2_psi = Net2_psi().to(device)


    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)

    # use the modules apply function to recursively apply the initialization
  
    #net2_psi.apply(init_normal) #initialize by transfer learning


    optimizer_psi = optim.Adam(net2_psi.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
   
   
    ############################################################
    def criterion_psi(x,y):


        x.requires_grad = True
        y.requires_grad = True
        
        net_in = torch.cat((x,y),1)
        psi = net2_psi(net_in)
        psi = psi.view(len(psi),-1)
        

        source_term = torch.exp(-2*y)
        psi_x = torch.autograd.grad(psi,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        psi_xx = torch.autograd.grad(psi_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        psi_y = torch.autograd.grad(psi,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        psi_yy = torch.autograd.grad(psi_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        psi_xy = torch.autograd.grad(psi_x,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        psi_xxx = torch.autograd.grad(psi_xx,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        psi_yyy = torch.autograd.grad(psi_yy,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        psi_xyy = torch.autograd.grad(psi_xy,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        psi_yxx = torch.autograd.grad(psi_xy,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        psi_xxxx = torch.autograd.grad(psi_xxx,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        psi_yyyy = torch.autograd.grad(psi_yyy,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        psi_xxy= torch.autograd.grad(psi_xx,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        psi_xxyy= torch.autograd.grad(psi_xxy,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        psi_yyx= torch.autograd.grad(psi_yy,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        psi_yyxx= torch.autograd.grad(psi_yyx,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
       

        loss = psi_xx + psi_yy + psi_y - source_term


        # MSE LOSS
        loss_f = nn.MSELoss()

        #Note our target is zero. It is residual so we use zeros_like
        loss = loss_f(loss,torch.zeros_like(loss))

        return loss    
############################################################
 ###############################################################
    def calculate_vel( psi,x, y):  #find velocity given psi
        x.requires_grad = True
        y.requires_grad = True
        psi_x = torch.autograd.grad(psi,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        psi_y = torch.autograd.grad(psi,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        return psi_y, (-1*psi_x)

    ###################################################################
    def Loss_BC(xb_in,yb_in,xb_out,yb_out,xb_up,yb_up,xb_down,yb_down,x,y):
         
        if(0):
          xb_in = torch.FloatTensor(xb_in).to(device)
          yb_in = torch.FloatTensor(yb_in).to(device)
          xb_out = torch.FloatTensor(xb_out).to(device)
          yb_out = torch.FloatTensor(yb_out).to(device)
          xb_up = torch.FloatTensor(xb_up).to(device)
          yb_up = torch.FloatTensor(yb_up).to(device)
          xb_down = torch.FloatTensor(xb_down).to(device)
          yb_down = torch.FloatTensor(yb_down).to(device)
          
        xb_in.requires_grad = True
        xb_out.requires_grad = True
        yb_in.requires_grad = True
        yb_out.requires_grad = True
        xb_up.requires_grad = True
        xb_down.requires_grad = True
        yb_up.requires_grad = True
        yb_down.requires_grad = True
        
        net_in_in = torch.cat((xb_in, yb_in), 1)
        out_in = net2_psi(net_in_in)
        out_in = out_in.view(len(out_in), -1)

        net_in_out = torch.cat((xb_out, yb_out), 1)
        out_out = net2_psi(net_in_out)
        out_out = out_out.view(len(out_out), -1)

        net_in_up = torch.cat((xb_up, yb_up), 1)
        out_up = net2_psi(net_in_up)
        out_up = out_up.view(len(out_up), -1)

        net_in_down = torch.cat((xb_down, yb_down), 1)
        out_down = net2_psi(net_in_down)
        out_down = out_down.view(len(out_down), -1)

       
        
        psi_x_in = torch.autograd.grad(out_in,xb_in,grad_outputs=torch.ones_like(xb_in),create_graph = True,only_inputs=True)[0]
        psi_y_in = torch.autograd.grad(out_in,yb_in,grad_outputs=torch.ones_like(yb_in),create_graph = True,only_inputs=True)[0]
        psi_x_out = torch.autograd.grad(out_out,xb_out,grad_outputs=torch.ones_like(xb_out),create_graph = True,only_inputs=True)[0]
        psi_y_out = torch.autograd.grad(out_out,yb_out,grad_outputs=torch.ones_like(yb_out),create_graph = True,only_inputs=True)[0]
        psi_x_up = torch.autograd.grad(out_up,xb_up,grad_outputs=torch.ones_like(xb_up),create_graph = True,only_inputs=True)[0]
        psi_y_up = torch.autograd.grad(out_up,yb_up,grad_outputs=torch.ones_like(yb_up),create_graph = True,only_inputs=True)[0]
        psi_x_down = torch.autograd.grad(out_down,xb_down,grad_outputs=torch.ones_like(xb_down),create_graph = True,only_inputs=True)[0]
        psi_y_down = torch.autograd.grad(out_down,yb_down,grad_outputs=torch.ones_like(yb_down),create_graph = True,only_inputs=True)[0]
        psi_xx_in = torch.autograd.grad(psi_x_in,xb_in,grad_outputs=torch.ones_like(xb_in),create_graph = True,only_inputs=True)[0]
        psi_yy_in = torch.autograd.grad(psi_y_in,yb_in,grad_outputs=torch.ones_like(yb_in),create_graph = True,only_inputs=True)[0]
        psi_xx_out = torch.autograd.grad(psi_x_out,xb_out,grad_outputs=torch.ones_like(xb_out),create_graph = True,only_inputs=True)[0]
        psi_yy_out = torch.autograd.grad(psi_y_out,yb_out,grad_outputs=torch.ones_like(yb_out),create_graph = True,only_inputs=True)[0]
        psi_xx_up = torch.autograd.grad(psi_x_up,xb_up,grad_outputs=torch.ones_like(xb_up),create_graph = True,only_inputs=True)[0]
        psi_yy_up = torch.autograd.grad(psi_y_up,yb_up,grad_outputs=torch.ones_like(yb_up),create_graph = True,only_inputs=True)[0]
        psi_xx_down = torch.autograd.grad(psi_x_down,xb_down,grad_outputs=torch.ones_like(xb_down),create_graph = True,only_inputs=True)[0]
        psi_yy_down = torch.autograd.grad(psi_y_down,yb_down,grad_outputs=torch.ones_like(yb_down),create_graph = True,only_inputs=True)[0]
        
        loss_f = nn.MSELoss()
       
        loss = loss_f(out_in,torch.zeros_like(out_in))+loss_f(out_out,torch.zeros_like(out_out))+loss_f(out_up,torch.zeros_like(out_up))+loss_f(out_down,torch.zeros_like(out_down))
        return loss


    # Main loop
    tic = time.time()
    
    #load low-fidelity results
    if (Flag_pretrain):
        print('Reading previous results')
        net2_psi.load_state_dict(torch.load(path+"psi_porous_media_with_rotation_from_fenics_100"+".pt"))
    
    net2_psi.eval()    
    # INSTANTIATE STEP LEARNING SCHEDULER CLASS
    if (Flag_schedule):
        
        scheduler_psi = torch.optim.lr_scheduler.StepLR(optimizer_psi, step_size=step_epoch, gamma=decay_rate)


    if(Flag_batch):# This one uses dataloader
           
            for epoch in range(epochs):
                loss_bc_n = 0
                loss_eqn_n = 0
                n = 0
                for batch_idx, (x_in,y_in) in enumerate(dataloader):
                    
                    net2_psi.zero_grad()
            
                    loss_eqn = criterion_psi(x_in,y_in) 
                    loss_bc = Loss_BC(xb_in,yb_in,xb_out,yb_out,xb_up,yb_up,xb_down,yb_down,x,y)
                    loss = loss_eqn + Lambda_BC* loss_bc
                    loss.backward()
                
                    optimizer_psi.step()
                    
                    loss_eqn_a =loss_eqn.detach().cpu().numpy()
                    loss_eqn_n += loss_eqn_a
                    loss_bc_a= loss_bc.detach().cpu().numpy()
                    loss_bc_n += loss_bc_a 
                    n += 1         
                      
                    if batch_idx % 40 ==0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f} Loss eqn {:.15f} Loss BC {:.15f}'.format(
                            epoch, batch_idx * len(x_in), len(dataloader.dataset),
                            100. * batch_idx / len(dataloader), loss.item(), loss_eqn.item(), loss_bc.item()))
                    
                if (Flag_schedule):
                    
                        scheduler_psi.step()

                
    
                mean_eqn = loss_eqn_n/n
                mean_bc = loss_bc_n/n
                print('***Total avg Loss : Loss eqn {:.15f} Loss BC {:.15f}'.format(mean_eqn, mean_bc) )
                print('****Epoch:', epoch,'learning rate is: ', optimizer_psi.param_groups[0]['lr'])
                
                
                if epoch % 1000 == 0:#save network
                 
                 torch.save(net2_psi.state_dict(),path+"fwd_psi_porous_media_with_rotation_low_fidelity_intialization_"+str(epoch)+".pt")
                
           
    else:
        for epoch in range(epochs):
            
            net2_psi.zero_grad()
            loss_eqn = criterion_psi(x_in,y_in)
            loss_bc = Loss_BC(xb_in,yb_in,xb_out,yb_out,xb_up,yb_up,xb_down,yb_down,x,y)
            if (Flag_BC_exact):
                loss = loss_eqn 
            else:
                loss = loss_eqn + Lambda_BC * loss_bc
            loss.backward()
           
            optimizer_psi.step() 
             
            if epoch % 1000 ==0:
                print('Train Epoch: {} \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
                    epoch, loss.item(),loss_bc.item()))

    toc = time.time()
    elapseTime = toc - tic
    print ("elapse time in parallel = ", elapseTime)
    ###################
    #plot
    net2_psi.eval()
    
    net_in = torch.cat((x.requires_grad_(),y.requires_grad_()),1)
    psi_out = net2_psi(net_in)
    output_u ,output_v = calculate_vel(psi_out,x,y)
    output_u = output_u.cpu().data.numpy() #need to convert to cpu before converting to numpy
    output_v = output_v.cpu().data.numpy()
    psi_out = psi_out.cpu().data.numpy()
    
    x = x.cpu()
    y = y.cpu()

    

    return

#######################################################
#Main code:
device = torch.device("cuda")

Flag_batch = True #False #USe batch or not  #With batch getting error...
Flag_BC_exact = False #If True enforces BC exactly HELPS ALOT!!! Not implemented in 2D
Flag_pretrain = True


Lambda_BC  = 20.00 


batchsize = 256 #Total number of batches 

epochs  = 10001 
Flag_schedule = True #If true change the learning rate at 3 levels
if (Flag_schedule):
    learning_rate = 3e-4
    step_epoch = 3000 
    decay_rate = 0.1


nPt = 161
xStart = 0.
xEnd = 1.0
yStart = 0.
yEnd = 1.0


#geometry

x = np.linspace(xStart, xEnd, nPt)    
y = np.linspace(yStart, yEnd, nPt)
x, y = np.meshgrid(x, y)
x = np.reshape(x, (np.size(x[:]),1))
y = np.reshape(y, (np.size(y[:]),1))


print('shape of x',x.shape)
print('shape of y',y.shape)


#boundary conditions
nPt_BC = 2 *nPt
xb_in = np.linspace(xStart, xStart, nPt_BC)
yb_in = np.linspace(yStart, yEnd, nPt_BC)
xb_out = np.linspace(xEnd, xEnd, nPt_BC)
yb_out = np.linspace(yStart, yEnd, nPt_BC)
xb_up = np.linspace(xStart, xEnd, nPt_BC)
yb_up = np.linspace(yEnd, yEnd, nPt_BC)
xb_down = np.linspace(xStart, xEnd, nPt_BC)
yb_down = np.linspace(yStart, yStart, nPt_BC)


xb_in= xb_in.reshape(-1, 1) #need to reshape to get 2D array
yb_in= yb_in.reshape(-1, 1) #need to reshape to get 2D array
xb_out= xb_out.reshape(-1, 1) #need to reshape to get 2D array
yb_out= yb_out.reshape(-1, 1) #need to reshape to get 2D array
xb_up= xb_up.reshape(-1, 1) #need to reshape to get 2D array
yb_up= yb_up.reshape(-1, 1) #need to reshape to get 2D array
xb_down= xb_down.reshape(-1, 1) #need to reshape to get 2D array
yb_down= yb_down.reshape(-1, 1) #need to reshape to get 2D array

print('shape of xb_in',xb_in.shape)
print('shape of yb_in',yb_in.shape)
print('shape of xb_out',xb_out.shape)
print('shape of yb_out',yb_out.shape)
print('shape of xb_up',xb_up.shape) 
print('shape of yb_up',yb_up.shape)
print('shape of xb_down',xb_down.shape)
print('shape of yb_down',yb_down.shape)


path = "Results/"


geo_train(device,x,y,xb_in,yb_in,xb_out,yb_out,xb_up,yb_up,xb_down,yb_down,batchsize,learning_rate,epochs,path,Flag_batch,Flag_BC_exact,Lambda_BC,nPt )









