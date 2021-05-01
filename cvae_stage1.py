import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
from random import randint

import matplotlib.pyplot as plt
import os
import numpy as np

path_directory = '/home/SharedData/Ashvini/ablation'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = open("./log/var_conv_ood_7.txt", "a")
print("Starting Training..............",file=log)
transforms = transforms.Compose([transforms.ToTensor()])

sliding_win_len = 7
train_folder = '/datasets/train_7/'
val_folder = '/datasets/val_7/'

################ Train Data Dictionary #########################
train_files=os.listdir(path_directory+train_folder)
dict_train={}
index=0
for file_3 in train_files:
    i=path_directory+train_folder+file_3
    dict_train[index]=i
    index=index+1

#print('No. of training examples',len(dict_train))


################# Validation Data Dictionary ###############
val_files=os.listdir(path_directory+val_folder)
dict_val={}
index=0
for file_3 in val_files:
	i=path_directory+val_folder+file_3
	dict_val[index]=i
	index=index+1
	#print(i)

#print('No. of validation examples',len(dict_val))


num_channels = 17 

############## Train Dataset ############################
class dataset_train(Dataset):
    
	def __init__(self,transform=None):
		self.dummy=0

	def __len__(self):
		return (len(dict_train))

	def __getitem__(self, idx):
		
		self.data=np.load(dict_train[idx])

#data has one numpy file loaded
		self.data_cord=self.data[:,3:3+2*num_channels].T 
		#print(self.data_cord[:,sliding_win_len:])
		self.data_cord_input=self.data_cord[:,sliding_win_len:]/500.0
		self.data_cord_condition = self.data_cord[:,:sliding_win_len]/500.0
		
		#data_cord_input has the future frames, to be used for recon. (Input to Encoder)
		#data_cord_cond has the present frames to be given as condition to encoder 

		self.data_conf=self.data[:,3+2*num_channels:3+3*num_channels].T
		
		self.data_conf_input=self.data_conf[:,sliding_win_len:]
		self.data_conf_condition = self.data_conf[:,:sliding_win_len]
		
		
#data_conf has the confidence values
		self.data_conf=np.concatenate((self.data_conf, self.data_conf), axis=0)
		self.data_conf_input=np.concatenate((self.data_conf_input, self.data_conf_input), axis=0)
		self.data_conf_condition=np.concatenate((self.data_conf_condition, self.data_conf_condition), axis=0)

		self.person_id=int(self.data[0,54])
		self.cam_id=int(self.data[0,0])
		self.vid_id=int(self.data[0,1])
		self.frames=self.data[:,2]



		return self.data_cord_input,self.data_conf_input,\
self.data_cord_condition,self.data_conf_condition

############## Val Dataset ############################
class dataset_val(Dataset):
    
	def __init__(self,transform=None):
		self.dummy=0

	def __len__(self):
		return (len(dict_val))

	def __getitem__(self, idx):
		
		self.data=np.load(dict_val[idx])

#data has one numpy file loaded
		self.data_cord=self.data[:,3:3+2*num_channels].T 
#data_cord has the x,y co-ordinates 

		self.data_cord_input=self.data_cord[:,sliding_win_len:]/500.0
		self.data_cord_condition = self.data_cord[:,:sliding_win_len]/500.0		
		#data_cord_input has the future frames, to be used for recon. (Input to Encoder)
		#data_cord_cond has the present frames to be given as condition to encoder 

		self.data_conf=self.data[:,3+2*num_channels:3+3*num_channels].T
		
		self.data_conf_input=self.data_conf[:,sliding_win_len:]
		self.data_conf_condition = self.data_conf[:,:sliding_win_len]
		
		
#data_conf has the confidence values
		self.data_conf=np.concatenate((self.data_conf, self.data_conf), axis=0)
		self.data_conf_input=np.concatenate((self.data_conf_input, self.data_conf_input), axis=0)
		self.data_conf_condition=np.concatenate((self.data_conf_condition, self.data_conf_condition), axis=0)

		self.person_id=int(self.data[0,54])
		self.cam_id=int(self.data[0,0])
		self.vid_id=int(self.data[0,1])
		self.frames=self.data[:,2]



		return self.data_cord_input,self.data_conf_input,\
self.data_cord_condition,self.data_conf_condition

##################### Encoder ################################
batch_size = 128
n_epochs = 2000

emb0 = 68
emb1 = 68
emb2 = 68

input_dim = 4*num_channels*1
hidden_dim1 = 4*num_channels*1
hidden_dim2 = 2*num_channels*1
latent_dim = num_channels*1
cond_dim = 4*num_channels*1

lr = 1e-3
#print("Learning rate: ", lr)

class Conv_Encoder1(nn.Module):
    def __init__(self):
        super(Conv_Encoder1,self).__init__()
        
        ###input shape = [batchsize = 32, channels = 34, sliding_len = 7 ]
        self.conv1 = nn.Conv1d(2*num_channels,emb0,3) ### size = [batchsize, 28,5]
        self.conv2 = nn.Conv1d(emb0,emb1,3) ### size = [batchsize,14,3]
        self.conv3 = nn.Conv1d(emb1,emb2,3) ### size = [batchsize,7,1] 
        
    def forward(self,x):
        #print(x.shape)
        st_0 = F.relu(self.conv1(x))
        #print(st_0.shape)
        st_1 = F.relu(self.conv2(st_0))
        st_2 = F.relu(self.conv3(st_1))
        
        return st_2


        
class Conv_Encoder2(nn.Module):
    def __init__(self):
        super(Conv_Encoder2,self).__init__()
        
        ###input shape = [batchsize = 32, channels = 34, sliding_len = 7 ]
        self.conv1 = nn.Conv1d(2*num_channels,emb0,3) ### size = [batchsize, 28,5]
        self.conv2 = nn.Conv1d(emb0,emb1,3) ### size = [batchsize,14,3]
        self.conv3 = nn.Conv1d(emb1,emb2,3) ### size = [batchsize,7,1] 
        
    def forward(self,x):
        #print(x.shape)
        st_0 = F.relu(self.conv1(x))
        #print(st_0.shape)
        st_1 = F.relu(self.conv2(st_0))
        st_2 = F.relu(self.conv3(st_1))
        
        return st_2
       


class Conv_Decoder(nn.Module):
    def __init__(self):
        super(Conv_Decoder,self).__init__()
        
        self.conv1tr = nn.ConvTranspose1d(emb2,emb1,3)
        self.conv2tr = nn.ConvTranspose1d(emb1,emb0,3)
        self.conv3tr = nn.ConvTranspose1d(emb0,2*num_channels,3)
        
    def forward(self,x):
        dec_0 = F.relu(self.conv1tr(x))
        dec_1 = F.relu(self.conv2tr(dec_0))
        dec_2 = F.relu(self.conv3tr(dec_1))
        
        return dec_2

class Encoder(nn.Module):
    def __init__(self,input_dim,hidden_dim1,hidden_dim2,latent_dim,cond_dim):
    
        super(Encoder,self).__init__()
        
        self.linear1 = nn.Linear(input_dim+cond_dim, hidden_dim1)
        self.linear2 = nn.Linear(hidden_dim1,hidden_dim2)
        self.mu = nn.Linear(hidden_dim2, latent_dim)
        self.var = nn.Linear(hidden_dim2, latent_dim)
    
    
    def forward(self,x):
        
        hidden1 = F.relu(self.linear1(x))
        hidden2 = F.relu(self.linear2(hidden1))
        mean = self.mu(hidden2)
        log_var = self.var(hidden2)
        
        return mean,log_var

class Decoder(nn.Module):
    def __init__(self,latent_dim,hidden_dim2,hidden_dim1,output_dim,cond_dim):
    
        super(Decoder,self).__init__()
        
        self.latent_to_hidden2 = nn.Linear(latent_dim+cond_dim,hidden_dim2)
        self.hidden2_to_hidden1 = nn.Linear(hidden_dim2,hidden_dim1)
        self.hidden_to_out = nn.Linear(hidden_dim1,output_dim)
        
        self.hidden_to_bce = nn.Linear(hidden_dim1,1) #adding the output layer for BCE
        
    def forward(self,x):
        
        hidden1 = F.relu(self.latent_to_hidden2(x))
        x = F.relu(self.hidden2_to_hidden1(hidden1))
        generated_x = self.hidden_to_out(x)
        
        class_p = torch.sigmoid(self.hidden_to_bce(x)) # class_p is the class probability of normal VS abnormal
        
        
        return generated_x, class_p

#MOG mean list
nu_of_gaussian = 10
mean = []
mean = [0,5,6,7,8,9,-5,-6,-7,-8,-9] # defining gaussian with mean and std=1
normal_mean = [0,0.5,0.75,1,1.5,2,-0.5,-0.75,-1,-1.5,-2]

class CVAE(nn.Module):
    def __init__(self,input_dim,hidden_dim1,hidden_dim2,latent_dim,cond_dim):
    
        super(CVAE,self).__init__()
        
        self.encoder_inp = Conv_Encoder1()
        self.encoder_cond = Conv_Encoder1()
        
        self.encoder = Encoder(input_dim,hidden_dim1,hidden_dim2,latent_dim,cond_dim)
        self.decoder = Decoder(latent_dim,hidden_dim2,hidden_dim1,input_dim,cond_dim)
        self.conv_decoder = Conv_Decoder()
    
    
    def forward(self,inp,cond,bce_flag):
        x = self.encoder_inp(inp)
        y = self.encoder_cond(cond)
        if(bce_flag == -1):
            x_hat = self.conv_decoder(x)
            y_hat = self.conv_decoder(y)
            return x_hat,y_hat
            
        elif(bce_flag==0):
                #print("shapes x y",x.shape,y.shape)
                x = x.view(-1,emb2*(sliding_win_len-6))
                y = y.view(-1,emb2*(sliding_win_len-6))        
                x = torch.cat((x,y),dim=1) #shape 6528(32*34*sliding_win_len*2)
                x = x.float()
                #print(x.shape)
                z_mu,z_var = self.encoder(x)
                
                std = torch.exp(z_var/2)
                eps = torch.randn_like(std)
                x_sample = eps.mul(std).add(z_mu)
                x_sample = x_sample.float()
                y = y.float()
                z = torch.cat((x_sample,y),dim=1)
                z = z.float()
                
                generated_x, class_p = self.decoder(z)
                generated_x = generated_x.view(-1,emb2,1)
                final_x = self.conv_decoder(generated_x)
                
                return final_x,class_p, z_mu, z_var
                
        elif(bce_flag==1):
                x = x.view(-1,emb2*(sliding_win_len-6))
                y = y.view(-1,emb2*(sliding_win_len-6))
                x = torch.cat((x,y),dim=1) #shape 6528(32*34*sliding_win_len*2)
                x = x.float()

                
                class_g = randint(1,10) #generate a random class number
                
                z_mu,z_var = self.encoder(x) #x is the input, y is the condition
                
                ############ sampling from MOG ################
                std = torch.exp(z_var/2) #used just for dimensional consistency
                
                eps = torch.randn_like(std)
                x_sample = eps.add(mean[class_g])
                
                x_sample = x_sample.float()
                #############################################
                
                
                y = y.float()
                z = torch.cat((x_sample,y),dim=1)
                z = z.float()
                
                generated_x, class_p_x = self.decoder(z)
                generated_x = generated_x.view(-1,emb2,1)
                final_x = self.conv_decoder(generated_x)
                
                return final_x, class_p_x,z_mu, z_var

        elif(bce_flag==2):
                x = x.view(-1,emb2*(sliding_win_len-6))
                y = y.view(-1,emb2*(sliding_win_len-6))
                x = torch.cat((x,y),dim=1) #shape 6528(32*34*sliding_win_len*2)
                x = x.float()

                
                class_g = randint(1,10) #generate a random class number
                
                z_mu,z_var = self.encoder(x) #x is the input, y is the condition
                
                ############ sampling from MOG ################
                std = torch.exp(z_var/2) #used just for dimensional consistency
                
                eps = torch.randn_like(std)
                #x_sample = eps.add(normal_mean[class_g])
                x_sample = eps.mul(std).add(z_mu)
                
                x_sample = x_sample.float()
                #############################################
                
                
                y = y.float()
                z = torch.cat((x_sample,y),dim=1)
                z = z.float()
                
                generated_x, class_p_x = self.decoder(z)
                generated_x = generated_x.view(-1,emb2,1)
                final_x = self.conv_decoder(generated_x)
                
                return final_x, class_p_x,z_mu, z_var
        

####################################### loading checkpoint to resume training ####################
       
        

model = CVAE(input_dim,hidden_dim1,hidden_dim2,latent_dim,cond_dim).to(device)
model = model.float()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=3)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1, gamma=0.99)
'''
epoch = 0

'''
checkpoint = torch.load('./checkpoints/checkpoints_var_conv_ood_7/pretrain_160.pth') #checkpoint_class0_400.pth
#print(checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch'] + 1
print("Loading Epoch:",epoch)


train_loss = checkpoint['train_loss']
val_loss = checkpoint['val_loss']
print(f'train_loss :: {train_loss}, val_loss :: {val_loss}')

#########################################################

def set_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(param_group['lr'])

transformed_dataset_train = dataset_train()

transformed_dataset_val = dataset_val()


trainloader = torch.utils.data.DataLoader(transformed_dataset_train, batch_size=batch_size,shuffle=True, num_workers=4,drop_last=True)
#print("Data_loader for Train is Ready ")


valloader = torch.utils.data.DataLoader(transformed_dataset_val, batch_size=batch_size,shuffle=True, num_workers=4,drop_last=False)
#print("Data_loader for Validation is Ready ")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def loss_calc(x,recons_x,mean,log_var):
    RCL = F.mse_loss(recons_x,x,size_average = False)
    #print(x)
    KLD = -0.5*torch.sum(1+log_var-mean.pow(2)-log_var.exp())
    
    
    
    return RCL+KLD
    
def mse_loss(x,recons_x):
    RCL = F.mse_loss(recons_x,x,size_average = False)
    
    return RCL
    

epsilon = 0    
def loss_bce(true_label, prob): # TRUE label: 0 for normal, 1 for abnormal
        
            
    #BCE =  -1*(true_label*(torch.log(prob+epsilon)) + (1-true_label)*(torch.log(1-prob-epsilon)))
    
    BCE_layer = nn.BCELoss()
    #true_label = torch.tensor(true_label).float().cuda()
    #print(type(prob), type(true_label))    
    BCE = BCE_layer(prob,true_label)
    #print(type(BCE))    
    return BCE
    
    

def pretrain():
    model.train()
    train_loss = 0
    for i,data in enumerate(trainloader):
        
        ip_cord, ip_conf, condition_cord, condition_conf  = data
        
        ip_cord = ip_cord.to(device) #future frames (4,5,6) # shape (32,34,sliding_win_len)
        ip_conf = ip_conf.to(device) #future frames 
        condition_cord = condition_cord.to(device) #condition frames (1,2,3)
        condition_conf = condition_conf.to(device) #future frames
        
        
        
        optimizer.zero_grad()
        #print(ip_cord.shape)
        x = ip_cord #flattening #shape 3264
        x = x.float() 
        y = condition_cord 
        y = y.float()
        
        x_hat,y_hat = model(x,y,-1)
        loss1 = mse_loss(x,x_hat)
        loss2 = mse_loss(y,y_hat)
        
        loss1.backward()
        loss2.backward()
        train_loss += loss1.item()+loss2.item()
        optimizer.step()
        
    return train_loss



def pretrain_val():
    model.eval()
    val_loss = 0
    for i,data in enumerate(valloader):
        
        ip_cord, ip_conf, condition_cord, condition_conf  = data
        
        ip_cord = ip_cord.to(device) #future frames (4,5,6) # shape (32,34,sliding_win_len)
        ip_conf = ip_conf.to(device) #future frames 
        condition_cord = condition_cord.to(device) #condition frames (1,2,3)
        condition_conf = condition_conf.to(device) #future frames
        
        
        
        optimizer.zero_grad()
        #print(ip_cord.shape)
        x = ip_cord #flattening #shape 3264
        x = x.float() 
        y = condition_cord 
        y = y.float()
        
        x_hat,y_hat = model(x,y,-1)
        loss1 = mse_loss(x,x_hat)
        loss2 = mse_loss(y,y_hat)
        val_loss += loss1.item()+loss2.item()
    
    return val_loss

def train():
    model.train()
    train_loss = 0
    
    for i,data in enumerate(trainloader):
        
        
        ip_cord, ip_conf, condition_cord, condition_conf  = data
        
        ip_cord = ip_cord.to(device) #future frames (4,5,6) # shape (32,34,sliding_win_len)
        ip_conf = ip_conf.to(device) #future frames 
        condition_cord = condition_cord.to(device) #condition frames (1,2,3)
        condition_conf = condition_conf.to(device) #future frames
        
        
        
        optimizer.zero_grad()
        
        x = ip_cord #flattening #shape 3264
        x = x.float() 
        y = condition_cord
        y = y.float()
        
        
        recons_x,class_p, z_mu, z_var = model(x,y,0) #one without bce
        
        
        recons_x = recons_x.float()
        z_mu = z_mu.float()
        z_var = z_var.float()
        
        loss = loss_calc(x,recons_x,z_mu,z_var)
        
        loss.backward()
        
        train_loss += loss.item()
        
        optimizer.step()
        #print(loss)
    
    return train_loss

def train_bce():
    model.train()
    train_loss = 0
    
    for i,data in enumerate(trainloader):
        
        ip_cord, ip_conf, condition_cord, condition_conf  = data
        
        ip_cord = ip_cord.to(device) #future frames (4,5,6) # shape (32,34,sliding_win_len)
        ip_conf = ip_conf.to(device) #future frames 
        condition_cord = condition_cord.to(device) #condition frames (1,2,3)
        condition_conf = condition_conf.to(device) #future frames
        
        
        
        optimizer.zero_grad()
        
        x = ip_cord #flattening #shape 3264
        x = x.float() 
        y = condition_cord
        y = y.float()
        
######### without bce############        
        recons_x,class_p, z_mu, z_var = model(x,y,2) #one without bce
        
        recons_x = recons_x.float()
        z_mu = z_mu.float()
        z_var = z_var.float()
        class_p = class_p.float()
        
        loss = loss_calc(x,recons_x,z_mu,z_var)
        
        class_0 = torch.zeros(class_p.shape).cuda()
        
        bce_loss = loss_bce(class_0,class_p)
        #print('bce-normal',bce_loss.item(),loss.item(), class_p.item())
        #print("type", bce_loss) #tensor
        
        
        
        total_loss = loss + bce_loss
        
        total_loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
############### with simulated abnormal data ##########

        recons_x,class_p, z_mu, z_var = model(x,y,1) #one with bce
        
        recons_x = recons_x.float()
        z_mu = z_mu.float()
        z_var = z_var.float()
        class_p = class_p.float()
        
        
        loss = mse_loss(x,recons_x)
        
        class_1 = torch.ones(class_p.shape).cuda()
        bce_loss = loss_bce(class_1,class_p)
        total = loss + bce_loss
        #print('bce_ab',bce_loss.item())
        #print('class_ab',bce_loss.item(),class_p.item())
        total.backward()
        
        optimizer.step()
    
    return train_loss

def val():
	model.eval()
	test_loss = 0
	
	with torch.no_grad():
		for i,data in enumerate(valloader):
			
			ip_cord, ip_conf, condition_cord, condition_conf  = data
			
			ip_cord = ip_cord.to(device) #future frames (4,5,6)
			ip_conf = ip_conf.to(device) #future frames
			condition_cord = condition_cord.to(device) #condition frames (1,2,3)
			condition_conf = condition_conf.to(device) #future frames
			
			x = ip_cord #flattening
			x = x.float().cuda()
			y = condition_cord
			y = y.float().cuda() 
			
			recons_x, class_p,z_mu, z_var = model(x,y,0)
			
			loss = loss_calc(x,recons_x,z_mu,z_var)
			test_loss += loss.item()
	
	return test_loss
	
trainlo_list = []
vallo_list = []

set_lr(optimizer)


for e in range(epoch,n_epochs):
        if(e<=160):
            train_loss = pretrain()
            train_loss /= len(dict_train)
            log = open("./log/var_conv_ood_7.txt", "a")
            if(e%10==0):
	            val_loss = pretrain_val()
	            val_loss /= len(dict_val)
	            scheduler.step(val_loss)
	            print(f'Epoch {e}, Learning_rate: {get_lr(optimizer)}')
	            print(f'Epoch {e}, Learning_rate: {get_lr(optimizer)}',file=log)
            print(f'Pretrain:: Epoch {e}, Train Loss:  {train_loss:.3f}, Val_loss: {val_loss:.3f}')
            print(f'Pretrain:: Epoch {e}, Train Loss:  {train_loss:.3f}, Val_loss: {val_loss:.3f}',file=log)
            if(e%20==0):
	            torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
            }, './checkpoints/checkpoints_var_conv_ood_7/pretrain_'+str(e)+'.pth')
        else:
            if(e==161):
                set_lr(optimizer)
                #for param in model.encoder_cond.parameters():
                 #   param.requires_grad = False
                #print(param)
            #print('Training_bce Epoch:',e)
            #print('simulating normal wih mu = 0')
            train_loss = train()
            train_loss /= len(dict_train)
            #for param in model.encoder_cond.parameters():
             #       print(param.requires_grad)
            print(f'Training_bce:: Epoch {e}, Train Loss:  {train_loss:.3f}')
                
            #print(f'Epoch {e}, Train Loss:  {train_loss:.3f}')
        


            if (e % 10 == 0):
	        
	            val_loss = val()
	            val_loss /= len(dict_val)
	            scheduler.step(val_loss)
	            print(f'Epoch {e}, Learning_rate: {get_lr(optimizer)}')

	            print(f'Epoch {e}, Val_loss: {val_loss:.3f}')
	            log = open("./log/var_conv_ood_7.txt", "a")
	        
	            print(f'Epoch {e}, Train Loss:  {train_loss:.3f}, Val_loss: {val_loss:.3f}, Learning_rate: {get_lr(optimizer)}',file=log)

	            torch.save({'epoch': e,'train_loss' : train_loss,'val_loss':val_loss, 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()},'./checkpoints/checkpoints_var_conv_ood_7_pca/checkpoint_'+str(e)+'.pth')
	            
        
	
	
