import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import os
import numpy as np
import sys

epoch_number_input=str(sys.argv[1])

sliding_win_len = 7

path_directory = '/home/SharedData/Ashvini/ablation' # base working directory
test_data = "./datasets/test_7" # folder in which test tracks are stored

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([transforms.ToTensor()])

batch_size = 128

emb0 = 68
emb1 = 68
emb2 = 68

num_channels = 17

input_dim = 4*num_channels*1
hidden_dim1 = 4*num_channels*1
hidden_dim2 = 2*num_channels*1
latent_dim = num_channels*1
cond_dim = 4*num_channels*1

test_folder=os.listdir(path_directory+'/test/'+ test_data +'/')

list_test=[]
for video_no in test_folder:
	track_path = path_directory+'/test/'+ test_data +'/'+video_no+'/'
	video=os.listdir(track_path) #comprises of a list all the tracks corresponding to a video
	
	for track in video:
		if(track!='result'):
			i=track_path+track
			list_test.append(i)

####################### test_loader ###############################################
class dataset_test(Dataset):
    
	def __init__(self,transform=None):
		self.dummy=0

	def __len__(self):
		return (len(list_test))

	def __getitem__(self, idx):
		
		self.data=np.load(list_test[idx])

#data has one numpy file loaded
		self.data_cord=self.data[:,3:3+2*num_channels].T 
#data_cord has the x,y co-ordinates 

		self.data_cord_gt=self.data_cord[:,sliding_win_len:]/500.0
		self.data_cord_condition = self.data_cord[:,:sliding_win_len]/500.0
		
		#data_cord_input has the future frames, to be used for recon. (Input to Encoder)
		#data_cord_cond has the present frames to be given as condition to encoder 

		self.data_conf=self.data[:,3+2*num_channels:3+3*num_channels].T
		
		self.data_conf_gt=self.data_conf[:,sliding_win_len:]
		self.data_conf_condition = self.data_conf[:,:sliding_win_len]
		
		
#data_conf has the confidence values
		self.data_conf=np.concatenate((self.data_conf, self.data_conf), axis=0)
		self.data_conf_gt=np.concatenate((self.data_conf_gt, self.data_conf_gt), axis=0)
		self.data_conf_condition=np.concatenate((self.data_conf_condition, self.data_conf_condition), axis=0)

		self.person_id=int(self.data[0,54])
		self.cam_id=int(self.data[0,0])
		self.vid_id=int(self.data[0,1])
		self.frames=self.data[:,2]
		self.name_file=list_test[idx].split('/')[-1]
		#print("name:",self.name_file)
		self.track_path = (list_test[idx].split('/')[-2]) 




		return self.data_cord_gt,self.data_conf_gt,\
self.data_cord_condition,self.data_conf_condition, self.frames, self.name_file, self.track_path
####################################################################################

transformed_dataset_test = dataset_test()

testloader = torch.utils.data.DataLoader(transformed_dataset_test, batch_size=batch_size,shuffle=False, num_workers=4,drop_last=False)
 

#####################################################################################

#######################  MODEL  ########################################

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
        class_p = torch.sigmoid(self.hidden_to_bce(x)) # class_p is the class probability of
        
        return generated_x, class_p


class CVAE(nn.Module):
    def __init__(self,input_dim,hidden_dim1,hidden_dim2,latent_dim,cond_dim):
    
        super(CVAE,self).__init__()
        
        self.encoder_inp = Conv_Encoder1()
        self.encoder_cond = Conv_Encoder1()
        
        self.encoder = Encoder(input_dim,hidden_dim1,hidden_dim2,latent_dim,cond_dim)
        self.decoder = Decoder(latent_dim,hidden_dim2,hidden_dim1,input_dim,cond_dim)
        
        self.conv_decoder = Conv_Decoder()
    
    
    def forward(self,inp,cond):
        x = self.encoder_inp(inp)
        y = self.encoder_cond(cond)
        x = torch.cat((x,y),dim=1) #shape 6528(32*34*sliding_win_len*2)
        x = x.float()
        #print(x.shape)
        z_mu,z_var = self.encoder(x)
        
        std = torch.exp(z_var/2)
        eps = torch.randn_like(std)
        
        ############ DURING TESTING SAMPLE FROM N(0,I)
        x_sample = eps  #.mul(std).add(z_mu)
        x_sample = x_sample.float()
        y = y.float()
        z = torch.cat((x_sample,y),dim=1)
        z = z.float()
        
        generated_x, class_p_x = self.decoder(z)
        generated_x = generated_x.view(-1,emb2,1)
        final_x = self.conv_decoder(generated_x)
        
        return final_x, z_mu, z_var
        
        

model = CVAE(input_dim,hidden_dim1,hidden_dim2,latent_dim,cond_dim).to(device)
model = model.float()

#model.load_state_dict(torch.load(path_directory+'/checkpoints/vae_epoch_'+epoch_number_input+'.pth'))

#CHK checkpoint_abnormal-bce_600
load_path = path_directory+'/checkpoints/checkpoints_var_conv_ood_7/checkpoint_' + epoch_number_input + '.pth'

checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint['model_state_dict'])
chkpt = checkpoint['epoch']




################################################################

def loss_calc(x,recons_x):
    RCL = F.mse_loss(recons_x,x,size_average = False)
    #print(x)
    
    return RCL
    
    
def mse_loss(res,gt):

	#print('Res Shape ::',res[:,0].shape) #[batch,2xnum_channels,Length of track]
	#print('GT Shape ::',gt.shape) #['']
        error_total=torch.zeros((res[:,0].shape)).cuda()#[batch,length of track]
	#print(error_total.shape)  [37,sliding_win_len]
        
        for counter_i in range(0,num_channels):
        
                error_total=error_total+((res[:,counter_i]-gt[:,counter_i])**2+(res[:,counter_i+num_channels]-gt[:,counter_i+num_channels])**2)
                
        return error_total


#################################################################

def test():
	model.eval()
	test_loss = 0
	print(" LOADED checkpoint_new_",epoch_number_input)
	
	with torch.no_grad():
		for i,data in enumerate(testloader):
			
			gt_cord, gt_conf, condition_cord, condition_conf, frame, name, track_path  = data
			
			gt_cord = gt_cord.to(device) #future frames (4,5,6)
			gt_conf = gt_conf.to(device) #future frames
			condition_cord = condition_cord.to(device) #condition frames (1,2,3)
			condition_conf = condition_conf.to(device) #condition frames
			frame = frame.to(device)			
			
			
			gt = gt_cord #flattening the gt
			gt = gt.float().cuda()
			#gt = model.encoder_inp(y)
			#print(gt.shape)
			#gt = gt.view(-1,emb2)
			
			cond = condition_cord
			cond = cond.float().cuda()
			x = model.encoder_cond(cond)
			x = x.view(-1,emb2) 
			z = torch.randn(x.shape[0],latent_dim).to(device) # random noise sample
			z = z.view(-1,latent_dim)
			#print(x.shape,z.shape)
			z = torch.cat((z,x),dim=1)
			
			#dummy_x = torch.cat((gt,x),dim=1)
			#mean, var = model.encoder(dummy_x)
			#mean_path = path_directory + "/mean/mean.txt"
			#log = log = open(mean_path, "a")
			#print(f'Mean:  {mean:.3f}',file=log)
			#print(mean)
			temp_x, class_p = model.decoder(z)
			temp_x = temp_x.view(-1,emb2,1)
			recons_x = model.conv_decoder(temp_x)
			
			
			#loss = loss_calc(gt,recons_x)
			
			x_hat = recons_x.reshape(-1,34,sliding_win_len)
			gt = gt.reshape(-1,34,sliding_win_len)
			
			error = mse_loss(x_hat,gt)
			error = error.data.cpu().numpy()
			
			for i in range(0,x.shape[0]):
		
		                name_a='_'.join([name[i].split('_')[0],name[i].split('_')[1]])#name_a=14_videoNUmber
		                g=os.listdir(path_directory+'/frames/'+name_a+'/')
		                npy=np.zeros((1,len(g))) #gives the length of a video
		
		                start = frame[i].data.cpu().numpy()
		                start = start.astype(int) 
		                #print(name[0],frame[i])
		
		                npy[0,start[sliding_win_len - 1]:start[-1]]=error[i]
		                #result1 = result[i,:,:]
		                pth = path_directory + '/test/'+test_data + '/'+track_path[i]+'/result/'+name[i]
		                #print(pth)
		                np.save(path_directory + '/test/'+test_data + '/'+track_path[i]+'/result/'+name[i],npy)

			
			
################################################################################	

test()
    
