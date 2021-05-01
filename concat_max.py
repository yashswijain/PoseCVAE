#Concatenates the person frame scores and takes a maximum and plots a graph
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

sliding_win_len = int(sys.argv[1])
save_folder = str(sys.argv[2])

path_directory='/home/SharedData/Ashvini/ablation/'
test_data_folder = 'datasets/test_'+str(sliding_win_len)+'/'

s=path_directory+'/accv20/test/'+ test_data_folder



video_lst=os.listdir(s)
for elemet_vid in video_lst:
	print(elemet_vid)
	k=os.listdir(s+elemet_vid+'/result/') #.npy file for each person, NOW A MAX WILL BE TAKEN after concat.
	#print('after')	

	e=os.listdir(path_directory+'/frames/14_00'+elemet_vid+'/') #original TEST FRAMES
	npy=np.zeros((1,len(e)))
	count = 0
        
	for ele in k:
		count = count + 1
		#print('path',count)
		g=np.load(s+elemet_vid+'/result/'+ele)
		#print(g.shape,npy.shape)
		npy=np.concatenate((npy,g)) #CONCATINATNG BEFORE TAKING THE MAX


	res=np.max(npy,axis=0)
	#print('taking max')



	np.save(path_directory+'/accv20/test/final_mse/conv_ood_'+str(save_folder)+'/14_00'+elemet_vid+'.npy',res)
###################################numpy_result consists of the MSE results video wise##########

	gt=np.load(path_directory+'/ground_truth/14_00'+elemet_vid+'.npy')
############### gt is the ground truth numpy file #################

	#print('res shape',gt.shape)
	####SCALES THE GT ACCORDING TO THE MSE VALUE
	
	#res = list(res)
	#res = (res - min(res))/(max(res)-min(res))
	#'''
	if(res.shape[0]==0):
		gt=gt*1
	else:
		gt=gt*np.max(res)

	
	plt.figure(figsize=(20,5))

	plt.plot(range(0,len(e)), res.tolist())

	if(elemet_vid.split('_')[0]=='14'):
		plt.plot(gt[:,0].tolist())
	else:
		plt.plot(gt.tolist())
	print(path_directory+'/accv20/test/mse_plots/conv_ood_'+str(save_folder)+'/'+elemet_vid+'.png')

	plt.savefig(path_directory+'/accv20/test/mse_plots/conv_ood_'+str(save_folder)+'/'+elemet_vid+'.png')
	plt.close()
	
