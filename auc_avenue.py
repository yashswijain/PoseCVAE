import os
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import os
import sys

arg_dir = sys.argv[1]

path_directory='/home/SharedData/Ashvini/ablation'

def read_numpy_concat_mean(path):

	k=os.listdir(path)
	count=0
	k.sort()
	for ele in k:
		#print(ele)


		
		current_file=np.load(path+ele)
		#np.mean(curren_file)
		#np.std(curren_file)

		element=current_file


		if(ele=='14_0017.npy'):
			#print('IT IS TRUE')
			element=np.delete(element, list(range(74,120))+list(range(389,436))+list(range(863,910))+list(range(930,1000)))
			#print(element.shape)

		if(ele=='14_0018.npy'):
			#print('IT IS TRUE')
			element=np.delete(element, list(range(271,319))+list(range(722,763)))
			#print(element.shape)

		if(ele=='14_0019.npy'):
			#print('IT IS TRUE')
			element=np.delete(element, list(range(292,340)))
			#print(element.shape)

		if(ele=='14_0022.npy'):
			#print('IT IS TRUE')
			element=np.delete(element, list(range(560,624))+list(range(813,1006)))
			#print(element.shape)

		if(ele=='14_0032.npy'):
			#print('IT IS TRUE')
			element=np.delete(element, list(range(727,739)))
			#print(element.shape)

		if(np.sum(element)>0):

			element=(element-np.min(element))/(np.max(element)-np.min(element))

			#element=element-np.mean(element)

			#element=(element-np.mean(element))/(np.std(element-np.mean(element)))

		else:

			element = np.zeros(element.shape)


		#print('Element Shape :: ',element.shape)


		if(count==0):
			p=element
			count=count+1
		else:
			p=np.concatenate((p,element))

		#print(p.shape)

	return p



def read_numpy_gt(path):

	k=os.listdir(path)
	count=0
	k.sort()
	for ele in k:
		#print(ele)
		current_file=np.load(path+ele)



		#np.mean(curren_file)
		#np.std(curren_file)


		element=current_file #(current_file-np.mean(current_file))/np.std(current_file)

		if(ele=='14_0017.npy'):
			#print('IT IS TRUE')
			element=np.delete(element, list(range(74,120))+list(range(389,436))+list(range(863,910))+list(range(930,1000)))
			#print(element.shape)

		if(ele=='14_0018.npy'):
			#print('IT IS TRUE')
			element=np.delete(element, list(range(271,319))+list(range(722,763)))
			#print(element.shape)

		if(ele=='14_0019.npy'):
			#print('IT IS TRUE')
			element=np.delete(element, list(range(292,340)))
			#print(element.shape)

		if(ele=='14_0022.npy'):
			#print('IT IS TRUE')
			element=np.delete(element, list(range(560,624))+list(range(813,1006)))
			#print(element.shape)

		if(ele=='14_0032.npy'):
			#print('IT IS TRUE')
			element=np.delete(element, list(range(727,739)))
			#print(element.shape)

		if(count==0):
			p=element
			count=count+1
		else:
			#print('pshape :: ',p.shape)
			#print('element shape :: ',element.shape)
			p=np.concatenate((p.reshape(p.shape[0],1),element.reshape(element.shape[0],1)))

		#print(p.shape)

	return p

path=path_directory+'/accv20/test/final_mse/'+str(arg_dir)+'/'
gt_path=path_directory+'/ground_truth/'

concat_all_video=read_numpy_concat_mean(path)

###########################Concat_all_video is already normalised between 0 and 1##########
concat_all_video[concat_all_video<0]=0

x=np.int32(concat_all_video>0) ########If mse is 0, then 0 otherwise 1, kind of binarisisng
x[x==0]=1

final_scores=concat_all_video/x
final_scores[np.isnan(final_scores)]=0

concat = concat_all_video.reshape((concat_all_video.shape[0],1))

gt_all_video=read_numpy_gt(gt_path)


y = label_binarize(np.int32(gt_all_video),classes=[0,1])
final_scores=final_scores.reshape((final_scores.shape[0],1))

auc_value=roc_auc_score(y, final_scores)
print('auc_score : {}'.format(auc_value))

save = np.array((auc_value))
#np.save('AUC_Epoch_'+epoch_num+'.npy',save)







