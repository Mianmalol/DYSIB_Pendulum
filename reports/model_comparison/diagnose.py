import sys, json, os, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import numpy as np, torch
from sklearn.neighbors import NearestNeighbors
from models import DYSIB,RandomFeatureProber
from training import encode_dataset,fit_ridge_head,predict_prober

torch.set_num_threads(2)
out=Path(tempfile.mkdtemp(prefix='dysib-diagnosis-'))
print('Temporary latent arrays:', out, flush=True)
d=np.load('data/pendulum_preprocessed.npz')
frames=torch.from_numpy(d['frames']).float()
theta=d['theta'][:,:58].reshape(-1)
y=torch.from_numpy(np.stack([np.cos(theta),np.sin(theta)],1)).float()
cut=1000*58
results={}
def rmse(pred,true):
 a=np.arctan2(pred[:,1],pred[:,0]);err=np.arctan2(np.sin(a-true),np.cos(a-true))
 return float(np.rad2deg(np.sqrt(np.mean(err**2))))
for name,file in [('original','kz2_nF2_str1_N1000_rep3.pt'),('new','my_run.pt')]:
 ck=torch.load('checkpoints/'+file,map_location='cpu',weights_only=False);p=ck['params']
 model=DYSIB(input_dim=p['input_dim'],dz=p['ndims'],num_frames=p['num_frames'])
 model.load_state_dict(ck['state_dict'])
 z,numD=encode_dataset(model,frames,num_frames=2,nF_stride=1,device='cpu')
 np.save(out/(name+'_z.npy'),z.numpy())
 result={'params':p,'latent_std':z.std(0).tolist(),'cov_eigenvalues':np.linalg.eigvalsh(np.cov(z.numpy().T)).tolist(),'rf':[],'knn':[]}
 if name=='new':
  saved=np.load('figure_data/phase_portrait.npz')['zx_latent'].reshape(-1,2)
  result['max_difference_saved']=float(np.abs(z.numpy()-saved).max())
 for seed in [0,1,2]:
  torch.manual_seed(seed);probe=RandomFeatureProber(2,output_dim=2)
  fit_ridge_head(probe,z[:cut],y[:cut],device='cpu')
  pred=predict_prober(probe,z,device='cpu').numpy()
  row={'seed':seed,'train_angle_rmse':rmse(pred[:cut],theta[:cut]),'test_angle_rmse':rmse(pred[cut:],theta[cut:]),'test_first_frame_rmse':rmse(pred[cut:][::58],theta[cut:][::58])}
  result['rf'].append(row);print(name,'RF',row,flush=True)
 nbr=NearestNeighbors(n_neighbors=20).fit(z[:cut].numpy())
 dist,idx=nbr.kneighbors(z[cut:].numpy())
 nearest_theta = theta[idx[:, 0]]
 angle_diff = np.rad2deg(np.abs(np.arctan2(np.sin(theta[cut:]-nearest_theta), np.cos(theta[cut:]-nearest_theta))))
 result['nearest_neighbor_fraction_angle_difference_above_90_deg'] = float((angle_diff > 90).mean())
 candidates = np.flatnonzero(angle_diff > 90)
 if len(candidates):
  j = int(candidates[np.argmin(dist[candidates, 0])]); i = int(idx[j, 0])
  result['ambiguous_pair'] = {'test_trial': (cut+j)//numD, 'test_frame': (cut+j)%numD,
                            'train_trial': i//numD, 'train_frame': i%numD,
                            'angle_difference_deg': float(angle_diff[j]), 'latent_distance': float(dist[j, 0])}
 for k in [1,5,20]:
  pred=y.numpy()[:cut][idx[:,:k]].mean(1)
  row={'k':k,'test_angle_rmse':rmse(pred,theta[cut:]),'test_first_frame_rmse':rmse(pred[::58],theta[cut:][::58])}
  result['knn'].append(row);print(name,'KNN',row,flush=True)
 results[name]=result
 (HERE/'diagnostic_results.json').write_text(json.dumps(results,indent=2))
 print(name,'std',result['latent_std'],'cov',result['cov_eigenvalues'],flush=True)
