import time
import numpy as np
from sklearn.datasets import make_s_curve
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch.nn.functional as F
import math
import os
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diffusion_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class MLPDiffusion(nn.Module):
    def __init__(self, input_dim, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()
        
        self.linears = nn.ModuleList([
            nn.Linear(input_dim, num_units),
            nn.Tanh(),
            nn.Linear(num_units, num_units),
            nn.Tanh(),
            nn.Linear(num_units, num_units),
            nn.Tanh(),
            nn.Linear(num_units, input_dim),
        ])
        
        # 确保嵌入层也在正确设备上
        self.step_embeddings = nn.ModuleList([
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units),
        ])
    
    def forward(self, x, t):
        # 确保所有操作都在同一设备上
        device = x.device
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t).to(device)
            x = self.linears[2*idx](x)
            x += t_embedding
            x = self.linears[2*idx+1](x)
            
        x = self.linears[-1](x)
        
        return x

def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    batch_size = x_0.shape[0]
    
    
    half_len = batch_size // 2
    t = torch.randint(0, n_steps, size=(half_len,)).to(device)
    
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    
    
    if batch_size % 2 == 1:
        t_extra = torch.randint(0, n_steps, size=(1,)).to(device)
        t = torch.cat([t, t_extra], dim=0)
    
    t = t.unsqueeze(-1)
    
    a = alphas_bar_sqrt[t]
    aml = one_minus_alphas_bar_sqrt[t]
        
    e = torch.randn_like(x_0).to(device)
    x = x_0 * a + e * aml
    output = model(x, t.squeeze(-1))
    
    return (e - output).square().mean()


def q_x(x_0,t):
    noise = torch.randn_like(x_0).to(device)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise)#

class EMA():
    def __init__(self,mu=0.01):
        self.mu = mu
        self.shadow = {}
        
    def register(self,name,val):
        self.shadow[name] = val.clone()
        
    def __call__(self,name,x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0-self.mu)*self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, cur_x):
    start_time = time.time()
    logger.info(f"Starting p_sample_loop with {n_steps} steps")
    
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        step_start_time = time.time()
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
        step_time = time.time() - step_start_time
        #logger.info(f"p_sample_loop step {i} completed in {step_time:.4f} seconds")
    
    total_time = time.time() - start_time
    logger.info(f"p_sample_loop completed in {total_time:.4f} seconds")
    return x_seq

def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    t = torch.tensor([t]).to(x.device)  # 确保与输入x在同一设备上
    
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    
    eps_theta = model(x, t)
    
    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
    
    z = torch.randn_like(x).to(x.device)  # 确保与输入x在同一设备上
    sigma_t = betas[t].sqrt()
    
    sample = mean + sigma_t * z
    
    return sample

def cau_model(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt, i, all_test):
    start_time = time.time()
    logger.info(f"Starting cau_model for feature {i}")
    
    
    device = next(model.parameters()).device
    
    # First p_sample_loop with noise
    noise_start_time = time.time()
    cur_xx = torch.randn(shape).to(device)
    x_seq_noise = p_sample_loop(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt, cur_xx)
    x_seq_noise_final = x_seq_noise[num_steps]
    noise_time = time.time() - noise_start_time
    logger.info(f"Noise sampling for feature {i} completed in {noise_time:.4f} seconds")
    
    # Second p_sample_loop with causal conditioning
    cau_start_time = time.time()
    cur = torch.from_numpy(all_test[:, i]).float().to(device)
    cur_xx[:, i] = cur
    x_seq_cau = p_sample_loop(model, shape, num_steps, betas, one_minus_alphas_bar_sqrt, cur_xx)
    x_seq_cau_final = x_seq_cau[num_steps]
    cau_time = time.time() - cau_start_time
    logger.info(f"Causal sampling for feature {i} completed in {cau_time:.4f} seconds")
    
    total_time = time.time() - start_time
    logger.info(f"cau_model for feature {i} completed in {total_time:.4f} seconds")
    
    return x_seq_cau_final, x_seq_noise_final

def hsic(Kx, Ky):
    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n**2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
    return h * n**2 / (n - 1)**2

def HSIC(x,y):
    Kx = np.expand_dims(x, 0) - np.expand_dims(x, 1)
    Kx = np.exp(- Kx**2)
    
    Ky = np.expand_dims(y, 0) - np.expand_dims(y, 1)
    Ky = np.exp(- Ky**2)
    return hsic(Kx, Ky)

def MSE(y,t):
    return np.sum((y-t)**2)/y.shape[0]

def dis_hisc(y_pre,y_test):
    y_test_pre = y_pre.detach().cpu().numpy()
    finalloss=MSE(y_test.reshape(-1),y_test_pre.reshape(-1))
    return finalloss

###dataread
expression = pd.read_csv('omics_0_feature_select_matrix.csv', sep=',')
expression=expression.iloc[:,1:]
input_dim = expression.shape[1]

logger.info(f"Dataset loaded with shape: {expression.shape}")

Data=np.array([])
x_target=0
y_target=0

s_curve=np.array(expression.values)
logger.info(f"shape of s: {np.shape(s_curve)}")

dataset = torch.Tensor(s_curve).float().to(device)
logger.info(f"Dataset converted to tensor: {dataset.shape}")

### parameter
num_steps = 200

betas = torch.linspace(-6,6,num_steps).to(device)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

alphas = 1-betas
alphas_prod = torch.cumprod(alphas,0)
alphas_prod_p = torch.cat([torch.tensor([1]).float().to(device),alphas_prod[:-1]],0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape
logger.info(f"all the same shape: {betas.shape}")

# 确保所有参数都在同一设备上
betas = betas.to(device)
alphas_bar_sqrt = alphas_bar_sqrt.to(device)
one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)

seed = 1234
    
logger.info('Training model...')
batch_size = 32
dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size)
num_epoch = 20

k_fold=5

x_now=np.array(expression.values)

f1_result=np.zeros((expression.shape[1],expression.shape[1],k_fold))
f2_result=np.zeros((expression.shape[1],expression.shape[1],k_fold))

final_count=0
cishu_count=1
final_result=np.zeros((cishu_count,6))

for cishu in range(cishu_count):
    kf = KFold(n_splits=k_fold,random_state=cishu+31,shuffle= True)
    kf.get_n_splits(x_now)
    kk=0
    
    logger.info(f"Starting iteration {cishu+1}/{cishu_count}")
    
    for train_index, test_index in kf.split(x_now):
        fold_start_time = time.time()
        logger.info(f"Starting fold {kk+1}/{k_fold}")
        
        #### k-fold
        all_train, all_test = x_now[train_index], x_now[test_index]
        all_test=np.array(all_test)
        all_train=np.array(all_train)
        dataset = torch.Tensor(all_train).float().to(device)

        ####train modle
        model = MLPDiffusion(input_dim,num_steps).to(device)

        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

        model.train()
        total_start_time = time.time()
            
        for t in range(num_epoch):
            epoch_start_time = time.time()
            epoch_loss = 0
            batch_count = 0
            
            for idx,batch_x in enumerate(dataloader):
                batch_start_time = time.time()
                loss = diffusion_loss_fn(model,batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                
                batch_time = time.time() - batch_start_time
                if idx % 5 == 0:  # Log every 100 batches
                    logger.info(f"Epoch {t}, Batch {idx}: loss={loss.item():.6f}, time={batch_time:.4f}s")
            
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            
            if t % 5 == 0:  # 每100个epoch输出一次，避免输出太多
                logger.info(f'Epoch {t}/{num_epoch}, Time: {epoch_time:.2f} seconds')
                
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        logger.info(f'Total training time for fold {kk}: {total_time:.2f} seconds')

        model.eval()
        
        dataset_test = torch.Tensor(all_test).float().to(device)
        
        # Time the evaluation process
        eval_start_time = time.time()
        for i in range(expression.shape[1]):
            feature_start_time = time.time()
            logger.info(f"Processing feature {i+1}/{expression.shape[1]}")
            
            final1,final2 = cau_model(model,dataset_test.shape,num_steps,betas,one_minus_alphas_bar_sqrt,i,all_test)

            for j in range(expression.shape[1]):
                if j == i:
                    continue
                else:
                    f1_result[i,j,kk]=dis_hisc(final1[:,j],all_test[:,j])
                    f2_result[i,j,kk]=dis_hisc(final2[:,j],all_test[:,j])
            
            feature_time = time.time() - feature_start_time
            logger.info(f"Feature {i} processing completed in {feature_time:.4f} seconds")
        
        eval_time = time.time() - eval_start_time
        logger.info(f"Evaluation for fold {kk} completed in {eval_time:.4f} seconds")
        
        fold_time = time.time() - fold_start_time
        logger.info(f"Fold {kk} completed in {fold_time:.4f} seconds")
        
        kk=kk+1
    
    # Save results for this iteration
    folder_path = 'network'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i in range(k_fold):
        pd.DataFrame(f1_result[:,:,i]).to_csv(f'{folder_path}/f1_{cishu}_{i}.csv')
        pd.DataFrame(f2_result[:,:,i]).to_csv(f'{folder_path}/f2_{cishu}_{i}.csv')
    
    logger.info(f'Iteration {cishu} completed successfully')

logger.info('All iterations completed successfully')
