# %%
import torch 
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data
import h5py
import os
import hdf5storage

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

datasetR = [] 
data_path = './'


# %%
'''Load Test Data'''
no_measurements = 100 
#filename = data_path+'data/gt_graph_random_large_outliers_real.h5'
filename = data_path+'data/gt_graph_random_large_outliers_test.h5'  
for item in range(no_measurements): 
    x = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/x', filename=filename, options=None), dtype=torch.float)
    xt = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/xt', filename=filename, options=None), dtype=torch.float)
    o = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/o', filename=filename, options=None), dtype=torch.float)
    edge_index = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/edge_index', filename=filename, options=None), dtype=torch.long)
    edge_attr = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/edge_feature', filename=filename, options=None), dtype=torch.float)
    y = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/y', filename=filename, options=None), dtype=torch.float)
    datasetR.append(Data(x=x, xt=xt, o=o, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)) 


# %%
'''Load Train Data'''
no_measurements = 1200 
filename = data_path+'data/gt_graph_random_large_outliers.h5'  
for item in range(no_measurements): 
    x = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/x', filename=filename, options=None), dtype=torch.float)
    xt = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/xt', filename=filename, options=None), dtype=torch.float)
    o = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/o', filename=filename, options=None), dtype=torch.float)
    edge_index = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/edge_index', filename=filename, options=None), dtype=torch.long)
    edge_attr = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/edge_feature', filename=filename, options=None), dtype=torch.float)
    y = torch.tensor(hdf5storage.read(path='/data/'+str(item+1)+'/y', filename=filename, options=None), dtype=torch.float)
    datasetR.append(Data(x=x, xt=xt, o=o, y=y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)) 


# %%
def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def inv_q(q):
    """
    Inverse quaternion(s) q .
    """
    assert q.shape[-1] == 4
    original_shape = q.shape
    return torch.stack((q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]), dim=1).view(original_shape)

# %%
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN, Dropout
from torch_geometric.nn import MessagePassing

class EdgeConvRot(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super(EdgeConvRot, self).__init__(aggr='mean', flow="target_to_source") #  "Max" aggregation.
        self.mlp0 = Seq(Linear(edge_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

        self.mlp = Seq(Linear(2*in_channels+edge_channels, out_channels),
               ReLU(),
               Linear(out_channels, out_channels))
            
    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr): 
        if x_i.size(1) > 5: 
            W = torch.cat([torch.cat([x_i, x_j], dim=1), edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]
            W = self.mlp(W) 
        else:
            W = edge_attr # torch.cat([torch.cat([x_i, x_j], dim=1), edge_attr], dim=1)  # tmp has shape [E, 2 * in_channels]            
            W = self.mlp0(W) 
        return W
            
    def propagate(self, edge_index, size, x, edge_attr):    
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0) 
        edge_out = self.message(x_i, x_j, edge_attr)
        out = scatter_(self.aggr, edge_out, edge_index[i], dim_size=size[i])
        return out, edge_out 

    


# %%
import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter_
from torch_geometric.nn import GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

def node_model(x, batch):
   # print(batch.shape)
    out, inverse_indices = torch.unique_consecutive(batch, return_inverse=True)
    quat_vals = x[inverse_indices] 
    q_ij = qmul(x, inv_q(quat_vals[batch]))  
    return q_ij 

def edge_model(x, edge_index):
    row, col = edge_index
    q_ij = qmul(x[col], inv_q(x[row]))  
    return q_ij 

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Linear(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])
class EdgePred(torch.nn.Module):
    def __init__(self, in_channels, edge_channels):
        super(EdgePred, self).__init__()
        self.mlp = Seq(Linear(2*in_channels+edge_channels, 8),
                       ReLU(),
                       Linear(8, 1)) 
    def forward(self, xn, edge_index, edge_attr): 
        row, col = edge_index
        xn = torch.cat([xn[row], xn[col], edge_attr], dim=1)
        xn = self.mlp(xn) 
        return torch.sigmoid(xn) 
    
class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn1, nn2):
        super(GlobalSAModule, self).__init__()
        self.nn1 = nn1
        self.nn2 = nn2

    def forward(self, x, batch): 
        xn = self.nn1(x)
      #  xn = F._max_pool1d(xn, x.size(1))
       # xn = scatter_('mean', xn, batch)
       # xn = xn[batch]  
        xn = torch.cat([xn, x], dim=1) 
     #   print(xn.shape)
      #  x = xn.unsqueeze(0).repeat(x.size(0), 1, 1) 
      #  batch = torch.arange(x.size(0), device=batch.device)
        return self.nn2(xn)
    
 

# %%
def update_attr(x, edge_index, edge_attr):
    row, col = edge_index
    x_i = x[row]
    x_j = inv_q(x[col])
    W=qmul(edge_attr, x_i) 
    W=qmul(x_j, W) 
    return W 


def smooth_l1_loss(input, beta=1. / 5, size_average=False):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input)
       
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()

def my_smooth_l1_loss(input, beta, alpha=0.05):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    nn = torch.sum(input ** 2, dim=1) 
    beta = torch.squeeze(beta) 
    nn = torch.mul(nn, beta) 

    return nn.sum()

class Net(torch.nn.Module):
    def __init__(self): 
        super(Net, self).__init__() 
        self.no_features = 32   # More features for large dataset 
        self.conv1 = EdgeConvRot(4, 4, self.no_features) 
        self.conv2 = EdgeConvRot(self.no_features, self.no_features+4, self.no_features)  
        self.conv3 = EdgeConvRot(2*self.no_features, 2*self.no_features, self.no_features) 
        self.conv4 = EdgeConvRot(2*self.no_features, 2*self.no_features, self.no_features) 

        self.lin01 = Linear(self.no_features, self.no_features) 
        self.lin02 = Linear(self.no_features, self.no_features) 
        self.lin1 = Linear(self.no_features, 4) 
        self.lin2 = Linear(self.no_features, 1) 
        
        self.m = torch.nn.Sigmoid() 
    def forward(self, data):
        x_org, edge_index, edge_attr, batch, beta = data.x, data.edge_index, data.edge_attr, data.batch, data.o  
        
        x1, edge_x1 = self.conv1(torch.zeros_like(x_org), edge_index, edge_attr)
        x1 = F.relu(x1)
        edge_x1 = F.relu(edge_x1)
        
        x2, edge_x2 = self.conv2(x1, edge_index, torch.cat([edge_attr, edge_x1], dim=1))
        x2 = F.relu(x2)
        edge_x2 = F.relu(edge_x2)

        x3, edge_x3 = self.conv3(torch.cat([x2, x1], dim=1), edge_index, torch.cat([edge_x2, edge_x1], dim=1))
        x3 = F.relu(x3)
        edge_x3 = F.relu(edge_x3)
        
        x4, edge_x4 = self.conv4(torch.cat([x3, x2], dim=1), edge_index, torch.cat([edge_x3, edge_x2], dim=1))
        edge_x4 = F.relu(edge_x4)
        
        out01 = self.lin01(edge_x4) 
        out02 = self.lin02(edge_x4) 
        
      #  print(out.shape) 
        edge_x = self.lin1(out01) + edge_attr
        edge_x = F.normalize(edge_x, p=2, dim=1) 
        
        return self.m(self.lin2(out02)), edge_x, beta   #x, loss1, beta   # node_model(x, batch),

# %%
PATH = 'checkpoint/outliers_detect_new2222.pth' 
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
for g in optimizer.param_groups:
    g['lr'] = 0.0001

# %%
PATH = 'checkpoint/outliers_detect_new222222.pth' 
import numpy as np 
no_training = 1200 #round(len(datasetR)*training_exmpl)
no_testing = 100 
print(no_training) 
test_loader  = DataLoader(datasetR[:no_testing], batch_size=4, shuffle=True,num_workers=2)
train_loader = DataLoader(datasetR[no_testing:], batch_size=4, shuffle=False,num_workers=2)

criterion = torch.nn.BCELoss()

model.train()
best_loss = 2000 
t = time.time() 
count = 0 
val = 14688
for epoch in range(250000):
    total_loss1 = 0 
    total_loss2 = 0 
    theta = []
    loss = 0 
    for idx, data in enumerate(train_loader):
        data_gpu = data.to(device)
        optimizer.zero_grad() 

   #     print(data_gpu.y.shape)

        out, edge_x, beta = model(data_gpu)
    
        loss1 = qmul(edge_x, inv_q(edge_model(data_gpu.y, data_gpu.edge_index)))  
        loss1 = smooth_l1_loss(loss1[:, 1:])
        
        loss2 = criterion(out, beta) 
        loss = 0.1*loss1 + 500*loss2
     #   if idx % 2 == 0: 
        loss.backward()
        optimizer.step()
       # print([idx, loss.item()])
       # time.sleep(0.01)
        
        if epoch % 2 == 0: 
           # loss1 = qmul(data_gpu.edge_attr, inv_q(edge_model(data_gpu.y, data_gpu.edge_index)))  
           # loss1 = smooth_l1_loss(loss1[:, 1:]) 
            total_loss1 = total_loss1 + loss1.item() 
            total_loss2 = total_loss2 + loss2.item() 
            
    if epoch % 10 == 0:
        for data in test_loader: 
            data_gpu = data.to(device)
            out, edge_x, beta = model(data_gpu)
            loss1 = qmul(edge_x, inv_q(edge_model(data_gpu.y, data_gpu.edge_index)))  
            loss1 = smooth_l1_loss(loss1[:, 1:])
            loss2 = criterion(out, beta) 
            total_loss1 = total_loss1 + loss1.item() 
            total_loss2 = total_loss2 + loss2.item()
            
        count = count + 1
    if epoch % 2 == 0: 
        print([epoch, "{0:.5f}".format(total_loss1/no_training), "{0:.5f}".format(total_loss2/no_training), "{0:.3f}".format(time.time() - t)])
        if epoch % 10 == 0:
            if val > total_loss1/no_training : 
                val = total_loss1/no_training 
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, PATH) 


# %%
import numpy as np 
import math 
import h5py
import time 
criterion = torch.nn.BCELoss()

data_path = './' # os.getcwd() 
 
PATH = 'checkpoint/outliers_detect_new22222.pth' 

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')
model = Net().to(device) 
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

train_loader = DataLoader(datasetR[100:], batch_size=1, shuffle=False)
#test_loader = DataLoader(datasetR[:100], batch_size=1, shuffle=False)
#test_loader = DataLoader(datasetR, batch_size=1, shuffle=False)
#model = best_model 
#print(best_loss)
#pred_rot = []
model.eval()
total_loss = 0 
count = 0 
total_time = 0
t = time.time() 
hf = h5py.File(data_path+'data/gt_graph_random_large_outliers_pred_rot.h5', 'w')
#hf = h5py.File(data_path+'data/gt_graph_random_large_outliers_test_pred_rot.h5', 'w')
theta = [] 

for data in train_loader: 
    print(data) 
    data_gpu = data.to(device)
    out, x, beta = model(data_gpu)
  #  x = edge_model(data_gpu.xt, data_gpu.edge_index) 
   # loss = criterion(out, beta)
  #  loss = (pred - data_gpu.y).pow(2).sum() 
  #  total_loss = total_loss + loss.item() 
 #   pred_rot = torch.cat([data.xt, pred, data.y], dim=1).data.cpu().numpy()
    hf.create_dataset('/data/'+str(count+1)+'/ot', data=out.data.cpu().numpy())
    hf.create_dataset('/data/'+str(count+1)+'/o', data=beta.data.cpu().numpy())
  #  hf.create_dataset('/data/'+str(count+1)+'/onode', data=data_gpu.onode.data.cpu().numpy())
  #  hf.create_dataset('/data/'+str(count+1)+'/omarker', data=data_gpu.omarker.data.cpu().numpy())
    hf.create_dataset('/data/'+str(count+1)+'/refined_qq', data=x.data.cpu().numpy())
    hf.create_dataset('/data/'+str(count+1)+'/y', data=data_gpu.y.data.cpu().numpy())
    hf.create_dataset('/data/'+str(count+1)+'/xt', data=data_gpu.xt.data.cpu().numpy())
    hf.create_dataset('/data/'+str(count+1)+'/edge_index', data=data_gpu.edge_index.data.cpu().numpy())
    hf.create_dataset('/data/'+str(count+1)+'/edge_feature', data=data_gpu.edge_attr.data.cpu().numpy())
    count = count + 1 
   # print([len(pred_rot), (time.time()-t)/len(pred_rot)])
#print([total_loss/len(test_loader), (time.time() - t)/(test_loader.batch_size*len(test_loader))])
hf.close() 
   # print([len(pred_rot), (time.time()-t)/len(pred_rot)])
print([total_loss/len(train_loader), total_time /(test_loader.batch_size*len(train_loader))]) 


# %%
