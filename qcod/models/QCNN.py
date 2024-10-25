import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torchquantum as tq
import random
from torchquantum.layer import U3CU3Layer0
from torchquantum.encoding import Encoder
from torchquantum.functional import func_name_dict
from typing import Iterable
from torchquantum.macro import C_DTYPE
from abc import ABCMeta
from qiskit import QuantumCircuit

class QConv2d(tq.QuantumModule):
    def __init__(self,in_channels = 1, out_channels = 10, kernel_size = 3, stride = 1,
                      n_wires = 9,num_shot = 256,tau = 1,padding=0,bias=None):
        # class initialization 
        super().__init__()
        self.n_wires      = n_wires  # num of q circuits
        self.filter_size  = kernel_size
        if self.filter_size ==1: 
            self.n_wires =1
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.padding      = padding
        self.stride       = stride
        self.num_shot     = num_shot
        self.Linear = nn.Linear(self.n_wires, self.out_channels)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires) 
        self.encoder = tq.QuantumModuleList([GeneralEncoder_HK(int(self.n_wires),func_list_1[f'9_qubit_{c + 1}']) 
    for c in range(3)]) 
        self.arch2 = { 'n_wires': self.n_wires, 'n_blocks': 1,  'n_layers_per_block': 2}
        self.q_layer2 = tq.QuantumModuleList([U3CU3Layer0(self.arch2) for _ in range(self.in_channels)])
        self.arch     = { 'n_wires': self.n_wires, 'n_blocks': 3,  'n_layers_per_block': 5}
        self.q_layer   =U3CU3Layer0(self.arch) 
        if self.filter_size == 1:
            self.encoder = tq.QuantumModuleList([GeneralEncoder_HK(int(self.n_wires),func_list_1[f'1_qubit_{c + 1}'])                             
    for c in range(3)])
            self.q_layer2 = tq.QuantumModuleList()
            self.q_layer   = tq.QuantumModuleList()
            for _ in range(3):
                self.q_layer.append(tq.RX(has_params=True, trainable=True))
                self.q_layer.append(tq.RY(has_params=True, trainable=True))
                self.q_layer.append(tq.RZ(has_params=True, trainable=True))
            for _ in range(self.in_channels):
                self.q_layer2.append(tq.RX(has_params=True, trainable=True))
        self.measure   = tq.MeasureAll(tq.PauliZ)
        self.bias = 0

    def process(self, data):
        data_pad = nn.ZeroPad2d(self.padding)(data)
        bsz, num_channels, x_size, y_size = data_pad.shape
        W_start = torch.arange(0, int((x_size - self.filter_size)) + 1, self.stride)
        H_start = torch.arange(0, int((y_size - self.filter_size)) + 1, self.stride)
        num_pts_W = len(W_start)
        num_pts_H = len(H_start)
        W_pts, H_pts = torch.tensor([], dtype =torch.long), torch.tensor([], dtype =torch.long)

        for i in range(self.filter_size):
            W_pts = torch.cat((W_pts, W_start + i), dim=0)
            H_pts = torch.cat((H_pts, H_start + i), dim=0)

        _, W_pts_indices = torch.sort(W_pts)
        _, H_pts_indices = torch.sort(H_pts)
        W_pts, H_pts = W_pts[W_pts_indices], H_pts[H_pts_indices]

        indices = []
        for i in range(num_pts_W):
            for j in range(num_pts_H):
                indices.append((i, j))
        indices = torch.tensor(indices)

        row_indices = indices[:, 0].repeat(self.filter_size**2)
        col_indices = indices[:, 1].repeat(self.filter_size**2)
        patch_indices = torch.stack([row_indices, col_indices], dim=-1)

        patch_indices = patch_indices.long()

        data_processed = data_pad[:, :, W_pts[patch_indices[:, 0]], H_pts[patch_indices[:, 1]]]
        
        # Reshape data_processed to get the desired output format
        data_processed = data_processed.reshape(bsz, num_channels, num_pts_W * num_pts_H, self.filter_size, self.filter_size)
        data_processed = data_processed.permute(0,2,1,3,4)
        # Reshape to get the final desired output format
        data_processed = data_processed.view(-1, num_channels, self.filter_size * self.filter_size)



        return data_processed, bsz, num_pts_W, num_pts_H

    def im2col(self,input_data):
        N, C, H, W = input_data.shape
        out_h = (H + 2*self.padding - self.filter_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.filter_size) // self.stride + 1

        img = torch.nn.functional.pad(input_data, pad=[self.padding, self.padding, self.padding, self.padding], mode='constant', value=0)
        col = torch.zeros((N, C, self.filter_size, self.filter_size, out_h, out_w))

        for y in range(self.filter_size):
            y_max = y + self.stride*out_h
            for x in range(self.filter_size):
                x_max = x + self.stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]

        col = col.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, C, self.filter_size**2)
        return col, N, out_h, out_w
  
    def forward(self,data, sample = False):
        Input, bsz, ty_size , tx_size = self.im2col(data)
        feature_list =[]
        for c in range(self.in_channels):
            if c ==0:
                self.q_device.reset_states(Input.shape[0])
            self.encoder[int(c%3)](self.q_device,Input[:,c,:])
            if self.filter_size!= 1:
                self.q_layer2[c](self.q_device)
            else:
                self.q_layer2[c](self.q_device, wires=0)
            #print(f'channel: {int(c%3)} :{self.q_device.get_states_1d()[1]}')
        if self.filter_size!= 1:
            self.q_layer(self.q_device)
        else:
            for z in range(len(self.q_layer)):
                self.q_layer[z](self.q_device, wires=0)
        feature  = self.measure(self.q_device)[:,:self.n_wires]
        #print(feature.shape)
        feature_list.append(feature)
        feature = torch.cat(feature_list,dim=-1) 
        feature  = feature.reshape(bsz,ty_size, tx_size,self.n_wires) #self.n_wires : Channel
        feature  = self.Linear(feature) #(bsz, ty, tx, self.outchannels)
        out_feature = feature.permute(0,3,1,2)
 
        
                                            
        return out_feature

class QConv2d_2x2(tq.QuantumModule):
    def __init__(self,in_channels = 1, out_channels = 10, kernel_size = 2, stride = 1,
                      n_wires = 4,num_shot = 256,tau = 1,padding=0,bias=None):
        # class initialization 
        super().__init__()
        self.n_wires      = n_wires  
        self.filter_size  = kernel_size
        if self.filter_size ==1: 
            self.n_wires =1
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.padding      = padding
        self.stride       = stride
        self.num_shot     = num_shot
        self.Linear = nn.Linear(self.n_wires, self.out_channels)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires) 
        self.encoder = tq.QuantumModuleList([GeneralEncoder_HK(int(self.n_wires),func_list_1[f'4_qubit_{c + 1}']) 
    for c in range(3)]) 
        self.arch2 = { 'n_wires': self.n_wires, 'n_blocks': 1,  'n_layers_per_block': 2}
        self.q_layer2 = tq.QuantumModuleList([U3CU3Layer0(self.arch2) for _ in range(self.in_channels)])
        self.arch     = { 'n_wires': self.n_wires, 'n_blocks': 3,  'n_layers_per_block': 5}
        self.q_layer   =U3CU3Layer0(self.arch) 
        if self.filter_size == 1:
            self.encoder = tq.QuantumModuleList([GeneralEncoder_HK(int(self.n_wires),func_list_1[f'1_qubit_{c + 1}'])                             
    for c in range(3)])
            self.q_layer2 = tq.QuantumModuleList()
            self.q_layer   = tq.QuantumModuleList()
            for _ in range(3):
                self.q_layer.append(tq.RX(has_params=True, trainable=True))
                self.q_layer.append(tq.RY(has_params=True, trainable=True))
                self.q_layer.append(tq.RZ(has_params=True, trainable=True))
            for _ in range(self.in_channels):
                self.q_layer2.append(tq.RX(has_params=True, trainable=True))
        self.measure   = tq.MeasureAll(tq.PauliZ)
        self.bias = 0

    def process(self, data):
        data_pad = nn.ZeroPad2d(self.padding)(data)
        bsz, num_channels, x_size, y_size = data_pad.shape
        W_start = torch.arange(0, int((x_size - self.filter_size)) + 1, self.stride)
        H_start = torch.arange(0, int((y_size - self.filter_size)) + 1, self.stride)
        num_pts_W = len(W_start)
        num_pts_H = len(H_start)
        W_pts, H_pts = torch.tensor([], dtype =torch.long), torch.tensor([], dtype =torch.long)

        for i in range(self.filter_size):
            W_pts = torch.cat((W_pts, W_start + i), dim=0)
            H_pts = torch.cat((H_pts, H_start + i), dim=0)

        _, W_pts_indices = torch.sort(W_pts)
        _, H_pts_indices = torch.sort(H_pts)
        W_pts, H_pts = W_pts[W_pts_indices], H_pts[H_pts_indices]

        indices = []
        for i in range(num_pts_W):
            for j in range(num_pts_H):
                indices.append((i, j))
        indices = torch.tensor(indices)

        row_indices = indices[:, 0].repeat(self.filter_size**2)
        col_indices = indices[:, 1].repeat(self.filter_size**2)
        patch_indices = torch.stack([row_indices, col_indices], dim=-1)

        patch_indices = patch_indices.long()

        data_processed = data_pad[:, :, W_pts[patch_indices[:, 0]], H_pts[patch_indices[:, 1]]]
        
        # Reshape data_processed to get the desired output format
        data_processed = data_processed.reshape(bsz, num_channels, num_pts_W * num_pts_H, self.filter_size, self.filter_size)
        data_processed = data_processed.permute(0,2,1,3,4)
        # Reshape to get the final desired output format
        data_processed = data_processed.view(-1, num_channels, self.filter_size * self.filter_size)



        return data_processed, bsz, num_pts_W, num_pts_H

    def im2col(self,input_data):
        N, C, H, W = input_data.shape
        out_h = (H + 2*self.padding - self.filter_size) // self.stride + 1
        out_w = (W + 2*self.padding - self.filter_size) // self.stride + 1

        img = torch.nn.functional.pad(input_data, pad=[self.padding, self.padding, self.padding, self.padding], mode='constant', value=0)
        col = torch.zeros((N, C, self.filter_size, self.filter_size, out_h, out_w))

        for y in range(self.filter_size):
            y_max = y + self.stride*out_h
            for x in range(self.filter_size):
                x_max = x + self.stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]

        col = col.permute(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, C, self.filter_size**2)
        return col, N, out_h, out_w
  
    def forward(self,data, sample = False):
        Input, bsz, ty_size , tx_size = self.im2col(data)
        feature_list =[]
        for c in range(self.in_channels):
            if c ==0:
                self.q_device.reset_states(Input.shape[0])
            self.encoder[int(c%3)](self.q_device,Input[:,c,:])
            if self.filter_size!= 1:
                self.q_layer2[c](self.q_device)
            else:
                self.q_layer2[c](self.q_device, wires=0)
            #print(f'channel: {int(c%3)} :{self.q_device.get_states_1d()[1]}')
        if self.filter_size!= 1:
            self.q_layer(self.q_device)
        else:
            for z in range(len(self.q_layer)):
                self.q_layer[z](self.q_device, wires=0)
        feature  = self.measure(self.q_device)[:,:self.n_wires]
        #print(feature.shape)
        feature_list.append(feature)
        feature = torch.cat(feature_list,dim=-1) 
        feature  = feature.reshape(bsz,ty_size, tx_size,self.n_wires) #self.n_wires : Channel
        feature  = self.Linear(feature) #(bsz, ty, tx, self.outchannels)
        out_feature = feature.permute(0,3,1,2)
 
        
                                            
        return out_feature    
  
#Encoding.py에 이식    
  
class GeneralEncoder(Encoder, metaclass=ABCMeta):
    def __init__(self, input_size,func_list):
        super().__init__()
        self.input_size = input_size
        self.func_list = func_list
        #print(self.func_list)
    
    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x):
        self.q_device = q_device
        #self.q_device.reset_states(x.shape[0])
        for info in self.func_list:
            if tq.op_name_dict[info['func']].num_params > 0:
                params = x[:, info['input_idx']]
            else:
                params = None
            func_name_dict[info['func']](
                self.q_device,
                wires=info['wires'],
                params=params,
                static=self.static_mode,
                parent_graph=self.graph
            )    
    

func_list_1 = {'4_qubit_1':
        [
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'ry', 'wires': [2]},
            {'input_idx': [3], 'func': 'ry', 'wires': [3]},
          
        ],
              '4_qubit_2':
        [
            {'input_idx': [0], 'func': 'rx', 'wires': [0]},
            {'input_idx': [1], 'func': 'rx', 'wires': [1]},
            {'input_idx': [2], 'func': 'rx', 'wires': [2]},
            {'input_idx': [3], 'func': 'rx', 'wires': [3]},
          
            
        ],
              '4_qubit_3':
        [
            {'input_idx': [0], 'func': 'rz', 'wires': [0]},
            {'input_idx': [1], 'func': 'rz', 'wires': [1]},
            {'input_idx': [2], 'func': 'rz', 'wires': [2]},
            {'input_idx': [3], 'func': 'rz', 'wires': [3]},
           
            
        ],
               
               
              '1_qubit_1':
        [
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
          
           
        ],
              '1_qubit_2':
        [
            {'input_idx': [0], 'func': 'rx', 'wires': [0]},
          
        ],
              '1_qubit_3':
        [
            {'input_idx': [0], 'func': 'rz', 'wires': [0]},
          
        ],
              '9_qubit_1':
        [
            {'input_idx': [0], 'func': 'ry', 'wires': [0]},
            {'input_idx': [1], 'func': 'ry', 'wires': [1]},
            {'input_idx': [2], 'func': 'ry', 'wires': [2]},
            {'input_idx': [3], 'func': 'ry', 'wires': [3]},
            {'input_idx': [4], 'func': 'ry', 'wires': [4]},
            {'input_idx': [5], 'func': 'ry', 'wires': [5]},
            {'input_idx': [6], 'func': 'ry', 'wires': [6]},
            {'input_idx': [7], 'func': 'ry', 'wires': [7]},
            {'input_idx': [8], 'func': 'ry', 'wires': [8]},
        ],
              '9_qubit_2':
        [
            {'input_idx': [0], 'func': 'rx', 'wires': [0]},
            {'input_idx': [1], 'func': 'rx', 'wires': [1]},
            {'input_idx': [2], 'func': 'rx', 'wires': [2]},
            {'input_idx': [3], 'func': 'rx', 'wires': [3]},
            {'input_idx': [4], 'func': 'rx', 'wires': [4]},
            {'input_idx': [5], 'func': 'rx', 'wires': [5]},
            {'input_idx': [6], 'func': 'rx', 'wires': [6]},
            {'input_idx': [7], 'func': 'rx', 'wires': [7]},
            {'input_idx': [8], 'func': 'rx', 'wires': [8]},
            
        ],
              '9_qubit_3':
        [
            {'input_idx': [0], 'func': 'rz', 'wires': [0]},
            {'input_idx': [1], 'func': 'rz', 'wires': [1]},
            {'input_idx': [2], 'func': 'rz', 'wires': [2]},
            {'input_idx': [3], 'func': 'rz', 'wires': [3]},
            {'input_idx': [4], 'func': 'rz', 'wires': [4]},
            {'input_idx': [5], 'func': 'rz', 'wires': [5]},
            {'input_idx': [6], 'func': 'rz', 'wires': [6]},
            {'input_idx': [7], 'func': 'rz', 'wires': [7]},
            {'input_idx': [8], 'func': 'rz', 'wires': [8]},
            
        ]}