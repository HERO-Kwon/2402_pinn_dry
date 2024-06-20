import torch
from torch.nn import MSELoss
from torch.nn.functional import conv2d, pad, interpolate
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class NSE_layer(torch.nn.Module):
    def __init__(
            self, nx=21, ny=21, length_x = 0.1, length_y=0.1, nu = 5*1e-2, bcs=None
    ):
        super(NSE_layer, self).__init__()
        self.length_x = length_x
        self.length_y = length_y
        self.nu = nu
        self.bcs = bcs
        # The weight 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.laplace_weight = torch.Tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]])
        self.dx_weight = torch.Tensor([[[[0,0,0],[-0.5,0,0.5],[0,0,0]]]])
        self.dy_weight = torch.Tensor([[[[0,-0.5,0],[0,0,0],[0,0.5,0]]]])
        self.fdx_weight = torch.Tensor([[[[0,0,0],[0,-1,1],[0,0,0]]]])
        self.fdy_weight = torch.Tensor([[[[0,1,0],[0,-1,0],[0,0,0]]]])
        self.bdx_weight = torch.Tensor([[[[0,0,0],[-1,1,0],[0,0,0]]]])
        self.bdy_weight = torch.Tensor([[[[0,0,0],[0,1,0],[0,-1,0]]]])
        # Padding
        self.nx = nx
        self.ny = ny
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        self.h = self.length_x / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        #self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def Dx(self,x):
        return conv2d(x, self.dx_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def Dy(self,x):
        return conv2d(x, self.dy_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def FDx(self,x):
        return conv2d(x, self.fdx_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def FDy(self,x):
        return conv2d(x, self.fdy_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def BDx(self,x):
        return conv2d(x, self.bdx_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def BDy(self,x):
        return conv2d(x, self.bdy_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def laplace(self, x):
        return conv2d(x, self.laplace_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def momentum_u(self,x,case):
        u = x[...,0,:,:]
        v = x[...,1,:,:]
        p = x[...,2,:,:]
        
        if (case==4)|(case==8)|(case==11):
            Dp = self.FDx(p)
        elif (case==6)|(case==9)|(case==10):
            Dp = self.BDx(p)
        else:
            Dp = self.Dx(p)

        return self.u*self.Dx(u)/self.h + self.v*self.Dy(u)/self.h + Dp/self.h - self.nu*(self.laplace(u))/self.h/self.h
        
    def momentum_v(self,x,case):
        u = x[...,0,:,:]
        v = x[...,1,:,:]
        p = x[...,2,:,:]

        if (case==7)|(case==10)|(case==11):
            Dp = self.FDy(p)
        elif (case==5)|(case==8)|(case==9):
            Dp = self.BDy(p)
        else:
            Dp = self.Dy(p)

        return self.u*self.Dx(v)/self.h + self.v*self.Dy(v)/self.h + Dp/self.h - self.nu*(self.laplace(v))/self.h/self.h
    
    def continuity(self,x):
        u = x[...,0,:,:]
        v = x[...,1,:,:]
        return self.Dx(u)/self.h + self.Dy(v)/self.h
    def NSE(self,x,case):
        return torch.stack([self.momentum_u(x,case), self.momentum_v(x,case), self.continuity(x)])


    def forward(self, layout, flow):
        # Source item
        f = 0#self.cof * layout
        # The nodes which are not in boundary
        #G = torch.ones_like(flow).detach()
        #G_nonslip = torch.zeros_like(flow).detach()
        G_bc_value = torch.zeros_like(flow).detach()
        G_bc = torch.ones_like(flow).detach()
        '''
        0: interior
        1: inflow
        2: non-slip
        3: outlet
        4: FD x, 5: BD y, 6: BD x, 7: FD y
        8: 4+5, 9: 5+6, 10: 6+7, 11: 4+7 
        '''
        # dirichlet bc 0
        layout = layout[...,1,:,:].squeeze()
        # non-slip values
        indices_ns = (layout == 2).nonzero(as_tuple=True)
        G_bc_value[..., 0, indices_ns[0], indices_ns[1]] = 0 # u
        G_bc_value[..., 1, indices_ns[0], indices_ns[1]] = 0 # v
        G_bc_value[..., 2, indices_ns[0], indices_ns[1]] = 0 # p
        G_bc[..., 0, indices_ns[0], indices_ns[1]] = 0 # u
        G_bc[..., 1, indices_ns[0], indices_ns[1]] = 0 # v
        G_bc[..., 2, indices_ns[0], indices_ns[1]] = 0 # p
        
        # inlet and outlet values
        indices_in = (layout == 1).nonzero(as_tuple=True)
        G_bc_value[..., 0, indices_in[0], 0] = 3 # u
        G_bc_value[..., 1, indices_in[0], 0] = 0 # v
        G_bc[..., 0, indices_in[0], 0] = 0 # u
        G_bc[..., 1, indices_in[0], 0] = 0 # v

        indices_out = (layout == 3).nonzero(as_tuple=True)
        G_bc_value[..., 2, indices_out[0], -1] = 0 # p outlet
        G_bc[..., 2, indices_out[0], -1] = 0 # p outlet

        #x = F.pad(flow * G * G_bc + G_nonslip + G_inout, [1, 1, 1, 1], mode='reflect')
        x = flow * G_bc + G_bc_value

        self.u = x[...,0,1:-1,1:-1]
        self.v = x[...,1,1:-1,1:-1]

        # mask
        indices_1diff = (layout > 3).nonzero(as_tuple=True)
        G_bc0 = G_bc.clone()
        G_bc0[...,:,indices_1diff[0],indices_1diff[1]] = 0
        
        indices_bc4 = (layout == 4).nonzero(as_tuple=True)
        G_bc4 = torch.zeros_like(flow).detach()
        G_bc4[...,:,indices_bc4[0],indices_bc4[1]] = 1
        indices_bc5 = (layout == 5).nonzero(as_tuple=True)
        G_bc5 = torch.zeros_like(flow).detach()
        G_bc5[...,:,indices_bc5[0],indices_bc5[1]] = 1
        indices_bc6 = (layout == 6).nonzero(as_tuple=True)
        G_bc6 = torch.zeros_like(flow).detach()
        G_bc6[...,:,indices_bc6[0],indices_bc6[1]] = 1
        indices_bc7 = (layout == 7).nonzero(as_tuple=True)
        G_bc7 = torch.zeros_like(flow).detach()
        G_bc7[...,:,indices_bc7[0],indices_bc7[1]] = 1
        indices_bc8 = (layout == 8).nonzero(as_tuple=True)
        G_bc8 = torch.zeros_like(flow).detach()
        G_bc8[...,:,indices_bc8[0],indices_bc8[1]] = 1
        indices_bc9 = (layout == 9).nonzero(as_tuple=True)
        G_bc9 = torch.zeros_like(flow).detach()
        G_bc9[...,:,indices_bc9[0],indices_bc9[1]] = 1
        indices_bc10 = (layout == 10).nonzero(as_tuple=True)
        G_bc10 = torch.zeros_like(flow).detach()
        G_bc10[...,:,indices_bc10[0],indices_bc10[1]] = 1
        indices_bc11 = (layout == 11).nonzero(as_tuple=True)
        G_bc11 = torch.zeros_like(flow).detach()
        G_bc11[...,:,indices_bc11[0],indices_bc11[1]] = 1
        #x = F.pad(x,[1,1,1,1], mode='reflect')
        #loss_nse = G_bc * (self.NSE(x) + f)
        loss_nse = G_bc0[...,1:-1,1:-1] * self.NSE(x,0)
        loss_nse += G_bc4[...,1:-1,1:-1] * self.NSE(x,4)
        loss_nse += G_bc5[...,1:-1,1:-1] * self.NSE(x,5)
        loss_nse += G_bc6[...,1:-1,1:-1] * self.NSE(x,6)
        loss_nse += G_bc7[...,1:-1,1:-1] * self.NSE(x,7)
        loss_nse += G_bc8[...,1:-1,1:-1] * self.NSE(x,8)
        loss_nse += G_bc9[...,1:-1,1:-1] * self.NSE(x,9)
        loss_nse += G_bc10[...,1:-1,1:-1] * self.NSE(x,10)
        loss_nse += G_bc11[...,1:-1,1:-1] * self.NSE(x,11)

        G_bc_mask = G_bc0+4*G_bc4+5*G_bc5+6*G_bc6+7*G_bc7+8*G_bc8+9*G_bc9+10*G_bc10+11*G_bc11
        G_bc_v = torch.stack([G_bc,G_bc_value])

        return loss_nse, G_bc_mask, G_bc_v

class Energy_layer(torch.nn.Module):
    def __init__(
            self, nx=21, ny=21, length_x = 0.1, length_y=0.1, nu = 5*1e-2, bcs=None
    ):
        super(NSE_layer, self).__init__()
        self.length_x = length_x
        self.length_y = length_y
        self.nu = nu
        self.bcs = bcs
        # The weight 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.laplace_weight = torch.Tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]])
        self.dx_weight = torch.Tensor([[[[0,0,0],[-0.5,0,0.5],[0,0,0]]]])
        self.dy_weight = torch.Tensor([[[[0,-0.5,0],[0,0,0],[0,0.5,0]]]])
        self.fdx_weight = torch.Tensor([[[[0,0,0],[0,-1,1],[0,0,0]]]])
        self.fdy_weight = torch.Tensor([[[[0,1,0],[0,-1,0],[0,0,0]]]])
        self.bdx_weight = torch.Tensor([[[[0,0,0],[-1,1,0],[0,0,0]]]])
        self.bdy_weight = torch.Tensor([[[[0,0,0],[0,1,0],[0,-1,0]]]])
        # Padding
        self.nx = nx
        self.ny = ny
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        self.h = self.length_x / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        #self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def Dx(self,x):
        return conv2d(x, self.dx_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def Dy(self,x):
        return conv2d(x, self.dy_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def FDx(self,x):
        return conv2d(x, self.fdx_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def FDy(self,x):
        return conv2d(x, self.fdy_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def BDx(self,x):
        return conv2d(x, self.bdx_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def BDy(self,x):
        return conv2d(x, self.bdy_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def laplace(self, x):
        return conv2d(x, self.laplace_weight.to(device=x.device), bias=None, stride=1, padding=0)
    
    def advection(self,x,case):
        temp = x[...,3,:,:]
        # diff x
        if (case==4)|(case==8)|(case==11):
            Dxtemp = self.FDx(temp)
        elif (case==6)|(case==9)|(case==10):
            Dxtemp = self.BDx(temp)
        else:
            Dxtemp = self.Dx(temp)
        # diff y
        if (case==7)|(case==10)|(case==11):
            Dytemp = self.FDy(temp)
        elif (case==5)|(case==8)|(case==9):
            Dtemp = self.BDy(temp)
        else:
            Dytemp = self.Dy(temp)

        return self.u*self.Dxtemp(temp)/self.h + self.v*self.Dytemp(temp)/self.h
        

    def Energy(self,x,case):
        return self.advection(x,case)

    def forward(self, layout, flow):
        # Source item
        f = 0#self.cof * layout
        # The nodes which are not in boundary
        #G = torch.ones_like(flow).detach()
        #G_nonslip = torch.zeros_like(flow).detach()
        G_bc_value = torch.zeros_like(flow).detach()
        G_bc = torch.ones_like(flow).detach()
        '''
        0: interior
        1: inflow
        2: non-slip
        3: outlet
        4: FD x, 5: BD y, 6: BD x, 7: FD y
        8: 4+5, 9: 5+6, 10: 6+7, 11: 4+7 
        '''
        # dirichlet bc 0
        layout = layout[...,1,:,:].squeeze()
        # non-slip values
        indices_ns = (layout == 2).nonzero(as_tuple=True)
        G_bc_value[..., 0, indices_ns[0], indices_ns[1]] = 0 # u
        G_bc_value[..., 1, indices_ns[0], indices_ns[1]] = 0 # v
        G_bc_value[..., 2, indices_ns[0], indices_ns[1]] = 0 # p
        G_bc[..., 0, indices_ns[0], indices_ns[1]] = 0 # u
        G_bc[..., 1, indices_ns[0], indices_ns[1]] = 0 # v
        G_bc[..., 2, indices_ns[0], indices_ns[1]] = 0 # p
        
        # inlet and outlet values
        indices_in = (layout == 1).nonzero(as_tuple=True)
        G_bc_value[..., 0, indices_in[0], 0] = 3 # u
        G_bc_value[..., 1, indices_in[0], 0] = 0 # v
        G_bc[..., 0, indices_in[0], 0] = 0 # u
        G_bc[..., 1, indices_in[0], 0] = 0 # v

        indices_out = (layout == 3).nonzero(as_tuple=True)
        G_bc_value[..., 2, indices_out[0], -1] = 0 # p outlet
        G_bc[..., 2, indices_out[0], -1] = 0 # p outlet

        #x = F.pad(flow * G * G_bc + G_nonslip + G_inout, [1, 1, 1, 1], mode='reflect')
        x = flow * G_bc + G_bc_value

        self.u = x[...,0,1:-1,1:-1]
        self.v = x[...,1,1:-1,1:-1]

        # mask
        indices_1diff = (layout > 3).nonzero(as_tuple=True)
        G_bc0 = G_bc.clone()
        G_bc0[...,:,indices_1diff[0],indices_1diff[1]] = 0
        
        indices_bc4 = (layout == 4).nonzero(as_tuple=True)
        G_bc4 = torch.zeros_like(flow).detach()
        G_bc4[...,:,indices_bc4[0],indices_bc4[1]] = 1
        indices_bc5 = (layout == 5).nonzero(as_tuple=True)
        G_bc5 = torch.zeros_like(flow).detach()
        G_bc5[...,:,indices_bc5[0],indices_bc5[1]] = 1
        indices_bc6 = (layout == 6).nonzero(as_tuple=True)
        G_bc6 = torch.zeros_like(flow).detach()
        G_bc6[...,:,indices_bc6[0],indices_bc6[1]] = 1
        indices_bc7 = (layout == 7).nonzero(as_tuple=True)
        G_bc7 = torch.zeros_like(flow).detach()
        G_bc7[...,:,indices_bc7[0],indices_bc7[1]] = 1
        indices_bc8 = (layout == 8).nonzero(as_tuple=True)
        G_bc8 = torch.zeros_like(flow).detach()
        G_bc8[...,:,indices_bc8[0],indices_bc8[1]] = 1
        indices_bc9 = (layout == 9).nonzero(as_tuple=True)
        G_bc9 = torch.zeros_like(flow).detach()
        G_bc9[...,:,indices_bc9[0],indices_bc9[1]] = 1
        indices_bc10 = (layout == 10).nonzero(as_tuple=True)
        G_bc10 = torch.zeros_like(flow).detach()
        G_bc10[...,:,indices_bc10[0],indices_bc10[1]] = 1
        indices_bc11 = (layout == 11).nonzero(as_tuple=True)
        G_bc11 = torch.zeros_like(flow).detach()
        G_bc11[...,:,indices_bc11[0],indices_bc11[1]] = 1
        #x = F.pad(x,[1,1,1,1], mode='reflect')
        #loss_nse = G_bc * (self.NSE(x) + f)
        loss_energy = G_bc0[...,1:-1,1:-1] * self.Energy(x,0)
        loss_energy += G_bc4[...,1:-1,1:-1] * self.Energy(x,4)
        loss_energy += G_bc5[...,1:-1,1:-1] * self.Energy(x,5)
        loss_energy += G_bc6[...,1:-1,1:-1] * self.Energy(x,6)
        loss_energy += G_bc7[...,1:-1,1:-1] * self.Energy(x,7)
        loss_energy += G_bc8[...,1:-1,1:-1] * self.Energy(x,8)
        loss_energy += G_bc9[...,1:-1,1:-1] * self.Energy(x,9)
        loss_energy += G_bc10[...,1:-1,1:-1] * self.Energy(x,10)
        loss_energy += G_bc11[...,1:-1,1:-1] * self.Energy(x,11)

        G_bc_mask = G_bc0+4*G_bc4+5*G_bc5+6*G_bc6+7*G_bc7+8*G_bc8+9*G_bc9+10*G_bc10+11*G_bc11
        G_bc_v = torch.stack([G_bc,G_bc_value])

        return loss_energy, G_bc_mask, G_bc_v

class OHEMF12d(torch.nn.Module):
    """
    Weighted Loss
    """

    def __init__(self, loss_fun, weight=None):
        super(OHEMF12d, self).__init__()
        self.weight = weight
        self.loss_fun = loss_fun

    def forward(self, inputs, targets):
        diff = self.loss_fun(inputs, targets, reduction='none').detach()
        min, max = torch.min(diff.view(diff.shape[0], -1), dim=1)[0], torch.max(diff.view(diff.shape[0], -1), dim=1)[0]
        if inputs.ndim == 4:
            min, max = min.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)
        elif inputs.ndim == 3:
            min, max = min.reshape(diff.shape[0], 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1).expand(diff.shape)
        diff = 10.0 * (diff - min) / (max - min)
        return torch.mean(torch.abs(diff * (inputs - targets)))
