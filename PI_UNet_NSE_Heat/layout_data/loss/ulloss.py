import torch
from torch.nn import MSELoss
from torch.nn.functional import conv2d, pad, interpolate
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

class NSE_layer(torch.nn.Module):
    def __init__(self, base_loss=MSELoss(reduction='mean'),
                nx=256, ny=128, length_x=6, length_y=3, nu=5*1e-2):
        super(NSE_layer, self).__init__()
        self.nx = nx
        self.ny = ny
        self.length_x = length_x
        self.length_y = length_y
        self.nu = nu
        self.h = self.length_x / self.nx
        self.scale_factor = 1  
        TEMPER_COEFFICIENT = 1
        self.base_loss = base_loss 

        # weights
        self.laplace_weight = torch.Tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]])
        self.dx_weight = torch.Tensor([[[[0,0,0],[-0.5,0,0.5],[0,0,0]]]])
        self.dy_weight = torch.Tensor([[[[0,-0.5,0],[0,0,0],[0,0.5,0]]]])
        self.fdx_weight = torch.Tensor([[[[0,0,0],[0,-1,1],[0,0,0]]]])
        self.fdy_weight = torch.Tensor([[[[0,1,0],[0,-1,0],[0,0,0]]]])
        self.bdx_weight = torch.Tensor([[[[0,0,0],[-1,1,0],[0,0,0]]]])
        self.bdy_weight = torch.Tensor([[[[0,0,0],[0,1,0],[0,-1,0]]]])

    # differential    
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
    
    # NSE
    def momentum_u(self,x,case):
        u = x[...,0,:,:]
        p = x[...,2,:,:]
        if (case==4)|(case==8)|(case==11): Dp = self.FDx(p)
        elif (case==6)|(case==9)|(case==10): Dp = self.BDx(p)
        else: Dp = self.Dx(p)
        
        return self.u*self.Dx(u)/self.h + self.v*self.Dy(u)/self.h + Dp/self.h - self.nu*(self.laplace(u))/self.h/self.h
    
    def momentum_v(self,x,case):
        v = x[...,1,:,:]
        p = x[...,2,:,:]
        
        if (case==7)|(case==10)|(case==11): Dp = self.FDy(p)
        elif (case==5)|(case==8)|(case==9): Dp = self.BDy(p)
        else: Dp = self.Dy(p)
        
        return self.u*self.Dx(v)/self.h + self.v*self.Dy(v)/self.h + Dp/self.h - self.nu*(self.laplace(v))/self.h/self.h
    
    def continuity(self,x):
        u = x[...,0,:,:]
        v = x[...,1,:,:]
        return self.Dx(u)/self.h + self.Dy(v)/self.h

    def NSE(self,x,case):
        return torch.stack([self.momentum_u(x,case), self.momentum_v(x,case), self.continuity(x)],dim=1)

    # boundary condition
    def set_bc(self, bc_num, outvar):
        var_num = {'u':0, 'v':1, 'p':2}
        indices = (self.boundary == bc_num).nonzero(as_tuple=True)
        for item in outvar:
            self.bc_mask[...,var_num[item], indices[0], indices[1]] = 0
            self.bc_value[..., var_num[item], indices[0], indices[1]] = outvar[item]
        return None
    
    def apply_NSE(self, x, eq_num):
        mask = torch.zeros_like(self.boundary)
        indices = (self.boundary == eq_num).nonzero(as_tuple=True)
        mask[...,indices[0],indices[1]] = 1
        self.eq_mask[...,indices[0],indices[1]] = eq_num
        return mask * self.NSE(x,eq_num)

    def forward(self, layout, flow):
        self.geom = layout[...,0,:,:].squeeze()
        self.boundary = layout[...,1,:,:].squeeze()
        
        # Source item
        f = 0 #self.cof * layout
        
        # Dirichlet Boundary
        # 0: interior, 1: inflow, 2: non-slip, 3: outlet
        self.bc_value = torch.zeros_like(flow).detach() # boundary setting value
        self.bc_mask = torch.ones_like(flow).detach() # The nodes which are not in boundary
        
        # apply hard BC
        self.set_bc(bc_num=2, outvar={'u':0, 'v':0, 'p':0}) # non slip
        self.set_bc(bc_num=1, outvar={'u':3, 'v':0}) # inlet
        self.set_bc(bc_num=3, outvar={'p':0}) # outlet

        flow_bc = flow * self.bc_mask + self.bc_value
        self.u = flow_bc[...,0,:,:]
        self.v = flow_bc[...,1,:,:]
        
        x = F.pad(flow_bc, [1, 1, 1, 1], mode='reflect')

        # physics equation
        ## 4: FD x, 5: BD y, 6: BD x, 7: FD y, 8: 4+5, 9: 5+6, 10: 6+7, 11: 4+7 
        self.eq_mask = torch.zeros_like(self.boundary)
        loss_eq = torch.zeros_like(flow)
        for eq_num in [0,4,5,6,7,8,9,10,11]: # interior
            loss_eq += self.apply_NSE(x,eq_num)

        # make loss
        loss_nse_m_u = self.base_loss(loss_eq[...,0,:,:], torch.zeros_like(loss_eq[...,0,:,:]))
        loss_nse_m_v = self.base_loss(loss_eq[...,1,:,:], torch.zeros_like(loss_eq[...,1,:,:]))
        loss_nse_d = self.base_loss(loss_eq[...,2,:,:], torch.zeros_like(loss_eq[...,2,:,:]))
        loss_nse_m = loss_nse_m_u + loss_nse_m_v
        loss_nse = loss_nse_m + loss_nse_d

        return loss_nse, flow_bc, self.eq_mask

class Energy_layer(torch.nn.Module):
    def __init__(
            self,base_loss=MSELoss(reduction='mean'), 
            nx=256, ny=128, length_x=6, length_y=3):
        super(Energy_layer, self).__init__()
        self.length_x = length_x
        self.length_y = length_y
        self.nx = nx
        self.ny = ny
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        self.h = self.length_x / self.nx
        self.cof = TEMPER_COEFFICIENT
        self.base_loss = base_loss
        self.diff_coeff = 0# 2.2 * 1e-5

        # The weight 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.weight = torch.Tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
        self.laplace_weight = torch.Tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]])
        self.dx_weight = torch.Tensor([[[[0,0,0],[-0.5,0,0.5],[0,0,0]]]])
        self.dy_weight = torch.Tensor([[[[0,-0.5,0],[0,0,0],[0,0.5,0]]]])
        self.fdx_weight = torch.Tensor([[[[0,0,0],[0,-1,1],[0,0,0]]]])
        self.fdy_weight = torch.Tensor([[[[0,1,0],[0,-1,0],[0,0,0]]]])
        self.bdx_weight = torch.Tensor([[[[0,0,0],[-1,1,0],[0,0,0]]]])
        self.bdy_weight = torch.Tensor([[[[0,0,0],[0,1,0],[0,-1,0]]]])
        self.boundary_weight = torch.Tensor([[[[0,1,0],[1,0,-1],[0,-1,0]]]])
        
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
    def jacobi(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)
    def advection_diffusion(self, x):
        return self.u*self.Dx(x)+self.v*self.Dy(x) - self.diff_coeff*(self.laplace(x))/self.h
    def newman_bc(self,x):
        return conv2d(x, self.boundary_weight.to(device=x.device), bias=None, stride=1, padding=0)

    def set_bc(self, bc_num, outvar):
        var_num = {'tmp':0}
        indices = (self.boundary == bc_num).nonzero(as_tuple=True)
        for item in outvar:
            self.bc_mask[...,var_num[item], indices[0], indices[1]] = 0
            self.bc_value[..., var_num[item], indices[0], indices[1]] = outvar[item]
        return None
    
    def apply_Energy(self, x, eq_num):
        mask = torch.zeros_like(self.boundary)
        indices = (self.boundary == eq_num).nonzero(as_tuple=True)
        mask[...,indices[0],indices[1]] = 1
        self.eq_mask[...,indices[0],indices[1]] = eq_num

        ## 4: FD x, 5: FD y, 6: BD x, 7: BD y, 8: 4+5, 9: 5+6, 10: 6+7, 11: 4+7 

        eq_energy = {0:self.advection_diffusion(x), #2:self.advection_diffusion(x),
        3: self.advection_diffusion(x),
        4: self.advection_diffusion(x)+self.FDx(x),
        5: self.advection_diffusion(x)+self.FDy(x),
        6: self.advection_diffusion(x)+self.BDx(x),
        7: self.advection_diffusion(x)+self.BDy(x),
        8: self.advection_diffusion(x)+(self.FDx(x)+self.FDy(x)),
        9: self.advection_diffusion(x)+(self.FDy(x)+self.BDx(x)),
        10: self.advection_diffusion(x)+(self.BDx(x)+self.BDy(x)),
        11: self.advection_diffusion(x)+(self.FDx(x)+self.BDy(x)),}

        return mask * eq_energy[eq_num]

    def apply_Flux(self, x, eq_num):
        mask = torch.zeros_like(self.boundary)
        indices = (self.boundary == eq_num).nonzero(as_tuple=True)
        mask[...,indices[0],indices[1]] = 1

        ## 4: FD x, 5: FD y, 6: BD x, 7: BD y, 8: 4+5, 9: 5+6, 10: 6+7, 11: 4+7 

        eq_flux = {4: torch.abs(self.FDx(x))-self.flux, 5: torch.abs(self.FDy(x))-self.flux,
        6: torch.abs(self.BDx(x))-self.flux, 
        7: torch.abs(self.BDy(x))-self.flux, 
        8: torch.abs(self.FDx(x)+self.FDy(x))-1.41421*(self.flux), 
        9: torch.abs(self.FDy(x)+self.BDx(x))-1.41421*(self.flux), 
        10: torch.abs(self.BDx(x)+self.BDy(x))-1.41421*(self.flux), 
        11: torch.abs(self.FDx(x)+self.BDy(x))-1.41421*(self.flux),}

        return mask * eq_flux[eq_num]

    def forward(self, layout, heat, flow):
        self.u = flow[...,0,:,:]
        self.v = flow[...,1,:,:]
        self.boundary = layout[...,1,:,:].clone().squeeze()
        self.geom = layout[...,0,:,:].clone()
        self.geom[...,0,:] = 1 # upper wall
        self.geom[...,-1,:] = 1 # lower wall
        self.boundary[...,1,1:] = 0 # upper wall 1diff
        self.boundary[...,-2,1:] = 0 # lower wall 1diff
        self.boundary[...,:,1] = 0 # inlet 1diff
        self.boundary[...,:,-1] = 3 # outlet
        self.boundary[...,0,:] = 3 # upper wall
        self.boundary[...,-1,:] = 3 # lower wall
        
        # Source item
        self.src = 0#300 * self.h #* self.h * self.h
        self.flux = 300 * self.h #-3000
        
        f = self.cof * abs(self.geom-1) * self.src #* self.h

        # Dirichlet boundary
        self.bc_value = torch.zeros_like(heat).detach()
        self.bc_mask = torch.ones_like(heat).detach()
        
        self.set_bc(bc_num=1, outvar={'tmp':0}) # inlet
        
        heat_bc = heat * self.bc_mask + self.bc_value

        x = F.pad(heat_bc, [1, 1, 1, 1], mode='reflect')  # constant, reflect, reflect
        
        # physics
        self.eq_mask = torch.zeros_like(self.boundary)
        loss_energy = torch.zeros_like(heat)
        loss_flux = torch.zeros_like(heat)

        for eq_num in [0,3,4,5,6,7,8,9,10,11]:
            loss_energy += self.apply_Energy(x,eq_num)
        mse_energy = self.base_loss(loss_energy, f)

        for eq_num in [4,5,6,7,8,9,10,11]:
            loss_flux += self.apply_Flux(x,eq_num)
        mse_flux = self.base_loss(loss_flux,torch.zeros_like(loss_flux))

        return mse_energy + mse_flux, heat_bc, self.eq_mask


class Jacobi_layer(torch.nn.Module):
    def __init__(
            self, nx=21, length=0.1, bcs=None
    ):
        super(Jacobi_layer, self).__init__()
        self.length = length
        self.bcs = bcs
        # The weight 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.weight = torch.Tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
        # Padding
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        STRIDE = self.length / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def jacobi(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat, n_iter):
        # Source item
        f = self.cof * layout
        # The nodes which are not in boundary
        G = torch.ones_like(heat).detach()

        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            pass
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, :1] = torch.zeros_like(G[..., idx_start:idx_end, :1])
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, -1:] = torch.zeros_like(G[..., idx_start:idx_end, -1:])
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., :1, idx_start:idx_end] = torch.zeros_like(G[..., :1, idx_start:idx_end])
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., -1:, idx_start:idx_end] = torch.zeros_like(G[..., -1:, idx_start:idx_end])
                else:
                    raise ValueError("bc error!")
        for i in range(n_iter):
            if i == 0:
                x = F.pad(heat * G, [1, 1, 1, 1], mode='reflect')
            else:
                x = F.pad(x, [1, 1, 1, 1], mode='reflect')
            x = G * (self.jacobi(x) + f)
        return x

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
