import torch
from torch.nn import MSELoss
from torch.nn.functional import conv2d, pad, interpolate
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

'''
class OutsideLoss(_Loss):
    def __init__(
            self, base_loss=MSELoss(reduction='mean'), length=0.1, u_D=298, bcs=None, nx=21
    ):
        super().__init__()
        self.base_loss = base_loss
        self.u_D = u_D
        self.bcs = bcs
        self.nx = nx
        self.length = length

    def forward(self, x):
        N, C, W, H = x.shape
        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all bcs are Dirichlet
            d1 = x[:, :, :1, :]
            d2 = x[:, :, -1:, :]
            d3 = x[:, :, 1:-1, :1]
            d4 = x[:, :, 1:-1, -1:]
            point = torch.cat([d1.flatten(), d2.flatten(), d3.flatten(), d4.flatten()], dim=0)
            return self.base_loss(point, torch.ones_like(point) * 0)
        loss = 0
        loss_consistency = 0
        for bc in self.bcs:
            if bc[0][1] == 0 and bc[1][1] == 0:
                idx_start = round(bc[0][0] * self.nx / self.length)
                idx_end = round(bc[1][0] * self.nx / self.length)
                point = x[..., idx_start:idx_end, :1]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            elif bc[0][1] == self.length and bc[1][1] == self.length:
                idx_start = round(bc[0][0] * self.nx / self.length)
                idx_end = round(bc[1][0] * self.nx / self.length)
                point = x[..., idx_start:idx_end, -1:]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            elif bc[0][0] == 0 and bc[1][0] == 0:
                idx_start = round(bc[0][1] * self.nx / self.length)
                idx_end = round(bc[1][1] * self.nx / self.length)
                point = x[..., :1, idx_start:idx_end]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            elif bc[0][0] == self.length and bc[1][0] == self.length:
                idx_start = round(bc[0][1] * self.nx / self.length)
                idx_end = round(bc[1][1] * self.nx / self.length)
                point = x[..., -1:, idx_start:idx_end]
                loss += self.base_loss(point, torch.ones_like(point) * 0)
            else:
                raise ValueError("bc error!")
        return loss


class LaplaceLoss(_Loss):
    def __init__(
            self, base_loss=MSELoss(reduction='mean'), nx=21,
            length=0.1, weight=[[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], bcs=None,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weight = torch.Tensor(weight)
        self.bcs = bcs
        self.length = length
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50.0
        STRIDE = self.length / self.nx
        self.cof = -1 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def laplace(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat):
        layout = interpolate(layout, scale_factor=self.scale_factor)

        heat = pad(heat, [1, 1, 1, 1], mode='reflect')  # constant, reflect, reflect
        layout_pred = self.laplace(heat)
        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            return self.base_loss(layout_pred[..., 1:-1, 1:-1], self.cof * layout[..., 1:-1, 1:-1])
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length) + 1
                    layout_pred[..., idx_start:idx_end, :1] = self.cof * layout[..., idx_start:idx_end, :1]
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    layout_pred[..., idx_start:idx_end, -1:] = self.cof * layout[..., idx_start:idx_end, -1:]
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    layout_pred[..., :1, idx_start:idx_end] = self.cof * layout[..., :1, idx_start:idx_end]
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    layout_pred[..., -1:, idx_start:idx_end] = self.cof * layout[..., -1:, idx_start:idx_end]
                else:
                    raise ValueError("bc error!")
        return self.base_loss(layout_pred, self.cof * layout)

'''
class NSE_layer(torch.nn.Module):
    def __init__(
            self, nx=21, length=0.1, nu = 5*1e-2, bcs=None
    ):
        super(NSE_layer, self).__init__()
        self.length = length
        self.nu = nu
        self.bcs = bcs
        # The weight 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.laplace_weight = torch.Tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]])
        self.dx_weight = torch.Tensor([[[[0,0,0],[-0.5,0,0.5],[0,0,0]]]])
        self.dy_weight = torch.Tensor([[[[0,-0.5,0],[0,0,0],[0,0.5,0]]]])
        # Padding
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        self.STRIDE = self.length / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        #self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def Dx(self,x):
        return conv2d(x, self.dx_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def Dy(self,x):
        return conv2d(x, self.dy_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def laplace(self, x):
        return conv2d(x, self.laplace_weight.to(device=x.device), bias=None, stride=1, padding=0)
    def momentum_u(self,x):
        u = x[...,0,:,:]
        v = x[...,1,:,:]
        p = x[...,2,:,:]
        return self.u*self.Dx(u) + self.v*self.Dy(u) + self.Dx(p) - self.nu*(self.laplace(u))/self.STRIDE
    def momentum_v(self,x):
        u = x[...,0,:,:]
        v = x[...,1,:,:]
        p = x[...,2,:,:]
        return self.u*self.Dx(v) + self.v*self.Dy(v) + self.Dy(p) - self.nu*(self.laplace(v))/self.STRIDE
    def continuity(self,x):
        u = x[...,0,:,:]
        v = x[...,1,:,:]
        return self.Dx(u) + self.Dy(v)
    def NSE(self,x):
        return self.momentum_u(x) + self.momentum_v(x) + self.continuity(x)

    def forward(self, layout, flow):
        # Source item
        f = 0#self.cof * layout
        # The nodes which are not in boundary
        G = torch.ones_like(flow).detach()
        G_nonslip = torch.zeros_like(flow).detach()
        G_inout = torch.zeros_like(flow).detach()
        G_bc = torch.ones_like(flow).detach()
        '''
        0: interior
        1: non-slip
        2: inlet
        3: outlet
        '''
        # dirichlet bc 0
        # layout에서 1인 위치를 찾습니다.
        indices = (layout == 9).nonzero(as_tuple=True)
        # 인덱스를 사용하여 G의 해당 위치의 값을 변경합니다.
        G[..., :, indices[0], indices[1]] = 0
        # non-slip values
        indices = (layout == 1).nonzero(as_tuple=True)
        G_inout[..., 0, indices[0], indices[1]] = 0 # u inlet
        G_inout[..., 1, indices[0], indices[1]] = 0 # v inlet
        G_bc[..., 0, indices[0], indices[1]] = 0 # u inlet
        G_bc[..., 1, indices[0], indices[1]] = 0 # v inlet
        
        # inlet and outlet values
        indices = (layout == 2).nonzero(as_tuple=True)
        G_inout[..., 0, indices[0], indices[1]] = 0.1 # u inlet
        G_inout[..., 1, indices[0], indices[1]] = 0 # v inlet
        G_bc[..., 0, indices[0], indices[1]] = 0 # u inlet
        G_bc[..., 1, indices[0], indices[1]] = 0 # v inlet

        indices = (layout == 3).nonzero(as_tuple=True)
        G_inout[..., 2, indices[0], indices[1]] = 0 # p outlet
        G_bc[..., 2, indices[0], indices[1]] = 0 # p outlet
        
        x = F.pad(flow * G * G_bc + G_nonslip + G_inout, [1, 1, 1, 1], mode='reflect')
                
        self.u = x[...,0,1:(self.nx + 1),1:(self.nx + 1)]
        self.v = x[...,1,1:(self.nx + 1),1:(self.nx + 1)]
        self.p = x[...,2,1:(self.nx + 1),1:(self.nx + 1)]
        x = G * G_bc * (self.NSE(x) + f)
        return x

'''
class Jacobi_layerSoft(torch.nn.Module):
    def __init__(
            self, nx=21, length=0.1, bcs=None
    ):
        super(Jacobi_layerSoft, self).__init__()
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
        # G: the nodes which are not in boundary
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
                x = F.pad(heat, [1, 1, 1, 1], mode='reflect')
            else:
                x = F.pad(x, [1, 1, 1, 1], mode='reflect')
            x = G * (self.jacobi(x) + f)
        return x
'''

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
