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

class Energy_Evp_layer(torch.nn.Module):
    def __init__(
            self,base_loss=MSELoss(reduction='mean'), 
            nx=256, ny=128, length_x=6, length_y=3):
        super(Energy_Evp_layer, self).__init__()
        self.length_x = length_x
        self.length_y = length_y
        self.nx = nx
        self.ny = ny
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        self.h = self.length_x / self.nx
        self.cof = TEMPER_COEFFICIENT
        self.base_loss = base_loss
        self.diff_coeff = 0#2.2 * 1e-5

        # evp
        # parameters

        TMP0 = 296.0 # initial temperature for slurry [K]
        WC0 = 651.830 # initial water contents in slurry [kg/m3]
        TH0 = 202.412 # inital thickness of coating layer [um]
        THcell = 2.0 
        self.C1 = 0.65248278 # volume fraction of solvent
        self.C2 = 0.00200 # (particle dia./ini. coating thickness)^2
        self.C4 = 400

        self.n = 0.5
        Ce_e = 0.1
        Ce_a = 0.9
        T_jet = 130
        T_boiling = 100
        self.CT = 1

        #Set_User_Memory_Name(0,"Water_Contents[kg]")
        #Set_User_Memory_Name(1, "Water_Contents[kg/m3]")
        #Set_User_Memory_Name(2, "Evaporation_Rate[kg/sec]")
        #Set_User_Memory_Name(3, "Evaporation_Rate[kg/m3/sec]")
        #Set_User_Memory_Name(4, "Wall_Heat_Flux")
        #Set_User_Memory_Name(5, "h_monitoring")

        self.wc0 = WC0*TH0*1e-3/THcell
        self.flux = 300
        

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
        return self.heat_ini - 0.001*(self.u*self.Dx(x)+self.v*self.Dy(x) - self.diff_coeff*(self.laplace(x))/self.h)
    def wc(self,x):
        return  self.wc0 - 0.001*self.m_dot
    def set_bc(self, bc_num, outvar):
        var_num = {'tmp':0}
        indices = (self.boundary == bc_num).nonzero(as_tuple=True)
        for item in outvar:
            self.bc_mask[...,var_num[item], indices[0], indices[1]] = 0
            self.bc_value[..., var_num[item], indices[0], indices[1]] = outvar[item]
        return None
        
    def evp(self,tmp,wc):
        u_temperature_f = tmp + 273.15 #T_1*273.15
        u_temperature_w = tmp + 273.15 #T_2s*273.15 #u_temperature_f
        #h = #T+273.15
        u_fraction = 0.00725 #E
        self.fluid_density = 1.1614 #kg/m3
        
        self.cp_a=(28.11+1.967*(1e-3)*u_temperature_f+4.802*(1e-6)*(u_temperature_f**2)-1.966*(1e-9)*(u_temperature_f**3))/28.970*1000
        # cp_a = 주변 공기의 비열 [J/KgK]
        cp_v=(32.24+1.923*(1e-3)*u_temperature_w+10.55*(1e-6)*(u_temperature_w**2)-3.595*(1e-9)*(u_temperature_w**3))/18.015*1000
        # cp_v = 계면 수증기의 비열 [J/KgK]
        P_v = torch.exp(23.2-3816.4/(u_temperature_w-46.1))
        # if ((u_temperature_w > 273)&&(u_temperature_w<473))
        u_pressure = 101325
        x_v = 0.62198*P_v/(u_pressure-P_v)
        # 표면에서는 순수 용매가 포화상태로 있다고 가정시 계면 수증기의 절대습도 x_v[kg/kg]
        x_a = u_fraction #주변 공기의 절대습도 [kg/kg]
        C3 = torch.pow((torch.mean(wc)/self.wc0),self.n) # solvent remaining coefficient
        porous_correction = self.C1*self.C2*C3*self.C4*self.CT
        
        #m_dot = heat_flux/(2317.0*(10**3))*porous_correction #the heat of vaporization = 2317 [kj/kg] at 350K
        self.m_dot = self.flux/(self.cp_a+cp_v*x_a)*(x_v-x_a)*porous_correction#/(1e-3*THcell) #kg/m3/sec
        #evp_water = self.m_dot/self.fluid_density #(1e-3*THcell) #공기의 무게 1m3당 1.2kg, plate area 0.46
               
        print(torch.mean(wc))
        return self.m_dot


    def evp_src(self,m_dot):
        
        # source term for energy equation
        hfg = 2257.0 #KJ/Kg
        src_h = -1*m_dot*hfg#*1000.0
        src_temp = src_h / (1e-3*self.cp_a) / self.fluid_density #/(le-3*THcell)
        #nd_src_temp = src_temp - 273.15
        
        return 0#src_temp
    

    def apply_Energy(self, wc, x, eq_num):
        mask = torch.zeros_like(self.boundary)
        indices = (self.boundary == eq_num).nonzero(as_tuple=True)
        mask[...,indices[0],indices[1]] = 1
        self.eq_mask[...,indices[0],indices[1]] = eq_num

        ## 4: FD x, 5: BD y, 6: BD x, 7: FD y, 8: 4+5, 9: 5+6, 10: 6+7, 11: 4+7 

        eq_energy = {0:self.advection_diffusion(x), 2:self.advection_diffusion(x),
        3: self.advection_diffusion(x),
        4: self.advection_diffusion(x)+self.flux*self.Dx(x)+self.wc(wc), 
        5: self.advection_diffusion(x)+self.flux*-1*self.Dy(x)+self.wc(wc),
        6: self.advection_diffusion(x)+self.flux*-1*self.Dx(x)+self.wc(wc), 
        7: self.advection_diffusion(x)+self.flux*self.Dy(x)+self.wc(wc), 
        8: self.advection_diffusion(x)+self.flux*(-1*self.Dx(x)+self.Dy(x))+self.wc(wc), 
        9: self.advection_diffusion(x)+self.flux*(self.Dy(x)+self.Dx(x))+self.wc(wc), 
        10: self.advection_diffusion(x)+self.flux*(self.Dx(x)-self.Dy(x))+self.wc(wc), 
        11: self.advection_diffusion(x)+self.flux*(-1*self.Dx(x)-self.Dy(x))+self.wc(wc),}

        return mask * eq_energy[eq_num]

    def forward(self, layout, heat_ini, wc, heat, flow):
        self.heat_ini = heat_ini
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

        wc_value = torch.zeros_like(wc).detach()
        wc_mask = torch.zeros_like(wc).detach()    
        wc_indices = (self.boundary > 3 ).nonzero(as_tuple=True)
        wc_mask[...,0, wc_indices[0], wc_indices[1]] = 1
        #wc_value[..., 0, wc_indices[0], wc_indices[1]] = self.wc0
        wc_bc = wc * wc_mask# + wc_value
        #wc_avg = torch.mean(wc_bc[wc_indices])
        #print(wc_avg)
        # Dirichlet boundary
        self.bc_value = torch.zeros_like(heat).detach()
        self.bc_mask = torch.ones_like(heat).detach()
        
        self.set_bc(bc_num=1, outvar={'tmp':0}) # inlet
        
        heat_bc = heat * self.bc_mask + self.bc_value

        x = F.pad(heat_bc, [1, 1, 1, 1], mode='reflect')  # constant, reflect, reflect
        
        # Source item
        self.src = 300 * self.h + self.evp_src(self.evp(heat_bc,wc_bc)) #* self.h * self.h
        self.flux = 300 * self.h + self.evp_src(self.evp(heat_bc,wc_bc)) #-3000
        
        f = self.cof * abs(self.geom-1) * self.src * self.h
        
        # physics
        self.eq_mask = torch.zeros_like(self.boundary)
        loss_eq = torch.zeros_like(heat)

        for eq_num in [0,2,3,4,5,6,7,8,9,10,11]:
            loss_eq += self.apply_Energy(wc_bc, x,eq_num)
        
        energy_loss = self.base_loss(loss_eq, f)
        wc_loss = self.base_loss((self.wc0*wc_mask-self.m_dot) , wc_bc)

        energy_loss += wc_loss
        
        return energy_loss, heat_bc, self.eq_mask



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
