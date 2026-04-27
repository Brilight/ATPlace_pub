import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
np.set_printoptions(suppress=True,precision=4)


class TModel(nn.Module):
    def __init__(self, L, W, num_chiplets, num_grid_x, num_grid_y):
        super(TModel, self).__init__()
        self.L, self.W = L, W
        self.num_chiplets = num_chiplets
        
        self.num_grid_x, self.num_grid_y = num_grid_x, num_grid_y
        xgrid = (torch.arange(num_grid_x)+0.5)/num_grid_x*self.L
        ygrid = (torch.arange(num_grid_y)+0.5)/num_grid_y*self.W
        self.xgrid, self.ygrid = torch.meshgrid(xgrid, ygrid, indexing='ij')
        self.amp = nn.Parameter(torch.ones(1,1,1,1)*1e3)
        self.bias = nn.Parameter(torch.rand(1,1,1,1))
        self.heff = nn.Parameter(torch.ones(1,1,1,1))
        self.decay = nn.Parameter(torch.ones(1,num_chiplets,1,2))

    def forward(self, input_data):
        num_chiplets = self.num_chiplets
        if len(input_data) == 5:
            x, y, length, width, power = input_data
        else:
            x, y, length, width, power, _masks = input_data
        X = self.xgrid[None,None,...].repeat(1, num_chiplets, 1, 1)
        Y = self.ygrid[None,None,...].repeat(1, num_chiplets, 1, 1)
        xc, yc = x.reshape(-1,num_chiplets,1,1), y.reshape(-1,num_chiplets,1,1)
        lc, wc = length.reshape(-1,num_chiplets,1,1), width.reshape(-1,num_chiplets,1,1)
        
        decay = self.decay
        
        def calc_main(a, xdis, ydis):
            val =  (self.Fabc(a, decay[...,:1]*(lc/2-xdis), decay[...,1:2]*(wc/2-ydis))+\
                    self.Fabc(a, decay[...,:1]*(lc/2-xdis), decay[...,1:2]*(wc/2+ydis))+\
                    self.Fabc(a, decay[...,:1]*(lc/2+xdis), decay[...,1:2]*(wc/2-ydis))+\
                    self.Fabc(a, decay[...,:1]*(lc/2+xdis), decay[...,1:2]*(wc/2+ydis)))
            return val/(lc)/(wc)
        
        power = power.reshape(-1,num_chiplets,1,1)
        val = calc_main(self.heff, X-xc, Y-yc)
        outputs = power*(self.amp*val+self.bias)
        outputs = outputs.sum(dim=1, keepdim=True).squeeze(1)
        return outputs

    def Fabc(self, a, b, c):
        a = a.double()
        b = b.double()
        c = c.double()
        delta = torch.sqrt(a**2+b**2+c**2)
        val = b*torch.log((c+delta)/(a**2+b**2)**0.5)+\
                c*torch.log((b+delta)/(a**2+c**2)**0.5)-a*torch.arctan(b*c/a/delta)
        return val.float()

class TModel_leak(nn.Module):
    def __init__(self, L, W, num_chiplets, num_grid_x, num_grid_y):
        super(TModel_leak, self).__init__()
        self.L, self.W = L, W
        self.num_chiplets = num_chiplets
        self.num_grid_x, self.num_grid_y = num_grid_x, num_grid_y
        
        # Create grid
        xgrid = (torch.arange(num_grid_x)+0.5)/num_grid_x*self.L
        ygrid = (torch.arange(num_grid_y)+0.5)/num_grid_y*self.W
        self.xgrid, self.ygrid = torch.meshgrid(xgrid, ygrid, indexing='ij')
        
        # Neural network to predict parameters
        self.param_net = nn.Sequential(
            nn.Linear(5, 32),  # input: x,y,width,height,power (5 features)
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # output: amp, bias, heff, decay (per chiplet)
        )
        
        # Initialize weights
        for layer in self.param_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.weight)

    def forward(self, input_data):
        if len(input_data) == 5:
            x, y, length, width, power = input_data
        else:
            x, y, length, width, power, _masks = input_data
        batch_size = x.shape[0]
        num_chiplets = self.num_chiplets
        
        # Prepare input features for parameter network
        # Shape: (batch_size, num_chiplets, 5)
        features = torch.stack([
            x, y, length, width, power.squeeze(-1)
        ], dim=-1)
        
        # Get parameters from neural network
        params = self.param_net(features.view(-1, 5)).view(batch_size, num_chiplets, -1)
        amp = params[..., 0].unsqueeze(-1).unsqueeze(-1)  # shape: (batch, num_chiplets, 1, 1)
        bias = params[..., 1].unsqueeze(-1).unsqueeze(-1)
        heff = params[..., 2].unsqueeze(-1).unsqueeze(-1)
        #decay = params[..., 3:].view(batch_size, num_chiplets, 1, 2)
        
        # Prepare grid coordinates
        X = self.xgrid[None,None,...].repeat(batch_size, num_chiplets, 1, 1)
        Y = self.ygrid[None,None,...].repeat(batch_size, num_chiplets, 1, 1)
        xc, yc = x.reshape(batch_size, num_chiplets, 1, 1), y.reshape(batch_size, num_chiplets, 1, 1)
        lc, wc = length.reshape(batch_size, num_chiplets, 1, 1), width.reshape(batch_size, num_chiplets, 1, 1)
        
        def calc_main(a, xdis, ydis):
            val =  (self.Fabc(a, (lc/2-xdis), (wc/2-ydis)) + \
                    self.Fabc(a, (lc/2-xdis), (wc/2+ydis)) + \
                    self.Fabc(a, (lc/2+xdis), (wc/2-ydis)) + \
                    self.Fabc(a, (lc/2+xdis), (wc/2+ydis)))
            return val/(lc)/(wc)
        
        power = power.reshape(batch_size, num_chiplets, 1, 1)
        val = calc_main(heff, X-xc, Y-yc)
        outputs = power*(amp*val + bias)
        outputs = outputs.sum(dim=1, keepdim=True).squeeze(1)
        return outputs

    def Fabc(self, a, b, c):
        a = a.double()
        b = b.double()
        c = c.double()
        delta = torch.sqrt(a**2 + b**2 + c**2)
        val = b*torch.log((c+delta)/(a**2+b**2)**0.5) + \
              c*torch.log((b+delta)/(a**2+c**2)**0.5) - \
              a*torch.arctan(b*c/a/delta)
        return val.float()

class TModel_fourier(nn.Module):
    def __init__(self, L, W, num_chiplets, num_grid_x, num_grid_y):
        super(TModel, self).__init__()
        self.L, self.W = L, W
        self.num_chiplets = num_chiplets

        xgrid = (torch.arange(num_grid_x) + 0.5) / num_grid_x * self.L
        ygrid = (torch.arange(num_grid_y) + 0.5) / num_grid_y * self.W
        self.xgrid, self.ygrid = torch.meshgrid(xgrid, ygrid, indexing='ij')

        self.amp = nn.Parameter(torch.ones(1, 1, 1, 1) * 1e3)
        self.bias = nn.Parameter(torch.rand(1, 1, 1, 1))
        self.heff = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.decay = nn.Parameter(torch.ones(1, num_chiplets, 1, 2))
        self.beta = torch.ones(1, num_chiplets, 1, 1) * 0.1

        self.register_buffer('fourier_kernel', self.get_fourier_kernel())

    def get_fourier_kernel(self):
        # 构造一个标准芯片的 W(r) 的频域表示
        xgrid = (torch.arange(self.num_grid_x) + 0.5) / self.num_grid_x * self.L
        ygrid = (torch.arange(self.num_grid_y) + 0.5) / self.num_grid_y * self.W
        X, Y = torch.meshgrid(xgrid, ygrid, indexing='ij')
        W_r = self.Fabc(self.heff, X, Y).squeeze()
        W_k = torch.fft.rfft2(W_r, norm='ortho')
        self.fourier_kernel = W_k[None, None, ...]  # (1, 1, H, W)
        
        
    def forward(self, input_data):
        num_chiplets = self.num_chiplets
        x, y, length, width, power, masks = input_data

        power = power.reshape(-1, num_chiplets, 1, 1)
        W_k = self.fourier_kernel

        # 扩展为 (B, num_chiplets, H, W)
        W_k_expanded = W_k.expand(power.shape[0], num_chiplets, -1, -1)

        # 频域格林函数 (1, 1, H, W)
        kx = torch.fft.fftfreq(self.num_grid_x, d=self.L / self.num_grid_x)
        ky = torch.fft.rfftfreq(self.num_grid_y, d=self.W / self.num_grid_y)
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
        K2 = Kx.pow(2) + Ky.pow(2)
        Phi_k = 4 * torch.pi / (K2 + 1e-8)
        Phi_k = Phi_k[None, None, ...].to(W_k.device)

        response_k = power * (
            self.amp * W_k_expanded / (1 - self.beta * Phi_k) + self.bias
        )
        total_response_k = response_k.sum(dim=1, keepdim=True)
        total_response = torch.fft.irfft2(
            total_response_k.squeeze(1), 
            s=(self.num_grid_x, self.num_grid_y), norm='ortho'
        )

        return total_response
    

class WModel(nn.Module):
    def __init__(self, L, W, num_chiplets, num_grid_x, num_grid_y):
        super().__init__()
        self.L, self.W = L, W
        self.num_chiplets = num_chiplets

        # Grid setup
        xgrid = (torch.arange(num_grid_x) + 0.5) / num_grid_x * self.L
        ygrid = (torch.arange(num_grid_y) + 0.5) / num_grid_y * self.W
        self.register_buffer('xgrid', xgrid)
        self.register_buffer('ygrid', ygrid)

        # Learnable parameters: per-chiplet amp and v3
        self.v3 = nn.Parameter(torch.ones(1, num_chiplets, 1, 1))
        self.amp = nn.Parameter(torch.ones(1, num_chiplets, 1, 1) * 1e-2)

    @staticmethod
    def _dir_l1(dx, dy, a, b, eps=1e-9):
        """
        Directional l1: distance from center to rectangle boundary along (dx,dy).
        dx, dy: [B, C, H, W]; a, b: [B, C, 1, 1]
        """
        r = torch.sqrt(dx*dx + dy*dy + eps)
        vx = dx / r
        vy = dy / r
        big = torch.tensor(1e30, device=dx.device, dtype=dx.dtype)
        tx = torch.where(vx.abs() > eps, a / vx.abs(), big)  # dist to left/right
        ty = torch.where(vy.abs() > eps, b / vy.abs(), big)  # dist to top/bottom
        l1_dir = torch.min(tx, ty)
        l1_iso = torch.min(a, b)  # fallback when r ~ 0
        l1_dir = torch.where(r < 1e-6, l1_iso, l1_dir)
        return l1_dir, r

    def forward(self, input_data):
        x, y, length, width, _, Temp = input_data
        C = self.num_chiplets

        # Build grid
        X = self.xgrid[None, None, :, None].expand(1, C, -1, self.ygrid.numel())
        Y = self.ygrid[None, None, None, :].expand(1, C, self.xgrid.numel(), -1)

        # Reshape chiplet positions and sizes
        xc, yc = x.reshape(-1, C, 1, 1), y.reshape(-1, C, 1, 1)
        lc, wc = length.reshape(-1, C, 1, 1), width.reshape(-1, C, 1, 1)

        # Compute directional l1 and radial distance
        l1, r = self._dir_l1(X - xc, Y - yc, lc * 0.5, wc * 0.5)

        # Near field: quadratic
        def Dz1(rr):
            return 0.5 * rr ** 2

        # Gradient for linear extrapolation
        def Dz1p(rr):
            return rr

        # Far field: linear + concave correction
        Dz2 = Dz1(l1) + Dz1p(l1) * (r - l1) + 0.5 * (-self.v3) * (r - l1) ** 2

        # Combine near/far field
        field = torch.where(r <= l1, Dz1(r), Dz2)

        # Apply per-chiplet temperature scaling
        if isinstance(Temp, float):
            # Scalar: scale output after sum
            out = (self.amp * field).sum(dim=1, keepdim=True).squeeze(1)
            out = out * Temp
        else:
            # Tensor Temp: shape (1, C, 1, 1), apply before sum
            assert Temp.shape == (1, C, 1, 1), f"Temp shape {Temp.shape} != (1, {C}, 1, 1)"
            field = field * Temp  # scale each chiplet's field
            out = (self.amp * field).sum(dim=1, keepdim=True).squeeze(1)

        return out
