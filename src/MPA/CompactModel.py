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
        xgrid, ygrid = torch.meshgrid(xgrid, ygrid, indexing='ij')
        self.register_buffer("xgrid", xgrid[None,None])
        self.register_buffer("ygrid", ygrid[None,None])
        
        self.amp = nn.Parameter(torch.ones(1)*1e3)
        self.bias = nn.Parameter(torch.zeros(1))
        self.heff = nn.Parameter(torch.ones(1))
        self.decay = nn.Parameter(torch.ones(1,num_chiplets,1,2))

    def forward(self, input_data):
        num_chiplets = self.num_chiplets
        x, y, length, width, power = input_data
        B, C = x.shape[0], self.num_chiplets
        xc, yc = x.view(-1,C,1,1), y.view(-1,C,1,1)
        lc, wc = length.view(-1,C,1,1), width.view(-1,C,1,1)
        X = self.xgrid.expand(B, C, -1, -1)
        Y = self.ygrid.expand(B, C, -1, -1)
        
        def calc_main(a, xdis, ydis):
            decay = self.decay
            val =  (self.Fabc(a, decay[...,:1]*(lc/2-xdis), decay[...,1:2]*(wc/2-ydis))+\
                    self.Fabc(a, decay[...,:1]*(lc/2-xdis), decay[...,1:2]*(wc/2+ydis))+\
                    self.Fabc(a, decay[...,:1]*(lc/2+xdis), decay[...,1:2]*(wc/2-ydis))+\
                    self.Fabc(a, decay[...,:1]*(lc/2+xdis), decay[...,1:2]*(wc/2+ydis)))
            return val/(lc)/(wc)
        
        power = power.reshape(-1,num_chiplets,1,1)
        val = calc_main(self.heff, X-xc, Y-yc)
        outputs = power*(self.amp*val+self.bias)
        outputs = outputs.sum(dim=1, keepdim=True)
        return outputs

    def Fabc(self, a, b, c):
        a = a.double()
        b = b.double()
        c = c.double()
        delta = torch.sqrt(a**2+b**2+c**2)
        val = b*torch.log((c+delta)/(a**2+b**2)**0.5)+\
                c*torch.log((b+delta)/(a**2+c**2)**0.5)-a*torch.arctan(b*c/a/delta)
        return val.float()
    
    
class TModel_float(nn.Module):
    def __init__(self, L, W, num_chiplets, gx, gy):
        super().__init__()
        self.L, self.W = L, W
        self.num_chiplets = num_chiplets

        # Precompute grid as buffers
        xg = (torch.arange(gx)+0.5)/gx * self.L
        yg = (torch.arange(gy)+0.5)/gy * self.W
        xg, yg = torch.meshgrid(xg, yg, indexing='ij')
        self.register_buffer("xgrid", xg[None,None])
        self.register_buffer("ygrid", yg[None,None])

        # Model parameters
        self.amp  = nn.Parameter(torch.tensor([1e3], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.heff = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.decay = nn.Parameter(torch.ones(1, num_chiplets, 1, 2))

    @staticmethod
    def Fabc_merged_old(a, b, c):
        a = a.float() #.double()
        b = b.float() #.double()
        c = c.float() #.double()
        a2 = a*a
        b2 = b*b
        c2 = c*c
        delta = torch.sqrt(a2+b2+c2)
        val = (
            b*torch.log((c+delta)/torch.sqrt(a2+b2))+\
            c*torch.log((b+delta)/torch.sqrt(a2+c2))-\
            a*torch.arctan(b*c/(a*delta))
        )
        return val.float()

    @staticmethod
    def Fabc_merged(a, b, c):
        """
        a: (1,)
        b,c: (...,4)
        全程 float32，利用 F(ka,kb,kc)=k*F(a,b,c) 做尺度归一化
        """
        # keep everything in float32
        a = a.float()
        b = b.float()
        c = c.float()

        # per-element scale factor s ≈ max(|a|, |b|, |c|)
        # shape will broadcast to (...,4)
        s = torch.maximum(torch.maximum(a.abs(), b.abs()), c.abs())
        s = torch.clamp(s, min=eps)  # avoid s=0

        # normalize to order ~1
        a1 = a / s
        b1 = b / s
        c1 = c / s

        a2 = a1 * a1
        b2 = b1 * b1
        c2 = c1 * c1

        delta = torch.sqrt(a2 + b2 + c2)

        # 原始公式在 (a1,b1,c1) 上计算（保持结构不变）
        num1 = c1 + delta
        den1 = torch.sqrt(a2 + b2 + eps)
        num2 = b1 + delta
        den2 = torch.sqrt(a2 + c2 + eps)

        term1 = b1 * torch.log(num1 / den1)
        term2 = c1 * torch.log(num2 / den2)
        term3 = a1 * torch.atan(b1 * c1 / (a1 * delta + eps))

        val1 = term1 + term2 - term3  # 这是 F(a1,b1,c1)

        # 利用齐次性：F(a,b,c) = s * F(a1,b1,c1)
        return val1 * s


    def forward(self, data):
        x, y, length, width, power = data
        B, C = x.shape[0], self.num_chiplets

        X = self.xgrid.expand(B, C, -1, -1)
        Y = self.ygrid.expand(B, C, -1, -1)
        xc, yc = x.view(B,C,1,1), y.view(B,C,1,1)
        lc, wc = length.view(B,C,1,1), width.view(B,C,1,1)
        xdis, ydis = X - xc, Y - yc

        decay_x = self.decay[...,0:1]
        decay_y = self.decay[...,1:2]
        lc2, wc2 = lc*0.5, wc*0.5

        # Build 4 corner (b,c) pairs into one tensor
        # shape → (B,C,H,W,4)
        b = torch.stack([
            decay_x*(lc2-xdis),
            decay_x*(lc2-xdis),
            decay_x*(lc2+xdis),
            decay_x*(lc2+xdis),
        ], dim=-1)

        c = torch.stack([
            decay_y*(wc2-ydis),
            decay_y*(wc2+ydis),
            decay_y*(wc2-ydis),
            decay_y*(wc2+ydis),
        ], dim=-1)

        # merged Fabc
        a = self.heff  # scalar
        val4 = self.Fabc_merged(a, b, c)   # (B,C,H,W,4)
        val = val4.sum(dim=-1) / (lc * wc) # reduce 4 corners

        power = power.view(B,C,1,1)
        out = power * (self.amp * val + self.bias)
        return out.sum(dim=1, keepdim=True)

    
class WModel(nn.Module):
    def __init__(self, L, W, num_chiplets, num_grid_x, num_grid_y):
        super().__init__()
        self.L, self.W = L/1e6, W/1e6
        self.num_chiplets = num_chiplets
        xgrid = (torch.arange(num_grid_x) + 0.5) / num_grid_x * self.L
        ygrid = (torch.arange(num_grid_y) + 0.5) / num_grid_y * self.W
        self.xgrid, self.ygrid = torch.meshgrid(xgrid, ygrid, indexing='ij')

        self.amp = nn.Parameter(torch.ones(1)*1e-1)
        self.k = nn.Parameter(torch.ones(1, num_chiplets+1, 1, 1, 2) * 1e-3)
        self.linear = nn.Parameter(torch.ones(1, num_chiplets+1, 1, 1))
        self.ref_temp = nn.Parameter(torch.zeros(1, num_chiplets+1, 1, 1))
        self.const = nn.Parameter(torch.zeros(1, num_chiplets+1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, input_data):
        C = self.num_chiplets
        x, y, length, width, Temp = input_data
        X = self.xgrid[None, None, ...].repeat(1, C+1, 1, 1)
        Y = self.ygrid[None, None, ...].repeat(1, C+1, 1, 1)

        def _to_BC11(t, size):
            t = t.reshape(-1, C, 1, 1) / 1e6
            return torch.cat((
                t, torch.full((t.shape[0], 1, 1, 1), size, dtype=t.dtype, device=t.device)), 
                dim=1)

        xc, yc = _to_BC11(x, self.L / 2), _to_BC11(y, self.W / 2)
        lc, wc = _to_BC11(length, self.L), _to_BC11(width, self.W)

        X_dis, Y_dis = self.k[..., 0] * (X - xc), self.k[..., 1] * (Y - yc)
        field = (X_dis**2 + Y_dis**2) + self.linear * (X_dis + Y_dis) + self.const
        out = self.amp * ((Temp+self.ref_temp)*field).sum(dim=1, keepdim=True) + self.bias
        return out
    
    
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
        x, y, length, width, power, masks = input_data
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
    

class WModel_old(nn.Module):
    def __init__(self, L, W, num_chiplets, num_grid_x, num_grid_y):
        super().__init__()
        self.L, self.W = L/1e3, W/1e3
        self.num_chiplets = num_chiplets

        xgrid = (torch.arange(num_grid_x) + 0.5) / num_grid_x * self.L
        ygrid = (torch.arange(num_grid_y) + 0.5) / num_grid_y * self.W
        self.xgrid, self.ygrid = torch.meshgrid(xgrid, ygrid, indexing='ij')

        self.v3 = nn.Parameter(torch.rand(1, num_chiplets+1, 1, 1))
        self.amb_temp = nn.Parameter(torch.zeros(1, num_chiplets+1, 1, 1))
        self.amp = nn.Parameter(torch.ones(1, num_chiplets, 1, 1) * 1e-4)
        self.bias = nn.Parameter(torch.zeros(1, num_chiplets, 1, 1))
        self.coeff = nn.Parameter(torch.ones(1, num_chiplets+1, 1, 1, 8))
        self.k = nn.Parameter(torch.rand(1, num_chiplets+1, 1, 1) * 1e-6)

    def forward(self, input_data):
        C = self.num_chiplets
        x, y, length, width, Temp = input_data

        B = x.shape[0]
        X = self.xgrid[None, None, ...].repeat(B, C+1, 1, 1)
        Y = self.ygrid[None, None, ...].repeat(B, C+1, 1, 1)

        def _to_BC11(t, size):
            t = torch.cat((
                t.reshape(B, -1, 1, 1) / 1e3,
                torch.full((B, 1, 1, 1), size, dtype=t.dtype, device=t.device)),
                dim=1)
            return t

        xc, yc = _to_BC11(x, self.L / 2), _to_BC11(y, self.W / 2)
        lc, wc = _to_BC11(length, self.L), _to_BC11(width, self.W)
        r = torch.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
        _, _, inside = self._nearest_boundary_point(X, Y, xc, yc, lc * 0.5, wc * 0.5)

        sum_temp = (Temp * inside).sum(dim=[2, 3])
        Temp = (sum_temp / inside.sum(dim=[2, 3])).unsqueeze(-1).unsqueeze(-1)
        print(Temp.squeeze())
        X_dis, Y_dis = (X - xc), (Y - yc)
        field_bg = (Temp - self.amb_temp) * (
            self.coeff[..., 0] * X_dis**2 + self.coeff[..., 1] * Y_dis**2 +
            self.coeff[..., 2] * X_dis + self.coeff[..., 3] * Y_dis + 
            #self.coeff[..., 4] * torch.cosh(self.k*(X_dis)) +
            #self.coeff[..., 5] * torch.cosh(self.k*(Y_dis)) + 
            self.coeff[..., 6]
        ) + self.coeff[..., 7]

        out = field_bg[:,:-1,...].sum(dim=1, keepdim=True)
        return out
    
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

    def forward_v1(self, input_data):
        x, y, length, width, Temp = input_data
        C = self.num_chiplets
        X = self.xgrid[None,None,...].repeat(1, C+1, 1, 1)
        Y = self.ygrid[None,None,...].repeat(1, C+1, 1, 1)
        
        # Reshape chiplet positions and sizes
        xc, yc = x.reshape(-1, C, 1, 1), y.reshape(-1, C, 1, 1)
        lc, wc = length.reshape(-1, C, 1, 1), width.reshape(-1, C, 1, 1)
        l1, r = self._dir_l1(X - xc, Y - yc, lc * 0.5, wc * 0.5)

        # Near field: quadratic
        def Dz1(rr):
            return (
                0.5 * rr ** 2 - 0
                #(torch.cosh(self.k*rr)-1)/self.k**2/torch.cosh(self.k*(self.L+self.W)/2)
            )

        # Gradient for linear extrapolation
        def Dz1p(rr):
            return (
                rr - 0
            )

        # Far field: linear + concave correction
        Dz2 = Dz1(l1) + Dz1p(l1) * (r - l1) + 0.5 * (-self.v3) * (r - l1) ** 2
        field = torch.where(r <= l1, Dz1(r), Dz2)

        # Apply per-chiplet temperature scaling
        if isinstance(Temp, float):
            #Temp += 273.15 if Temp<300 else 0
            field = self.amp * field * Temp + self.bias
        elif len(Temp.shape)==4:
            #Temp += 273.15 if Temp.max()<300 else 0
            if Temp.shape[1]!=C:
                mask = (((X-xc).abs() <= lc / 2) & ((Y-yc).abs() <= wc / 2)).int()
                sum_temp = (Temp * mask).sum(dim=[2, 3])
                count = mask.sum(dim=[2, 3])
                Temp = (sum_temp / count).unsqueeze(-1).unsqueeze(-1)
            field = self.amp * field * (self.amb_temp-Temp) + self.bias
        else:
            field = self.amp * field + self.bias
            
        out = field.sum(dim=1, keepdim=True)
        return out
    
    def forward_v2(self, input_data):
        C = self.num_chiplets
        x, y, length, width, Temp = input_data
        X = self.xgrid[None,None,...].repeat(1, C+1, 1, 1)
        Y = self.ygrid[None,None,...].repeat(1, C+1, 1, 1)
        
        # Reshape chiplet positions and sizes
        xc = torch.cat((
            x.reshape(x.shape[0], -1, 1, 1)/1e3, 
            torch.full((x.shape[0], 1, 1, 1), self.L/2, dtype=x.dtype, device=x.device)), 
            dim=1)
        yc = torch.cat((
            y.reshape(y.shape[0], -1, 1, 1)/1e3, 
            torch.full((y.shape[0], 1, 1, 1), self.W/2, dtype=y.dtype, device=y.device)), 
            dim=1)
        lc = torch.cat((
            length.reshape(length.shape[0], -1, 1, 1)/1e3, 
            torch.full((length.shape[0], 1, 1, 1), self.L, dtype=length.dtype, device=length.device)), 
            dim=1)
        wc = torch.cat((
            width.reshape(width.shape[0], -1, 1, 1)/1e3, 
            torch.full((width.shape[0], 1, 1, 1), self.W, dtype=width.dtype, device=width.device)), 
            dim=1)
        l1, r = self._dir_l1(X - xc, Y - yc, lc * 0.5, wc * 0.5)

        # Near field: quadratic
        def Dz1(rr):
            return (
                0.5 * rr ** 2 - 
                (torch.cosh(self.k*rr)-1)/self.k**2/torch.cosh(self.k*(self.L+self.W)/4)
            )

        # Gradient for linear extrapolation
        def Dz1p(rr):
            return (
                rr - 0
                #torch.sinh(self.k*rr)/self.k/torch.cosh(self.k*(self.L+self.W)/4)
            )
        #print(((torch.cosh(self.k*r)-1)/self.k**2).max(), ((torch.sinh(self.k*r))/self.k).max(), (torch.cosh(self.k*(self.L+self.W)/4)), self.k, self.L, self.W)
        # Far field: linear + concave correction
        Dz2 = Dz1(l1) + Dz1p(l1) * (r - l1) + 0.5 * (-self.v3) * (r - l1) ** 2
        field = torch.where(r <= l1, Dz1(r), Dz2)

        # Apply per-chiplet temperature scaling
        if isinstance(Temp, float):
            #Temp += 273.15 if Temp<300 else 0
            field = self.amp * field * Temp + self.bias
        elif len(Temp.shape)==4:
            #Temp += 273.15 if Temp.max()<300 else 0
            if Temp.shape[1]!=C+1:
                mask = (((X-xc).abs() <= lc / 2) & ((Y-yc).abs() <= wc / 2)).int()
                sum_temp = (Temp * mask).sum(dim=[2, 3])
                count = mask.sum(dim=[2, 3])
                Temp = (sum_temp / count).unsqueeze(-1).unsqueeze(-1)
            field = self.amp * field  + self.bias
        else:
            field = self.amp * field + self.bias
            
        out = field.sum(dim=1, keepdim=True)
        return out
    
    @staticmethod
    def _nearest_boundary_point(X, Y, xc, yc, a, b):
        x_min, x_max = xc - a, xc + a
        y_min, y_max = yc - b, yc + b
        inside = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
        xi_out = X.clamp(min=x_min, max=x_max)
        yi_out = Y.clamp(min=y_min, max=y_max)
        return xi_out, yi_out, inside
    
    @staticmethod
    def _nearest_boundary_dir(X, Y, xc, yc, a, b, eps=1e-9):
        x_min, x_max = xc - a, xc + a
        y_min, y_max = yc - b, yc + b
        inside = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

        dx, dy = xc - X, yc - Y
        t_left = torch.where(dx.abs() > eps, (x_min - X) / dx, float('inf'))  # 距离左边界的交点
        t_right = torch.where(dx.abs() > eps, (x_max - X) / dx, float('inf'))  # 距离右边界的交点
        t_bottom = torch.where(dy.abs() > eps, (y_min - Y) / dy, float('inf'))  # 距离下边界的交点
        t_top = torch.where(dy.abs() > eps, (y_max - Y) / dy, float('inf'))  # 距离上边界的交点

        xi_left = X + t_left * dx  # 距离左边界的交点
        yi_left = Y + t_left * dy  # 距离左边界的交点

        xi_right = X + t_right * dx  # 距离右边界的交点
        yi_right = Y + t_right * dy  # 距离右边界的交点

        xi_bottom = X + t_bottom * dx  # 距离下边界的交点
        yi_bottom = Y + t_bottom * dy  # 距离下边界的交点

        xi_top = X + t_top * dx  # 距离上边界的交点
        yi_top = Y + t_top * dy  # 距离上边界的交点

        dist_to_points = torch.stack([
            (xi_left - X) ** 2 + (yi_left - Y) ** 2,
            (xi_right - X) ** 2 + (yi_right - Y) ** 2,
            (xi_bottom - X) ** 2 + (yi_bottom - Y) ** 2,
            (xi_top - X) ** 2 + (yi_top - Y) ** 2
        ], dim=1)  # [B, 4, H, W], 每个点到 4 个交点的距离

        # 找到最小的距离并返回对应的交点
        min_idx = torch.argmin(dist_to_points, dim=1)  # 获取最小距离的索引 [B, H, W]

        # 选择最小距离对应的交点
        xi = torch.gather(torch.stack([xi_left, xi_right, xi_bottom, xi_top], dim=1), 1, min_idx.unsqueeze(1))  # [B, H, W]
        yi = torch.gather(torch.stack([yi_left, yi_right, yi_bottom, yi_top], dim=1), 1, min_idx.unsqueeze(1))  # [B, H, W]

        return xi.squeeze(1), yi.squeeze(1), inside

    def forwardv3(self, input_data):

        C = self.num_chiplets
        x, y, length, width, Temp = input_data

        B = x.shape[0]
        X = self.xgrid[None, None, ...].repeat(B, C+1, 1, 1)
        Y = self.ygrid[None, None, ...].repeat(B, C+1, 1, 1)
        def _to_BC11(t, size):
            t = torch.cat((
                t.reshape(B, -1, 1, 1)/1e3, 
                torch.full((B, 1, 1, 1), size, dtype=t.dtype, device=t.device)), 
                dim=1)
            #t = torch.as_tensor(t.reshape(B, -1, 1, 1)/1e3, device=t.device)
            return t
        
        xc, yc = _to_BC11(x, self.L/2), _to_BC11(y, self.W/2)
        lc, wc = _to_BC11(length, self.L), _to_BC11(width, self.W)
        r  = torch.sqrt((X - xc)**2 + (Y - yc)**2)
        xi, yi, inside = self._nearest_boundary_point(X, Y, xc, yc, lc * 0.5, wc * 0.5)
        rb  = torch.sqrt((xi - xc)**2 + (yi - yc)**2)
        d_perp = torch.sqrt((X - xi)**2 + (Y - yi)**2)
        #print(xc.max(), X.max(), xi.max(), r.max(), rb.max(), d_perp.max())
        # Near field: quadratic
        def Dz1(rr):
            return (
                0.5 * rr ** 2 - 0
                #(torch.cosh(self.k*rr)-1)/self.k**2/torch.cosh(self.k*(self.L+self.W)/4)
            )

        # Gradient for linear extrapolation
        def Dz1p(rr):
            return (
                rr - 0
                #torch.sinh(self.k*rr)/self.k/torch.cosh(self.k*(self.L+self.W)/4)
            )
        #Dz_inside  = Dz1(r)
        #Dz_outside = Dz1(rb) + Dz1p(rb) * d_perp + 0.5 * (-self.v3) * (d_perp ** 2)
        #field = torch.where(inside, Dz_inside, Dz_outside)
        #field_chip = self.amp * (
        #    torch.where(inside, Dz_inside, 0)[:, :-1, ...]*Temp + self.bias)
        X_dis, Y_dis = (X-xc).abs(), (Y-yc).abs()
        #torch.where(inside, (X-xc).abs(), 0), torch.where(inside, (Y-yc).abs(), 0)
        field_bg = (#Temp.mean(dim=1, keepdim=True) * (
            self.coeff[...,0]*X_dis**2 + self.coeff[...,1]*Y_dis**2 + 
            self.coeff[...,2]*X_dis + self.coeff[...,3]*Y_dis + self.coeff[...,4] 
        )#+ self.coeff[...,5] 
        #field = torch.cat([field_chip, field_bg], dim=1)
        #print(field_chip.mean(dim=(-1,-2)), field_bg.mean(dim=(-1,-2)))
        #field   = Dz_inside[:, -1:, ...]
        #field = self.amp[:, -1:, ...]**2 * Dz_inside[:, -1:, ...] - self.bias[:, -1:, ...]**2
        out = field_bg.sum(dim=1, keepdim=True)
        return out
    
        if isinstance(Temp, (float, int)):
            field = self.amp * field * float(Temp) + self.bias

        elif torch.is_tensor(Temp) and Temp.dim() == 4:
            # Temp 形状可为 [B,1,H,W] 或 [1,1,H,W] 或 [B,C,H,W]
            if Temp.size(1) in (1, C, C+1):
                mask_in = ((X - xc).abs() <= a) & ((Y - yc).abs() <= b)
                mask_in = mask_in.expand(Temp.size(0), C+1, *Temp.shape[-2:])
                TempBC = (Temp * inside.float()).sum(dim=(-1,-2))/inside.float().sum(dim=(-1,-2))
                TempBC = TempBC.unsqueeze(-1).unsqueeze(-1)
                
            field[:,:-1,...] = self.amp * field[:,:-1,...] * (self.amb_temp - Temp) + self.bias
            field[:,-1:,...] = self.amp * field[:,-1:,...] + self.bias

        else:
            field = self.amp * field + self.bias

        out = field.sum(dim=1, keepdim=True)  # 聚合所有 chiplet
        return out
    
