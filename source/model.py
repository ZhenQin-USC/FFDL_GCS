import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from os.path import join
import os
import json
from collections import OrderedDict
from typing import List


class ResConv_block(nn.Module):
    def __init__(self, 
                 in_channels, 
                 kernel_size=3, 
                 padding=1, 
                 stride=1, 
                 padding_mode='zeros', 
                 norm_type='group', 
                 num_groups=None
                 ):
        super(ResConv_block, self).__init__()
        self.conv0 = nn.Conv3d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               padding_mode=padding_mode)
        
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               padding_mode=padding_mode)

        if norm_type == 'batch': 
            self.norm0 = nn.BatchNorm3d(in_channels)
            self.norm1 = nn.BatchNorm3d(in_channels)
        elif norm_type == 'group':
            if num_groups == None:
                num_groups = in_channels
            self.norm0 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        
        self.nonlinear = nn.ReLU()

    def forward(self, x):
        a = self.conv0(x)
        a = self.nonlinear(self.norm0(a))
        a = self.conv1(a)
        y = self.norm1(a)
        return x + y


class Up(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 groups: int=2, 
                 mode: str='trilinear', 
                 align_corners: bool=True
                 ):
        super(Up, self).__init__()
        self.deconv = nn.Upsample(scale_factor=2, mode=mode, align_corners=align_corners)
        self.padding = nn.ReflectionPad3d(1)
        self.conv = nn.Conv3d(groups*in_channels, groups*out_channels, groups=groups,
                              kernel_size=3, padding=0, stride=1, padding_mode='zeros')
    def forward(self, x):
        x = self.deconv(x)
        x = self.padding(x)
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 groups, 
                 stride:int=2
                 ):
        super(Down, self).__init__()
        self.upscaler = nn.Conv3d(groups*in_channels, groups*out_channels, 
                                  kernel_size=3, stride=stride, groups=groups, 
                                  padding=1, padding_mode='zeros', bias=True)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=groups*out_channels)

    def forward(self, x):
        x = self.upscaler(x)
        return self.norm(x)


class Conv3dDerivative(nn.Module):
    def __init__(self, 
                 DiffFilter, 
                 deno:float=1.0):
        '''
        :param DerFilter: constructed derivative filter, e.g. Laplace filter
        :param deno: resolution of the filter, used to divide the output, e.g. c*dt, c*dx or c*dx^2
        :param kernel_size:
        '''
        super(Conv3dDerivative, self).__init__()
        self.deno = deno  # constant in the finite difference
        self.input_channels = DiffFilter.shape[1]
        self.output_channels = DiffFilter.shape[0]
        self.kernel_size = DiffFilter.shape[-1]
        self.filter = nn.Conv3d(self.input_channels, self.output_channels, self.kernel_size, 1, padding=0, bias=False)
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DiffFilter, dtype=torch.float32), requires_grad=False)
        
    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.deno


class Density_Function(nn.Module):
    def __init__(self, 
                 piece_wise_feature:int=10):
        super(Density_Function, self).__init__()
        self.piece_wise_feature = piece_wise_feature
        self.Wh1_p = nn.Conv3d(in_channels=1, out_channels=self.piece_wise_feature, 
                               kernel_size=1, stride=1, padding=0, bias=True, )
        self.Wh2_p = nn.Conv3d(in_channels=self.piece_wise_feature, out_channels=2, 
                               kernel_size=1, stride=1, padding=0, bias=True, )
        self.Nonlinear = nn.GELU()
    
    def forward(self, p):
        z = self.Wh1_p(p)
        z = self.Nonlinear(z)
        z = self.Wh2_p(z)
        return self.Nonlinear(z)


class Physical_Function(nn.Module):
    def __init__(self, 
                 piece_wise_feature:int=10):
        
        super(Physical_Function, self).__init__()
        
        self.piece_wise_feature = piece_wise_feature
        self.Nonlinear1 = nn.GELU()
        self.Nonlinear2 = nn.Sigmoid() # GELU('tanh')
        self.Wh1_p = nn.Conv3d(in_channels=1, out_channels=self.piece_wise_feature, 
                               kernel_size=1, stride=1, padding=0, bias=True, )
        self.Wh2_p = nn.Conv3d(in_channels=self.piece_wise_feature, out_channels=1, 
                               kernel_size=1, stride=1, padding=0, bias=True, )
        self.Wh3_p = nn.Conv3d(in_channels=1, out_channels=self.piece_wise_feature, 
                               kernel_size=1, stride=1, padding=0, bias=True, )
        self.Wh4_p = nn.Conv3d(in_channels=self.piece_wise_feature, out_channels=2, 
                               kernel_size=1, stride=1, padding=0, bias=True, )
        self.Wh1_s = nn.Conv3d(in_channels=1, out_channels=self.piece_wise_feature, 
                               kernel_size=1, stride=1, padding=0, bias=True, )
        self.Wh2_s = nn.Conv3d(in_channels=self.piece_wise_feature, out_channels=2, 
                               kernel_size=1, stride=1, padding=0, bias=True, )
        
    def forward(self, p, sn):
        zp1 = self.Nonlinear1(self.Wh1_p(p))
        cphi= self.Nonlinear1(self.Wh2_p(zp1))
        
        zp2 = self.Nonlinear1(self.Wh3_p(p))
        imu = self.Nonlinear1(self.Wh4_p(zp2))
        
        zs = self.Nonlinear2(self.Wh1_s(sn))
        kr = self.Nonlinear2(self.Wh2_s(zs))
        
        # imuw, imun = torch.split(imu, split_size_or_sections, dim=0)
        lam = kr * imu 
        return torch.cat((lam, cphi), dim=1)


class Physical_Operator(nn.Module):
    def __init__(self, 
                 piece_wise_feature:int=10):
        
        super(Physical_Operator, self).__init__()
        diff_filter = self.generate_diff_filters()
        reflection_pad = (1, 1, 1, 1, 1, 1)
        self.p_pad_func = nn.ReplicationPad3d(reflection_pad) 
        self.z_pad_func = nn.ReplicationPad3d(reflection_pad)
        self.density_func  = Density_Function(piece_wise_feature)
        self.physical_func = Physical_Function(piece_wise_feature)
        self.derivative_func = Conv3dDerivative(diff_filter)
    
    def forward(self, x):
        p, sn, dz = torch.split(x, (1, 1, 1), dim=1)
        # State-based Operator
        rho = self.density_func(p)
        rhow, rhon = torch.split(rho, (1, 1), dim=1)
        prop = self.physical_func(p, sn)
        lamw, lamn, cphi = torch.split(prop, (1, 1, 1), dim=1)

        # Spatial-based Operator: {Padding; Derivative}
        p_pad = self.p_pad_func(p)
        z_pad = self.z_pad_func(dz)
        rho_pad = self.density_func(p_pad)
        Dz = rho_pad*z_pad
        Dzw, Dzn = torch.split(Dz, (1, 1), dim=1)

        dp = self.derivative_func(p_pad)
        dDzw = self.derivative_func(Dzw)
        dDzn = self.derivative_func(Dzn)
        
        return torch.cat((dp, dDzw, dDzn, lamw, lamn, rhow, rhon, cphi), dim=1)
    
    def generate_diff_filters(self):
        stencil = (1/2)*np.array([-1, 0, 1])
        diff_2nd_filter_3d = np.zeros((6, 1, 3, 3, 3), dtype=np.float32) # [-1, 0, 1]/2 -- df/dx ~= [f(i+1,j)-f(i-1,j)]/(2*dx)
        diff_2nd_filter_3d[0, 0, 1, :, 1] = stencil # x-axis direction
        diff_2nd_filter_3d[1, 0, 1, :, 1] = stencil # x-axis direction

        diff_2nd_filter_3d[2, 0, :, 1, 1] = stencil # y-axis direction
        diff_2nd_filter_3d[3, 0, :, 1, 1] = stencil # y-axis direction

        diff_2nd_filter_3d[4, 0, 1, 1, :] = stencil # z-axis direction
        diff_2nd_filter_3d[5, 0, 1, 1, :] = stencil # z-axis direction
        
        return diff_2nd_filter_3d


class State_Encoder_Layer(nn.Module):
    def __init__(self, hidden_operator_dim:int, in_channels:int=1, 
                 local_kernel_size:int=1, spatial_kernel_size:int=3, 
                 stride:int=1, norm_type:str='group'):

        super(State_Encoder_Layer, self).__init__()
        
        self.split_sizes_0 = (in_channels, in_channels, 6*in_channels, 23*in_channels)
        self.split_sizes_1 = (6*in_channels, 6*in_channels, 6*in_channels,
                              in_channels, in_channels, in_channels,
                              in_channels, in_channels)
        # Activation
        self.Nonlinear0 = nn.Sigmoid()
        self.Nonlinear1 = nn.GELU()

        # State-based Operator
        self.F_sn   = nn.Conv3d(in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        self.F_rhow = nn.Conv3d(in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        self.F_rhon = nn.Conv3d(in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        self.F_lamw = nn.Conv3d(in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        self.F_lamn = nn.Conv3d(in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        self.F_cphi = nn.Conv3d(in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        
        # Spatial-based Operator
        self.F_at   = nn.Conv3d(in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        self.F_dp   = nn.Conv3d(6*in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        self.F_dDzw = nn.Conv3d(6*in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        self.F_dDzn = nn.Conv3d(6*in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        self.F_tgm  = nn.Conv3d(6*in_channels, hidden_operator_dim, local_kernel_size, 
                                  stride=stride, padding=0, bias=True)
        self.F_tdyw = nn.Conv3d(hidden_operator_dim, hidden_operator_dim, spatial_kernel_size, 
                                  stride=1, padding=1, padding_mode='reflect', bias=False)
        self.F_tdyn = nn.Conv3d(hidden_operator_dim, hidden_operator_dim, spatial_kernel_size, 
                                  stride=1, padding=1, padding_mode='reflect', bias=False)

        # resnet
        self.res_acc0 = ResConv_block(2*hidden_operator_dim, norm_type=norm_type, num_groups=2)
        self.res_adv0 = ResConv_block(2*hidden_operator_dim, norm_type=norm_type, num_groups=2)
        self.res_acc1 = ResConv_block(2*hidden_operator_dim, norm_type=norm_type, num_groups=2)
        self.res_adv1 = ResConv_block(2*hidden_operator_dim, norm_type=norm_type, num_groups=2)

    def forward(self, x):
        
        # Split Input
        sn, at, tgm, props = torch.split(x, self.split_sizes_0, dim=1)
        dp, dDzw, dDzn, lamw, lamn, rhow, rhon, cphi = torch.split(
            props, self.split_sizes_1, dim=1)

        # State-based Operator
        sn = self.Nonlinear0(self.F_sn(sn))
        at = self.Nonlinear1(self.F_at(at))
        rhow = self.Nonlinear1(self.F_rhow(rhow))
        rhon = self.Nonlinear1(self.F_rhon(rhon))
        lamw = self.Nonlinear1(self.F_lamw(lamw))
        lamn = self.Nonlinear1(self.F_lamn(lamn))
        cphi = self.Nonlinear1(self.F_cphi(cphi))
        dp   = self.Nonlinear1(self.F_dp(dp))

        # Space-based Operator
        dDzw = self.Nonlinear1(self.F_dDzw(dDzw))
        dDzn = self.Nonlinear1(self.F_dDzn(dDzn))
        tgm = self.F_tgm(tgm)
        tdyw = self.Nonlinear1(self.F_tdyw(rhow*lamw))
        tdyn = self.Nonlinear1(self.F_tdyn(rhon*lamn))
        
        # Non-parametric Operator
        dpzmw = dp - dDzw
        dpzmn = dp - dDzn
        accw = cphi*(1-sn)*rhow
        accn = cphi*sn*rhon
        advw = tgm*tdyw*dpzmw*at
        advn = tgm*tdyn*dpzmn*at
        
        # Resnet
        acc = self.res_acc0(torch.cat((accw, accn), dim=1))
        adv = self.res_adv0(torch.cat((advw, advn), dim=1))
        acc = self.res_acc1(acc)
        adv = self.res_adv1(adv)
        
        return torch.cat((acc, adv), dim=1)


class Latent_Processor_Layer(nn.Module):
    # 2*hidden_operator_dim, latent_processor_dim
    def __init__(self, latent_outer_dim:int, latent_inner_dim=None, group_multipler:int=1, 
                 tanh_clip:float=2.0):
        
        super(Latent_Processor_Layer, self).__init__()
        if latent_inner_dim == None:
            latent_inner_dim = latent_outer_dim
        self.latent_outer_dim = latent_outer_dim
        self.latent_inner_dim = latent_inner_dim

        # Upscaling (coarsening)
        self.Wz_up  = nn.Conv3d(4*latent_outer_dim, 4*latent_inner_dim, kernel_size=3, groups=4,
                                stride=2, padding=1, padding_mode='zeros', bias=False)

        # Residual 
        self.Wr_acc = nn.Conv3d(latent_inner_dim, latent_inner_dim, kernel_size=3, # groups=2, 
                                padding=1, padding_mode='zeros', bias=False)
        self.Wr_adv = nn.Conv3d(latent_inner_dim, latent_inner_dim, kernel_size=3, # groups=2, 
                                padding=1, padding_mode='zeros', bias=False)
        self.Wr_src = nn.Conv3d(latent_inner_dim, latent_inner_dim, kernel_size=3, # groups=2, 
                                padding=1, padding_mode='zeros', bias=False)
        
        # Difference
        self.Wd_acc = nn.Conv3d(latent_inner_dim, latent_inner_dim, kernel_size=3, 
                                padding=1, padding_mode='zeros', bias=True)
        self.Wd_adv = nn.Conv3d(latent_inner_dim, latent_inner_dim, kernel_size=3, 
                                padding=1, padding_mode='zeros', bias=True)
        self.Wd_src = nn.Conv3d(latent_inner_dim, latent_inner_dim, kernel_size=3, 
                                padding=1, padding_mode='zeros', bias=True)
        
        self.GN_dz = nn.GroupNorm(num_groups=6*group_multipler, num_channels=3*latent_inner_dim)
        self.GN_rz = nn.GroupNorm(num_groups=6*group_multipler, num_channels=3*latent_inner_dim)
        
        # Downscaling 
        self.W_dzt = nn.Conv3d(3*latent_inner_dim, 3*latent_inner_dim, kernel_size=1, 
                               stride=1, padding=0, padding_mode='zeros', bias=True)
        self.Wz_down = nn.Sequential( OrderedDict([
                ('Upsampling', nn.Upsample(scale_factor=2)), #, mode='trilinear', align_corners=True
                ('Padding', nn.ReflectionPad3d(1)), 
                ('Conv', nn.Conv3d(3*latent_inner_dim, 3*latent_outer_dim, kernel_size=3, groups=3, 
                                   stride=1, padding=0, bias=False))])
                                   )
        self.GN_dzt = nn.GroupNorm(num_groups=6*group_multipler, num_channels=3*latent_outer_dim)
        self.Nonlinear_rz = nn.Tanh()
        self.Nonlinear_dz = nn.Hardtanh(-tanh_clip, tanh_clip)

    def forward(self, x):

        latent_outer_dim = self.latent_outer_dim
        latent_inner_dim = self.latent_inner_dim

        # Split Input
        _, z0 = torch.split(x, (latent_outer_dim, 3*latent_outer_dim), dim=1)

        # Downsampling 
        zlow = self.Wz_up(x)

        acc_tm1, acc, adv, src = torch.split(
            zlow, (latent_inner_dim, latent_inner_dim, latent_inner_dim, latent_inner_dim), dim=1)

        # Update
        res = acc - acc_tm1 + adv + src
        r_acc = self.Wr_acc(res*acc)
        r_adv = self.Wr_adv(res*adv)
        r_src = self.Wr_src(res*src)

        d_acc = self.Wd_acc(acc)
        d_adv = self.Wd_adv(adv)
        d_src = self.Wd_src(src)
        
        dz = torch.cat((d_acc, d_adv, d_src), dim=1)
        rz = torch.cat((r_acc, r_adv, r_src), dim=1)
        
        rz = self.Nonlinear_rz(rz)
        dz = self.GN_dz(dz)
        rz = self.GN_rz(rz)

        dzt = self.W_dzt(dz*rz)
        dzt = self.GN_dzt(self.Wz_down(dzt))
        dzt = self.Nonlinear_dz(dzt)

        return z0 + dzt


class Multistep_Control_Encoder(nn.Module):
    def __init__(self, 
                 physic_operator_dim:int=16,
                 hidden_operator_dim:int=32, 
                 in_channels:int=1, 
                 spatial_kernel_size:int=3, 
                 local_kernel_size:int=1, 
                 stride:int=1):
        super(Multistep_Control_Encoder, self).__init__()
        self.F_volw = nn.Conv3d(in_channels, physic_operator_dim, spatial_kernel_size,
                                  stride=stride, padding=1, padding_mode='zeros', bias=False)
        self.F_voln = nn.Conv3d(in_channels, physic_operator_dim, spatial_kernel_size,
                                  stride=stride, padding=1, padding_mode='zeros', bias=False)
        self.F_at = nn.Conv3d(in_channels, physic_operator_dim, local_kernel_size,
                                stride=stride, padding=0, bias=True)
        self.down_src = Down(in_channels=physic_operator_dim, groups=2,
                             out_channels=hidden_operator_dim)

    def forward(self, x, at):
        B = at.size(0)
        T = x.size(0)//B

        uw, un = torch.split(x, (1, 1), dim=1)
        volw = self.F_volw(uw)
        voln = self.F_voln(un)
        ats = torch.repeat_interleave(self.F_at(at), repeats=T, dim=0)
        srcw = volw*ats
        srcn = voln*ats
        srcs = self.down_src(torch.cat((srcw, srcn), dim=1))
        return srcs.view(B, T, srcs.size(1), srcs.size(2), srcs.size(3), srcs.size(4))


class Multistep_Decoder_Layer(nn.Module):
    
    def __init__(self, hidden_operator_dim, decoder_output_dim, out_dim:int=1):

        super(Multistep_Decoder_Layer, self).__init__()

        out_channels_a, out_channels_b, out_channels_c = decoder_output_dim

        # Layer 1: 2*hidden_operator_dim --> 2*out_channels_a
        self.W_tgma = nn.Conv3d(6, out_channels_a, kernel_size=1, 
                                padding=0, padding_mode='zeros', bias=True)
        self.W_at   = nn.Conv3d(1, out_channels_a, kernel_size=1, 
                                padding=0, padding_mode='zeros', bias=True)
        
        self.W_acc0 = nn.Conv3d(2*hidden_operator_dim, 2*out_channels_a, kernel_size=1, groups=2,
                                padding=0, padding_mode='zeros', bias=True)
        self.W_acc1 = nn.Conv3d(2*hidden_operator_dim, 2*out_channels_a, kernel_size=1, 
                                padding=0, padding_mode='zeros', bias=True)

        self.W_adv0 = nn.Conv3d(2*hidden_operator_dim, 2*out_channels_a, kernel_size=1, groups=2,
                                padding=0, padding_mode='zeros', bias=True)
        self.W_adv1 = nn.Conv3d(2*hidden_operator_dim, 2*out_channels_a, kernel_size=3, groups=2,
                                padding=1, padding_mode='zeros', bias=True)

        # Layer 2: 
        self.W_zp0 = nn.Conv3d(4*out_channels_a, 2*out_channels_b, kernel_size=1, groups=2,
                               padding=0, padding_mode='zeros', bias=True)
        self.W_zs0 = nn.Conv3d(4*out_channels_a, 2*out_channels_b, kernel_size=1, groups=2,
                               padding=0, padding_mode='zeros', bias=True)
        
        # Layer 3:
        self.W_zp1 = nn.Conv3d(2*out_channels_b, 2*out_channels_c, kernel_size=1,
                               padding=0, padding_mode='zeros', bias=True)
        self.W_zs1 = nn.Conv3d(2*out_channels_b, 2*out_channels_c, kernel_size=1,
                               padding=0, padding_mode='zeros', bias=True)

        # Layer 4:
        self.W_zp2 = nn.Conv3d(2*out_channels_c, out_dim, kernel_size=1,
                               padding=0, padding_mode='zeros', bias=True)
        self.W_zs2 = nn.Conv3d(2*out_channels_c, out_dim, kernel_size=1,
                               padding=0, padding_mode='zeros', bias=True)

        self.Nonlinear1 = nn.GELU()
        self.Nonlinear2 = nn.Sigmoid()

    def forward(self, acc, adv, at, tgm):
        B = at.size(0)
        T = acc.size(0)//B

        tgm_0 = self.Nonlinear1(self.W_tgma(tgm)) # 6 --> hidden_operator_dim
        tgm_a = torch.cat((tgm_0, tgm_0), dim=1)
        tgm_a = torch.repeat_interleave(tgm_a, repeats=T, dim=0)

        at_0  = self.Nonlinear1(self.W_at(at)) # 1 --> hidden_operator_dim
        at_a  = torch.cat((at_0, at_0), dim=1)
        at_a = torch.repeat_interleave(at_a, repeats=T, dim=0)

        # Layer 1: 2*hidden_operator_dim --> 2*out_channels_a
        acc_0 = self.Nonlinear1(self.W_acc0(acc))
        acc_1 = self.Nonlinear1(self.W_acc1(acc))
        
        adv_0 = self.W_adv0(adv)-tgm_a-at_a
        adv_1 = self.W_adv1(adv)-tgm_a-at_a

        acc_2 = acc_0 + acc_1
        adv_2 = adv_0 + adv_1
        z = torch.cat((acc_2, adv_2), dim=1)

        # Layer 2: 2*out_channels_a --> 2*out_channels_b
        zp0 = self.Nonlinear1(self.W_zp0(z))
        zs0 = self.Nonlinear1(self.W_zs0(z))

        zp1 = self.Nonlinear1(self.W_zp1(zp0))
        zs1 = self.Nonlinear1(self.W_zs1(zs0))

        pt  = self.Nonlinear1(self.W_zp2(zp1))
        snt = self.Nonlinear2(self.W_zs2(zs1))

        out = torch.cat((pt, snt), dim=1)
        return out


class Initial_State_Encoder(nn.Module): 
    def __init__(self, 
                 physic_operator_dim: int = 16,
                 hidden_operator_dim: int = 32,
                 group_multipler: int = 1,
                ):
        super(Initial_State_Encoder, self).__init__()

        self.encoder_upscaler = nn.Sequential(
            OrderedDict([
                ('enc', State_Encoder_Layer(physic_operator_dim)),
                ('upscale', Down(in_channels=physic_operator_dim, out_channels=hidden_operator_dim, groups=4)),
                ('norm', nn.GroupNorm(num_groups=4*group_multipler, num_channels=4*hidden_operator_dim)),
                ]))
        self.z_split_size = (2*hidden_operator_dim, 2*hidden_operator_dim)

    def forward(self, states, static, props):
        # input
        p, sn = torch.split(states, (1, 1), dim=1)
        at, dz, tgm = torch.split(static, (1, 1, 6), dim=1)

        # encoder
        z0 = torch.cat((sn, at, tgm, props), dim=1)
        z_tm1 = self.encoder_upscaler(z0)
        acc_tm1, _ = torch.split(z_tm1, self.z_split_size, dim=1)
        return acc_tm1, z_tm1


class Latent_Recurrent_Processor(nn.Module):
    def __init__(self, 
                 num_processor: int = 3, 
                 hidden_operator_dim: int = 32,
                 latent_processor_dim: int = 48,
                 group_multipler: int = 1,
                 tanh_clip: float = 2.0,
                 ):
        super(Latent_Recurrent_Processor, self).__init__()
        self.latent_split_size = (2*hidden_operator_dim, 
                                  2*hidden_operator_dim, 
                                  2*hidden_operator_dim)

        processor_layer = []
        processor_norms = []
        for _ in range(num_processor):
            processor_layer.append(
                Latent_Processor_Layer(2*hidden_operator_dim, latent_processor_dim, 
                                       group_multipler=group_multipler, tanh_clip=tanh_clip))
            processor_norms.append(
                nn.GroupNorm(num_groups=6*group_multipler, 
                             num_channels=6*hidden_operator_dim))
        self.processors = nn.ModuleList(processor_layer)
        self.proc_norms = nn.ModuleList(processor_norms)


    def forward(self, acc_tm1, z_tm1, srcs):
        steps = srcs.size(1)
        acc_seq = []
        adv_seq = []
        for step in range(steps):
            # Forecast
            zt = self.latent_processor(srcs[:, step], [acc_tm1, z_tm1])
            # Reset States
            acc, adv, src = torch.split(zt, self.latent_split_size, dim=1)
            acc_tm1 = acc
            z_tm1 = torch.cat((acc, adv), dim=1)
            # Store
            acc_seq.append(acc)
            adv_seq.append(adv)
        return torch.stack(acc_seq, dim=1), torch.stack(adv_seq, dim=1),  
    
    def latent_processor(self, src, hidden_states):
        acc_tm1, z_tm1 = hidden_states
        zt = torch.cat((z_tm1, src), dim=1)
        for processor, proc_norm in zip(self.processors, self.proc_norms):
            zt = processor(torch.cat((acc_tm1, zt), dim=1))
            zt = proc_norm(zt)
        return zt
    

class State_Decoder(nn.Module):
    def __init__(self, 
                 hidden_operator_dim: int = 32, 
                 decoder_output_dim: List[int] = [16, 8, 4],
                 ):
        super(State_Decoder, self).__init__()
        self.up_acc = Up(in_channels=hidden_operator_dim, out_channels=hidden_operator_dim, groups=2)
        self.up_adv = Up(in_channels=hidden_operator_dim, out_channels=hidden_operator_dim, groups=2)
        self.decoder = Multistep_Decoder_Layer(hidden_operator_dim, decoder_output_dim)

    def forward(self, accs, advs, at, tgm):
        B, T, L, X, Y, Z = accs.size()
        acc = self.up_acc(accs.view(B * T, L, X, Y, Z)) 
        adv = self.up_adv(advs.view(B * T, L, X, Y, Z)) 

        states = self.decoder(acc, adv, at, tgm)
        return states.view(B, T, 2, 2*X, 2*Y, 2*Z)


class ffdl3M(nn.Module):
    def __init__(self,
                 physic_operator_dim: int = 16,
                 hidden_operator_dim: int = 32,
                 latent_processor_dim: int = 48,
                 decoder_output_dim: List[int] = [16, 8, 4],
                 num_processor: int = 3,
                 piece_wise_feature: int = 40,
                #  group_multipler: int = 1,
                 tanh_clip: float = 2.0,
                 ):

        super(ffdl3M, self).__init__()
        self.hidden_operator_dim = hidden_operator_dim
        self.num_processor = num_processor
        self.latent_split_size = (2*hidden_operator_dim, 
                                  2*hidden_operator_dim, 
                                  2*hidden_operator_dim)
        self.z_split_size = (2*hidden_operator_dim, 2*hidden_operator_dim)
        
        self.physics = Physical_Operator(piece_wise_feature=piece_wise_feature)

        self.state_encoder = Initial_State_Encoder(physic_operator_dim=physic_operator_dim,
                                                   hidden_operator_dim=hidden_operator_dim)

        self.control_encoder = Multistep_Control_Encoder(physic_operator_dim=physic_operator_dim,
                                                         hidden_operator_dim=hidden_operator_dim)
        
        self.latent_processor = Latent_Recurrent_Processor(
            num_processor=num_processor, hidden_operator_dim=hidden_operator_dim, 
            latent_processor_dim=latent_processor_dim, tanh_clip=tanh_clip)

        self.state_decoder = State_Decoder(hidden_operator_dim, decoder_output_dim)

    def forward(self, contrl, states, static):
        # ut: (B, Nt, Nc, Nx, Ny, Nz) = (uw, un)
        # states: (B, Nc, Nx, Ny, Nz) = (p, sn)
        # static: (B, Nc, Nx, Ny, Nz) = (at, z, tgm)
        
        B, steps, Nc, Nx, Ny, Nz = contrl.size()

        # input
        p, sn = torch.split(states, (1, 1), dim=1)
        at, dz, tgm = torch.split(static, (1, 1, 6), dim=1)

        # encoder
        props = self.physics(torch.cat((p, sn, dz), dim=1))
        acc_tm1, z_tm1 = self.state_encoder(states, static, props)

        # control encoder
        srcs = self.control_encoder(contrl.view(B*steps, Nc, Nx, Ny, Nz), at)
        
        # processors
        accs, advs = self.latent_processor(acc_tm1, z_tm1, srcs)

        # decoder
        outputs = self.state_decoder(accs, advs, at, tgm)

        return outputs, outputs
