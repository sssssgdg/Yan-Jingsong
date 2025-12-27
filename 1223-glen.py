import taichi as ti
import numpy as np
import time
import math
import os
from plyfile import *
from plyImporter import PlyImporter
import xlwt
import xlrd

np.set_printoptions(suppress=True)
ply3 = PlyImporter("zhibai_mpm_1217.ply")
n_particles = ply3.get_count()


def excel2matrix(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows
    ncols = table.ncols
    datamatrix = np.zeros((nrows, ncols))
    for i in range(ncols):
        cols = table.col_values(i)
        datamatrix[:, i] = cols
    return datamatrix


pathX = 'G:\\YJS/resource/EW 0.0005间隔.xls'
# 如果需要Y和Z轴数据，请取消注释并确保文件存在
accX = excel2matrix(pathX)


@ti.data_oriented
class MPMSolver:
    def __init__(self):
        self.flip_velocity_adjustment = 0.97
        self.flip_position_adjustment_min = 0.0
        self.flip_position_adjustment_max = 1.0
        self.apic_affine_stretching = 1.0
        self.apic_affine_rotation = 1.0
        self.particle_collision = 1.0
        self.max_particle_num = n_particles
        self.grid_num = 128
        self.screen_ratio = 5300
        self.dx = self.screen_ratio / self.grid_num
        self.inv_dx = self.grid_num / self.screen_ratio
        self.dt = 5e-4

        self.d = 3
        self.pi = ti.math.pi
        self.gravity = ti.Vector([0, -9.8, 0])
        self.p_vol = (self.dx * 0.5) ** 3
        self.n = 0.4
        self.e = self.n / (1 - self.n)
        self.state = ti.field(ti.i32, shape=self.max_particle_num)
        self.block_dim_1 = 32
        self.block_size = 16

        # --- Glen Flow Law Constants (from PDF Table 2) ---
        self.A0 = 1.916e3
        self.Q_act = 139000.0  # J/mol
        self.R_gas = 8.314
        self.Temp_C = -3.0
        self.Temp_K = 273.15 + self.Temp_C
        self.n_exp = 3.0
        self.tau_th = 100000.0  # 100 kPa
        self.B_enhance = 1.5

        self.totalparticles = ti.Struct.field({
            "material": ti.f64,
            "position": ti.types.vector(3, ti.f64),
        }, shape=self.max_particle_num)

        # 在粒子结构体中添加 damage 字段
        self.particles = ti.Struct.field({
            "position": ti.types.vector(3, ti.f64),
            "position0": ti.types.vector(3, ti.f64),
            "rgba": ti.types.vector(4, ti.f64),
            "pos": ti.f64,
            "vel": ti.f64,
            "color": ti.types.vector(3, ti.f64),
            "g": ti.types.vector(3, ti.f64),
            "a": ti.types.vector(3, ti.f64),
            "a_p": ti.types.vector(3, ti.f64),
            "rho": ti.f64,
            "mass": ti.f64,
            "velocity": ti.types.vector(3, ti.f64),
            "stresssolid": ti.types.matrix(3, 3, ti.f64),
            "F": ti.types.matrix(3, 3, ti.f64),
            "C": ti.types.matrix(3, 3, ti.f64),
            "vc_s": ti.f64,
            "phi_s": ti.f64,
            "material": ti.i32,
            "c_C0": ti.f64,
            "c_C": ti.f64,
            "h0": ti.f64,
            "h1": ti.f64,
            "h2": ti.f64,
            "h3": ti.f64,
            "phi": ti.f64,
            "alpha_s": ti.f64,
            "E_s": ti.f64,
            "E_s_tr": ti.f64,
            "nu_s": ti.f64,
            "mu_s": ti.f64,
            "lambda_s": ti.f64,
            "q_s": ti.f64,
            "k": ti.f64,
            "J_w": ti.f64,
            "w_k": ti.f64,
            "w_gamma": ti.f64,
            "J": ti.f64,
            "damage": ti.f64,  # 新增：损伤变量 D
        }, shape=self.max_particle_num)

        self.b_particles = ti.Struct.field({
            "position": ti.types.vector(3, ti.f64),
            "rgba": ti.types.vector(4, ti.f64),
            "pos": ti.f64,
            "vel": ti.f64,
            "color": ti.types.vector(3, ti.f64),
            "rho": ti.f64,
            "mass": ti.f64,
            "velocity": ti.types.vector(3, ti.f64),
            "material": ti.f64,
        }, shape=self.max_particle_num)

        self.solid_m = ti.field(ti.f64)
        self.solid_v = ti.Vector.field(3, ti.f64)
        self.solid_v0 = ti.Vector.field(3, ti.f64)
        self.solid_f = ti.Vector.field(3, ti.f64)
        self.solid_acc = ti.Vector.field(3, ti.f64)
        self.solid_a = ti.Vector.field(3, ti.f64)
        self.solid_gradient = ti.Vector.field(3, ti.f64)
        self.solid_num = ti.field(ti.f64)
        self.grid_damping = ti.field(ti.f64)

        self.boundary_m = ti.field(ti.f64)
        self.boundary_v = ti.Vector.field(3, ti.f64)
        self.boundary_gradient = ti.Vector.field(3, ti.f64)

        self.root = ti.root.pointer(ti.ijk, (self.grid_num // self.block_size,) * 3)

        self.root.dense(ti.ijk, self.block_size).place(
            self.solid_m,
            self.solid_v,
            self.solid_v0,
            self.solid_f,
            self.solid_acc,
            self.solid_a,
            self.solid_gradient,
            self.solid_num,
            self.grid_damping,
            self.boundary_m,
            self.boundary_v,
            self.boundary_gradient
        )

        self.total_particles = ti.field(ti.i32, shape=())
        self.n_s_particles = ti.field(ti.i32, shape=())
        self.boundry_particles = ti.field(ti.i32, shape=())
        self.bear_points = ti.Vector.field(4, ti.f64, shape=self.max_particle_num)

    def init_bear(self):
        self.bear_points.from_numpy(ply3.get_array())

    @ti.kernel
    def generate_bear(self):
        for i in range(self.max_particle_num):
            n = ti.atomic_add(self.total_particles[None], 1)
            self.totalparticles[n].material = self.bear_points[n][3]
            zoom = self.bear_points[n][0:3]
            position = ti.Vector(
                [zoom[0] * self.screen_ratio, zoom[1] * self.screen_ratio, zoom[2] * self.screen_ratio])

            if self.totalparticles[n].material == 0:  # 冰 (Glen流变 + 损伤)
                k = ti.atomic_add(self.n_s_particles[None], 1)
                self.particles[k].velocity = ti.Vector([0.0, -0.0, 0.0])
                self.particles[k].position = position
                self.particles[k].position0 = position
                self.particles[k].F = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                self.particles[k].rho = 910  # PDF Table 2 Value
                self.particles[k].mass = self.particles[k].rho * self.p_vol
                self.particles[k].rgba = [0.6, 0.8, 0.9, 1.0]  # 浅蓝色
                self.particles[k].material = 0
                self.particles[k].damage = 0.0  # 初始损伤
                # 冰的弹性参数用于压力计算
                self.particles[k].E_s = 1e8
                self.particles[k].nu_s = 0.33

            elif self.totalparticles[n].material == 1:  # 岩石 (保持原有 Drucker-Prager)
                k = ti.atomic_add(self.n_s_particles[None], 1)
                self.particles[k].velocity = ti.Vector([0.0, -0.0, 0.0])
                self.particles[k].position = position
                self.particles[k].position0 = position
                self.particles[k].F = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                self.particles[k].rho = 2500  # PDF Table 1 Value
                self.particles[k].mass = self.particles[k].rho * self.p_vol
                self.particles[k].c_C0 = -0.001
                self.particles[k].phi = 35  # 摩擦角
                self.particles[k].alpha_s = ti.sqrt(2 / 3) * (2 * ti.sin(self.particles[k].phi)) \
                                            / (3 - ti.sin(self.particles[k].phi))
                self.particles[k].rgba = [0.5, 0.5, 0.5, 1.0]
                self.particles[k].E_s = 4e10  # PDF Table 1 Value
                self.particles[k].nu_s = 0.25
                self.particles[k].mu_s = self.particles[k].E_s / (2 * (1 + self.particles[k].nu_s))
                self.particles[k].lambda_s = self.particles[k].E_s * self.particles[k].nu_s / (
                        (1 + self.particles[k].nu_s) * (1 - 2 * self.particles[k].nu_s))
                self.particles[k].material = 1
                self.particles[k].damage = 0.0

            if self.totalparticles[n].material == 4:  # 边界层
                j = ti.atomic_add(self.boundry_particles[None], 1)
                self.b_particles[j].position = position
                self.b_particles[j].velocity = ti.Vector([0.0, 0.0, 0.0])
                self.b_particles[j].rho = 2000
                self.b_particles[j].mass = self.b_particles[j].rho * self.p_vol
                self.b_particles[j].material = 4

    @ti.func
    def contact_boundary_rock(self, i: ti.i32, j: ti.i32, k: ti.i32):
        # ... (原有边界处理逻辑保持不变) ...
        normal_s = ti.Vector.zero(float, 3)
        normal_r = ti.Vector.zero(float, 3)

        if self.solid_m[i, j, k] > 0 and self.boundary_m[i, j, k] > 0:
            normal_r = (- self.boundary_gradient[i, j, k] + self.solid_gradient[i, j, k]) / (
                    - self.boundary_gradient[i, j, k] + self.solid_gradient[i, j, k]).norm()
            normal_s = -normal_r

        if (- self.boundary_v[i, j, k] + self.solid_v[i, j, k]).dot(normal_r) > 0:
            f_contact_r = ((self.solid_m[i, j, k] * self.boundary_m[i, j, k]) / (
                    (self.solid_m[i, j, k] + self.boundary_m[i, j, k]) * self.dt)) * (
                                  - self.solid_v[i, j, k] + self.boundary_v[i, j, k])
            f_contact_s = - f_contact_r
            f_normal_r = (f_contact_r.dot(normal_r)) * normal_r
            f_normal_s = (f_contact_s.dot(normal_s)) * normal_s

            f_tangent_r = f_contact_r - f_normal_r
            f_tangent_s = f_contact_s - f_normal_s
            f_index = 0.26
            if f_tangent_r.norm() >= f_index * f_normal_r.norm():
                f_contact_r = f_normal_r + f_index * f_normal_r.norm() * f_tangent_r / f_tangent_r.norm()
            if f_tangent_s.norm() >= f_index * f_normal_s.norm():
                f_contact_s = f_normal_s + f_index * f_normal_s.norm() * f_tangent_s / f_tangent_s.norm()

            self.boundary_v[i, j, k] += self.dt * f_contact_s / self.boundary_m[i, j, k]
            self.solid_v[i, j, k] += self.dt * f_contact_r / self.solid_m[i, j, k]

    @ti.kernel
    def init_pml(self, Lx: ti.i32, Ly: ti.i32, Lz: ti.i32, alpha: ti.f64, beta: ti.f64):
        # ... (原有PML逻辑保持不变) ...
        for i, j, k in self.grid_damping:
            val_x = 0.0
            val_y = 0.0
            val_z = 0.0
            dist_x = min(i, self.grid_num - 1 - i)
            if dist_x < Lx:
                t = (Lx - dist_x) / Lx
                val_x = alpha * (t ** beta)
            dist_y = min(j, self.grid_num - 1 - j)
            if dist_y < Ly:
                t = (Ly - dist_y) / Ly
                val_y = alpha * (t ** beta)
            dist_z = min(k, self.grid_num - 1 - k)
            if dist_z < Lz:
                t = (Lz - dist_z) / Lz
                val_z = alpha * (t ** beta)
            self.grid_damping[i, j, k] = ti.max(val_x, val_y, val_z)

    @ti.kernel
    def substep(self, frame: ti.i32, acX: ti.f64):
        b = frame + 1
        self.solid_m.fill(0.0)
        self.solid_v.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.solid_v0.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.solid_f.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.solid_a.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.solid_acc.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.solid_gradient.fill(ti.Vector([0.0, 0.0, 0.0]))
        self.solid_num.fill(0.0)

        # ... (边界P2G逻辑保持不变) ...
        ti.loop_config(block_dim=self.block_dim_1)
        for p in range(self.boundry_particles[None]):
            base = (self.b_particles[p].position * self.inv_dx - 0.5).cast(int)
            if base[0] < 0 or base[1] < 0 or base[2] < 0 or base[0] >= self.grid_num - 2 or base[
                1] >= self.grid_num - 2 or base[2] >= self.grid_num - 2: continue
            self.solid_num[base[0], base[1], base[2]] = 1
            fx = self.b_particles[p].position * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                self.boundary_m[base + offset] += weight * self.b_particles[p].mass
                self.boundary_gradient[base + offset] += weight * self.b_particles[
                    p].mass * 4 * self.inv_dx * dpos * self.inv_dx

        # P2G (Solid: Ice and Rock)
        ti.loop_config(block_dim=self.block_dim_1)
        for p in range(self.n_s_particles[None]):
            base = (self.particles[p].position * self.inv_dx - 0.5).cast(int)
            if base[0] < 0 or base[1] < 0 or base[2] < 0 or base[0] >= self.grid_num - 2 or base[
                1] >= self.grid_num - 2 or base[2] >= self.grid_num - 2: continue
            fx = self.particles[p].position * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            # --- 核心修改：本构模型处理 ---
            stress = ti.Matrix.zero(float, 3, 3)
            if self.particles[p].material == 0:  # 冰 (Glen Law)
                # 1. 计算应变率张量 D (Strain Rate)
                # C 近似为速度梯度 L. D = 0.5(L + L^T)
                D_rate = 0.5 * (self.particles[p].C + self.particles[p].C.transpose())
                trace_D = D_rate.trace()
                D_dev = D_rate - trace_D / 3.0 * ti.Matrix.identity(float, 3)  # 偏应变率

                # 2. 有效应变率 (Eq. 14)
                # e_dot_e = sqrt(0.5 * D_dev : D_dev)
                e_dot = ti.sqrt(0.5 * (
                            D_dev[0, 0] ** 2 + D_dev[1, 1] ** 2 + D_dev[2, 2] ** 2 + 2 * D_dev[0, 1] ** 2 + 2 * D_dev[
                        0, 2] ** 2 + 2 * D_dev[1, 2] ** 2)) + 1e-15

                # 3. 损伤演化 (Eq. 17-19)
                old_D = self.particles[p].damage

                # 计算基准黏度 (未损伤)
                # term = A0 * exp(-Q/RT)
                term_A = self.A0 * ti.exp(-self.Q_act / (self.R_gas * self.Temp_K))
                # eta_bar = 0.5 * term^(-1/n) * e_dot^((1-n)/n)
                eta_bar = 0.5 * ti.pow(term_A, -1.0 / self.n_exp) * ti.pow(e_dot, (1.0 - self.n_exp) / self.n_exp)

                # 计算驱动损伤的剪应力 (Effective Shear Stress)
                # s_tilde = 2 * eta_bar * D_dev
                # tau_eff = sqrt(0.5 * s_tilde : s_tilde) = 2 * eta_bar * e_dot
                tau_eff = 2.0 * eta_bar * e_dot

                # 损伤准则
                omega = 0.0
                # Eq 19: max(0, tau_xy/(1-D) - tau_th). 这里用有效剪应力 tau_eff
                if tau_eff > self.tau_th:
                    omega = tau_eff - self.tau_th

                # 更新损伤
                new_D = old_D + self.dt * self.B_enhance * omega
                new_D = ti.min(ti.max(new_D, 0.0), 0.99)  # 限制最大损伤
                self.particles[p].damage = new_D

                # 4. 计算最终黏度 (Eq. 20: s = (1-D) * s_tilde)
                # 意味着黏度折减
                eta_final = eta_bar * (1.0 - new_D)
                eta_final = ti.min(eta_final, 1e9)  # 限制最大黏度防止数值爆炸

                # 5. 计算偏应力 s
                s_dev = 2.0 * eta_final * D_dev

                # 6. 计算压力 p (使用弹性EOS保持体积守恒)
                # K = E / 3(1-2nu)
                K_ice = self.particles[p].E_s / (3 * (1 - 2 * self.particles[p].nu_s))
                J = self.particles[p].F.determinant()
                p = -K_ice * (J - 1.0)  # 弱可压缩压力

                # 7. 合成Cauchy应力: sigma = s - pI
                stress_cauchy = s_dev + p * ti.Matrix.identity(float, 3)

                # MPM力计算需要 Kirchhoff 应力 (或者对应 Force Kernel 的形式)
                # stresssolid 变量在后续用于 grid force: f = stresssolid @ dpos
                # 此处我们将 J * sigma 存入 stresssolid
                stress = J * stress_cauchy

            else:  # 岩石 (原有的 Drucker-Prager)
                U, sig, V = ti.svd(self.particles[p].F)
                inv_sig = sig.inverse()
                e = ti.Matrix([[ti.log(sig[0, 0]), 0, 0], [0, ti.log(sig[1, 1]), 0], [0, 0, ti.log(sig[2, 2])]])
                stress = U @ (2 * self.particles[p].mu_s * inv_sig @ e + self.particles[
                    p].lambda_s * e.trace() * inv_sig) @ V.transpose()
                # Drucker-Prager 塑性部分在 G2P 结尾处理，此处计算的是 Trial Stress

            # 存储应力用于力汇聚
            self.particles[p].stresssolid = (-self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress @ self.particles[
                p].F.transpose()

            affine = self.particles[p].mass * self.particles[p].C
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                self.solid_v[base + offset] += weight * (
                            self.particles[p].mass * self.particles[p].velocity + affine @ dpos)
                self.solid_v0[base + offset] += weight * (
                            self.particles[p].mass * self.particles[p].velocity + affine @ dpos)  # 注意: 简化affine
                self.solid_m[base + offset] += weight * self.particles[p].mass
                self.solid_f[base + offset] += weight * self.particles[p].stresssolid @ dpos
                self.solid_gradient[base + offset] += weight * self.particles[
                    p].mass * 4 * self.inv_dx * dpos * self.inv_dx

        # Update Grids Momentum
        ti.loop_config(block_dim=self.block_dim_1)
        for i, j, k in self.solid_m:
            gravity = ti.Vector([0.0, -9.8, 0.0])
            if b < 10000:
                gravity = ti.Vector([0.0, -((9.8 / 10000) * b), 0.0])
            else:
                gravity = ti.Vector([0.0, -9.8, 0.0])

            if 20000 < b < 120000 and 15 < j < 20:
                self.solid_acc[i, j, k] = ti.Vector([acX, 0.0, 0.0])
            else:
                self.solid_acc[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

            if self.solid_m[i, j, k] > 0:
                self.solid_v[i, j, k] = (1 / self.solid_m[i, j, k]) * self.solid_v[i, j, k]
                self.solid_v[i, j, k] += self.dt * (
                            self.solid_acc[i, j, k] + gravity + self.solid_f[i, j, k] / self.solid_m[i, j, k])
                self.solid_v0[i, j, k] *= 1 / self.solid_m[i, j, k]
                self.solid_a[i, j, k] = (
                            self.solid_acc[i, j, k] + gravity + self.solid_f[i, j, k] / self.solid_m[i, j, k])

            # # PML
            # damping = self.grid_damping[i, j, k]
            # if damping > 0:
            #     decay = ti.exp(-damping)
            #     self.solid_v[i, j, k] *= decay

            # Boundary conditions
            if self.solid_m[i, j, k] > 0:
                if i < 17 or i > self.grid_num - 17: self.solid_v[i, j, k] = [0, 0, 0]
                if j < 13 and self.solid_v[i, j, k][1] < 0.0: self.solid_v[i, j, k] = [0, 0, 0]
                if j > self.grid_num - 6: self.solid_v[i, j, k] = [0, 0, 0]
                if k < 4 or k > self.grid_num - 4: self.solid_v[i, j, k] = [0, 0, 0]

        # G2P (Sand/Rock and Ice)
        param_flip_vel_adj = 0.97  # 0.0 是 APIC  0.97 是 ASFLIP
        param_flip_pos_adj_min = 0.0
        param_flip_pos_adj_max = 1.0
        param_part_col = 1.0
        ti.loop_config(block_dim=self.block_dim_1)
        for p in range(self.n_s_particles[None]):
            xp = self.particles[p].position
            base = (self.particles[p].position * self.inv_dx - 0.5).cast(int)
            if base[0] < 0 or base[1] < 0 or base[2] < 0 or base[0] >= self.grid_num - 2 or base[
                1] >= self.grid_num - 2 or base[2] >= self.grid_num - 2: continue
            fx = self.particles[p].position * self.inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(ti.f64, 3)
            new_a = ti.Vector.zero(ti.f64, 3)
            new_C = ti.Matrix.zero(ti.f64, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = offset.cast(float) - fx
                g_v_s = self.solid_v[base + offset]
                g_a_s = self.solid_a[base + offset]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v_s
                new_C += 4 * self.inv_dx * weight * g_v_s.outer_product(dpos)
                new_a += weight * g_a_s
            self.particles[p].F = (ti.Matrix.identity(float, 3) + self.dt * new_C) @ self.particles[p].F
            self.particles[p].a_p = new_a
            self.particles[p].J = self.particles[p].F.determinant()

            if param_flip_vel_adj > 0.0:
                vp = self.particles[p].velocity
                flip_pos_adj = param_flip_pos_adj_max
                # if not collided, check if the particle is separating
                if param_flip_pos_adj_min < flip_pos_adj:
                    logdJ = self.particles[p].C.trace() * self.dt
                    J = self.particles[p].J
                    if ti.log(max(1e-6, J)) + logdJ < -0.001:  # if not separating
                        flip_pos_adj = param_flip_pos_adj_min
                # interpolate to get old nodal velocity
                old_v = ti.Vector.zero(float, 3)
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k])
                    g_sv0 = self.solid_v0[base + offset]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    old_v += weight * g_sv0
                # apply generalized FLIP advection
                diff_vel = vp - old_v
                self.particles[p].velocity = new_v + param_flip_vel_adj * diff_vel
                self.particles[p].position = xp + (new_v + flip_pos_adj * param_flip_vel_adj * diff_vel) * self.dt
            else:
                # apply PIC advection
                self.particles[p].velocity = new_v
                self.particles[p].position = xp + new_v * self.dt
            self.particles[p].C = new_C
            self.particles[p].position += self.dt * self.particles[p].velocity

            # --- 塑性回归映射 (仅针对岩石 Material 1) ---
            if self.particles[p].material == 1:
                U, sig, V = ti.svd(self.particles[p].F)
                self.particles[p].E_s_tr = self.particles[p].E_s
                self.particles[p].c_C = self.particles[p].c_C0
                self.particles[p].phi = self.particles[p].phi
                Qnorm_max = 0.5 * max(abs(self.particles[p].F[0, 0] - self.particles[p].F[1, 1]),
                                      abs(self.particles[p].F[2, 2] - self.particles[p].F[1, 1]),
                                      abs(self.particles[p].F[0, 0] - self.particles[p].F[2, 2]), )
                YIy = Qnorm_max * 100 / 0.012
                self.particles[p].E_s_tr = self.particles[p].E_s / (1 + YIy ** (0.49 * YIy ** (-0.14)))
                eqp = ti.sqrt((2 / 3) * self.particles[p].q_s ** 2)
                self.particles[p].c_C = 0.1 * self.particles[p].c_C0 + (
                        self.particles[p].c_C0 - 0.1 * self.particles[p].c_C0) * ti.exp(-10 * eqp)

                e = ti.Matrix([[ti.log(sig[0, 0]), 0, 0], [0, ti.log(sig[1, 1]), 0], [0, 0, ti.log(sig[2, 2])]])
                e1 = e + self.particles[p].vc_s / 3 * ti.Matrix.identity(float, 3)  # volume correction treatment
                e1 += self.particles[p].c_C * (1.0 - self.particles[p].phi_s) / (
                            3 * self.particles[p].alpha_s) * ti.Matrix.identity(float,
                                                                                3)  # effects of cohesion
                ehat = e - e.trace() / 3 * ti.Matrix.identity(float, 3)
                Fnorm = ti.sqrt(ehat[0, 0] ** 2 + ehat[1, 1] ** 2 + ehat[2, 2] ** 2)  # Frobenius norm
                self.particles[p].mu_s = self.particles[p].E_s_tr / (2 * (1 + self.particles[p].nu_s))
                self.particles[p].lambda_s = self.particles[p].E_s_tr * self.particles[p].nu_s / (
                        (1 + self.particles[p].nu_s) * (1 - 2 * self.particles[p].nu_s))
                yp = Fnorm + (3 * self.particles[p].lambda_s + 2 * self.particles[p].mu_s) / (
                        2 * self.particles[p].mu_s) * e1.trace() * self.particles[
                         p].alpha_s  # delta gamma
                new_e = ti.Matrix.zero(float, 3, 3)
                dq = 0.0
                if Fnorm < 0 or e1.trace() > 0:  # Case II: 红
                    new_e = ti.Matrix.zero(float, 3, 3)
                    dq = ti.sqrt(e1[0, 0] ** 2 + e1[1, 1] ** 2 + e1[2, 2] ** 2)
                elif Fnorm == 0 or yp <= 0:  # Case I: 绿
                    new_e = e  # return initial matrix without vo lume correction and cohesive effect
                    dq = 0
                else:  # Case III: 蓝
                    new_e = e1 - yp / Fnorm * ehat
                    dq = yp
                new_e, dq = new_e, dq

                self.particles[p].q_s += dq
                phi = 0.5 * self.particles[p].phi + (self.particles[p].phi - 0.5 * self.particles[p].phi) * ti.exp(
                    -5 * eqp)
                phi = phi / 180 * self.pi
                sin_phi = ti.sin(phi)
                self.particles[p].alpha_s = ti.sqrt(2 / 3) * (2 * sin_phi) / (3 - sin_phi)
                new_F = U @ ti.Matrix([[ti.exp(new_e[0, 0]), 0, 0], [0, ti.exp(new_e[1, 1]), 0],
                                       [0, 0, ti.exp(new_e[2, 2])]]) @ V.transpose()
                self.particles[p].vc_s += -ti.log(new_F.determinant()) + ti.log(
                    self.particles[p].F.determinant())  # formula (26)
                self.particles[p].F = new_F

    def run(self, write_ply, frame):
        # ... (原有Run逻辑) ...
        start = time.time()
        self.init_pml(Lx=20, Ly=0, Lz=7, alpha=10, beta=1.0)
        for i in range(frame * 200, (frame + 1) * 200):
            if 20000 < i < 70000:
                acX = float(accX[:, 0][i - 20000])
            elif 70000 < i < 120000:
                acX = float(accX[:, 1][i - 70000])
            else:
                acX = 0
            self.substep(i, acX)

        if write_ply:
            # ... (输出逻辑) ...
            pos = self.particles.position.to_numpy()[:self.n_s_particles[None]]

            phi = self.particles.phi_s.to_numpy()[:self.n_s_particles[None]]
            vel = self.particles.velocity.to_numpy()[:self.n_s_particles[None]]
            np_material = self.particles.material.to_numpy()[:self.n_s_particles[None]]
            a_p = self.particles.a_p.to_numpy()[:self.n_s_particles[None]]

            damage = self.particles.damage.to_numpy()[:self.n_s_particles[None]]  # 新增输出
            c_c0 = self.particles.c_C0.to_numpy()[:self.n_s_particles[None]]
            rho = self.particles.rho.to_numpy()[:self.n_s_particles[None]]
            qs = self.particles.q_s.to_numpy()[:self.n_s_particles[None]]
            E_s = self.particles.E_s_tr.to_numpy()[:self.n_s_particles[None]]
            stress = self.particles.stresssolid.to_numpy()[:self.n_s_particles[None]]
            solid_index = np.array(np.where((np_material == 0) | (np_material == 1)))

            if len(solid_index[0]) > 0:
                writer = ti.tools.PLYWriter(num_vertices=len(solid_index[0]))
                writer.add_vertex_pos(pos[solid_index][0][:, 0], pos[solid_index][0][:, 1], pos[solid_index][0][:, 2])
                writer.add_vertex_channel("damage", "float", damage[solid_index][0])  # 输出损伤以便Paraview查看
                writer.add_vertex_channel("material", "float", np_material[solid_index][0])
                writer.add_vertex_channel("velocity0", "float", vel[solid_index][0][:, 0])
                writer.add_vertex_channel("velocity1", "float", vel[solid_index][0][:, 1])
                writer.add_vertex_channel("velocity2", "float", vel[solid_index][0][:, 2])
                writer.add_vertex_channel("phi", "float", phi[solid_index][0])
                writer.add_vertex_channel("qs", "float", qs[solid_index][0])
                writer.add_vertex_channel("c_C0", "float", c_c0[solid_index][0])
                writer.add_vertex_channel("E_s", "float", E_s[solid_index][0])
                writer.add_vertex_channel("Acc", "float", a_p[solid_index][0])
                writer.add_vertex_channel("stress00", "float", stress[solid_index][0][:, 0, 0])
                writer.add_vertex_channel("stress11", "float", stress[solid_index][0][:, 1, 1])
                writer.add_vertex_channel("stress22", "float", stress[solid_index][0][:, 2, 2])
                writer.add_vertex_channel("rho", "float", rho[solid_index][0])

                writer.export_frame(frame, 'G:\\YJS\\PLY' + '/1217-MPM.ply')  # 请修改为您的输出路径

        end = time.time()
        print('frame=' + str(frame), str(end - start) + "s")


if __name__ == "__main__":
    ti.init(arch=ti.gpu, device_memory_fraction=0.95, default_fp=ti.f64, default_ip=ti.i32, debug=False)
    mpm_solver = MPMSolver()
    mpm_solver.init_bear()
    mpm_solver.generate_bear()
    frame = 0
    while True:
        mpm_solver.run(1, frame)
        frame += 1
        if frame > 1000: break