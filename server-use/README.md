# 服务器使用指南

## 第一部分：NUS VPN 安装

1. **安装指南**（需 NUS 账号登录查看）：https://nusit-dwp.onbmc.com/dwp/app/#/knowledge/KBA00027608/rkm

2. **账号信息**
   - 账号：`e1520673@u.nus.edu`
   - 密码：**[保管好自己的密码，勿上传到 Git]**

3. **MFA 认证**：安装指南中有一步需要 Microsoft Authenticator 验证。如果是他人使用，请先尝试自行完成认证；如果无法通过，则每次需要由本人在手机上确认验证。

---

## 第二部分：HPC (Hopper) 使用

> **维护通知 (2026-04-13 ~ 04-20)：** /scratch 存储维护，4月10日(周五) 20:00 队列停止，4月13日起无法登录，4月20日恢复。请提前备份重要数据。

### 登录

```bash
ssh e1520673@hopper.nus.edu.sg
# 输入你的 HPC 密码（勿上传明文密码到 Git）
```

Web Portal: https://hpcportal.nus.edu.sg/

## 账号状态（已确认可用）

| 项目 | 总额度 | 剩余 GPU 小时 | 用途 |
|---|---|---|---|
| **CFP04-CF-095** | 5,000,000 | ~49,989 小时 | 额度充足，日常优先用这个 |
| **CFP04-SF-109** | 1,000,000 | ~9,812 小时 | 备用 |

```bash
hpc project    # 查看最新额度
```

## 集群配置

- **节点数量:** 40 个 GPU 节点（hopper-07 ~ hopper-46）
- **每节点:** 8x NVIDIA H200 (143,771 MiB / ~141 GB) / 112 CPU 核 / 2 TB 内存
- **总 GPU 数:** 320 块 H200
- **驱动:** NVIDIA 575.57.08 / CUDA 13.0
- **每 GPU 分配:** 12 CPU + 225 GB 内存（自动分配，不用手动设）

### 作业限制

| 类型 | GPU 数/作业 | 最大同时运行 | 最长运行时间 |
|---|---|---|---|
| interactive | 1~2 | 1 个 | 6 小时 |
| small (batch) | 1~2 | 2 个 | 144 小时（6天） |
| medium (batch) | 3~7 | 2 个 | 96 小时（4天） |
| large (batch) | 8~16 | 1 个 | 48 小时（2天） |

队列自动路由，不需要在脚本里指定 queue。

```bash
hpc gstat    # 查看集群实时负载和空闲 GPU
```

## 工作目录

| 路径 | 空间 | 备份 | 说明 |
|---|---|---|---|
| `/home/svu/e1520673` | 20 GB | 有（10天快照） | 放代码和配置 |
| `/hpctmp` | 500 GB | 无 | 放数据和中间结果，**超过 60 天自动删除** |

```bash
hpc s     # 查看磁盘配额
```

## 提交 GPU 作业

### 交互式（调试用，最长 6 小时）

```bash
# 1 块 H200
qsub -I -P CFP04-CF-095 -l select=1:ngpus=1 -l walltime=01:00:00

# 2 块 H200
qsub -I -P CFP04-CF-095 -l select=1:ngpus=2 -l walltime=01:00:00
```

进入节点后：
```bash
module load singularity
singularity exec -e /app1/common/singularity-img/hopper/pytorch/pytorch_2.8.0_cuda_13.0.sif bash

nvidia-smi          # 确认分配到几块 GPU
python train.py     # 运行代码
```

### 批处理（训练用）

创建 `job.pbs`（以 1 卡为例）：
```bash
#!/bin/bash
#PBS -P CFP04-CF-095
#PBS -j oe
#PBS -k oed
#PBS -N my_training_job
#PBS -l walltime=24:00:00
#PBS -l select=1:ngpus=1

cd $PBS_O_WORKDIR

image="/app1/common/singularity-img/hopper/pytorch/pytorch_2.8.0_cuda_13.0.sif"
module load singularity

singularity exec $image bash << EOF > stdout.$PBS_JOBID 2> stderr.$PBS_JOBID

python train.py

EOF
```

改卡数只需要改 `select` 那一行：
```bash
#PBS -l select=1:ngpus=1     # 1 卡（12 CPU / 225GB）   — small，最长 144 小时
#PBS -l select=1:ngpus=2     # 2 卡（24 CPU / 450GB）   — small，最长 144 小时
#PBS -l select=1:ngpus=4     # 4 卡（48 CPU / 900GB）   — medium，最长 96 小时
#PBS -l select=1:ngpus=7     # 7 卡（84 CPU / 1.5TB）   — medium，最长 96 小时
#PBS -l select=1:ngpus=8     # 8 卡（整个节点 2TB）     — large，最长 48 小时
#PBS -l select=2:ngpus=8     # 16 卡（跨 2 节点）       — large，最长 48 小时
```

提交和管理：
```bash
qsub job.pbs              # 提交
qstat -u $USER            # 查看状态
qgpu_smi <JOB_ID>         # 查看 GPU 负载
qdel <JOB_ID>             # 取消作业
```

> **不要在代码里设 `CUDA_VISIBLE_DEVICES`，PBS 自动分配 GPU。**

## 可用容器

### PyTorch
| 镜像 | 路径 |
|---|---|
| PyTorch 2.8.0 + CUDA 13.0 | `/app1/common/singularity-img/hopper/pytorch/pytorch_2.8.0_cuda_13.0.sif` |
| PyTorch 2.7.0 + CUDA 12.8 | `/app1/common/singularity-img/hopper/pytorch/pytorch_2.7.0_cuda_12.8.sif` |
| PyTorch 2.6.0 + CUDA 12.8 | `/app1/common/singularity-img/hopper/pytorch/pytorch_2.6.0_cuda_12.8.sif` |
| Unsloth (PyTorch 2.8) | `/app1/common/singularity-img/hopper/pytorch/unslothtorch28.sif` |

### CUDA
| 镜像 | 路径 |
|---|---|
| CUDA 12.4.1 + cuDNN | `/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif` |
| CUDA 12.1.1 + cuDNN8 | `/app1/common/singularity-img/hopper/cuda/cuda_12.1.1-cudnn8-devel-ubuntu22.04.sif` |

其他容器（TensorFlow、AlphaFold）在 `/app1/common/singularity-img/hopper/` 下。

## 常用命令

```bash
hpc                  # 所有可用命令
hpc ai               # AI 作业提交说明
hpc project          # 项目额度
hpc gstat            # 集群状态
hpc s                # 磁盘配额

qsub job.pbs         # 提交作业
qstat -u $USER       # 查看自己的作业
qstat -awn1          # 所有作业（含节点信息）
qstat -fx <JOB_ID>   # 作业详情
qgpu_smi <JOB_ID>    # 运行中作业的 GPU 负载
qdel <JOB_ID>        # 删除作业
```

## 文件传输

```bash
# 上传
scp local_file.py e1520673@hopper.nus.edu.sg:/hpctmp/

# 下载
scp e1520673@hopper.nus.edu.sg:/hpctmp/results.tar.gz ./
```

Windows：校内拖拽到 U: 盘或访问 `\\hpcnas.nus.edu.sg\svu\e1520673`，校外用 FileZilla SFTP。

## VS Code Remote SSH（推荐工作流）

> 最舒服的方式：本地 VS Code + Copilot 直接编辑和运行远程代码，消除文件同步和网络延迟

### 快速开始

1. **VS Code 安装扩展**
   - Extension ID: `ms-vscode-remote.remote-ssh`
   - 或在命令面板输入 `Extensions: Install Extensions`，搜索 "Remote - SSH"

2. **连接 hopper**
   - 命令面板：`Remote-SSH: Connect to Host...`
   - 选择 `hopper`
   - 首次连接时接受主机指纹（会弹出确认框）
   - 自动免密登录

3. **开发工作流**
   - 左下角显示 `SSH: hopper`，代表已连接
   - `Ctrl + ~` 打开远程终端
   - 文件自动保存到 `/home/svu/e1520673`

### 快速任务示例

#### 交互式调试（6小时内）
```bash
# 终端里直接运行
qsub -I -P CFP04-CF-095 -l select=1:ngpus=1 -l walltime=01:00:00

# 进入节点后（VS Code 里直接编辑和运行）
module load singularity
singularity exec -e /app1/common/singularity-img/hopper/pytorch/pytorch_2.8.0_cuda_13.0.sif bash

nvidia-smi
python your_script.py
```

#### 长期作业（创建 job.pbs → 提交）
```bash
# VS Code 里新建 job.pbs，改数字后提交
qsub job.pbs

# 监控
qstat -u $USER
qgpu_smi <JOB_ID>
```

### 配置说明

- SSH 私钥已生成并配置：`~/.ssh/hopper`
- SSH 配置文件：`~/.ssh/config`（主机别名 hopper 可直连）
- 如需重新配置，运行命令：`ssh-keygen -t ed25519 -f ~/.ssh/hopper -N ""`

---

## 遇到问题？

- HPC 主页：https://nusit.nus.edu.sg/hpc/
- 提交工单：https://ask.nus.edu.sg/（附上报错信息）
- nTouch：搜索 "HPC Enquiries"
- Remote SSH 故障排除：https://code.visualstudio.com/docs/remote/troubleshooting
