# Bringing PyTorch Monarch to AMD GPUs: Single-Controller Distributed Training on ROCm

April 22, 2026 by 
AMD: Chaojun Hou, Liz Li, Zachary Streeter, Xinyu Kang, Lei Zhang, Yuankai Chen, Yao Fu, Wen Chen, Zhenyu Gu, Andy Luo.
Meta: Matthias Reso, Hamid Shojanazeri, Monarch team

10 min read. | 2150 total words.

**Applications & models:** AI/ML, LLM, Kubernetes, PyTorch, SLURM
**Tags:** AI, Developers

Training state-of-the-art large language models (LLMs) with billions of parameters requires distributed training across hundreds or thousands of GPUs. At this scale, hardware failures are not exceptional events—they are expected. A single GPU memory error, network partition, or node crash can bring down an entire training run that has been progressing for days or weeks. While our previous work demonstrated linear scaling of FP8 training at scale (achieving 96.16% scaling efficiency on a 1024-GPU MI325 cluster with DeepSeekV3-671B), the key challenge remains: reliability at scale.

*(Note: PTC 2025 FP8 scaling chart omitted; refer to the previous blog for scaling efficiency details.)* To address these challenges, we have brought PyTorch Monarch to AMD Instinct GPUs with ROCm, expanding the single-controller model beyond CUDA environments and bringing this emerging runtime to a broader hardware ecosystem.

In this blog, we will explore the architecture of PyTorch Monarch, walk through the engineering effort required to port Monarch's GPU runtime and distributed communication stack to ROCm, and demonstrate how the system dynamically recovers from node failures without halting the entire training job. By the end, you will understand how Monarch enables elastic, fault-tolerant distributed training on AMD GPUs and why this represents a significant step toward stable, large-scale AI infrastructure.

## The Challenge: Reliability at Scale

Traditional fault-tolerance strategies rely heavily on periodic checkpointing: saving the full model state to persistent storage at regular intervals. When a failure occurs, the entire job restarts from the last checkpoint. While conceptually simple, this approach has significant drawbacks.

| Challenge | Impact |
| :--- | :--- |
| **Checkpoint overhead** | Writing hundreds of gigabytes of model state to storage consumes time and I/O bandwidth. |
| **Wasted computation** | All progress since the last checkpoint is lost upon failure. |
| **Cluster idle time** | The entire cluster sits idle while the failed node is replaced and the job restarts. |
| **Scalability limits** | As cluster size grows, the probability of failure during any checkpoint interval increases. |

For truly large-scale training, scaling is not enough—training must also recover from failures. We need a more dynamic approach, one that allows healthy nodes to continue training while failed nodes recover and rejoin, minimizing wasted computation and maximizing GPU utilization. This is where PyTorch Monarch comes in.

## What is PyTorch Monarch?

PyTorch Monarch introduces a new distributed programming paradigm that enables developers to orchestrate entire GPU clusters from a single Python program. With its actor-based runtime, process mesh abstraction, and asynchronous execution model, Monarch simplifies large-scale distributed training and enables complex workflows that combine training, evaluation, and reinforcement learning within one unified script.

The architecture operates at multiple distinct levels:
1. **Python API**: A single-program interface where developers write simple Python code to get distributed GPU execution.
2. **Monarch Runtime**: Manages actors and meshes, supervision trees, and tensor sharding.
3. **Rust Runtime (Tokio)**: Ensures high performance and memory safety.
4. **Infrastructure**: Integrates with RDMA, RCCL/NCCL, SLURM, Kubernetes, and SkyPilot.

![Monarch Architecture Overview](https://files.manuscdn.com/user_upload_by_module/session_file/310519663279446120/iudCixSZotosvIIw.png)
*Figure 1: PyTorch Monarch architecture decoupling Python API from the Rust runtime and infrastructure.*

By decoupling the parallelism strategy used within each training replica from the fault-tolerance mechanism used across replicas, Monarch provides a cleaner fault-tolerance model. Failures are isolated (actors have private state, crashes do not propagate), hierarchical (handled at the lowest possible level), and recovery is fast (seconds for local restart, minutes only if escalated).

![How Monarch Handles Failures](https://files.manuscdn.com/user_upload_by_module/session_file/310519663279446120/JaEtIYGkUwRswKmS.png)
*Figure 2: Monarch's hierarchical fault-handling model and supervision tree.*

## Porting Monarch to ROCm: Ecosystem Integration

Bringing Monarch to AMD GPUs required significant engineering effort to port the GPU runtime and distributed communication stack to ROCm.

We successfully implemented three main porting paths:

1. **Collective Communications**: We used `hipify_torch` to convert the C++ bridge code from CUDA to HIP and linked against RCCL, which mirrors NCCL's API.
2. **GPU Memory Management**: We extended the build system to auto-detect the platform and route CUDA driver API calls through their HIP equivalents.
3. **RDMA Integration**: Configuring `GPU_PLATFORM=rocm` keeps the `libibverbs`-based RDMA path intact while swapping the GPU-side bindings from CUDA to HIP for GPU-direct transfers.

![Porting Monarch to ROCm](https://files.manuscdn.com/user_upload_by_module/session_file/310519663279446120/SpXoskYABWpMmbjP.png)
*Figure 3: Porting Monarch from CUDA to ROCm via hipify_torch and auto-detection.*

Moreover, two cross-cutting issues shaped the port and deserve a closer look:

1. **No static link for the HIP runtime**: NVIDIA ships `libcudart_static.a`, so the CUDA path links `cudart_static` directly. ROCm ships no static equivalent for `libamdhip64`, so the ROCm build links `amdhip64` dynamically. Both platforms additionally `dlopen` the GPU driver API functions, including `hipMemCreate`, `cuMemCreate`, and related calls, keeping the runtime contract identical on either side.
2. **Rust compatibility shim instead of forking the bindings**: Once `hipify_torch` rewrites the C/C++ headers, `bindgen` emits HIP-named types such as `hipError_t`, `hipDeviceptr_t`, and `hipStream_t`. Rather than add `#ifdef` branches at every Rust call site, we added a `rocm_compat` module in `nccl-sys` and `rdmaxcel-sys` that re-exports HIP symbols under their CUDA names, for example `pub type cudaError_t = hipError_t` and `pub use hipSetDevice as cudaSetDevice`. The rest of the Rust code stays platform-agnostic.

These efforts culminated in the introduction of HIP type aliases in Rust, with all 1,171 tests passing, ensuring full support for ROCm 7.0+. We have upstreamed these contributions to the open-source community (see PR [#2393](https://github.com/meta-pytorch/monarch/pull/2393) and PR [#2891](https://github.com/meta-pytorch/monarch/pull/2891)).

Today, Monarch on ROCm provides full ecosystem support, including the Actor runtime, RDMA, Supervision, and Tensor sharding. It runs seamlessly on SLURM (HPC), Kubernetes (Cloud native), and SkyPilot (Multi-cloud), enabling downstream engines like TorchTitan (Training engine) and TorchFT (Fault tolerance) for production workloads.

## Case Study: Fault-Tolerant Training at Scale

To demonstrate the power of Monarch on AMD GPUs, we integrated it with TorchTitan and TorchFT to build a resilient, checkpoint-less distributed training architecture.

### Architecture Overview

The architecture consists of three layers:
- **Monarch**: Acts as the orchestrator, managing process and cluster orchestration. It spawns ReplicaActors and a Lighthouse service, organizing GPUs into Process Meshes.
- **TorchFT**: Handles fault tolerance at the step level. It contacts the Lighthouse for quorum coordination, performs Quorum AllReduce, and skips failed nodes.
- **TorchTitan**: Serves as the training engine, executing the Forward (FSDP), Backward, and Optimizer steps, while managing checkpoints and metrics.

![The Resilient Training Stack](https://files.manuscdn.com/user_upload_by_module/session_file/310519663279446120/QMHTZojitowgvDdm.png)
*Figure 4: The resilient training stack on AMD GPUs integrating Monarch, TorchFT, and TorchTitan.*

In this setup, Monarch provides a supervision tree for fine-grained fault detection and isolation. When a failure is injected into the training actors, it is detected by the Lighthouse and handled by TorchFT. The healthy replicas continue training independently despite peer failures, without requiring a global interruption.

### Dynamic Fault Recovery Workflow

Let us walk through a concrete scenario with four replica groups to understand the recovery workflow.

1. **Normal Training**: The OrchestrationManager spawns 4 ReplicaActors (Monarch Supervisors) and a Lighthouse. Each ReplicaActor spawns a Replica with 8 GPU processes running TorchTitan trainers. All 4 replicas are ready (`quorum_id=1`), and DiLoCo gradient synchronization occurs every 20 steps.
2. **Failure Detection**: A GPU process in Replica 0 crashes. The Monarch supervisor captures the `report_training_error` (with full traceback) before the process dies. Replicas 1, 2, and 3 are marked as unaffected and continue training.
3. **Local Restart**: ReplicaActor 0 initiates an in-place restart (`_stop_and_restart()`), stopping the old process mesh and spawning a new one. Meanwhile, the other 3 replicas continue syncing (`quorum_id=2`).
4. **Peer Checkpoint Transfer**: The Lighthouse selects Replica 1 as the donor. A peer checkpoint transfer (model, optimizer, scheduler, and trainer state) is initiated from Replica 1 to the recovering Replica 0. All replicas pause briefly at the quorum boundary while the new quorum forms.
5. **Resumed Training**: Once Replica 0 is synced, the new quorum (`quorum_id=3`) is established with all 4 replicas, and DiLoCo synchronization resumes.

![Dynamic Fault Recovery Workflow](https://files.manuscdn.com/user_upload_by_module/session_file/310519663279446120/MLKRPdfaGsoCafOM.png)
*Figure 5: Dynamic fault recovery workflow demonstrating peer checkpoint transfer without global checkpoint reload.*

The entire recovery process completes without any manual intervention, without full checkpoint restarts, and with minimal disruption to the overall training throughput.

## Performance Characteristics

We validated this approach on both SLURM and Kubernetes environments using AMD Instinct MI300-class clusters.

### SLURM 16-Node MI300 Cluster (128 GPUs)

We trained a Llama 3 8B model on a 16-node SLURM cluster (totaling 128 MI300 GPUs), injecting RCCL failures every 180 seconds with a quorum sync every 20 steps. The results were remarkable:
- The number of active workers fluctuated dynamically (between 8 and 16) due to the injected failures.
- Training continued seamlessly despite frequent failures—there was no full restart.
- The loss curve showed steady convergence, perfectly matching the baseline run without failure injection.

![Results on SLURM 16 Nodes MI300 Cluster](https://files.manuscdn.com/user_upload_by_module/session_file/310519663279446120/gIHYwPTvtsvjlAvy.png)
*Figure 6: Training continues despite frequent failures on a 16-node SLURM MI300 cluster. Note that not a single replica was down for 30+ minutes; recovery was fast, and different replicas were killed and recovered dynamically.*

### Kubernetes 32-Node MI355 Cluster (256 GPUs)

We scaled the experiment to a 32-node Kubernetes cluster (totaling 256 MI355 GPUs). The number of participants remained highly stable (fluctuating slightly between 30 and 32 during recovery events), and the global average loss decreased smoothly from 12 to approximately 4. This proves that the Monarch fault-tolerance model works flawlessly on both SLURM and Kubernetes at scale.

![Results on Kubernetes 32 Nodes MI355 Cluster](https://files.manuscdn.com/user_upload_by_module/session_file/310519663279446120/IkFnSaCXLgyVBNce.png)
*Figure 7: Stable recovery and smooth loss convergence on a 32-node Kubernetes MI355 cluster.*

## Summary and Future Directions

Training large AI models at scale requires more than raw computational power—it demands resilient infrastructure that can gracefully handle the inevitable hardware failures. By bringing PyTorch Monarch to AMD Instinct GPUs with ROCm, we have demonstrated a practical approach to fault-tolerant distributed training that minimizes wasted computation and maximizes GPU utilization.

This integration represents several significant achievements for large-scale training on AMD GPUs:
- **First large-scale validation on AMD hardware**: We successfully deployed Monarch with TorchTitan and TorchFT on AMD GPUs, demonstrating that the ROCm software stack fully supports advanced fault-tolerance mechanisms.
- **Cleaner fault-tolerance model**: Monarch provides a robust supervision tree and process mesh abstraction, isolating failures and enabling rapid local recovery.
- **Ecosystem readiness**: The approach works seamlessly across SLURM and Kubernetes, making it ready for production workloads.

The key architectural insight is the use of Monarch's actor-based runtime and supervision trees to isolate failures, combined with TorchFT's quorum-based synchronization to allow healthy nodes to continue training. For teams running large-scale training workloads on AMD GPUs, this integration offers a path toward more stable, efficient, and cost-effective model development.

Looking ahead, our next steps include:
- Extending NIC support and improving runtime performance.
- Expanding Monarch to support more pre-training and reinforcement learning (RL) frameworks on ROCm.
- Further optimizing fault tolerance performance, specifically reducing rejoin reload latency and overlapping recovery with compute.
- Continuing our open-source collaboration with the PyTorch community.

## Additional Resources

- [PyTorch Monarch GitHub Repository](https://github.com/meta-pytorch/monarch)
- [TorchTitan Documentation](https://github.com/pytorch/torchtitan)
- [TorchFT GitHub Repository](https://github.com/pytorch/torchft)
- [Resilient Large-Scale Training: Integrating TorchFT with TorchTitan on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/primus-torchft/README.html)

## Disclaimers

Third-party content is licensed to you directly by the third party that owns the content and is not licensed to you by AMD. ALL LINKED THIRD-PARTY CONTENT IS PROVIDED “AS IS” WITHOUT A WARRANTY OF ANY KIND. USE OF SUCH THIRD-PARTY CONTENT IS DONE AT YOUR SOLE DISCRETION AND UNDER NO CIRCUMSTANCES WILL AMD BE LIABLE TO YOU FOR ANY THIRD-PARTY CONTENT. YOU ASSUME ALL RISK AND ARE SOLELY RESPONSIBLE FOR ANY DAMAGES THAT MAY ARISE FROM YOUR USE OF THIRD-PARTY CONTENT.

---
*© 2026 Advanced Micro Devices, Inc.*
