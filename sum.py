#!/usr/bin/env python3
import torch

if __name__ == "__main__":    
    backends = ['cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 've', 'fpga', 'maia', 'xla', 'lazy', 'vulkan', 'mps', 'meta', 'hpu', 'mtia']
    
    print("=" * 80)
    print("Testing Addition (A + B)")
    print("=" * 80)
    for b in backends:
        try:
            r = torch.device(b)
            A = torch.zeros((1, 1024), device=r)
            B = torch.zeros((1, 1024), device=r)
            print(f"{b}: {A + B}")
        except Exception as e:
            print(f"ERROR {b}: {e}")
    
    print("\n" + "=" * 80)
    print("Testing Matrix Multiplication (A @ B or A * B)")
    print("=" * 80)
    for b in backends:
        try:
            r = torch.device(b)
            # Matrix multiplication: (M, K) @ (K, N) = (M, N)
            A = torch.randn((64, 128), device=r)
            B = torch.randn((128, 64), device=r)
            C = torch.matmul(A, B)  # or A @ B
            print(f"{b}: matmul successful, result shape: {C.shape}")
            
            # Element-wise multiplication
            D = torch.randn((64, 64), device=r)
            E = torch.randn((64, 64), device=r)
            F = D * E
            print(f"{b}: element-wise mul successful, result shape: {F.shape}")
        except Exception as e:
            print(f"ERROR {b}: {e}")
    
    # Specific Vulkan example
    print("\n" + "=" * 80)
    print("Detailed Vulkan Backend Example")
    print("=" * 80)
    try:
        # Check if Vulkan is available
        if not torch.is_vulkan_available():
            print("WARNING: Vulkan backend is not available!")
            print("PyTorch needs to be compiled with USE_VULKAN=1")
        else:
            print("✓ Vulkan backend is available!")
            
            # Create tensors on CPU first
            A_cpu = torch.randn((4, 5))
            B_cpu = torch.randn((5, 3))
            
            print(f"\nA (CPU):\n{A_cpu}")
            print(f"\nB (CPU):\n{B_cpu}")
            
            # Move to Vulkan
            A_vulkan = A_cpu.vulkan()
            B_vulkan = B_cpu.vulkan()
            
            print(f"\nA device: {A_vulkan.device}")
            print(f"B device: {B_vulkan.device}")
            
            # Matrix multiplication on Vulkan
            C_vulkan = torch.matmul(A_vulkan, B_vulkan)
            
            # Move result back to CPU for printing
            C_cpu = C_vulkan.cpu()
            
            print(f"\nC = A @ B (result):\n{C_cpu}")
            print(f"Result shape: {C_cpu.shape}")
            
            # Verify correctness
            C_expected = torch.matmul(A_cpu, B_cpu)
            if torch.allclose(C_cpu, C_expected, rtol=1e-3):
                print("\n✓ Vulkan computation matches CPU result!")
            else:
                print("\n✗ Warning: Results differ from CPU computation")
                print(f"Max difference: {torch.max(torch.abs(C_cpu - C_expected))}")
                
    except Exception as e:
        print(f"ERROR with Vulkan: {e}")
        import traceback
        traceback.print_exc()    
