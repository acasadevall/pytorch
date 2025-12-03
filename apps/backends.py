import torch

if __name__ == "__main__":    
    backends = ['cpu', 'cuda', 'ipu', 'xpu', 'mkldnn', 'opengl', 'opencl', 'ideep', 'hip', 've', 'fpga', 'maia', 'xla', 'lazy', 'vulkan', 'mps', 'meta', 'hpu', 'mtia']
    
    for b in backends:
        try:
            r = torch.randn((), device=torch.device(b), dtype=torch.float)
            print(f"OK: {b}")
        except:
            print(f"ERROR: {b}")
