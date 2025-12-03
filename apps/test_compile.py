import torch

def foo(x1, x2):
    a = torch.neg(x1)
    b = torch.maximum(x2, a)
    y = torch.cat([b], dim=0)
    return y

if __name__ == "__main__":
    x1 = torch.randint(256, (1, 8), device=torch.device('xpu'), dtype=torch.uint8)
    x2 = torch.randint(256, (8390, 8), device=torch.device('xpu'), dtype=torch.uint8)

    # compiled_foo = torch.compile(foo, backend="inductor", options={"triton.cudagraphs": False})
    compiled_foo = torch.compile(foo, backend="eager")

    result = None
    for _ in range(1000):
        result = compiled_foo(x1, x2)

    print(result)

    # TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1 TORCH_INSTALL_PATH=$HOME/pytorch/install/standalone_xpu_vulkan PYTHONPATH=$HOME/pytorch:$PYTHONPATH python app/compile_test.py
