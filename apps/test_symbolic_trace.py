if __name__ == "__main__":
    import torch
    from torch.fx import symbolic_trace

    def f(x):
        return torch.relu(x + 2)

    traced = symbolic_trace(f)
    print(traced.graph)
