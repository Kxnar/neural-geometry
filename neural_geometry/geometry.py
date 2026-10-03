"""Analytic SDFs in [-1,1]^3; negative inside, zero on the surface."""
import torch


SHAPES = ("sphere", "torus", "thin_box", "double_sphere")


def sdf(points, shape="torus", feature=0.08):
    if shape not in SHAPES:
        raise ValueError(f"Unknown shape: {shape}")
    if not 0.005 <= feature <= 0.25:
        raise ValueError("feature must be in [0.005, 0.25]")
    if shape == "sphere":
        return points.norm(dim=-1, keepdim=True) - 0.55
    if shape == "torus":
        radial = points[:, :2].norm(dim=-1) - 0.55
        return torch.stack((radial, points[:, 2]), -1).norm(dim=-1, keepdim=True) - feature
    if shape == "thin_box":
        half_size = points.new_tensor([0.6, 0.6, feature])
        q = points.abs() - half_size
        return q.clamp_min(0).norm(dim=-1, keepdim=True) + q.amax(-1, keepdim=True).clamp_max(0)
    # Disjoint spheres: exact signed distance to their union. feature is gap width.
    center = 0.3 + feature / 2
    a = points - points.new_tensor([center, 0, 0])
    b = points + points.new_tensor([center, 0, 0])
    return torch.minimum(a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)) - 0.3


def surface_points(n, shape, feature, generator, device):
    """Sample analytic surface points. Torus angles are uniform, not surface area."""
    def uniform(*size):
        return torch.rand(*size, generator=generator, device=device)
    if shape in ("sphere", "double_sphere"):
        p = torch.randn(n, 3, generator=generator, device=device)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if shape == "sphere":
            return p * 0.55
        p = p * 0.3
        p[:, 0] += torch.where(uniform(n) < 0.5, -1.0, 1.0) * (0.3 + feature / 2)
        return p
    if shape == "torus":
        a, b = uniform(n) * (2 * torch.pi), uniform(n) * (2 * torch.pi)
        radius = 0.55 + feature * b.cos()
        return torch.stack((radius * a.cos(), radius * a.sin(), feature * b.sin()), -1)
    if shape == "thin_box":
        half = torch.tensor([0.6, 0.6, feature], device=device)
        p = (uniform(n, 3) * 2 - 1) * half
        # Choose a face proportional to its area.
        weights = half.prod() / half
        axis = torch.multinomial(weights, n, replacement=True, generator=generator)
        sign = torch.where(uniform(n) < 0.5, -1.0, 1.0)
        p[torch.arange(n, device=device), axis] = half[axis] * sign
        return p
    raise ValueError(shape)


def sample(n, shape, feature, mode, generator, device):
    if mode not in ("uniform", "mixed"):
        raise ValueError(mode)
    points = torch.rand(n, 3, generator=generator, device=device) * 2 - 1
    if mode == "mixed":
        count = n // 2
        surface = surface_points(count, shape, feature, generator, device)
        noise = torch.randn(count, 3, generator=generator, device=device) * (feature / 2)
        points[:count] = (surface + noise).clamp(-1, 1)
    return points, sdf(points, shape, feature)


def feature_probes(shape, feature, device):
    """Known interior/exterior probes; diagnostic, NOT a topology certificate."""
    if shape == "torus":
        a = torch.linspace(0, 2 * torch.pi, 257, device=device)[:-1]
        inside = torch.stack((0.55 * a.cos(), 0.55 * a.sin(), a * 0), -1)
        outside = torch.stack(((0.55 + 1.5 * feature) * a.cos(),
                               (0.55 + 1.5 * feature) * a.sin(), a * 0), -1)
    elif shape == "thin_box":
        t = torch.linspace(-0.5, 0.5, 16, device=device)
        x, y = torch.meshgrid(t, t, indexing="ij")
        inside = torch.stack((x.flatten(), y.flatten(), torch.zeros_like(x).flatten()), -1)
        outside = inside.clone()
        outside[:, 2] = 1.5 * feature
    elif shape == "double_sphere":
        inside = torch.tensor([[-0.3-feature/2, 0, 0], [0.3+feature/2, 0, 0]], device=device)
        t = torch.linspace(-0.4 * feature, 0.4 * feature, 128, device=device)
        outside = torch.stack((t, t * 0, t * 0), -1)
    else:
        inside = torch.tensor([[0., 0., 0.]], device=device)
        outside = torch.tensor([[0.7, 0., 0.]], device=device)
    return inside, outside
