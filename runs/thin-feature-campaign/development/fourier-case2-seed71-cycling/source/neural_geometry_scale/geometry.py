"""Exact disjoint-box SDFs and paired, area-uniform sampling controls.

Local scale is an oracle box half-thickness, capped by half the gap on inward
parallel-plate faces. It is not an estimated medial-axis local feature size.
"""
from dataclasses import dataclass
import math

import torch


POLICIES = ("global_narrow", "global_wide", "multiscale", "local")
SCENES = ("mixed_thickness", "parallel_plates")


@dataclass(frozen=True)
class Scene:
    name: str
    feature: float
    angle: float = 0.0

    def __post_init__(self):
        if self.name not in SCENES or not math.isfinite(self.feature) or not .005 <= self.feature <= .08:
            raise ValueError("Unknown scene or feature outside [0.005, 0.08]")
        if not math.isfinite(self.angle):
            raise ValueError("Angle must be finite")

    def boxes(self, device="cpu", dtype=torch.float32):
        if self.name == "mixed_thickness":
            centers = [[-.43, 0, 0], [.43, 0, 0]]
            halves = [[.24, .5, self.feature], [.24, .5, .12]]
        else:
            centers = [[0, 0, -(.1 + self.feature / 2)], [0, 0, .1 + self.feature / 2]]
            halves = [[.6, .6, .1], [.6, .6, .1]]
        return (torch.tensor(centers, device=device, dtype=dtype),
                torch.tensor(halves, device=device, dtype=dtype))

    def rotation(self, points):
        a = math.radians(self.angle)
        return points.new_tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                                  [-math.sin(a), 0, math.cos(a)]])

    def sdf(self, points):
        local = points @ self.rotation(points)
        centers, half = self.boxes(points.device, points.dtype)
        q = (local[:, None] - centers).abs() - half
        distances = q.clamp_min(0).norm(dim=-1) + q.amax(-1).clamp_max(0)
        return distances.amin(-1, keepdim=True)

    def surface(self, n, generator, device="cpu"):
        """Area-uniform samples, component IDs and oracle scales before rotation."""
        centers, half = self.boxes(device)
        # Flatten component x normal-axis x face-sign; each area is 4*h_j*h_k.
        areas = (4 * half.prod(-1, keepdim=True) / half).repeat_interleave(2, dim=1).flatten()
        face = torch.multinomial(areas, n, replacement=True, generator=generator)
        component, axis = face // 6, (face % 6) // 2
        sign = (face % 2) * 2 - 1
        points = (torch.rand(n, 3, generator=generator, device=device) * 2 - 1) * half[component]
        points[torch.arange(n, device=device), axis] = sign * half[component, axis]
        points += centers[component]
        scale = half[component].amin(-1)
        if self.name == "parallel_plates":
            inward = (axis == 2) & (((component == 0) & (sign == 1)) | ((component == 1) & (sign == -1)))
            scale = torch.where(inward, scale.clamp_max(self.feature / 2), scale)
        return points @ self.rotation(points).T, component, scale

    @property
    def scale_bounds(self):
        return (self.feature, .12) if self.name == "mixed_thickness" else (self.feature / 2, .1)

    def probes(self, device="cpu"):
        centers, half = self.boxes(device)
        axis = torch.linspace(-.8, .8, 16, device=device)
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        plane = torch.stack((x.flatten(), y.flatten(), torch.zeros_like(x).flatten()), -1)
        inside = [plane * half[k] + centers[k] for k in range(2)]
        if self.name == "parallel_plates":
            gap = plane * half[0]
        else:
            gap = plane.clone()
            gap[:, 0], gap[:, 1], gap[:, 2] = 0, x.flatten() * .4, y.flatten() * self.feature * .8
        return {**{f"component_{k}": p @ self.rotation(p).T for k, p in enumerate(inside)},
                "gap": gap @ self.rotation(gap).T}


def batch(scene, n, policy, generator, device="cpu"):
    if policy not in POLICIES or n < 2:
        raise ValueError("Valid policy and batch size >=2 required")
    points = torch.rand(n, 3, generator=generator, device=device) * 2 - 1
    count = n // 2
    surface, components, local = scene.surface(count, generator, device)
    noise = torch.randn(count, 3, generator=generator, device=device)
    # Draw in every arm, including fixed-scale arms, to preserve RNG pairing.
    u = torch.rand(count, generator=generator, device=device)
    low, high = scene.scale_bounds
    scale = {"global_narrow": torch.full_like(local, low),
             "global_wide": torch.full_like(local, high),
             "multiscale": low * (high / low) ** u,
             "local": local}[policy]
    near = surface + noise * scale[:, None] / 2
    clipped = (near.abs() > 1).any(-1)
    points[:count] = near.clamp(-1, 1)
    return points, scene.sdf(points), {"components": components, "scales": scale,
                                      "clipped": clipped}
