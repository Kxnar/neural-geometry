import pytest
import torch
from neural_geometry.geometry import sdf, surface_points, SHAPES, feature_probes
from neural_geometry.models import Field


@pytest.mark.parametrize("shape", SHAPES)
def test_surface_samples_are_on_analytic_surface(shape):
    g = torch.Generator().manual_seed(42)
    p = surface_points(1000, shape, 0.08, g, torch.device("cpu"))
    assert sdf(p, shape, 0.08).abs().max() < 2e-6


@pytest.mark.parametrize("shape", SHAPES)
def test_feature_probes_have_correct_sign(shape):
    inside, outside = feature_probes(shape, 0.04, torch.device("cpu"))
    assert (sdf(inside, shape, 0.04) < 0).all()
    assert (sdf(outside, shape, 0.04) > 0).all()


@pytest.mark.parametrize("kind", ["siren", "fourier"])
def test_model_gradients_and_tiny_fit(kind):
    torch.set_num_threads(2)
    torch.manual_seed(12)
    model = Field(kind, width=32, depth=2)
    points = torch.rand(128, 3) * 2 - 1
    target = sdf(points, "sphere")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = (model(points) - target).square().mean().item()
    for _ in range(150):
        optimizer.zero_grad()
        loss = (model(points) - target).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert loss.item() < initial * 0.25


def test_analytic_distances():
    p = torch.tensor([[0., 0., 0.], [0.55, 0., 0.], [1., 0., 0.]])
    assert torch.allclose(sdf(p, "sphere").flatten(), torch.tensor([-0.55, 0., 0.45]), atol=1e-6)
    assert sdf(torch.zeros(1, 3), "double_sphere", 0.08).item() == pytest.approx(0.04, abs=1e-6)
