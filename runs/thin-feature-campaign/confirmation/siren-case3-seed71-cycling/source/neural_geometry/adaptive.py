"""Error-prioritised replay on top of the ordinary mixed sampler.

Every step queries a full fresh batch. A quarter of gradient examples are replaced
by labelled replay points; discarded fresh labels still count against the budget.
Priorities are refreshed every 50 steps using additional forward passes.
"""
import torch
from .geometry import sample


class ErrorReplay:
    def __init__(self, shape, feature, generator, device, fraction=0.25,
                 capacity=4096, refresh=50):
        self.shape, self.feature = shape, feature
        self.generator, self.device = generator, device
        self.fraction, self.capacity, self.refresh = fraction, capacity, refresh
        self.points = self.targets = self.weights = None
        self.label_queries = self.scoring_points = self.replayed_points = 0

    @torch.no_grad()
    def batch(self, model, n, step):
        fresh, truth = sample(n, self.shape, self.feature, "mixed", self.generator, self.device)
        self.label_queries += n
        # Construct the gradient batch before admitting current fresh points to memory.
        if self.points is not None:
            replay_count = int(n * self.fraction)
            index = torch.multinomial(self.weights, replay_count, replacement=True,
                                      generator=self.generator)
            order = torch.randperm(n, generator=self.generator, device=self.device)[:n - replay_count]
            batch = torch.cat((fresh[order], self.points[index]))
            target = torch.cat((truth[order], self.targets[index]))
            self.replayed_points += replay_count
        else:
            batch, target = fresh, truth
        if step == 1 or (step - 1) % self.refresh == 0:
            points = fresh if self.points is None else torch.cat((self.points, fresh))
            targets = truth if self.targets is None else torch.cat((self.targets, truth))
            errors = (model(points) - targets).abs().flatten()
            self.scoring_points += len(points)
            index = errors.topk(min(self.capacity, len(points))).indices
            self.points, self.targets = points[index], targets[index]
            self.weights = errors[index].clamp_min(1e-6)
        return batch, target
