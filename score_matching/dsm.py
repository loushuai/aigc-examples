import torch
import numpy as np
import torch.autograd as autograd
from tqdm import tqdm


def dsm_score_estimation(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    # scores = scorenet(perturbed_samples, torch.tensor([0]*samples.shape[0]).to(samples.device))
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss


# def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
#     used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
#     perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
#     target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
#     scores = scorenet(perturbed_samples, labels/len(sigmas))
#     target = target.view(target.shape[0], -1)
#     scores = scores.view(scores.shape[0], -1)
#     loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

#     return loss.mean(dim=0)


def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels/len(sigmas))
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)


# def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
#     used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
#     epsilon = torch.randn_like(samples) * used_sigmas
#     perturbed_samples = samples + epsilon
#     # target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
#     scores = scorenet(perturbed_samples, labels)
#     # target = target.view(target.shape[0], -1)
#     # scores = scores.view(scores.shape[0], -1)
#     # loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power
#     epsilon = epsilon.view(epsilon.shape[0], -1)
#     scores = scores.view(scores.shape[0], -1)
#     loss = (1. / (2. * used_sigmas.squeeze())) * ((scores + epsilon)**2.).sum(dim=-1)

#     return loss.mean(dim=0)


def Langevin_dynamics(x_mod, scorenet, n_steps=200, step_lr=0.00005):
    images = []

    labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
    labels = labels.long()

    with torch.no_grad():
        for _ in range(n_steps):
            images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
            noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
            grad = scorenet(x_mod, labels)
            x_mod = x_mod + step_lr * grad + noise
            x_mod = x_mod
            # print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

        return images


def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
    images = []

    with torch.no_grad():
        for c, sigma in tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            # labels = labels.long()
            labels = labels / len(sigmas)
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                # images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                # images.append(torch.clamp(x_mod, 0.0, 1.0).to(x_mod.device))
                images.append(torch.clamp(x_mod, -1.0, 1.0).to(x_mod.device))
                noise = torch.randn_like(x_mod) * torch.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                # x_mod = x_mod + step_size * grad + noise
                x_mod = x_mod + 1.0 * step_size * grad + 1.0 * noise
                # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                #                                                          grad.abs().max()))

        return images
