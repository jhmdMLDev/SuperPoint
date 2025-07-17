import torch


def NCCLoss(input, target):
    # subtract the mean of each image
    input_mean = torch.mean(input)
    target_mean = torch.mean(target)
    input = input - input_mean
    target = target - target_mean

    # calculate the numerator and denominator of the NCC formula
    numerator = torch.sum(input * target)
    input_norm = torch.norm(input)
    target_norm = torch.norm(target)
    denominator = input_norm * target_norm

    # calculate the NCC
    NCC = numerator / denominator

    return 1 - NCC


if __name__ == "__main__":
    x1 = torch.randn((512, 512))
    x2 = torch.randn((512, 512))
    loss = NCCLoss(x1, x2)
    print(loss)
