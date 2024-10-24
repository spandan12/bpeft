import torch
import torch.nn.functional as F

def get_device():
    device = 0#args.device
    return device






















def get_confidence(output, label):
    conf = []
    with torch.no_grad():
        evidence = get_evidence(output)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim = True)
        probs = alpha / S
        # print(probs.shape, label.shape)
        probs = probs.cpu().numpy()
        label = label.cpu().numpy()
        for ind, l in enumerate(label):
            conf.append(probs[ind, l])
        
    return conf

def get_vacuity(output):
    num_classes = 100 # Cifar100
    evidence = get_evidence(output)
    alpha = evidence + 1

    S = torch.sum(alpha, dim=1)
    
    with torch.no_grad():
        vacuity = num_classes / S.detach()
    
    return vacuity