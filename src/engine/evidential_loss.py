import torch
import torch.nn.functional as F
class Evidential_loss():
    def __init__(self, args ={'unc_act':'exp', 'ev_unc_type':'log', 'use_vac_reg':True, 'kl_strength':10000.0}):
        '''
        self.unc_act ['relu', 'softplus', 'exp', 'selu']
        self.ev_unc_type ['mse', 'ce', 'log']
        '''
        self.args_evid = args

    
    def get_evidence(self, y):
        if self.args_evid['unc_act'] == 'relu':
            return F.relu(y)
        elif self.args_evid['unc_act'] == 'softplus':
            return F.softplus(y)
        elif self.args_evid['unc_act'] == 'exp':
            return torch.exp(y)
        elif self.args_evid['unc_act'] == 'none':
            return y
        elif self.args_evid['unc_act'] == 'elushift':
            return torch.nn.ELU()(y) + 1
        else:
            print("The evidence function is not accurate.")
            raise NotImplementedError()
        
    
    def kl_divergence(self, alpha, num_classes):
        device = alpha.device

        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl


    def mse_loss(self, target, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((target - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood
    
    def edl_loss(self, func, target, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(target * (func(S) - func(alpha)), dim=1, keepdim=True)
        
        return A
        
    def one_hot_embedding(self, labels, num_classes=10):
        # Convert to One Hot Encoding
        y = torch.eye(num_classes).to(labels.device)
        return y[labels]

    def edl_overall_loss(self, output, target, epoch_num=1, annealing_step=1):
       
        if output.device != target.device:
            print("Check Device")
            raise NotImplementedError

        num_classes = output.shape[-1]

        #Make one hot
        target = self.one_hot_embedding(target, num_classes)
        
        evidence = self.get_evidence(output)
        alpha = evidence + 1
        # print("Evidential Loss")

        if self.args_evid['ev_unc_type'] == 'mse':
            loss_val = self.mse_loss(target, alpha)
        elif self.args_evid['ev_unc_type'] == 'ce':
            loss_val = self.edl_loss(torch.digamma, target, alpha)
        elif self.args_evid['ev_unc_type'] == 'log':
            loss_val = self.edl_loss(torch.log, target, alpha)
        else:
            print("Loss not implimented")
            raise NotImplementedError
        
        #Kl loss part
        annealing_coef = torch.min(
                    torch.tensor(1.0, dtype=torch.float32),
                    torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
                ).to(target.device)
        
        kl_alpha = (alpha - 1) * (1 - target) + 1
        kl_div = self.args_evid['kl_strength'] * annealing_coef * self.kl_divergence(kl_alpha, num_classes)
        loss_val += kl_div

        # print("kl_div: ", kl_div)
        # print("kl_div: ", torch.mean(kl_div))
        # print("alpha: ", kl_alpha)

        #EDL Loss
        loss = torch.mean(loss_val)

        if self.args_evid['use_vac_reg']:
            #Vacuity, detached
            with torch.no_grad():
                S = torch.sum(alpha, dim=1, keepdim=True)
                vacuity = num_classes / S.detach()
    
            output_correct = output*target
            output_vac = vacuity * output_correct
            output_negative = output_vac[output_vac<=0]
            loss -= torch.sum(output_negative)/output.shape[0]

        return loss
