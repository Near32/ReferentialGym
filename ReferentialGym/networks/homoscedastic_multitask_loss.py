import torch
import torch.nn as nn


class HomoscedasticMultiTasksLoss(nn.Module):
    def __init__(self, nbr_tasks=2, use_cuda=False):
        super(HomoscedasticMultiTasksLoss,self).__init__()

        self.nbr_tasks = nbr_tasks
        self.use_cuda = use_cuda
        self.homoscedastic_log_vars = torch.nn.Parameter(torch.zeros(self.nbr_tasks))
        
        if use_cuda:
            self = self.cuda()

    def forward(self, loss_dict) :
        '''
        :param loss_dict: Dict[str, Tuple(float, torch.Tensor)] that associates loss names
                            with their pair of (linear coefficient, loss), where the loss
                            is in batched shape: (batch_size, 1)
        '''
        nbr_tasks_ineffect = len(loss_dict)
        k0 = list(loss_dict.keys())[0]
        batch_size = loss_dict[k0][1].size()[0]

        if nbr_tasks_ineffect > self.nbr_tasks:
            self.nbr_tasks = nbr_tasks_ineffect
            self.homoscedastic_log_vars.data = torch.zeros(self.nbr_tasks).to(self.homoscedastic_log_vars.device)

        inv_uncertainty2 = torch.exp( -self.homoscedastic_log_vars[:self.nbr_tasks])
        # (nbr_tasks)
        # (batch_size, self.nbr_tasks)
        batched_multiloss = {}
        for idx_loss, (kn, l) in enumerate(loss_dict.items()):
            batched_multiloss[kn] = inv_uncertainty2[idx_loss]*l[1]+self.homoscedastic_log_vars[idx_loss].repeat(l[1].shape[0])
        
        for kn in loss_dict:
            loss_dict[kn].append( batched_multiloss[kn])

        return loss_dict
        