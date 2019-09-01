import torch
import torch.nn as nn


class HomoscedasticMultiTasksLoss(nn.Module):
    def __init__(self, nbr_tasks=2):
        super(HomoscedasticMultiTasksLoss,self).__init__()

        self.nbr_tasks = nbr_tasks
        self.log_vars = torch.nn.Parameter(torch.zeros(1,self.nbr_tasks))
        self.register_parameter( name='Homoscedastic_log_vars', param=self.log_vars )
        
        
    def forward(self, loss_dict) :
        nbr_tasks_ineffect = len(loss_dict)
        k0 = list(loss_dict.keys())[0]
        batch_size = loss_dict[k0][1].size()[0]

        if nbr_tasks_ineffect > self.nbr_tasks:
            new_log_vars = torch.nn.Parameter(torch.zeros(1,nbr_tasks_ineffect))
            new_log_vars[:,:self.nbr_tasks] = self.log_vars
            self.nbr_tasks = nbr_tasks_ineffect
            self.log_vars = new_log_vars
            self.register_parameter( name='Homoscedastic_log_vars', param=self.log_vars )
            
        precision = torch.exp( -self.log_vars[0,:self.nbr_tasks] ).view((1,self.nbr_tasks))
        bp = torch.cat( [precision]*batch_size,dim=0)
        blv = torch.cat( [self.log_vars[0,:self.nbr_tasks].view((1,self.nbr_tasks))]*batch_size,dim=0)
        
        loss_inputs = torch.cat([ l[1].unsqueeze(1) for l in loss_dict.values()], dim=-1)
        # (batch_size, self.nbr_tasks)
        batched_multiloss = bp * loss_inputs + blv
        # (batch_size, self.nbr_tasks)
        for idx, k in enumerate(loss_dict):
            loss_dict[k].append( batched_multiloss[:, idx])

        return loss_dict
        