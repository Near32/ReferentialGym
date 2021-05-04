import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.2, beta=1.0) :
        self.length = 0
        self.counter = 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        self.capacity = int(capacity)
        self.tree = np.zeros(2*self.capacity-1)
        self.data = np.zeros(self.capacity, dtype=object)
        self.sumPi_alpha = 0.0

    def reset(self):
        self.__init__(capacity=self.capacity, alpha=self.alpha)

    def add(self, exp, priority):
        if np.isnan(priority) or np.isinf(priority) :
            priority = self.total()/self.capacity

        idx = self.counter + self.capacity -1

        self.data[self.counter] = exp

        self.counter += 1
        self.length = min(self.length+1, self.capacity)
        if self.counter >= self.capacity :
            self.counter = 0

        self.sumPi_alpha += priority
        self.update(idx,priority)

    def priority(self, error) :
        return (error+self.epsilon)**self.alpha

    def update(self, idx, priority) :
        if np.isnan(priority) or np.isinf(priority) :
            priority = self.total()/self.capacity

        change = priority - self.tree[idx]

        previous_priority = self.tree[idx]
        self.sumPi_alpha -= previous_priority

        self.sumPi_alpha += priority
        self.tree[idx] = priority

        self._propagate(idx,change)

    def _propagate(self, idx, change) :
        parentidx = (idx - 1) // 2

        self.tree[parentidx] += change

        if parentidx != 0 :
            self._propagate(parentidx, change)

    def __call__(self, s) :
        idx = self._retrieve(0,s)
        dataidx = idx-self.capacity+1
        data = self.data[dataidx]
        priority = self.tree[idx]

        return (idx, priority, data)

    def get(self, s) :
        idx = self._retrieve(0,s)
        dataidx = idx-self.capacity+1

        data = self.data[dataidx]
        
        priority = self.tree[idx]

        return (idx, priority, data)

    def get_importance_sampling_weight(priority,beta=1.0) :
        return pow( self.capacity * priority , -beta )

    def _retrieve(self,idx,s) :
         leftidx = 2*idx+1
         rightidx = leftidx+1

         if leftidx >= len(self.tree) :
            return idx

         if s <= self.tree[leftidx] :
            return self._retrieve(leftidx, s)
         else :
            return self._retrieve(rightidx, s-self.tree[leftidx])

    def total(self) :
        return self.tree[0]

    def __len__(self) :
        return self.length

    def sample(self, batch_size):
        prioritysum = self.total()
        # Random Experience Sampling with priority
        low = 0.0
        step = (prioritysum-low) / batch_size
        randexp = np.arange(low,prioritysum,step)[:batch_size]+np.random.uniform(low=0.0,high=step,size=(batch_size))

        transitions = list()
        priorities = []
        for i in range(batch_size):
            '''
            Sampling from this replayBuffer requires it to be fully populated.
            Otherwise, we might end up trying to sample a leaf ot the binary sumtree
            that does not contain any data, thus throwing a TypeError.
            '''
            try :
                el = self.get(randexp[i])
                priorities.append( el[1] )
                transitions.append(el)
            except TypeError as e :
                continue

        # Importance Sampling Weighting:
        priorities = np.array(priorities, dtype=np.float32)
        importanceSamplingWeights = np.power( len(self) * priorities , -self.beta)

        return transitions, importanceSamplingWeights