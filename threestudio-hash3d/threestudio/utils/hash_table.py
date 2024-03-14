import torch
from collections import deque
import torch.nn.functional as F

class GridBasedHashTable:
    def __init__(self, delta_c, delta_t, N, max_queue_length, hash_table_size):
        """
        Initializes a GridBasedHashTable object.

        Args:
            delta_c (list): List of grid sizes for each spatial dimension.
            delta_t (float): Time grid size.
            N (list): List of constants for hash formula [N1, N2, N3].
            max_queue_length (int): Maximum length of the queue for each hash table entry.
            hash_table_size (int): Size of the hash table.

        Returns:
            None
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.delta_c = torch.tensor(delta_c).to(self.device)
        self.delta_t = delta_t
        self.N = N
        self.max_queue_length = max_queue_length
        self.hash_table_size = hash_table_size
        self.hash_table = [deque(maxlen=max_queue_length) for _ in range(hash_table_size)]

    def compute_hash_index_raw(self, key):
        """
        Computes the raw hash index for a given key.

        Args:
            key (torch.Tensor): Key tensor of the form [c1, c2, c3, t].

        Returns:
            int: The raw hash index.
        """
        i, j, k = torch.floor(key[:3] / self.delta_c).type(torch.int64)
        l = torch.floor(key[3] / self.delta_t).type(torch.int64)
        idx = i + self.N[0] * j + self.N[1] * k + self.N[2] * l
        return idx 

    def compute_hash_index(self, key):
        """
        Computes the hash index for a given key.

        Args:
            key (torch.Tensor): Key tensor of the form [c1, c2, c3, t].

        Returns:
            int: The hash index.
        """
        idx = self.compute_hash_index_raw(key)
        return idx % self.hash_table_size
    
    def append(self, key, meta_key=None, feature=None):
        """
        Appends a key-value pair to the hash table.

        Args:
            key (torch.Tensor): Key tensor of the form [c1, c2, c3, t].
            feature (torch.Tensor): Feature tensor.

        Returns:
            None
        """
        idx = self.compute_hash_index(key)
        self.hash_table[idx].append((key, feature))

    def query(self, key, meta_key=None):
        """
        Queries the hash table for a given key and returns the aggregated output.

        Args:
            key (torch.Tensor): Key tensor of the form [c1, c2, c3, t].

        Returns:
            torch.Tensor: The aggregated output.
        """
        idx = self.compute_hash_index(key)
        queue = self.hash_table[idx]

        if len(queue) == 0:
            return None

        keys, features = zip(*queue)
        keys_tensor = torch.stack(keys).float()
        key_float = key.float().unsqueeze(0)
        
        distances = torch.norm(keys_tensor[:, :3] - key_float[:, :3], dim=1)
        weights = F.softmax(-distances, dim=0)
        weights = weights.view(-1, 1, 1, 1, 1)
        features = torch.stack(features)

        aggregated_output = torch.sum(features * weights, dim=0)
        return aggregated_output
    
class GridBasedHashTable_Sim(GridBasedHashTable):
    
    def append(self, key, meta_key, feature):
        """
        Appends a key-value pair to the hash table.

        Args:
            key (torch.Tensor): Key tensor of the form [c1, c2, c3, t].
            meta_key (torch.Tensor): Meta key tensor.
            feature (torch.Tensor): Feature tensor.

        Returns:
            None
        """
        idx = self.compute_hash_index(key)
        self.hash_table[idx].append((key, meta_key, feature))
        
    def query(self, key, meta_key):
        """
        Queries the hash table for a given key and meta key, returns the aggregated output.

        Args:
            key (torch.Tensor): Key tensor of the form [c1, c2, c3, t].
            meta_key (torch.Tensor): Meta key tensor for query.

        Returns:
            torch.Tensor or None: The aggregated output if available, otherwise None.
        """
        idx = self.compute_hash_index(key)
        queue = self.hash_table[idx]

        if len(queue) == 0:
            return None

        _, meta_keys, features = zip(*queue)
        keys_tensor = torch.stack(meta_keys).float()
        key_float = meta_key.float().reshape(1, -1)
        

        keys_tensor = keys_tensor.view(keys_tensor.size(0), -1)
            
        distances = torch.norm(keys_tensor - key_float, dim=1)
        weights = F.softmax(-distances, dim=0)
        weights = weights.view(-1, 1, 1, 1, 1)

        features = torch.stack(features)

        aggregated_output = torch.sum(features * weights, dim=0)
        return aggregated_output
    
class AdaptiveGridBasedHashTable(GridBasedHashTable_Sim):
    def __init__(self, delta_c, delta_t, N, max_queue_length, hash_table_size, learning_rate=0.001):
        super(AdaptiveGridBasedHashTable, self).__init__(delta_c, delta_t, N, max_queue_length, hash_table_size)
        self._delta_c = torch.tensor(delta_c, requires_grad=True)
        self.delta_c = self._delta_c.to(self.device)
        self._delta_t =  torch.tensor([float(delta_t)], requires_grad=True)
        self.delta_t = self._delta_t.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam([self._delta_c, self._delta_t], lr=learning_rate)
        
    def append(self, key, meta_key, feature):
        """
        Appends a key-feature pair to the hash table and updates grid sizes.

        Args:
            key (torch.Tensor): Key tensor of the form [c1, c2, c3, t].
            feature (torch.Tensor): Feature tensor.

        Returns:
            None
        """
        self.optimizer.zero_grad()
        # Use a differentiable operation for learning grid sizes
        # This could be a custom loss function based on the distribution of keys, for example
        loss = self.compute_loss_for_grid_size_adjustment(key, meta_key, feature)
        print("Loss: ", loss.item())
        loss.backward()
        self.optimizer.step()
        
        print("Updated grid sizes: ", self.delta_c, self.delta_t)
        
        idx = self.compute_hash_index(key)
        self.hash_table[idx].append((key, meta_key, feature))
        
    def compute_loss_for_grid_size_adjustment(self, key, meta_key, feature):
        """
        Defines a custom loss function to adjust grid sizes.
        """
        pseudo_aggregated_output, soft_hash_indices = self.differentiable_query(key, meta_key)
        if pseudo_aggregated_output is None:
            return torch.tensor(0.0, requires_grad=True)
        loss = F.mse_loss(pseudo_aggregated_output, feature) * soft_hash_indices.sum()
        return loss
    
    def compute_hash_logits(self, key):
        """
        Computes the logits for each hash bucket based on the key.
        This is a placeholder function; you need to define how to compute these logits.

        Args:
            key (torch.Tensor): The input key tensor.

        Returns:
            torch.Tensor: The logits for each hash bucket.
        """
        print("Key: ", key, key.requires_grad)
        key.requires_grad_()
        print("Key: ", key, key.requires_grad)
        print(key/2)
        print(key.pow(2))
        print(self.delta_c/2)
        print("Delta c: ", self.delta_c, self.delta_c.requires_grad)
        i, j, k, l = torch.div(key, torch.cat([self.delta_c, self.delta_t], dim=0))
        print(key[0], key[0]/self.delta_c[0], key[0]/self.delta_c[0].requires_grad)
        
        print("i, j, k, l: ", i, j, k, l, i.requires_grad, j.requires_grad, k.requires_grad, l.requires_grad)
        exit()
        idx = i + self.N[0] * j + self.N[1] * k + self.N[2] * l
        # print("Idx: ", idx, idx.requires_grad)
        idx = idx % self.hash_table_size
        distance = torch.abs(torch.arange(self.hash_table_size, device=key.device) - idx)
        return - distance
        
    def differentiable_query(self, key, meta_key):
        """
        Queries the hash table for a given key and meta key, returns the aggregated output.

        Args:
            key (torch.Tensor): Key tensor of the form [c1, c2, c3, t].
            meta_key (torch.Tensor): Meta key tensor for query.

        Returns:
            torch.Tensor or None: The aggregated output if available, otherwise None.
        """
        logits = self.compute_hash_logits(key)

        soft_hash_indices = F.gumbel_softmax(logits, tau=1, hard=False)
        # print("Soft hash indices: ", soft_hash_indices, soft_hash_indices.requires_grad)
        
        # use the soft hash indices to query the hash table
        idx = torch.argmax(soft_hash_indices)
        # print("Selected idx: ", soft_hash_indices[idx], soft_hash_indices[idx].requires_grad)
        # print("Selected idx: ", idx)
        queue = self.hash_table[idx]

        if len(queue) == 0:
            return None, soft_hash_indices

        _, meta_keys, features = zip(*queue)
        keys_tensor = torch.stack(meta_keys).float()
        key_float = meta_key.float().reshape(1, -1)
        

        keys_tensor = keys_tensor.view(keys_tensor.size(0), -1)
            
        distances = torch.norm(keys_tensor - key_float, dim=1)
        weights = F.softmax(-distances, dim=0)
        weights = weights.view(-1, 1, 1, 1, 1)

        features = torch.stack(features)

        aggregated_output = torch.sum(features * weights, dim=0)
        return aggregated_output, soft_hash_indices
        