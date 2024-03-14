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
    
    def append(self, key, feature):
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

    def query(self, key):
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
    
    