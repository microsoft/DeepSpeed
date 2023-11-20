
import numpy as np
# This class creates the layout for all the devices used for the training system;
# It gets the world-size, together wit all the different parallelism degree (tp, pp, ep, sp, and dp)
# Then it creates a multi-dimensional list based on the order of the parallelism provided
# We use this list later to create the process groups for each of the paralllel ranks

class Layout:
    def __init__(self, world_size,
                 tp_size=1,
                 pp_size=1,
                 ep_size=1,
                 dp_size=1,
                 sp_size=1,
                 order="despt"):
        self.world_size_ = world_size
        ranks = np.arange(0, world_size)

        initial_layout = np.reshape(ranks, [dp_size, ep_size, sp_size, pp_size, tp_size])

        self.layout_ = np.einsum(f"despt->{order}", initial_layout)
        shape_list = self.layout_.shape
        self.layout_order_ = order
        self.parallel_keys_ = [order[i] for i in range(len(order))]
        self.parallel_map_ = dict(zip(self.parallel_keys_, shape_list))

    def get_neighbors(self, parallel_key):
        neighbors = np.copy(self.layout_)
        paralle_size = self.parallel_map_[parallel_key]
        all_rest_dim = self.world_size_ // paralle_size
        transposed_layout = ''.join([pk for pk in self.parallel_keys_ if pk != parallel_key])
        transposed_layout = parallel_key + transposed_layout
        neighbors = np.einsum(f'{self.layout_order_}->{transposed_layout}', neighbors)
        neighbors = neighbors.reshape(paralle_size, all_rest_dim)
        return [neighbors[:,i].tolist() for i in range(all_rest_dim)]

    def get_tp_ranks(self):
        return self.get_neighbors('t')
    
    def get_sp_ranks(self):
        return self.get_neighbors('s')
        
    def get_dp_ranks(self):
        return self.get_neighbors('d')
        
    def get_pp_ranks(self):
        return self.get_neighbors('p')

    def get_ep_ranks(self):
        return self.get_neighbors('e')

