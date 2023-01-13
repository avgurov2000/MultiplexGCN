import torch
from torch import nn
from torch_scatter import scatter_std, scatter_mean, scatter_add

from typing import NamedTuple, Dict, List, Tuple
from collections import defaultdict

from utils import explicit_broadcast

class GCNMultiplex(nn.Module):
    """
    Graph Convolution class for a multiplex network. 
    The concept is similar to regular GCN. Like the general GCN, the model is based on a message passing mechanism. 
    A message from each node is sent to neighboring nodes and aggregated. Unlike the ordinar GCN, 
    the model is able to work with a multilayer network. 
    It can simultaneously and independently work at each layer and then combine several embedding into one, 
    also it can use the shared parameters and work with all layers at the same time.
    """
    nodes_dim = 1
    src_nodes_dim = 0
    trg_nodes_dim = 1
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_layers: int, 
        activation: nn.Module=nn.LeakyReLU(0.2),
        dropout_prob: float=0.6, 
        bias: bool=True,
        add_self_loops: bool=True,
        add_interlayer_loops: bool=True
    ) -> None:
        """
        Initialize GCN layer with following parameters:
        :param in_features: initial node's features count.
        :param out_features: output node's features count.
        :param num_layers: multiplex network layer's count.
        :param activation: activation function.
        :param dropout_prob: dropout probability.
        :param bias: add bias or not.
        :param add_self_loops: add inlayer self-loops between nodes itself or not.
        :param add_interlayer_loops: add interlayer self-loops between nodes itself or not.
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.activation = activation
        self.add_self_loops = add_self_loops
        self.add_interlayer_loops = add_interlayer_loops
        
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_proj = nn.Linear(in_features, num_layers * out_features, bias=False) #bias=True
        self.merge_proj = nn.Linear(num_layers * out_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_layers, out_features))
        else:
            self.register_parameter('bias', None)
        
        #self.init_params()
        
    def init_params(self):
        """
        Weight initialization.
        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.merge_proj.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, data):
        
        #
        # Step 1: Linear Projection + regularization
        #
        
        in_nodes_features, edge_index = data  #unpack data
        
        batch_size = in_nodes_features.size(0)
        num_nodes = in_nodes_features.shape[self.nodes_dim]
        
        assert len(edge_index) == self.num_layers, f'Expected edge index list of length={self.num_layers} got {len(edge_index)}'
        
        #shape = (None, N, FIN) where:
        #None - batch size, 
        #N - number of nodes in graph, 
        #FIN - number of input features per node
        
        # We apply the dropout to all of the input node features
        in_nodes_features = self.dropout(in_nodes_features)
        
        #shape = (None, N, FIN)*(FIN, NL*FOUT) = (None, F, NL, FOUT) where
        #NL - number of layers,
        #FOUT - number of output features per node
        # We project the input node features into NL independent output features (one for each layer)
        nodes_features_proj = self.linear_proj(in_nodes_features)
        nodes_features_proj = torch.concat(nodes_features_proj.split(self.out_features, -1), self.nodes_dim)
        nodes_features_proj = self.dropout(nodes_features_proj) 
        
        #
        # Step 2: Edge index preparation and degree normalization
        #
        
        #We merge edge indexes of list(edge_index_0, edge_index_1, ..., edge_index_n) to tensor of all edges
        #Additionaly we're going to normalize the sum of recieved messages for each node
        edge_index = self.merge_edges_indexes(edge_index, num_nodes)
        in_degree, out_degree = self.get_nodes_degree(edge_index, num_nodes)
        nodes_features_proj_normalize = nodes_features_proj * explicit_broadcast(out_degree, nodes_features_proj)
        
        # We simply copy (lift) the normalized nodes projection for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # nodes_features_proj_lifted shape = (None, E, NL, FOUT), E - number of edges in the graph
        nodes_features_proj_lifted = self.lift(nodes_features_proj_normalize, edge_index)
        
        #
        # Step 3: Aggregate neighbors, normalize once again, add bias and scale
        #
        
        #aggregate neighbors
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted, edge_index, num_nodes)
        
        #normalize wirh input node degree
        out_nodes_features = out_nodes_features * explicit_broadcast(in_degree, out_nodes_features)
        out_nodes_features = torch.stack(out_nodes_features.split(num_nodes, self.nodes_dim), self.nodes_dim+1)
        
        
        #add bias and activation
        if self.bias is not None:
            out_nodes_features = out_nodes_features + self.bias
        out_nodes_features = self.activation(out_nodes_features)
        
        #scaling
        out_nodes_features_scaled = self.merge_proj(out_nodes_features.flatten(-2))
        return out_nodes_features_scaled
    
    def aggregate_neighbors(self, nodes_features_proj_lifted, edge_index, num_nodes):
        size = list(nodes_features_proj_lifted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = int(num_nodes*self.num_layers)
        out_nodes_features = torch.zeros(size, dtype=nodes_features_proj_lifted.dtype, device=nodes_features_proj_lifted.device)
        
        # shape = (None, E) -> (None, E, FOUT)
        trg_index_broadcasted = explicit_broadcast(edge_index[self.trg_nodes_dim].unsqueeze(0), nodes_features_proj_lifted)
        
        # aggregation step - we accumulate projected node features
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted)
        return out_nodes_features
    
        
    def lift(
        self, 
        nodes_features_proj, 
        edge_index
    ) -> torch.Tensor:
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        
        :param nodes_features_proj: node's vector representation.
        :param edge_index: edges in form of couples of sorce and target node indexes.
        """
        
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        nodes_features_proj_lifted = nodes_features_proj.index_select(self.nodes_dim, src_nodes_index)
        
        return nodes_features_proj_lifted
    
    def merge_edges_indexes(
        self, 
        edge_index: List[torch.Tensor], 
        num_nodes: int
    ) -> torch.Tensor:
        """
        Function for merging edges indexes for all layer into one global edges indexes. 
        If it is nessesary to use self-loops it adds new required edges indexes.
        Similarly, it adds extra edges if it is nessesary to opperate all layers simultaniously.
        
        :param edge_index: edges in form of couples of sorce and target node indexes.
        :param num_nodes: number of the whole nodes in the network.
        """
        all_edges = torch.cat([edge_index[i]+i*num_nodes for i in range(self.num_layers)], dim=1)
        dtype = all_edges.dtype
        device = all_edges.device
        
        if self.add_self_loops:
            all_edges = self._add_self_loops(all_edges, num_nodes)
            
        if self.add_interlayer_loops:
            all_edges = self._add_interlayer_loops(all_edges, num_nodes)
        
        return all_edges
    
    def _add_self_loops(
        self, 
        edge_index: List[torch.Tensor], 
        num_nodes: int
    ) -> torch.Tensor:
        """
        Function for adding self-loops into graph. Thus after that each node will has one additional edge with itself.
        
        :param edge_index: edges in form of couples of sorce and target node indexes.
        :param num_nodes: number of the whole nodes in the network.
        """
        dtype = edge_index.dtype
        device = edge_index.device
        self_loops = torch.stack([torch.arange(self.num_layers * num_nodes, dtype=dtype, device=device) for i in range(2)])
        all_edges = torch.cat([edge_index, self_loops], dim=1)
        
        return all_edges
    
    def _add_interlayer_loops(
        self,
        edge_index: List[torch.Tensor], 
        num_nodes: int
    ) -> torch.Tensor:
        """
        Function for adding additional inter-layers connections (edges). Thus after that each node will has several 
        additional edge with representation of the node itself on other layers.
        
        :param edge_index: edges in form of couples of sorce and target node indexes.
        :param num_nodes: number of the whole nodes in the network.
        """
        dtype = edge_index.dtype
        device = edge_index.device
        
        layer_combination = torch.combinations(torch.arange(self.num_layers, dtype=dtype, device=device), r=2, with_replacement=False)
        layer_combination = torch.concat([layer_combination, torch.flip(layer_combination, [0, 1])], 0).T

        all_nodes = torch.arange(self.num_layers * num_nodes, dtype=dtype, device=device).reshape(self.num_layers, num_nodes)
        src_layers = all_nodes.index_select(0, layer_combination[0]).flatten()
        trg_layers = all_nodes.index_select(0, layer_combination[1]).flatten()
        
        interlayer_loops = torch.stack([src_layers, trg_layers], dim=0)
        all_edges = torch.cat([edge_index, interlayer_loops], dim=1)
        
        return all_edges
        
    def get_nodes_degree(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.tensor]:
        
        """
        Function for calculating node degree for further normalization. Note that if the network is undirected 
        then input and output degrees will be the same, otherwise they will be different.
        
        :param edge_index: edges in form of couples of sorce and target node indexes.
        :param num_nodes: number of the whole nodes in the network.
        """
        
        in_degree = torch.zeros(num_nodes*self.num_layers, dtype=torch.float, device=edge_index.device)
        out_degree = torch.zeros(num_nodes*self.num_layers, dtype=torch.float, device=edge_index.device)
        one_in = torch.ones(edge_index.size(1), device=edge_index.device)
        one_out = torch.ones(edge_index.size(1), device=edge_index.device)
        
        in_degree.scatter_add_(0, edge_index[0], one_in)
        out_degree.scatter_add_(0, edge_index[1], one_out)
        
        in_degree = 1/in_degree.sqrt().unsqueeze(0)
        out_degree = 1/out_degree.sqrt().unsqueeze(0)
        
        return (in_degree, out_degree)
    
    


class CDTripletLoss(nn.Module):
    """Custom Loss function."""
    
    nodes_dim = 1
    src_nodes_dim = 0
    trg_community_dim = 1
    
    def __init__(
        self,
        p: int = 2,
        std: float = 1.,
        alpha: float = 0.25
    ) -> None:
        super().__init__()
        self.p = p
        self.std = std
        self.alpha = alpha
        
    def forward(self, node_features, community_belong_list):
        #
        # Step 1: order communities, calculate its representations
        #
        permute_community_belong_list = self._community_inverse_permute(node_features, community_belong_list)
        
        #calculate community sum representation 
        #sum - sum of the nodes representation for each community and count of its members, shape (None, C, FOUT), where C - number of communities
        community_representations = self._groupby_sum(node_features, permute_community_belong_list)
        
        #calculate community members count
        #cnt - count of community members, shape (None, C, FOUT), where C - number of communities
        community_members = self._groupby_count(node_features, permute_community_belong_list)
        
        #calculate pairwise distances b/w nodes representations and mean community representations 
        community_representations_mean = community_representations / community_members
        distance = torch.cdist(node_features, community_representations_mean, p=self.p)
        
        #
        # Step 2: calculate communities representations for each nodes munis representations of corresponding nodes itself
        #
        
        #calculate community representations per each node
        community_representations_per_node = community_representations.index_select(self.nodes_dim, permute_community_belong_list[self.trg_community_dim])
        
        #calculate community members count-1 for communities which has greater than one node, otherwise remain community members count
        community_members_per_node = community_members.index_select(self.nodes_dim, permute_community_belong_list[self.trg_community_dim])
        community_members_per_node = torch.max(community_members_per_node-1, torch.ones_like(community_members_per_node))
        
        #calculate nodes representations for substractive from its community representation for communities, 
        #which has greater than one node, otherwise remain 0
        node_representations_per_node = node_features.index_select(self.nodes_dim, permute_community_belong_list[self.src_nodes_dim])
        node_representations_per_node_minus = torch.clone(node_representations_per_node)
        node_representations_per_node_minus[community_members_per_node==1] = 0
        
        #calculate distances b/w nodes representations and its edited community representations
        community_representations_per_node_mean = (community_representations_per_node - node_representations_per_node_minus)/community_members_per_node
        positive_distance = nn.PairwiseDistance(self.p)(community_representations_per_node_mean, node_representations_per_node)
        
        positive_index = self._community_membership_index(distance, permute_community_belong_list)
        negative_indexes = ~positive_index
        negative_shape = list(distance.shape)
        negative_shape[-1] -= 1
        negative_distance = distance[negative_indexes].reshape(negative_shape)
        
        mean_negative = negative_distance.mean(-1)
        hard_negative = negative_distance.min(-1)[0]
        
        triplet_mean = positive_distance - mean_negative + self.alpha
        triplet_mean = torch.max(triplet_mean, torch.zeros_like(triplet_mean))
        
        triplet_min = positive_distance - hard_negative + self.alpha
        triplet_min = torch.max(triplet_min, torch.zeros_like(triplet_min))
        
        std = self._groupby_std(node_features, permute_community_belong_list)
        
        return triplet_mean.mean(), triplet_min.mean(), (std-self.std).pow(2).mean()
    
    def _community_inverse_permute(self, node_features, community_belong_list):
        
        """
        Calculate permutation inversion vector and order community_belong_list, e.g. 
        for community_belong_list ([2, 0, 1], [0, 1, 2]) get vector [1, 2, 0] to order community_belong_list to ([0, 1, 2], [1, 2, 0])
        """
        
        permute_src = torch.arange(node_features.shape[self.nodes_dim], device=node_features.device, dtype=community_belong_list.dtype)
        permute_index = community_belong_list[self.src_nodes_dim]
        permute_out = torch.zeros_like(permute_src)
        permute_out.scatter_add_(0, permute_index, permute_src)
        permute_community_belong_list = community_belong_list.index_select(1, permute_out)
        
        return permute_community_belong_list
        
    def _community_membership_index(self, distance, community_belong_list):
        src_size = list(distance.shape)[self.nodes_dim:]
        src = torch.zeros(src_size, device=distance.device, dtype=torch.bool)
        
        index = community_belong_list.split(1, dim=0)
        src[index] = True
        src = explicit_broadcast(src.unsqueeze(0), distance)
        return src
    
    
    def _prepare_indexes(self, indexes, broadcast_to = None):
        indexes_unsqueezed = indexes.unsqueeze(0).unsqueeze(-1)
        if broadcast_to is not None:
            indexes_unsqueezed = explicit_broadcast(indexes_unsqueezed, broadcast_to)
        return indexes_unsqueezed
    
    def _groupby_template(self, node_features, community_belong_list):
        size = list(node_features.shape)
        size[self.nodes_dim] = len(community_belong_list[self.trg_community_dim].unique())
        out = torch.zeros(size, dtype=node_features.dtype, device=node_features.device)
        indexes = self._prepare_indexes(community_belong_list[self.trg_community_dim], node_features)
        src = node_features.index_select(1, community_belong_list[self.src_nodes_dim])
        return out, src, indexes
    
    def _groupby_count(self, node_features, community_belong_list):
        out, _, indexes = self._groupby_template(node_features, community_belong_list)
        src = torch.ones(indexes.shape, dtype=node_features.dtype, device=node_features.device)
        out.scatter_add_(1, indexes, src)
        return out
    
    def _groupby_mean(self, node_features, community_belong_list):
        out, src, indexes = self._groupby_template(node_features, community_belong_list)
        out = scatter_mean(src, indexes, 1, out=out)
        return out
    
    def _groupby_sum(self, node_features, community_belong_list):
        out, src, indexes = self._groupby_template(node_features, community_belong_list)
        out = scatter_add(src, indexes, 1, out=out)
        return out
    
    def _groupby_std(self, node_features, community_belong_list):
        out, src, indexes = self._groupby_template(node_features, community_belong_list)
        out = scatter_std(src, indexes, 1, out=out)
        return out