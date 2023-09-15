from typing import Dict, Any, Optional, List, Callable, Union
import torch
import numpy as np
from functools import partial
import copy


def is_leaf(node: Dict):
    return any([ not isinstance(node[key], dict) for key in node.keys()])

def concat_fn(x):
    """
    All elements should either all be the same shape or mainly all have 
    different/iteratively increasing shapes.

    """
    if all([x[0].shape==xi.shape for xi in x]):
        return torch.cat(x, dim=1)
    nplist = np.empty(len(x), dtype=object)
    for idx, v in enumerate(x):
        nplist[idx] = v
    return nplist

def archi_concat_fn(x):
    if all([isinstance(xi, torch.Tensor) for xi in x]):
        return torch.cat(x, dim=0)   # concatenate on the unrolling dimension (axis=1).
    
    for idx, xi in enumerate(x):
        if isinstance(xi, torch.Tensor):
            x[idx] = np.empty([xi.shape[1]], dtype=object)
            for ridx in range(xi.shape[1]):
                x[idx] = xi[:,ridx:ridx+1,...]

    assert all([isinstance(xi, np.ndarray) and isinstance(xi[0], torch.Tensor) for xi in x])

    batch_size = len(x)

    # Identify the dimension(s) where torch.Tensors are different:
    # Store the maximal value...
    # Expand the other 

    max_dim_value_shape = None
    max_dim_value_shapes = {}
    nbr_dims = len(x[0][0].shape)

    diff_dims = {bidx:[] for bidx in range(batch_size)}
    max_dim_values = {bidx:{} for bidx in range(batch_size)}
    for bidx in range(len(x)):
        max_dim_value_shapes[bidx] = list(x[bidx][0].shape)
        for idx_dim in range(nbr_dims):
            list_dim_values = [x[bidx][idx_t].shape[idx_dim] for idx_t in range(len(x[bidx]))]
            
            if any([dv!=x[bidx][0].shape[idx_dim] for dv in list_dim_values]):
                diff_dims[bidx].append(idx_dim)
                max_dim_values[bidx][idx_dim] = max(list_dim_values)
                max_dim_value_shapes[bidx][idx_dim] = max_dim_values[bidx][idx_dim]

    # Assumption: it is on memory element dimension only that there is discrepancies:
    assert all([ diff_dim[0]==2  for diff_dim in diff_dims.values() if len(diff_dim)==1])
    max_dim_value_shape = list(max_dim_value_shapes.values())[0]
    max_dim_value_shape[2] = max([mdvs[2] for mdvs in max_dim_value_shapes.values()])

    for bidx in range(len(x)):
        for unroll_id, el in enumerate(x[bidx]):
            new_x = torch.zeros(max_dim_value_shape)
            new_x[:,:,:x[bidx][unroll_id].shape[2],...] = x[bidx][unroll_id]
            x[bidx][unroll_id] = new_x
        # unrolling dim concatenation:
        x[bidx] = torch.cat(x[bidx].tolist(), dim=1)
    # batch concatenation:
    r = torch.cat(x, dim=0)
    return r

def recursive_inplace_update(
    in_dict: Dict,
    extra_dict: Union[Dict, torch.Tensor],
    batch_mask_indices: Optional[torch.Tensor]=None,
    preprocess_fn: Optional[Callable] = None,
    assign_fn: Optional[Callable] = None):
    '''
    Taking both :param: in_dict, extra_dict as tree structures,
    adds the nodes of extra_dict into in_dict via tree traversal.
    Extra leaf keys are created if and only if the update is over the whole batch, i.e. :param
    batch_mask_indices: is None.
    :param batch_mask_indices: torch.Tensor of shape (batch_size,), containing batch indices that
                        needs recursive inplace update. If None, everything is updated.
    '''
    if in_dict is None: return None
    if is_leaf(extra_dict):
        for leaf_key in extra_dict:
            # In order to make sure that the lack of deepcopy at this point will not endanger
            # the consistency of the data (since we are slicing at some other parts),
            # or, in other words, to make sure that this is yielding a copy rather than
            # a reference, proceed with caution:
            # WARNING: the following makes a referrence of the elements:
            # listvalue = extra_dict[node_key][leaf_key]
            # RATHER, to generate copies that lets gradient flow but do not share
            # the same data space (i.e. modifying one will leave the other intact), make
            # sure to use the clone() method, as list comprehension does not create new tensors.
            listvalue = [value.clone() for value in extra_dict[leaf_key]]
            in_dict[leaf_key] = listvalue
        return 

    for node_key in extra_dict:
        if node_key not in in_dict: in_dict[node_key] = {}
        if not is_leaf(extra_dict[node_key]):
            recursive_inplace_update(
                in_dict=in_dict[node_key], 
                extra_dict=extra_dict[node_key],
                batch_mask_indices=batch_mask_indices,
                preprocess_fn=preprocess_fn,
                assign_fn=assign_fn,
            )
        else:
            for leaf_key in extra_dict[node_key]:
                # In order to make sure that the lack of deepcopy at this point will not endanger
                # the consistancy of the data (since we are slicing at some other parts),
                # or, in other words, to make sure that this is yielding a copy rather than
                # a reference, proceed with caution:
                # WARNING: the following makes a referrence of the elements:
                # listvalue = extra_dict[node_key][leaf_key]
                # RATHER, to generate copies that lets gradient flow but do not share
                # the same data space (i.e. modifying one will leave the other intact), make
                # sure to use the clone() method, as list comprehension does not create new tensors.
                
                listvalue = [value.clone() for value in extra_dict[node_key][leaf_key]]
                # TODO: identify the issue that the following line was aiming to solve:
                #listvalue = [value.clone() for value in extra_dict[node_key][leaf_key] if value != {}]
                if leaf_key not in in_dict[node_key]:
                    # initializing here, and preprocessing below...
                    in_dict[node_key][leaf_key] = listvalue
                if batch_mask_indices is None or batch_mask_indices==[]:
                    in_dict[node_key][leaf_key]= listvalue
                else:
                    for vidx in range(len(in_dict[node_key][leaf_key])):
                        v = listvalue[vidx]
                        if leaf_key not in in_dict[node_key]:   continue
                        
                        # SPARSE-NESS : check & record
                        sparse_v = False
                        if getattr(v, "is_sparse", False):
                            sparse_v = True
                            v = v.to_dense()
                        
                        # PREPROCESSING :
                        new_v = v[batch_mask_indices, ...].clone().to(in_dict[node_key][leaf_key][vidx].device)
                        if preprocess_fn is not None:   new_v = preprocess_fn(new_v)
                        
                        # SPARSE-NESS : init
                        if in_dict[node_key][leaf_key][vidx].is_sparse:
                            in_dict[node_key][leaf_key][vidx] = in_dict[node_key][leaf_key][vidx].to_dense()
                        # ASSIGNMENT:
                        if assign_fn is not None:
                            assign_fn(
                                dest_d=in_dict,
                                node_key=node_key,
                                leaf_key=leaf_key,
                                vidx=vidx,
                                batch_mask_indices=batch_mask_indices,
                                new_v=new_v,
                            )
                        else:
                            in_dict[node_key][leaf_key][vidx][batch_mask_indices, ...] = new_v
                        
                        # SPARSE-NESS / POST-PROCESSING:
                        if sparse_v:
                            v = v.to_sparse()
                            in_dict[node_key][leaf_key][vidx] = in_dict[node_key][leaf_key][vidx].to_sparse()

def copy_hdict(in_dict: Dict):
    '''
    Makes a copy of :param in_dict:.
    '''
    if in_dict is None: return None
    
    out_dict = {key: {} for key in in_dict}
    need_reg = False
    if isinstance(in_dict, list):
        out_dict = {'dummy':{}}
        in_dict = {'dummy':in_dict}
        need_reg = True 

    recursive_inplace_update(
        in_dict=out_dict,
        extra_dict=in_dict,
    )

    if need_reg:
        out_dict = out_dict['dummy']

    return out_dict

def extract_subtree(in_dict: Dict,
                    node_id: str):
    '''
    Extracts a copy of subtree whose root is named :param node_id: from :param in_dict:.
    '''
    queue = [in_dict]
    pointer = None

    while len(queue):
        pointer = queue.pop(0)
        if not isinstance(pointer, dict): continue
        for k in pointer.keys():
            if node_id==k:
                return copy_hdict(pointer[k])
            else:
                queue.append(pointer[k])

    return {}


def _extract_from_rnn_states(rnn_states_batched: Dict,
                             batch_idx: Optional[int]=None,
                             map_keys: Optional[List]=None,
                             post_process_fn:Callable=(lambda x:x)): #['hidden', 'cell']):
    '''
    :param map_keys: List of keys we map the operation to.
    '''
    rnn_states = {k: {} for k in rnn_states_batched}
    for recurrent_submodule_name in rnn_states_batched:
        # It is possible that an initial rnn states dict has states for actor and critic, separately,
        # but only the actor will be operated during the take_action interface.
        # Here, we allow the critic rnn states to be skipped:
        if rnn_states_batched[recurrent_submodule_name] is None:    continue
        if is_leaf(rnn_states_batched[recurrent_submodule_name]):
            rnn_states[recurrent_submodule_name] = {}
            eff_map_keys = map_keys if map_keys is not None else rnn_states_batched[recurrent_submodule_name].keys()
            for key in eff_map_keys:
                if key in rnn_states_batched[recurrent_submodule_name]:
                    rnn_states[recurrent_submodule_name][key] = []
                    for idx in range(len(rnn_states_batched[recurrent_submodule_name][key])):
                        value = rnn_states_batched[recurrent_submodule_name][key][idx]
                        sparse_v = False
                        if value.is_sparse:
                            sparse_v = True
                            value = value.to_dense()
                        if batch_idx is not None:
                            value = value[batch_idx,...].unsqueeze(0)
                        new_value = post_process_fn(value)
                        if sparse_v:
                            value = value.to_sparse()
                            new_value = new_value.to_sparse()
                        rnn_states[recurrent_submodule_name][key].append(new_value)
        else:
            rnn_states[recurrent_submodule_name] = _extract_from_rnn_states(
                rnn_states_batched=rnn_states_batched[recurrent_submodule_name], 
                batch_idx=batch_idx,
                post_process_fn=post_process_fn
            )
    return rnn_states


def _extract_rnn_states_from_batch_indices(
    rnn_states_batched: Dict,
    batch_indices: torch.Tensor,
    use_cuda: bool=False,
    map_keys: Optional[List]=None,
): 
    if rnn_states_batched is None:  return None

    rnn_states = {k: {} for k in rnn_states_batched}
    for recurrent_submodule_name in rnn_states_batched:
        if is_leaf(rnn_states_batched[recurrent_submodule_name]):
            rnn_states[recurrent_submodule_name] = {}
            eff_map_keys = map_keys if map_keys is not None else rnn_states_batched[recurrent_submodule_name].keys()
            for key in eff_map_keys:
                if key in rnn_states_batched[recurrent_submodule_name]:
                    rnn_states[recurrent_submodule_name][key] = []
                    for idx in range(len(rnn_states_batched[recurrent_submodule_name][key])):
                        value = rnn_states_batched[recurrent_submodule_name][key][idx]
                        sparse_v = False
                        if value.is_sparse:
                            sparse_v = True
                            value = value.to_dense()
                        new_value = value[batch_indices,...]
                        if use_cuda: new_value = new_value.cuda()
                        if sparse_v:
                            new_value = new_value.to_sparse()
                        rnn_states[recurrent_submodule_name][key].append(new_value)
        else:
            rnn_states[recurrent_submodule_name] = _extract_rnn_states_from_batch_indices(
                rnn_states_batched=rnn_states_batched[recurrent_submodule_name],
                batch_indices=batch_indices,
                use_cuda=use_cuda,
                map_keys=map_keys
            )
    return rnn_states


def _extract_rnn_states_from_seq_indices(
    rnn_states_batched: Dict,
    seq_indices: torch.Tensor,
    use_cuda: bool=False,
    map_keys: Optional[List]=None,
): 
    if rnn_states_batched is None:  return None

    rnn_states = {k: {} for k in rnn_states_batched}
    for recurrent_submodule_name in rnn_states_batched:
        if is_leaf(rnn_states_batched[recurrent_submodule_name]):
            rnn_states[recurrent_submodule_name] = {}
            eff_map_keys = map_keys if map_keys is not None else rnn_states_batched[recurrent_submodule_name].keys()
            for key in eff_map_keys:
                if key in rnn_states_batched[recurrent_submodule_name]:
                    rnn_states[recurrent_submodule_name][key] = []
                    for idx in range(len(rnn_states_batched[recurrent_submodule_name][key])):
                        value = rnn_states_batched[recurrent_submodule_name][key][idx]
                        sparse_v = False
                        if value.is_sparse:
                            sparse_v = True
                            value = value.to_dense()
                        new_value = value[:, seq_indices,...]
                        if len(seq_indices) == 1:
                            new_value = new_value.squeeze(1)
                        if use_cuda: new_value = new_value.cuda()
                        if sparse_v:
                            new_value = new_value.to_sparse()
                        rnn_states[recurrent_submodule_name][key].append(new_value)
        else:
            rnn_states[recurrent_submodule_name] = _extract_rnn_states_from_seq_indices(
                rnn_states_batched=rnn_states_batched[recurrent_submodule_name],
                seq_indices=seq_indices,
                use_cuda=use_cuda,
                map_keys=map_keys
            )
    return rnn_states


def _concatenate_hdict(hd1: Union[Dict, List],
                       hds: List,
                       map_keys: List,
                       concat_fn: Optional[Callable] = partial(torch.cat, dim=0),
                       preprocess_fn: Optional[Callable] = (lambda x:x) ):
    if not(isinstance(hd1, dict)):
        return _concatenate_hdict(
            hd1=hds.pop(0),
            hds=hds,
            map_keys=map_keys,
            concat_fn=concat_fn,
            preprocess_fn=preprocess_fn
        )

    out_hd = {}
    for key in hd1:
        out_hd[key] = {}
        map_key_not_found_at_this_level = True
        for map_key in map_keys:
            if map_key in hd1[key]:
                map_key_not_found_at_this_level = False
                out_hd[key][map_key] = []
                for idx in range(len(hd1[key][map_key])):
                    concat_list = [preprocess_fn(hd1[key][map_key][idx])]
                    for hd in hds:
                        concat_list.append(preprocess_fn(hd[key][map_key][idx]))
                    out_hd[key][map_key].append(concat_fn(concat_list))
        if map_key_not_found_at_this_level:
            out_hd[key] = _concatenate_hdict(
                hd1=hd1[key],
                hds=[hd[key] for hd in hds],
                map_keys=map_keys,
                concat_fn=concat_fn,
                preprocess_fn=preprocess_fn,
            )
    return out_hd

def SPARSE_concatenate_list_hdict(
    lhds: List[Dict],
    concat_fn: Optional[Callable] = partial(torch.cat, dim=0),
    preprocess_fn: Optional[Callable] = (lambda x:
        torch.from_numpy(x).unsqueeze(0) if isinstance(x, np.ndarray) else torch.ones(1, 1)*x
        )
    ):
    out_hd = {key: {} for key in lhds[0]}

    queue = [lhds]
    pointers = None

    out_queue = [out_hd]
    out_pointer = None

    while len(queue):
        pointers = [hds for hds in queue.pop(0)]
        out_pointer = out_queue.pop(0)

        if not is_leaf(pointers[0]):
            for k in pointers[0]:
                queue_element = [pointer[k] for pointer in pointers if k in pointer]
                queue.insert(0, queue_element)

                out_pointer[k] = {}
                out_queue.insert(0, out_pointer[k])
        else:
            for k in pointers[0]:
                #try:
                # Since we are at a leaf then value is
                # either numpy or numpy.float64
                # or list of tensors:
                if isinstance(pointers[0][k], list):
                    out_pointer[k] = []
                    for idx in range(len(pointers[0][k])):
                        concat_list = [ 
                                pointer[k][idx].to_dense() if getattr(pointer[k][idx], "is_sparse", False) else pointer[k][idx]
                                for pointer in pointers if k in pointer
                        ]
                        concat_list = [
                            preprocess_fn(v)
                            for v in concat_list
                        ]
                        out_v = concat_fn(concat_list)
                        if getattr(pointers[0][k][idx], "is_sparse", False):
                            out_v = out_v.to_sparse()
                        out_pointer[k].append(out_v)
                elif isinstance(pointers[0][k], np.ndarray) \
                or isinstance(pointers[0][k], torch.Tensor):
                    out_pointer[k] = []
                    concat_list = [ 
                            pointer[k].to_dense() if getattr(pointer[k], "is_sparse", False) else pointer[k]
                            for pointer in pointers if k in pointer
                    ]
                    concat_list = [
                        preprocess_fn(v)
                        for v in concat_list
                    ]
                    out_v = concat_fn(concat_list)
                    if getattr(pointers[0][k], "is_sparse", False):
                        out_v = out_v.to_sparse()
                    out_pointer[k] = out_v
                else:
                    #print(f"Key {k} found of type : {type(pointers[0][k])}")
                    continue
                #except Exception as e:
                #        # the concat_fn may fail, silently...
                #        # e.g.: neither a list nor a compatible stuff....
                #        pass
    return out_hd


def _concatenate_list_hdict(
    lhds: List[Dict],
    concat_fn: Optional[Callable] = partial(torch.cat, dim=0),
    preprocess_fn: Optional[Callable] = (lambda x:
        torch.from_numpy(x).unsqueeze(0) if isinstance(x, np.ndarray) else torch.ones(1, 1)*x
        )
    ):
    out_hd = {key: {} for key in lhds[0]}

    queue = [lhds]
    pointers = None

    out_queue = [out_hd]
    out_pointer = None

    while len(queue):
        pointers = [hds for hds in queue.pop(0)]
        out_pointer = out_queue.pop(0)

        if not is_leaf(pointers[0]):
            for k in pointers[0]:
                queue_element = [pointer[k] for pointer in pointers if k in pointer]
                queue.insert(0, queue_element)

                out_pointer[k] = {}
                out_queue.insert(0, out_pointer[k])
        else:
            for k in pointers[0]:
                out_pointer[k] = []
                # Since we are at a leaf then value is
                # either numpy or numpy.float64
                # or list of tensors:
                if isinstance(pointers[0][k], list):
                    for idx in range(len(pointers[0][k])):
                        concat_list = [ 
                                pointer[k][idx].to_dense() if getattr(pointer[k][idx], "is_sparse", False) else pointer[k][idx]
                                for pointer in pointers if k in pointer
                        ]
                        concat_list = [
                            preprocess_fn(v)
                            for v in concat_list
                        ]
                        out_v = concat_fn(concat_list)
                        if getattr(pointers[0][k][idx], "is_sparse", False):
                            out_v = out_v.to_sparse()
                        out_pointer[k].append(out_v)
                elif isinstance(pointers[0][k], np.ndarray) \
                or isinstance(pointers[0][k], torch.Tensor):
                    concat_list = [ 
                            pointer[k].to_dense() if getattr(pointer[k], "is_sparse", False) else pointer[k]
                            for pointer in pointers if k in pointer
                    ]
                    concat_list = [
                        preprocess_fn(v)
                        for v in concat_list
                    ]
                    out_v = concat_fn(concat_list)
                    if getattr(pointers[0][k], "is_sparse", False):
                        out_v = out_v.to_sparse()
                    out_pointer[k] = out_v
                else:
                    #print(f"CONCAT: Skipped {k} of type {type(pointers[0][k])}")
                    continue
    return out_hd


def apply_on_hdict(
    hdict: Dict,
    fn: Optional[Callable] = lambda x: x,
    ):
    out_hd = {key: {} for key in hdict}

    queue = [hdict]
    pointer = None

    out_queue = [out_hd]
    out_pointer = None

    while len(queue):
        pointer = queue.pop(0)
        out_pointer = out_queue.pop(0)

        if not is_leaf(pointer):
            for k in pointer:
                queue_element = pointer[k]
                queue.insert(0, queue_element)

                out_pointer[k] = {}
                out_queue.insert(0, out_pointer[k])
        else:
            for k in pointer:
                out_pointer[k] = []
                # Since we are at a leaf then value is
                # either numpy or numpy.float64
                # or list of tensors:
                if isinstance(pointer[k], list):
                    for idx in range(len(pointer[k])):
                        out_pointer[k].append(
                            fn(pointer[k][idx])
                        )
                else:
                    out_pointer[k] = fn(pointer[k])
    return out_hd
