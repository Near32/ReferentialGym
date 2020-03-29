from typing import Dict, List

import copy

class StreamHandler(object):
    def __init__(self):
        self.placeholders = {}

    def register(self, placeholder_id:str):
        self.update(placeholder_id=placeholder_id, stream_data={})

    def update(self, 
               placeholder_id:str, 
               stream_data:Dict[str,object], 
               p_ptr:Dict[str,object]=None):
        '''
        Updates the streams of a given placeholder.
        Hierarchically explores the placeholders and their streams.

        :params placeholder_id: string formatted with ':' between the name of the streaming module/placeholder and the name of the stream.
        :param stream_data: data or dict of str and torch.Tensor/List to update the stream with.
        :param p_ptr: None, except when called by self in a recurrent fashion.
        '''

        init = False
        if p_ptr is None:   
            init = True
            p_ptr = self.placeholders
        
        if init and stream_data is {}:   return
        
        previous_placeholder = {}
        if ':' not in placeholder_id:
            if placeholder_id not in p_ptr:
                p_ptr[placeholder_id] = {}
            
            # Not possible to copy leaves tensor at the moment with PyTorch...
            previous_placeholder = None #copy.deepcopy(p_ptr[placeholder_id])

            if isinstance(stream_data, dict):
                for k,v in stream_data.items():
                    p_ptr[placeholder_id][k] = v 
            else:
                p_ptr[placeholder_id] = stream_data
            return previous_placeholder
        else:
            ptr, next_placeholder_id = placeholder_id.split(":", 1)
            if ptr not in p_ptr:    p_ptr[ptr] = {}
            previous_placeholder_data = self.update(placeholder_id=next_placeholder_id, 
                                                   stream_data=stream_data, 
                                                   p_ptr=p_ptr[ptr])

        # One-iteration backup...
        # TODO: extend this class into a logging class that consumes those previous values...?
        if init and previous_placeholder_data != {}:
            if "current" in placeholder_id:
                new_placeholder_id = placeholder_id.replace("current", "previous")
                self.update(placeholder_id=new_placeholder_id, 
                            stream_data=previous_placeholder_data,
                            p_ptr=self.placeholders)

    def serve(self, pipeline:List[object]):
        for module in pipeline:
            module_input_stream_dict = self._serve_module(module)    
            module_output_stream_dict = module.compute(input_streams_dict=module_input_stream_dict)
            self.update(f"modules:{module.get_id()}", module_output_stream_dict)

    def _serve_module(self, module:object):
        module_input_stream_keys = module.get_input_stream_keys()
        module_input_stream_ids = module.get_input_stream_ids()
        module_input_stream_dict = {}
        for k_in in module_input_stream_keys:
            k_out = module_input_stream_ids[k_in]
            module_input_stream_dict[k_out] = self[k_in]
        return module_input_stream_dict
    
    def __getitem__(self, stream_id):
        '''
        Hierarchically explores the streaming modules/placeholders and their streams.

        :params stream_id: string formatted with ':' between the name of the streaming module/placeholder and the name of the stream.
        '''
        stream_id = stream_id.split(":")
        p_ptr = self.placeholders
        for ptr in stream_id[:-1]:
            if ptr not in p_ptr.keys(): raise AssertionError("The required stream does not exists...")
            p_ptr = p_ptr[ptr]

        # Do we need to perform some operations on the data stream?
        operations = []
        if '.' in stream_id[-1]:
            operations = stream_id[-1].split(".")
            stream_id[-1] = operations.pop(0)
        
        if hasattr(p_ptr, stream_id[-1]):
            output = getattr(p_ptr, stream_id[-1])
        elif stream_id[-1] in p_ptr:
            output = p_ptr[stream_id[-1]]
        else:
            #raise AssertionError("The required stream does not exists...")
            output = None

        return self._operate(output, operations)

    def _operate(self, data:object, operations:List[str]) -> object:
        for operation in operations:
            op = getattr(data, operation, None)
            if op is not None:
                data = op()

        return data