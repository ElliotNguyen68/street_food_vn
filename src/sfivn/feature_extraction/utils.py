import numpy as np
from scipy.sparse import csr_matrix

def sparse_value_from_dict_to_dense_vector(
    input_info:dict,
    dim_output:int=150
)->np.ndarray:
    list_index_x=[]
    list_value=[]
    for key in input_info:
        val=input_info[key]
        list_index_x.append(key)
        list_value.append(val)
    return np.squeeze(csr_matrix(
        (list_value,
        (np.repeat([0],len(list_value)),list_index_x)),
        shape=(1, dim_output)
    ).toarray())