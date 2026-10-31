import numpy as np
import torch
import random
import functools
import sympy
from sympy.utilities.lambdify import lambdify

def restores_global_rng(fn):
    '''
    the global generators of torch (CPU), numpy and random are restored after fn: fn may seed them for its own draws
    without changing the caller's streams (pykan 0.2.8 reseeded them in the constructor, LBFGS, create_dataset and at
    import, so a caller's later draws repeated across what the caller took for different seeds, #424, #351)
    '''
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        np_state, py_state = np.random.get_state(), random.getstate()
        with torch.random.fork_rng(devices=[]):
            try:
                return fn(*args, **kwargs)
            finally:
                np.random.set_state(np_state)
                random.setstate(py_state)
    return wrapper


@restores_global_rng
def create_dataset(f,  
                   n_var=2, 
                   f_mode = 'col',
                   ranges = (-1, 1),
                   train_num=1000, 
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   device='cpu',
                   seed=0):
    '''
    create dataset
    
    Args:
    -----
        f : function
            the symbolic formula used to create the synthetic dataset
        ranges : list or np.array; shape (2,) or (n_var, 2)
            the range of input variables. Default: [-1,1].
        train_num : int
            the number of training samples. Default: 1000.
        test_num : int
            the number of test samples. Default: 1000.
        normalize_input : bool
            If True, apply normalization to inputs. Default: False.
        normalize_label : bool
            If True, apply normalization to labels. Default: False.
        device : str
            device. Default: 'cpu'.
        seed : int
            random seed. Default: 0.
        
    Returns:
    --------
        train, test : dict
            the inputs under 'x' and the labels under 'y' (the batches of xkan.train)
         
    Example
    -------
    >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    >>> train, test = create_dataset(f, n_var=2, train_num=100)
    >>> train['x'].shape
    torch.Size([100, 2])
    '''

    # seeded for this call only (restores_global_rng); the CPU generator, where the draws are made
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)

    if len(np.array(ranges).shape) == 1:
        # np.tile: also for numpy ranges, which ranges * n_var multiplied (pykan 0.2.8 raised)
        ranges = np.tile(np.asarray(ranges), n_var).reshape(n_var,2)
    else:
        ranges = np.array(ranges)
        
    
    train_input = torch.zeros(train_num, n_var)
    test_input = torch.zeros(test_num, n_var)
    for i in range(n_var):
        train_input[:,i] = torch.rand(train_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
        test_input[:,i] = torch.rand(test_num,)*(ranges[i,1]-ranges[i,0])+ranges[i,0]
                
    if f_mode == 'col':
        train_label = f(train_input)
        test_label = f(test_input)
    elif f_mode == 'row':
        train_label = f(train_input.T)
        test_label = f(test_input.T)
    else:
        raise ValueError(f"f_mode {f_mode!r} not recognized; 'col' or 'row'")
        
    # if has only 1 dimension
    if len(train_label.shape) == 1:
        train_label = train_label.unsqueeze(dim=1)
        test_label = test_label.unsqueeze(dim=1)
        
    def normalize(data, mean, std):
            # a constant column is centred only (pykan 0.2.8 divided by 0: NaN)
            return (data-mean)/torch.where(std == 0, torch.ones_like(std), std)
            
    if normalize_input == True:
        mean_input = torch.mean(train_input, dim=0, keepdim=True)
        std_input = torch.std(train_input, dim=0, keepdim=True)
        train_input = normalize(train_input, mean_input, std_input)
        test_input = normalize(test_input, mean_input, std_input)
        
    if normalize_label == True:
        mean_label = torch.mean(train_label, dim=0, keepdim=True)
        std_label = torch.std(train_label, dim=0, keepdim=True)
        train_label = normalize(train_label, mean_label, std_label)
        test_label = normalize(test_label, mean_label, std_label)

    return ({'x': train_input.to(device), 'y': train_label.to(device)},
            {'x': test_input.to(device), 'y': test_label.to(device)})



def sparse_mask(in_dim, out_dim):
    '''
    get sparse mask
    '''
    in_coord = torch.arange(in_dim) * 1/in_dim + 1/(2*in_dim)
    out_coord = torch.arange(out_dim) * 1/out_dim + 1/(2*out_dim)

    dist_mat = torch.abs(out_coord[:,None] - in_coord[None,:])
    in_nearest = torch.argmin(dist_mat, dim=0)
    in_connection = torch.stack([torch.arange(in_dim), in_nearest]).permute(1,0)
    out_nearest = torch.argmin(dist_mat, dim=1)
    out_connection = torch.stack([out_nearest, torch.arange(out_dim)]).permute(1,0)
    all_connection = torch.cat([in_connection, out_connection], dim=0)
    mask = torch.zeros(in_dim, out_dim)
    mask[all_connection[:,0], all_connection[:,1]] = 1.
    
    return mask


def ex_round(ex1, n_digit):
    '''
    rounding the numbers in an expression to certain floating points
    
    Args:
    -----
        ex1 : sympy expression
        n_digit : int
        
    Returns:
    --------
        ex2 : sympy expression
    
    Example
    -------
    >>> from xkan.utils import *
    >>> from sympy import *
    >>> input_vars = a, b = symbols('a b')
    >>> expression = 3.14534242 * exp(sin(pi*a) + b**2) - 2.32345402
    >>> ex_round(expression, 2)
    '''
    def rounded(a):
        # SymPy's round keeps only the bits the rounded value needs: 0.0095 to 2 digits became 0.009766 at 3 bits,
        # printed as 0.e-2; a Float of the decimal string has the precision of its digits
        r = round(a, n_digit)
        return r if r == 0 else sympy.Float(f"{float(r):.{n_digit}f}", "")
    ex2 = ex1
    for a in sympy.preorder_traversal(ex1):
        if isinstance(a, sympy.Float):
            ex2 = ex2.subs(a, rounded(a))
    return ex2


def augment_input(orig_vars, aux_vars, x):
    '''
    augment inputs
    
    Args:
    -----
        orig_vars : list of sympy symbols
        aux_vars : list of auxiliary symbols
        x : inputs (batch, len(orig_vars)), a data dict (its "x" augmented, a new dict), or a tuple of them
            (create_dataset's (train, test))
        
    Returns:
    --------
        augmented inputs (the auxiliary columns first), of the same kind as x
    
    Example
    -------
    >>> from xkan.utils import *
    >>> from sympy import *
    >>> orig_vars = a, b = symbols('a b')
    >>> aux_vars = [a + b, a * b]
    >>> x = torch.rand(100, 2)
    >>> augment_input(orig_vars, aux_vars, x).shape
    '''
    # if x is a tensor
    if isinstance(x, torch.Tensor):
        
        aux_values = torch.tensor([]).to(x.device)

        for aux_var in aux_vars:
            func = lambdify(orig_vars, aux_var,'numpy') # returns a numpy-ready function
            aux_value = torch.from_numpy(func(*[x[:,[i]].detach().cpu().numpy() for i in range(len(orig_vars))])).to(x.device)
            aux_values = torch.cat([aux_values, aux_value], dim=1)
            
        x = torch.cat([aux_values, x], dim=1)

    elif isinstance(x, dict):
        x = {**x, 'x': augment_input(orig_vars, aux_vars, x['x'])}
    elif isinstance(x, tuple):
        x = tuple(augment_input(orig_vars, aux_vars, part) for part in x)
    else:
        raise TypeError(f"augment_input takes a tensor, a data dict or a tuple of them, not {type(x).__name__}")
    return x


def batch_jacobian(func, x, create_graph=False, mode='scalar'):
    '''
    jacobian
    
    Args:
    -----
        func : function or model
        x : inputs
        create_graph : bool
        
    Returns:
    --------
        jacobian
    
    Example
    -------
    >>> from xkan.utils import batch_jacobian
    >>> x = torch.normal(0,1,size=(100,2))
    >>> model = lambda x: x[:,[0]] + x[:,[1]]
    >>> batch_jacobian(model, x)
    '''
    # x in shape (Batch, Length)
    def _func_sum(x):
        return func(x).sum(dim=0)
    if mode == 'scalar':
        return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph)[0]
    elif mode == 'vector':
        return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)
    else:
        raise ValueError(f"unknown mode {mode!r}; 'scalar' or 'vector'")

def batch_hessian(model, x, create_graph=False):
    '''
    hessian
    
    Args:
    -----
        func : function or model
        x : inputs
        create_graph : bool
        
    Returns:
    --------
        jacobian
    
    Example
    -------
    >>> from xkan.utils import batch_hessian
    >>> x = torch.normal(0,1,size=(100,2))
    >>> model = lambda x: x[:,[0]]**2 + x[:,[1]]**2
    >>> batch_hessian(model, x)
    '''
    # x in shape (Batch, Length)
    jac = lambda x: batch_jacobian(model, x, create_graph=True)
    def _jac_sum(x):
        return jac(x).sum(dim=0)
    return torch.autograd.functional.jacobian(_jac_sum, x, create_graph=create_graph).permute(1,0,2)


def create_dataset_from_data(inputs, labels, train_ratio=0.8, device='cpu'):
    '''
    create dataset from data
    
    Args:
    -----
        inputs : 2D torch.float
        labels : 2D torch.float
        train_ratio : float
            the ratio of training fraction
        device : str
        
    Returns:
    --------
        train, test : dict
            the inputs under 'x' and the labels under 'y'
    
    Example
    -------
    >>> from xkan.utils import create_dataset_from_data
    >>> x = torch.normal(0,1,size=(100,2))
    >>> y = torch.normal(0,1,size=(100,1))
    >>> train, test = create_dataset_from_data(x, y)
    >>> train['x'].shape
    '''
    num = inputs.shape[0]
    train_id = np.random.choice(num, int(num*train_ratio), replace=False)
    test_id = list(set(np.arange(num)) - set(train_id))
    return ({'x': inputs[train_id].detach().to(device), 'y': labels[train_id].detach().to(device)},
            {'x': inputs[test_id].detach().to(device), 'y': labels[test_id].detach().to(device)})


def get_derivative(model, inputs, labels, derivative='hessian', loss_mode='pred', regularizer=None, lamb=0.):
    '''
    the jacobian (1, n) or hessian (1, n, n) of a loss of model with respect to all its parameters (n entries, in the
    order of model.named_parameters(), frozen ones included), through torch.func.functional_call

    Args:
    -----
        model : a module (a KANNetwork, say)
        inputs, labels : 2D torch.tensor
        derivative : 'hessian' or 'jacobian'
        loss_mode : 'pred', the mean squared error of model(inputs) against labels; 'reg', regularizer(model, rec) on
            a recorded forward of inputs (default regularizer: xkan.losses.KANRegularizer()); 'all', 'pred' + lamb * 'reg'
    '''
    from torch.func import functional_call
    if loss_mode not in ('pred', 'reg', 'all'):
        raise ValueError(f"unknown loss_mode {loss_mode!r}; 'pred', 'reg' or 'all'")
    if derivative not in ('hessian', 'jacobian'):
        raise ValueError(f"unknown derivative {derivative!r}; 'hessian' or 'jacobian'")
    if loss_mode != 'pred' and regularizer is None:
        from .losses import KANRegularizer
        regularizer = KANRegularizer()
    named = [(n, p.detach()) for n, p in model.named_parameters()]
    names, shapes, sizes = [n for n, _ in named], [p.shape for _, p in named], [p.numel() for _, p in named]

    class Objective(torch.nn.Module):
        # the loss inside one module call, so that the regulariser sees the substituted parameters too
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x, y):
            if loss_mode == 'pred':
                return torch.mean((self.model(x) - y) ** 2)
            from .interp.recording import record_activations
            with record_activations(self.model) as rec:
                out = self.model(x)
            reg = regularizer(self.model, rec)
            return reg if loss_mode == 'reg' else torch.mean((out - y) ** 2) + lamb * reg

    objective = Objective()

    def loss(flat):
        values = torch.split(flat[0], sizes)
        params = {'model.' + n: v.reshape(s) for n, v, s in zip(names, values, shapes)}
        return functional_call(objective, params, (inputs, labels)).reshape(1, 1)

    p = torch.cat([v.reshape(-1) for _, v in named])[None, :]
    return batch_hessian(loss, p) if derivative == 'hessian' else batch_jacobian(loss, p)


def model2param(model):
    '''
    turn model parameters into a flattened vector
    '''
    p = torch.tensor([]).to(model.device)
    for params in model.parameters():
        p = torch.cat([p, params.reshape(-1,)], dim=0)
    return p
