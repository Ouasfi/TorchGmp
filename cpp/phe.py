
import torch
from torch.utils.cpp_extension import load
ops = load(name="lltm_cpp", sources=["lltm.cpp"],extra_ldflags  = ['-lhcs' ,'-lgmp'], verbose=True)
import random

class P1:

    def __init__(self, public_key, q = 1 ):
        self.q = q
        self.pk = public_key
    def add(self, E_a, E_b):
        shape = E_a.shape
        return ops.modular_add(E_a.flatten(), E_b.flatten(), self.pk).reshape(shape)
    def sub(self, E_a, E_b):
        shape = E_a.shape
        return ops.modular_sub(E_a.flatten(), E_b.flatten(), self.pk).reshape(shape)
    def sum(self, E_a):
        return ops.modular_sum(E_a.flatten(), self.pk)
    def send(self, E_a , E_r):
        return ops.modular_mul(E_a, E_r , self.pk[-1], self.q)
    def rec_div(self,E_c, d, r):
        return ops.div_enc(E_c, d, r, self.pk)
    def send_comp (self, E_a, E_r, r2):
        E_a_r = ops.modular_pow(E_a, self.pk, r2)
        E_a_r =ops.modular_add(E_a_r, E_r,self.pk )
        return E_a_r
    def rec_mul(self,E_a, E_b, E_M, r_b, r_a):
        return ops.retrieve_mul(E_a, E_b,E_M, self.pk, r_b, r_a )
    def rec_matmul(self,E_a, E_b, E_M, r_b, r_a):
        return  ops.retrieve_matmul(E_a.reshape(-1,1),E_b.reshape(-1,1),E_M,self.pk, r_b, r_a)
class P2:
    def __init__(self, private_key , public_key, q = 1):
        self.vk = private_key
        self.pk = public_key
        self.q = q
    def decrypt(self, M):
        shape = M.shape
        return ops.decrypt (M.flatten(), self.vk).reshape(shape)
    def encrypt(self, M):
        shape = M.shape
        return ops.encrypt (M.flatten(), self.pk).reshape(shape)
    def sum(self, A, axis = 0, delta = 0):#todo: replace by p1 method
        sum_A = self.decrypt(A).sum(axis)+delta
        return self.encrypt(sum_A)
    def send_div(self, E_z , d):
        z = ops.decrypt(E_z, self.vk)
        c = torch.div(z, d).long()
        return ops.encrypt(c,self.pk)
    def send_mul(self,E_a_prime, E_b_prime):
        return ops.mul_enc(E_a_prime, E_b_prime, self.pk, self.vk , self.q)
    def send_sum(self,E_a_prime, E_b_prime): 
        E_sum_A = self.sum( E_a_prime, axis = 1 )
        E_sum_B = self.sum( E_b_prime, axis = 0)
        return E_sum_A, E_sum_B
    def send_matmul(self,E_a_prime, E_b_prime):
        E_M = ops.dummy_matmul(E_a_prime,E_b_prime, self.pk, self.vk , self.q )
        return E_M
    def send_max(self, E_a_prime,E_b_prime ):
        a_prime = ops.decrypt(E_a_prime, self.vk)
        b_prime = ops.decrypt(E_b_prime, self.vk)
        comp = (a_prime >b_prime)*1
        return ops.encrypt(comp, self.pk)

def flatten_tensor():
    def decorator_func(func):
        def wrapper_func(*args, **kwargs):
            # Invoke the wrapped function first
            args = list(args)
            shape = args[1].shape
            args[1]= args[1].flatten()
            args[2]= args[1].flatten()
            
            retval = func(*args, **kwargs)
            # Now do something here with retval and/or action
            return retval.reshape(shape)
        return wrapper_func
    return decorator_func   


class pallier_ops:
    def __init__(self, cloud , source, pres = 4):
        self.cloud =cloud
        self.source = source
        self.pres = pres
    def broadcast(self, a , b):
        s_a = a.shape
        s_b = b.shape
        if (len(s_a)==len(s_b) and s_a[0]==s_b[0]):
            if s_b[-1]==1:
                return a, b.repeat(1,s_a[-1])
            if s_a[-1]==1:
                return a.repeat(1,s_a[-1]), b
        if len(s_b)==1: return a, b.unsqueeze(0).repeat(s_a[0],1)
        return a, b
    def sum(self,E_a,  axis):
        if axis ==0:
            return torch.tensor([self.cloud.sum(E_a[i]) for i in range(E_a.shape[axis])])
    def decrypt(self, M): return self.source.decrypt(M)
    def encrypt(self, M): return self.source.encrypt(M)
    def add(self, E_a, E_b):
        E_a, E_b = self.broadcast(E_a, E_b)
        #print(E_a.shape, E_b.shape)
        return self.cloud.add(E_a, E_b)
    def sub(self, E_a, E_b): 
        E_a, E_b = self.broadcast(E_a, E_b)
        return self.cloud.sub(E_a, E_b)
    @flatten_tensor()
    def mul(self, E_a, E_b):

        r_a  = 0#random.randint(0,2**self.pres )
        r_b  = 0#random.randint(0,2**self.pres )
        E_r_a = ops.encrypt_s(r_a,  self.cloud.pk)
        E_r_b = ops.encrypt_s(r_b,  self.cloud.pk)
        #print("done")
        E_a_prime = self.cloud.send(E_a , E_r_a)
        E_b_prime = self.cloud.send( E_b , E_r_b)
        #print("Sent")
        E_M = self.source.send_mul (E_a_prime,E_b_prime )
        #print("Sentmul")
        #print(E_a[1], E_b[1], E_M[1])
        return self.cloud.rec_mul(E_a.long(), E_b.long(), E_M.long(), r_b, r_a)
    def matmul(self, E_a, E_b):
        r_a  = random.randint(0,2**self.pres )
        r_b  = random.randint(0,2**self.pres )
        E_r_a = ops.encrypt_s(r_a,  self.cloud.pk)
        E_r_b = ops.encrypt_s(r_b,  self.cloud.pk)
        a_shape = E_a.shape
        b_shape = E_b.shape
        E_a_prime = self.cloud.send(E_a.flatten() , E_r_a).reshape(a_shape)
        E_b_prime = self.cloud.send(E_b.flatten() , E_r_b).reshape(b_shape)
        E_M = self.source.send_matmul (E_a_prime,E_b_prime )
        E_sum_A, E_sum_B = self.source.send_sum(E_a,E_b )
        r = r_a*(E_a_prime.size(1)-1)
        E_r = ops.encrypt_s(r,  self.cloud.pk)
        E_sum_A_ = self.cloud.send(E_sum_A , E_r)
        return self.cloud.rec_matmul(E_sum_A_, E_sum_B, E_M, r_b, r_a)
    def max_(self, E_a, E_b):
        r_a  = random.randint(0,500 )
        r_b  = random.randint(50000,100000 )
        r = torch.tensor(r_a).repeat(E_a.size(0)) 
        E_r_a = self.encrypt(2*r)
        E_r_b = self.encrypt(r)
        E_a_r = self.cloud.send_comp ( E_a, E_r_a, r_b)
        E_b_r = self.cloud.send_comp (E_b, E_r_b, r_b)
        E_i = self.source.send_max(  E_a_r, E_b_r) #E_i
        E_diff = self.sub(E_a, E_b) # E_a*E_b^-1
        E_idiff = self.mul(E_diff, E_i) # E(i(a-b))
        return self.add(E_idiff, E_b) # E(i(a-b) +b)
    def max(self, E_a, E_b):
        shape = E_a.shape
        return self.max_(E_a.flatten(), E_b.flatten()).reshape(shape)
    
    def relu_(self, E_a, val = 0):
        r = torch.tensor(val).repeat(E_a.size(0)) 
        E_r = self.encrypt(r)
        return self.max_( E_a, E_r)
    def relu(self, E_a, val = 0):
        return self.relu_(E_a.flatten(), val).reshape(E_a.shape)
    def einsum_ij(self, a, b):
        batch = a.size(0)
        n_a = a.size(1)
        n_b = b.size(1)
        return self.mul(a.repeat(n_b, 1,1).permute(1,2,0).flatten(),b.repeat(n_a,1,1).permute(1,0,2).flatten()).reshape(batch, n_a,n_b)
    #@flatten_tensor()
    def div(self, E_a, d, *args, **kwargs):

        shape = E_a.shape
        E_a = E_a.flatten()
        d = torch.ones(shape).flatten()*d
        r_a = random.randint(0,2**self.pres )
        E_r_a = ops.encrypt_s(r_a,  self.cloud.pk)
        E_z = self.cloud.send( E_a , E_r_a)
        E_c = self.source.send_div( E_z , d)
        E_a_d = self.cloud.rec_div(E_c, d, r_a)
        return E_a_d.reshape(shape)

def encrypted_operations(pres = 30, q =2**20):
    keys = ops.pcs_keys(pres)
    public_key = keys[0]
    private_key = keys[1]
    phe = pallier_ops(P1(public_key, q = q), P2(private_key, public_key, q = q), pres = 12)
    return phe


if __name__ == "__main__":
    keys = ops.pcs_keys(30)
    public_key = keys[0]
    private_key = keys[1]

    phe = pallier_ops(P1(public_key), P2(private_key, public_key), pres = 12)
    N_samples = 8
    a = torch.randint(0,100, (N_samples,)).to(torch.int64)
    b = a.clone()#torch.randint(0,100, (N,)).to(torch.int64)
    print('\ndata :', a)
    
    E_a = ops.encrypt(a, public_key)
    E_b = ops.encrypt(b, public_key)

    E_a_d = phe.div(E_a, 3)
    print('\na/d :', ops.decrypt(E_a_d, private_key))


    E_a_b = phe.mul( E_a, E_b)
    print('\na = ', a)
    print('\nb = ', b)
    print('\na.b = ', a*b)
    
    print('a*b', ops.decrypt(E_a_b, private_key))
    assert torch.allclose(ops.decrypt(E_a_b, private_key), a*b)
    batch_size, feature_size = 5, 100
    weights = torch.randint(1,1000, (feature_size,10)).to(torch.int64)
    E_weights = ops.encrypt(weights.flatten(), public_key).reshape((feature_size,10))
    examples = torch.randint(1, 1000, (batch_size, feature_size) )
    E_examples = ops.encrypt(examples.flatten(), public_key).reshape((batch_size, feature_size))
    E_weights = E_weights
    
    D_mul = torch.matmul(examples,weights)
    
    E_ab = phe.matmul(E_examples,E_weights )
 
    D_ab = phe.decrypt(E_ab)
 
    assert torch.allclose(D_mul, D_ab)
   