import torch
import tensorly
from tensorly import tucker_to_tensor
import copy
from utils.eigen import eigen_pairs
tensorly.set_backend('pytorch')

def mask_map_2_single_vector_index(mask_map):
    # mask_map: bool map
    return

def nvector_index_2_single_vector_index():
    return

class posterior_output_decorator(torch.nn.Module):
    def __init__(self, K, y_predict, y_groundtrue, mask_indexes, train_data) -> None:
        super().__init__()
        self.K = K
        # self.K_inv = torch.inverse(tensorly.tenalg.kronecker(self.K)) # memory exceed
        self.K_eigen = []
        for i in range(len(K)):
            self.K_eigen.append(eigen_pairs(self.K[i]))

        self.y_predict = copy.deepcopy(y_predict.data)
        # self.y_predict.requires_grad = False 
        self.y_groundtrue = y_groundtrue
        self.mask_indexes = mask_indexes
        self.train_data = train_data

        self.d_y = []
        y_size = y_predict.data.numel()
        sample_y = y_predict.data.reshape(y_size)
        target_y = self.y_groundtrue.reshape(y_size)
        for i in range(y_size):
            if i in self.mask_indexes:
                self.d_y.append(target_y[i])
            else:
                self.d_y.append(torch.nn.Parameter(sample_y[i]))

        self.grad_d_y = []
        for _v in self.d_y:
            if _v.requires_grad:
                self.grad_d_y.append(_v)
        # self.y_decorated = torch.nn.Parameter(torch.tensor(self.y_predict))
        # with torch.no_grad():
        #     for _index in self.mask_indexes:
        #         self.y_decorated[_index] = self.y_groundtrue[_index]

        # self.d_y = torch.stack(self.d_y, 0)
        # self.d_y = self.d_y.reshape_as(self.y_predict)
        self.optimizer = torch.optim.Adam(self.grad_d_y, lr= 0.001)


    def train(self):
        # new_y = torch.stack(self.train_data+[torch.stack(self.d_y, 0).reshape_as(self.y_predict)], len(self.y_predict.shape))
        new_y = torch.cat([self.train_data, torch.stack(self.d_y, 0).reshape([*self.y_predict.shape,1])], -1)
        # loss = torch.log(self.K[0]+1e-5) + new_y@torch.inverse(tensorly.tenalg.kronecker(self.K))@new_y.T
        
        # loss = new_y@torch.inverse(tensorly.tenalg.kronecker(self.K))@new_y.T
        # loss = new_y.reshape(-1)@self.K_inv@new_y.T.reshape(-1)

        _init_value = torch.tensor([1.0]).reshape(*[1 for i in self.K])
        lambda_list = [eigen.value.reshape(-1, 1) for eigen in self.K_eigen]

        # lazy compute
        '''
        A = tucker_to_tensor((_init_value, lambda_list))
        A = A + self.noise.pow(-1)* tensorly.ones(A.shape)

        T_1 = tensorly.tenalg.multi_mode_dot(new_y, [eigen.vector.T for eigen in self.K_eigen])
        T_2 = T_1 * A.pow(-1/2)
        T_3 = tensorly.tenalg.multi_mode_dot(T_2, [eigen.vector for eigen in self.K_eigen])
        b = tensorly.tensor_to_vec(T_3)
        '''

        # # fast compute
        A = tucker_to_tensor((_init_value, lambda_list))
        A = self.noise.pow(-1)* tensorly.ones(A.shape)
        A = A[:,:,-1:]

        # for test, no change
        # tt = [eigen.vector.T for eigen in self.K_eigen]
        # tt[2] = tt[2][:500,:500]
        # test_T1 = tensorly.tenalg.multi_mode_dot(new_y[:,:,:500], tt)
        # print(test_T1.sum()) -> keep same all the time
    
        eigen_matix = [eigen.vector.T for eigen in self.K_eigen]
        # eigen_matix[2] = eigen_matix[2][-1:,:]
        T_1 = tensorly.tenalg.multi_mode_dot(new_y, [*eigen_matix[:-1], eigen_matix[-1][-1:,:]])
        T_2 = T_1 * A.pow(-1/2)
        T_3 = tensorly.tenalg.multi_mode_dot(T_2, [*eigen_matix[:-1], eigen_matix[-1][:,-1:]])
        b = tensorly.tensor_to_vec(T_3)

        loss = b.T@ b

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return

    def eval(self):
        # with torch.no_grad():
        #     predict_l2 = (self.y_predict - self.y_groundtrue).pow(2).sum()
        #     # decorate_l2 = (self.y_decorated - self.y_groundtrue).pow(2).sum()
        #     decorate_l2 = (torch.stack(self.d_y, 0).reshape_as(self.y_groundtrue) - self.y_groundtrue).pow(2).sum()
        #     print('predict l2:', predict_l2)
        #     print('decorate l2:', decorate_l2)
        # return decorate_l2< predict_l2

        return torch.stack(self.d_y, 0).reshape_as(self.y_groundtrue)