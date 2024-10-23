import torch.nn as nn

from kan import *
from copy import deepcopy
from torch import Tensor
from data_handler import DataHandler

# set default dtype
torch.set_default_dtype(torch.float32)

# GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_type(args):
    if args.model == "PureKAN":
        MODEL = PureKAN
    return MODEL

'''
default_params = {
    'opt'         : 'LBFGS',
    'update_grid' : True,
    'batch_size'  : 10000,
    'lamb'        : 0.003,
    'lr'          : 0.00447,
    'epoch'       : 100,
    'model_kwargs' :{
        'width' : [8, 8, 1],
        'grid'  : 3,
        'k'     : 3,
        'seed'  : 42,
        'device': device,
    }
}
'''

class PureKAN():
    def __init__(self, args):
        self.args = args

    def _init_model(self) -> KAN:
        self.model = KAN(width  = self.args.width,
                         grid   = self.args.grid,
                         k      = self.args.k,
                         seed   = self.args.seed,
                         device = self.args.device)
    
    def train_model(self, dataset:Tensor):
        # dataset = DataHandler.get_dataset()
        self._init_model()

        opt         = self.args.opt
        update_grid = self.args.update_grid
        batch       = self.args.batch
        lamb        = self.args.lamb
        lr          = self.args.lr
        n_epoch     = self.args.n_epoch

        if batch != -1:
            steps = dataset['train_input'].shape[0] // batch
        else:
            steps = self.args.steps

        mse_loss = nn.MSELoss()

        # results = {'train_loss' : train_loss, 'test_loss' : test_loss, 'reg' : reg}
        # results:dict = model.fit(dataset, opt=opt, loss_fn=mse_loss, update_grid=update_grid, batch=batch, steps=steps, lamb=lamb, lr=lr)
        print('training...')
        for _ in range(n_epoch):
            self.model.fit(dataset, opt=opt, loss_fn=mse_loss, update_grid=update_grid, batch=batch, steps=steps, lamb=lamb, lr=lr)

    def plot_model(self):
        self.model.plot()
        plt.gcf().savefig(f'./image_{self.model.round}_{self.model.state_id}.png', dpi=600)

    def predict(self, dataset:Tensor) -> Tensor:
        pred_result = self.model(dataset)
        return pred_result
