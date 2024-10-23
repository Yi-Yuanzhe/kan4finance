from arguments import *
from models import *
from data_handler import DataHandler

# get args from arguments
args = arg_parse()
# use pretty table to print args
print_args(args)

args.device = torch.device(args.device)

MODEL = get_model_type(args)
model = MODEL(args)

data_handler = DataHandler(args.datapath)
dataset      = data_handler.get_dataset()

model.train_model(dataset)
model.plot_model()
