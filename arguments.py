import argparse

from prettytable import PrettyTable

def print_args(parse_args):
    x = PrettyTable()
    x.field_names = ["Arg.", "Value"]
    for arg in vars(parse_args):
        x.add_row([arg, getattr(parse_args, arg)])
    print(x)

def arg_parse():
    parser = argparse.ArgumentParser(description='KAN 4 Quant')

    parser.add_argument('--model', type=str, dest='model', help='model type')
    parser.add_argument('--opt', type=str, dest='opt', help='optimizer "LBFGS" or "Adam"')
    parser.add_argument('--update_grid', type=bool, dest='update_grid', help='update grid or not')
    parser.add_argument('--batch', type=int, dest='batch', help='batch size')
    parser.add_argument('--lamb', type=float, dest='lamb', help='overall penalty strength')
    parser.add_argument('--lr', type=float, dest='lr', help='learning rate')
    parser.add_argument('--n_epoch', type=int, dest='n_epoch', help='number of training epochs')
    parser.add_argument('--steps', type=int, dest='steps', help='training steps')
    parser.add_argument('--width', type=list, dest='width', help='number of neuron in each layer')
    parser.add_argument('--grid', type=int, dest='grid', help='number of grid intervals')
    parser.add_argument('--k', type=int, dest='k', help='the order of piecewise polynomial')
    parser.add_argument('--seed', type=int, dest='seed', help='42')
    parser.add_argument('--device', type=str, dest='device', help='pytorch device')
    parser.add_argument('--datapath', type=str, dest='datapath', help='path of data')

    parser.set_defaults(
        model       = 'PureKAN',
        opt         = 'LBFGS',
        update_grid = True,
        batch       = 10000,
        lamb        = 0.003,
        lr          = 0.00447,
        n_epoch     = 100,
        steps       = 100,
        width       = [8, 8, 1],
        grid        = 3,
        k           = 3,
        seed        = 42,
        device      = 'cuda',
        datapath    = "./data"
    )

    args = parser.parse_args()
    return args
