import os
import numpy as np
from utils.mlgp_result_record import MLGP_record_parser
from matplotlib import pyplot as plt




if __name__ == '__main__':
    test_file = './record_ref.txt'

    _parser = MLGP_record_parser(test_file)
    data = _parser.get_data()

    for i in range(len(data)):
        result = data[i]['@record_result@']
        
        epoch_index = result[0].index('epoch')
        epoch = [_l[epoch_index] for _l in result[1:]]

        rmse_index = result[0].index('rmse')
        rmse = [_l[rmse_index] for _l in result[1:]]

        plt.plot(epoch, rmse)
        plt.show()


    