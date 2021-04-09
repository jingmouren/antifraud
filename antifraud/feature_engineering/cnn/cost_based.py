import numpy as np
import pandas as pd

def c_main(cutoff = 0.25, score_list = None):
    c = cutoff
    try:
        y_val = np.array(score_list)
    except:
        a = pd.read_excel('Save_Excel.xlsx')
        y_val = a[0].tolist()
    b = pd.read_csv('transaction.csv')
    y_label = b['label'].tolist()
    t = len(y_label)
    cost = [0.0 for _ in range(t)]
    for i in range(t):
        j_legi = 0 ;j_fr = 0
        for j in range(t):
            tmp_dsn = abs(y_val[i] - y_val[j]) - c
            if y_label[j] == 0:
                if tmp_dsn < 0:
                    j_legi +=1
            if y_label[j] == 1:
                if tmp_dsn < 0:
                    j_fr +=1
        print('Start generate the cost of transaction {}'.format(str(i+1)))
        cost[i] = j_legi/j_fr

    cost_dataframe = pd.DataFrame(cost)
    writer = pd.ExcelWriter('cost.{}.xlsx'.format(c))
    cost_dataframe.to_excel(writer,'page_1',float_format='%.5f') # float_format 控制精度
    writer.save()
    return writer, cost

def transf_cost(costfilename):
    cc = pd.read_excel(costfilename)
    cost_obj = cc[0].tolist()
    return cost_obj


if __name__ == '__main__':
    c_main()
