import numpy as np

def FreedmanDiaconisBinSize(feature_values):
    q75, q25 = np.percentile(feature_values, [75, 25])
    IRQ = q75-q25
    return 2*(IRQ)*len(feature_values)**(-1.0/3.0)

def customizeBinSize(feature_values,numBins=500):
    minVal=min(feature_values)
    maxVal=max(feature_values)
    return (maxVal-minVal)/numBins

# 产生Bins
def make_histogram_bins(data, bin_size_calculator=FreedmanDiaconisBinSize,col_index=0,numBins=500):
    data = data.reshape(-1,1)
    # 取出某一列的值
    feature_vals = [row[col_index] for row in data]
    
    # 计算该列上应该的binsize
    bin_range = bin_size_calculator(feature_vals)
    # print(bin_range)
    
    # 如果bin_range为0，则将该值设为1
    if bin_range == 0:
        bin_range = 1

    # 基于该列对数据的每行进行排序
    data_tuples =  list(enumerate(data)) 
    sorted_data_tuples = sorted(data_tuples, key=lambda tup:tup[1][col_index]) #将所有的行按照col_index所在列的值升序排序

    # 统计出该列的最值
    max_val = max(data, key = lambda datum:datum[col_index])[col_index]
    min_val = min(data, key = lambda datum:datum[col_index])[col_index]

    index_bins = [] # 用存放每个bins中的行号
    val_ranges = []
    curr = min_val
    # 将该列的值分段
    while curr<=max_val:
        index_bins.append([])
        val_ranges.append((curr,curr+bin_range))
        curr+=bin_range
    
    #逐行判断属于哪一段
    for row_index, row in sorted_data_tuples:
        for bin_num, val_range in enumerate(val_ranges):
            if val_range[0] <= row[col_index] < val_range[1]:
                index_bins[bin_num].append(row_index)
                break
    
    index_bins = [b for b in index_bins if b]
    return index_bins,val_ranges