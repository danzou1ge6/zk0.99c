
def get_gmem_time(global_bandwidth, data_size): # GB/s, bytes
    data_size = data_size * 2
    return data_size / (global_bandwidth * 1024**3)

def get_smem_time(shared_bandwidth, data_size, deg, clocks, SMs): # bytes/clock, KB, no, MHz
    data_size = data_size * 2
    data_size = data_size * (i + 2)
    num_clock = data_size / (shared_bandwidth)
    return num_clock / (SMs * clocks * 10**6)

def get_compute_time(TOPS, deg, mul_op, add_op, data_size):
    num_mul = deg * data_size / 2# + (60) * data_size
    num_add = deg * data_size
    index_compute = data_size * 15
    num_ops = num_add * add_op + num_mul * mul_op + index_compute
    return num_ops  / (TOPS * 10**12)



if __name__ == "__main__":
    max_deg = 24
    words = 24
    frequency = 1410
    sms = 108
    global_bandwidth = 1555
    shared_bandwidth = 128
    int_tops = 19.5
    deg = [6,6,4,4,4]

    total_time = 0
    for i in deg:
        data_size = (2 ** max_deg)
        global_mem_time = get_gmem_time(global_bandwidth, data_size * words * 4)
        print(global_mem_time * 1000)
        total_time += global_mem_time
        share_mem_time = get_smem_time(shared_bandwidth, data_size * words * 4, i, frequency, sms)
        print(share_mem_time * 1000)
        total_time += share_mem_time
        compute_time = get_compute_time(int_tops, i, words * words * 2, words * 2, data_size)
        print(compute_time * 1000)
        total_time += compute_time

    print(total_time * 1000)
