from numpy import array, full


def linkedlist2d(x: array, y: array, nx=None, ny=None) -> (int, array):
    '''

    :param x:
    :param y:
    :param nx:
    :param ny:
    :return:
    '''

    if nx is None:
        nx = max(x)
    if ny is None:
        ny = max(y)

    first = full((nx, ny), -1)
    last = full((nx, ny), -1)
    lkdlist = full(len(x), -1)

    for index, item in enumerate(x):
        i = min(max(int(item), 0), nx)
        j = min(max(int(y[index]), 0), ny)

        if first[i, j] == -1:
            first[i, j] = index
        else:
            lkdlist[last[i, j]] = index
        last[i, j] = index

    # Creating a single list
    index_last = start = -1
    for i in range(0, nx-1):
        for j in range(0, ny-1):
            index_first = first[i, j]
            if index_first != -1:
                if index_last != -1:
                    lkdlist[index_last] = index_first
                else:
                    start = index_first
                index_last = last[i, j]

    return start, lkdlist
