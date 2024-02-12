from numpy import array, full


def linkedlist2d(x: array, y: array, nx=None, ny=None) -> list:
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
        i = min(max(int(item), 0), nx - 1)
        j = min(max(int(y[index]), 0), ny - 1)

        if first[i, j] == -1:
            first[i, j] = index
        else:
            lkdlist[last[i, j]] = index
        last[i, j] = index
    

    # Creating a single list
    result = []
    for i in range(0, nx):
        for j in range(0, ny):
            index = first[i, j]
            if index != -1:
                result.append(index)
                index = lkdlist[index]
                while index != -1:
                    result.append(index)
                    index = lkdlist[index]
    
    return result
