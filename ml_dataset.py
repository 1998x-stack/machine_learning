from sklearn.datasets import load_iris, load_digits, make_blobs

def load_data(name, **kwargs):
    if name == 'iris':
        data = load_iris(**kwargs)
    elif name == 'digits':
        data = load_digits(**kwargs)
    elif name == 'blobs':
        data = make_blobs(**kwargs)
    else:
        raise ValueError('Unknown dataset')
    return data

