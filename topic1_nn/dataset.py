import numpy as np

def load_mnist_images(filename):

    with open(filename, 'rb') as f:

        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        print("magic:", magic)
        print("num:", num)
        print("size:", rows, cols)

        data = np.frombuffer(f.read(), dtype=np.uint8)

        images = data.reshape(num, rows, cols)

        return images
    
def load_mnist_labels(filename):

    with open(filename, 'rb') as f:

        magic = int.from_bytes(f.read(4), 'big')
        num = int.from_bytes(f.read(4), 'big')

        print("magic:", magic)
        print("num:", num)

        labels = np.frombuffer(f.read(), dtype=np.uint8)

        return labels