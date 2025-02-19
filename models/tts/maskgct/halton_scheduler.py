def halton_sequence(b):
    """Generator function for Halton sequence."""
    n, d = 0, 1
    while True:
        x = d - n
        if x == 1:
            n = 1
            d *= b
        else:
            y = d // b
            while x <= y:
                y //= b
            n = (b + 1) * y - x
        yield n / d

def discrete_halton_sequence(b, size):
    seq = halton_sequence(b)
    min_value = 0
    max_value = size
    discrete_sequence = []

    while len(discrete_sequence) < size:
        # list.add(int(next(seq) * (max_value - min_value)))
        value_scaled = int(next(seq) * (max_value - min_value))
        if value_scaled not in discrete_sequence:
            discrete_sequence.append(value_scaled)
    
    return discrete_sequence

if __name__ == '__main__':
    discrete_sequence = discrete_halton_sequence(2, 900)
    print(len(set(discrete_sequence)))