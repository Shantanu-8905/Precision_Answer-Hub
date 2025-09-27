def save_data(data, filename):
    """Saves a list of strings to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item + '\n')
