import gzip

def unzip_gz(n):
    output_file_path = f'graph{n}.g6'
    gz_file_path = f'{output_file_path}.gz'
    print(f'Unzipping {gz_file_path} to {output_file_path}')
    with gzip.open(gz_file_path, 'rb') as gz_file:
        with open(output_file_path, 'wb') as output_file:
            output_file.write(gz_file.read())