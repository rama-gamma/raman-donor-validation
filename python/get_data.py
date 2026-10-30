"""Download the public dataset and deposited MATLAB code (Zenodo record 21901443, version 2, CC-BY 4.0)
into ../data/ and verify MD5 checksums. The data are not redistributed in this archive."""
import hashlib, os, urllib.parse, urllib.request, zipfile
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FILES = {'DATASET_ID UPDATED v2.xlsx': 'db23f4bec6abdaec62ac8c446d5bd33b',
         'code.zip': '68731750dae28b7d15ce35f9062c8d30'}
os.makedirs(D, exist_ok=True)
for name, md5 in FILES.items():
    path = os.path.join(D, name)
    if not os.path.exists(path):
        url = 'https://zenodo.org/api/records/21901443/files/' + urllib.parse.quote(name) + '/content'
        print('downloading', name); urllib.request.urlretrieve(url, path)
    h = hashlib.md5(open(path, 'rb').read()).hexdigest()
    print(name, 'md5 ok' if h == md5 else f'MD5 MISMATCH ({h})')
    assert h == md5
zipfile.ZipFile(os.path.join(D, 'code.zip')).extractall(D)   # data/code/pcasvm.m, pcalda_k.m, main_code.mlx
