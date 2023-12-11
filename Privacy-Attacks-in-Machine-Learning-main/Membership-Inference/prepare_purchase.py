import tarfile
import yaml
import os
import numpy as np
import wget

config_file = './env.yml'
with open(config_file, 'r') as stream:
    yamlfile = yaml.safe_load(stream)
    root_dir = yamlfile['root_dir']
##prepare

if not os.path.exists(os.path.join(root_dir, 'tmp')):
    os.makedirs(os.path.join(root_dir, 'tmp'))

####prepare purchase dataset
if not os.path.isfile(os.path.join(root_dir, 'tmp', 'dataset_purchase.tgz')):
    print("Dowloading purchase dataset...")
    wget.download("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz", os.path.join(root_dir, 'tmp', 'dataset_purchase.tgz'))
    print('Dataset Downloaded')



print("Prepare Purchase100 dataset")
tar = tarfile.open(os.path.join(root_dir, 'tmp', 'dataset_purchase.tgz'))
tar.extractall(path=os.path.join(root_dir, 'tmp'))
data_set =np.genfromtxt(os.path.join(root_dir, 'tmp', 'dataset_purchase'), delimiter=',')


X = data_set[:,1:].astype(np.float64)
Y = (data_set[:,0]).astype(np.int32)-1

DATASET_PATH = os.path.join(root_dir, 'purchase', 'data')
np.save(os.path.join(DATASET_PATH, 'X.npy'), X)
np.save(os.path.join(DATASET_PATH,'Y.npy'), Y)


####prepare purchase dataset
