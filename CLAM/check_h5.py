import h5py

h5_file_path = 'chief_features/h5_files/4242.h5'

with h5py.File(h5_file_path, 'r') as f:
    print("Datasets in the file:", list(f.keys()))
    features = f['features'][:]
    coordinates = f['coords'][:]
    print("Features shape:", features.shape)  # (16935, 1024)
    print("Coordinates shape:", coordinates.shape)  # (16935, 2)
    # print(coordinates)
import torch

pt_file_path = 'chief_features/pt_files/4242.pt'

features_tensor = torch.load(pt_file_path)
print("Features tensor shape:", features_tensor.shape)  # (16935, 1024)
