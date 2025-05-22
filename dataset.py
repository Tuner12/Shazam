import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Parameters:
            csv_file (str): Path to the CSV file. It must contain 'image_id' and 'top_label_name' columns.
            root_dir (str): Path to the root directory containing the category subfolders.
            transform (callable, optional): Image preprocessing and augmentation.
        """
        # Load CSV file into a pandas DataFrame
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir  # Root directory for images
        self.transform = transform  # Image preprocessing pipeline

        # Extract image paths and labels
        self.image_paths = self.data['image_id'].tolist()
        self.labels = self.data['top_label_name'].tolist()
        
        # Map label names to numerical indices
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}
        self.labels = [self.label_to_idx[label] for label in self.labels]

        # Automatically retrieve category folders
        self.category_folders = self._get_category_folders()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Parameters:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (image, label), where image is the preprocessed image tensor, and label is the numerical class index.
        """
        # Retrieve the image name and corresponding label
        image_name = self.image_paths[idx]
        category_folder = self._get_category_folder(image_name)  # Identify the folder based on image name
        img_path = os.path.join(self.root_dir, category_folder, image_name)
        
        # Load the image and convert to RGB format
        image = Image.open(img_path).convert('RGB')
        
        # Apply the transformation pipeline if provided
        if self.transform:
            image = self.transform(image)
        
        # Fetch the numerical label
        label = self.labels[idx]
        return image, label

    def _get_category_folders(self):
        """
        Automatically retrieves all subfolders in the root directory.

        Returns:
            list: List of category folder names.
        """
        return [folder for folder in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, folder))]

    def _get_category_folder(self, image_name):
        """
        Determines the category folder based on the image name.

        Parameters:
            image_name (str): Name of the image file.

        Returns:
            str: Corresponding category folder name.
        """
        # Search for the image in all category folders
        for folder in self.category_folders:
            if os.path.exists(os.path.join(self.root_dir, folder, image_name)):
                return folder
        
        # Raise an error if no category folder is found
        raise ValueError(f"Cannot determine category folder for {image_name}")

# # Example usage for testing purposes
# if __name__ == "__main__":
#     root_dir = '/ailab/public/pjlab-smarthealth03/leiwenhui/Data/unitho/800'  # Root directory for the dataset
#     train_csv = os.path.join(root_dir, 'train.csv')
#     test_csv = os.path.join(root_dir, 'test.csv')

if __name__ == "__main__":
    root_dir = '/ailab/public/pjlab-smarthealth03/leiwenhui/Data/unitho/800'  # Root directory for the dataset
    train_csv = os.path.join(root_dir, 'train.csv')
    test_csv = os.path.join(root_dir, 'test.csv')

    # Define image preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize image
    ])

    # Create dataset instances for train and test datasets
    train_dataset = ImageDataset(train_csv, root_dir, transform=transform)
    test_dataset = ImageDataset(test_csv, root_dir, transform=transform)

    # Test dataset length and a sample
    print(f"Number of training samples: {len(train_dataset)}")
    img, label = train_dataset[0]
    print(img.size)
    print(f"Image shape: {img.shape}, Label: {label}")
    
    print(f"Number of training samples: {len(test_dataset)}")
    img, label = test_dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")
    
    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        print(f"Sample {idx}: Image shape: {img.shape}, Label: {label}")