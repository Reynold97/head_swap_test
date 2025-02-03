from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import wget
import zipfile
import shutil  # Add this import for rmtree

class ModelDownloader:
    def __init__(self):
        self.save_dir = 'pretrained_models'
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'BFM'), exist_ok=True)
        
        # File IDs from Google Drive
        self.model_paths = {
            "epoch_20.pth": "1BlDBB4dLLrlN3cJhVL4nmrd_g6Jx6uP0",
            "face.zip": "1-0xOf6g58OmtKtEWJlU3VlnfRqPN9Uq7",  # Contains PIRender model
        }

    def download_bfm(self):
        """Download BFM related files"""
        bfm_url = "https://raw.githubusercontent.com/sicxu/Deep3DFaceRecon_pytorch/master/BFM/similarity_Lm3D_all.mat"
        bfm_path = os.path.join(self.save_dir, 'BFM', 'similarity_Lm3D_all.mat')
        
        if not os.path.exists(bfm_path):
            print("Downloading similarity_Lm3D_all.mat...")
            wget.download(bfm_url, bfm_path)

    def download_with_gdown(self, file_id, output_path):
        """Download file from Google Drive using gdown"""
        try:
            import gdown
            if not os.path.exists(output_path):
                print(f"Downloading to {output_path}...")
                gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
            else:
                print(f"{output_path} already exists!")
        except Exception as e:
            print(f"Error downloading {output_path}: {str(e)}")

    def extract_pirender(self):
        """Extract PIRender model from face.zip"""
        zip_path = os.path.join(self.save_dir, 'face.zip')
        face_dir = os.path.join(self.save_dir, 'face')
        
        if os.path.exists(zip_path):
            print("Extracting face.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.save_dir)
            
            # Move the model file to the correct location
            src = os.path.join(face_dir, 'epoch_00190_iteration_000400000_checkpoint.pt')
            dst = os.path.join(self.save_dir, 'epoch_00190_iteration_000400000_checkpoint.pt')
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)  # Remove existing file if it exists
                os.rename(src, dst)
                print("Moved PIRender model to correct location")
            
            # Cleanup
            os.remove(zip_path)
            if os.path.exists(face_dir):
                shutil.rmtree(face_dir)  # Remove directory and all its contents

    def run_downloads(self):
        """Run all downloads"""
        # Install gdown if not present
        os.system('pip install -q gdown')
        
        # Download BFM files
        self.download_bfm()
        
        # Download models from Google Drive
        for name, file_id in self.model_paths.items():
            self.download_with_gdown(
                file_id, 
                os.path.join(self.save_dir, name)
            )
        
        # Extract PIRender model
        self.extract_pirender()
        
        print("\nAll downloads completed!")

if __name__ == "__main__":
    downloader = ModelDownloader()
    downloader.run_downloads()