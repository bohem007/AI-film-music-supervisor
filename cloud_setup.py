import os
import mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import boto3
from botocore.config import Config

# Load .env for Account ID, Keys, and Bucket Name
load_dotenv()

def get_r2_client():
    """Initializes a thread-safe Boto3 client for Cloudflare R2."""
    # max_pool_connections should match or exceed your thread count
    return boto3.client(
        service_name="s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", max_pool_connections=25)
    )

def upload_worker(file_path, root_path, bucket_name, s3_client):
    """Function to handle the individual file upload logic."""
    # STRIPPING THE ROOT: relative_to(root_path) removes 'musicWavMp3' from the path
    # Example: musicWavMp3/Electronic/beep.mp3 -> Electronic/beep.mp3
    remote_key = str(file_path.relative_to(root_path)).replace("\\", "/")
    
    content_type, _ = mimetypes.guess_type(file_path)
    extra_args = {'ContentType': content_type} if content_type else {}

    try:
        s3_client.upload_file(
            Filename=str(file_path),
            Bucket=bucket_name,
            Key=remote_key,
            ExtraArgs=extra_args
        )
        return True, None
    except Exception as e:
        return False, f"Failed {remote_key}: {str(e)}"

def main():
    local_directory = os.getenv("MUSIC_DIR")
    root_path = Path(local_directory)
    bucket_name = os.getenv("BUCKET_NAME")
    
    if not root_path.exists():
        print(f"❌ Error: Folder '{local_directory}' not found in current directory.")
        return

    # 1. Gather all files (recursive glob)
    all_files = [f for f in root_path.rglob("*") if f.is_file()]
    
    if not all_files:
        print("No files found to upload.")
        return

    print(f"⚙️  Optimized Upload: {len(all_files)} files | 15 Threads | Bucket: {bucket_name}")

    # 2. Execute with ThreadPool and Progress Bar
    s3_client = get_r2_client()
    
    with tqdm(total=len(all_files), unit="file", desc="🚀 Uploading to R2") as pbar:
        with ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks
            futures = [
                executor.submit(upload_worker, f, root_path, bucket_name, s3_client) 
                for f in all_files
            ]
            
            # Update progress bar as they finish
            for future in as_completed(futures):
                success, error_msg = future.result()
                if not success:
                    tqdm.write(f"⚠️ {error_msg}")
                pbar.update(1)

    print("\n✨ Migration complete. Root folder prefix stripped.")

if __name__ == "__main__":
    main()