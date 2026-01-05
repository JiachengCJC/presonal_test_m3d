from huggingface_hub import snapshot_download, login

# 1. PASTE YOUR TOKEN HERE
# Replace 'hf_xxxxx' with the actual token you copied in Step 2
my_token = "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" 

# 2. Login using the script (Bypasses the broken CLI)
login(token=my_token)

print("Authentication successful. Starting download...")

# 3. Download
snapshot_download(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    local_dir="./LaMed/pretrained_model/llama-2-7b-chat",
    local_dir_use_symlinks=False,
    token=my_token  # Explicitly pass the token here as well
)

print("Download complete!")