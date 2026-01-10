import os
from modelscope import snapshot_download

# 下载模型
print("开始下载 Qwen2.5-1.5B 模型...")
print("使用 ModelScope (阿里云) 下载...")

try:
    # 使用 modelscope 下载模型
    # 注意：modelscope 的 snapshot_download 参数与 huggingface_hub 略有不同
    downloaded_path = snapshot_download(
        model_id="Qwen/Qwen2.5-1.5B",  # 或者使用 repo_id 参数
        cache_dir="./model",           # 缓存目录，相当于 local_dir
        local_files_only=False,        # 不强制使用本地文件
        # resume_download 参数在 modelscope 中不支持，会自动处理断点续传
        # token 参数可选，如果需要认证可以传入
        # token="your_token_here",
        # 文件过滤参数
        ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.git*"],
        allow_patterns=[
            "*.json", "*.pt", "*.bin", "*.model",
            "*.safetensors", "*.txt", "*.py", "*.md",
            "*.tokenizer*", "*.config*", "*.xml", "*.csv"
        ],
        max_workers=2,  # 并发下载线程数
    )
    print(f"✅ 模型下载完成！保存路径: {downloaded_path}")
    print(f"✅ 模型已缓存到: {downloaded_path}")
    
    # 检查下载的文件
    import glob
    files = glob.glob(os.path.join(downloaded_path, "**"), recursive=True)
    file_count = len([f for f in files if os.path.isfile(f)])
    print(f"📊 下载文件数: {file_count}")
    
    # 显示关键文件
    print("📋 关键文件:")
    key_patterns = ["config.json", "model.safetensors", "tokenizer", "generation_config"]
    for pattern in key_patterns:
        found_files = [f for f in files if os.path.isfile(f) and pattern in os.path.basename(f)]
        if found_files:
            for f in found_files[:2]:  # 显示前两个匹配文件
                size_mb = os.path.getsize(f) / (1024 * 1024)
                print(f"  ✅ {os.path.basename(f)} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {pattern}* (未找到)")
            
except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("请检查网络连接或 modelscope 是否安装")
    print("安装命令: pip install modelscope")
    raise
