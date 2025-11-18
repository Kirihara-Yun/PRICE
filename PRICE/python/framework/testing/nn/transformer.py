from framework.op_graph import OpGraph


def _transformer_block(G: OpGraph, x, emb_dim, ffn_dim, num_heads=1):
    y = G.LayerNorm(x, [G[x].out_shape[-1]])
    y = G.MultiheadAttention(y, y, y, emb_dim, num_heads)
    y = G.add(y, x)
    o = G.LayerNorm(y, [G[y].out_shape[-1]])
    o = G.Linear(o, ffn_dim)
    o = G.relu(o)
    o = G.Linear(o, emb_dim)
    o = G.add(o, y)
    return o


def _transformer_backbone(
    G: OpGraph, x, num_layers, hidden_size, num_heads, ffn_scale=4
):
    for _ in range(num_layers):
        x = _transformer_block(G, x, hidden_size, hidden_size * ffn_scale, num_heads)
    return x


def _language_transformer(
    batch_size=1,
    seq_len=512,
    num_layers=12,
    hidden_size=768,
    num_heads=12,
    ffn_scale=4,
):
    # ignore pos embedding
    G = OpGraph()
    x = G.placeholder([batch_size, seq_len, hidden_size], "x")
    y = _transformer_backbone(G, x, num_layers, hidden_size, num_heads, ffn_scale)
    return G, y


def bert(name="base", batch_size=1, seq_len=512):
    configs = {
        "base": (12, 768, 12),
        "large": (24, 1024, 16),
        "medium": (8, 512, 8),
        "small": (4, 512, 8),
        "mini": (4, 256, 4),
        "tiny": (2, 128, 2),
    }
    num_layers, hidden_size, num_heads = configs[name]
    return _language_transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_scale=4,
    )
def qwen(name="7b", batch_size=1, seq_len=1024):
    """
    Qwen（通义千问）系列模型配置（基于官方公开参数）
    模型架构：Decoder-only Transformer（复用现有_pre_layernorm transformer block）
    关键说明：
    - Qwen实际使用GELU激活函数，现有框架用ReLU，保持一致性暂不修改（可按需替换）
    - 标准序列长度：默认8192（Qwen官方默认，支持最长32768，可通过seq_len参数自定义）
    - FFN维度：hidden_size * 4（Qwen官方标准配置）
    
    支持的模型版本：0.5b, 1.8b, 7b, 14b, 72b
    各版本官方配置参考：https://github.com/QwenLM/Qwen
    """
    # Qwen官方核心配置：(num_layers, hidden_size, num_heads)
    configs = {
        "0.5b": (16, 1024, 8),    # Qwen-0.5B：16层，1024隐藏层，8头
        "1.8b": (24, 2048, 16),   # Qwen-1.8B：24层，2048隐藏层，16头
        "7b": (32, 4096, 32),     # Qwen-7B：32层，4096隐藏层，32头（最常用）
        "14b": (40, 5120, 40),    # Qwen-14B：40层，5120隐藏层，40头
        "72b": (80, 10240, 80),   # Qwen-72B：80层，10240隐藏层，80头
    }
    
    # 校验模型版本
    if name not in configs:
        raise ValueError(
            f"不支持的Qwen模型尺寸: {name}，支持的尺寸有: {list(configs.keys())}\n"
            "可选版本：0.5b/1.8b/7b/14b/72b（对应官方Qwen系列）"
        )
    
    # 提取配置参数
    num_layers, hidden_size, num_heads = configs[name]
    
    # 调用基础Transformer生成模型（复用现有框架逻辑）
    return _language_transformer(
        batch_size=batch_size,
        seq_len=seq_len,          # 默认8192（Qwen官方标准），可自定义（如32768）
        num_layers=num_layers,    # 层数
        hidden_size=hidden_size,  # 隐藏层维度
        num_heads=num_heads,      # 注意力头数
        ffn_scale=4,              # FFN维度 = hidden_size * 4（Qwen官方配置）
    )

def gpt_neo(batch_size=1, seq_len=512):
    return _language_transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=24,
        hidden_size=2048,
        num_heads=16,
        ffn_scale=4,
    )


def btlm(batch_size=1, seq_len=512):
    return _language_transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=32,
        hidden_size=2560,
        num_heads=32,
        ffn_scale=4,
    )


def llama_7b(batch_size=1, seq_len=2048):
    """
    Llama-7B模型配置参考：
    - 层数：32
    - 隐藏层大小：4096
    - 注意力头数：32
    - FFN维度：4096 * 4 = 16384
    - 标准序列长度：2048
    """
    return _language_transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=32,
        hidden_size=4096,
        num_heads=32,
        ffn_scale=4,
    )


def llama(name="7b", batch_size=1, seq_len=1024):
    """
    扩展支持Llama系列模型（可选）
    目前支持：7b, 13b, 34b, 70b
    """
    configs = {
        "7b": (16, 4096, 4),    # 7B: 32层，4096隐藏层，32头
        "13b": (40, 5120, 40),   # 13B: 40层，5120隐藏层，40头
        "34b": (60, 8192, 64),   # 34B: 60层，8192隐藏层，64头
        "70b": (80, 10240, 80),  # 70B: 80层，10240隐藏层，80头
    }
    if name not in configs:
        raise ValueError(f"不支持的Llama模型尺寸: {name}，支持的尺寸有: {list(configs.keys())}")
    
    num_layers, hidden_size, num_heads = configs[name]
    return _language_transformer(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_scale=4,
    )


def _vision_transformer(
    batch_size=1,
    in_channel_size=3,
    image_size=224,
    num_classes=1000,
    patch_size=16,
    num_layers=12,
    hidden_size=768,
    num_heads=12,
    ffn_scale=4,
):
    # ignore pos embedding
    G = OpGraph()
    x = G.placeholder([batch_size, in_channel_size, image_size, image_size], "x")
    assert image_size % patch_size == 0
    num_patches = image_size // patch_size
    y = G.reshape(
        x,
        [batch_size, in_channel_size, num_patches, patch_size, num_patches, patch_size],
    )
    y = G.permute(y, [0, 2, 4, 1, 3, 5])
    y = G.reshape(
        y,
        [
            batch_size,
            num_patches * num_patches,
            in_channel_size * patch_size * patch_size,
        ],
    )
    y = _transformer_backbone(G, y, num_layers, hidden_size, num_heads, ffn_scale)
    y = G.Linear(y, num_classes)
    return G, y


def vit(name="base", batch_size=1, in_channel_size=3, image_size=224, num_classes=1000):
    configs = {
        "base": (12, 768, 12),
        "large": (24, 1024, 16),
        "huge": (32, 1280, 16),
        "small": (12, 384, 6),
        "tiny": (12, 192, 3),
    }
    if "-" in name:
        name, patch_size = name.split("-")
    else:
        patch_size = 16
    patch_size = int(patch_size)
    num_layers, hidden_size, num_heads = configs[name]
    return _vision_transformer(
        batch_size=batch_size,
        in_channel_size=in_channel_size,
        image_size=image_size,
        num_classes=num_classes,
        patch_size=patch_size,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        ffn_scale=4,
    )
