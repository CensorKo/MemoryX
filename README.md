# MemoryX Python SDK

让 AI Agents 轻松拥有持久记忆

## 快速开始

```python
from memoryx import connect_memory

# 自动注册并连接
memory = connect_memory()

# 存储记忆
memory.add("用户喜欢深色模式")

# 搜索记忆
results = memory.search("用户偏好")
```

## 功能特性

- 🔧 **自动注册** - Agent 自动注册，无需手动配置
- 💾 **永久存储** - 记忆永久保存到向量数据库
- 🔍 **智能搜索** - 基于语义的相似度搜索
- 🏷️ **认知分类** - 自动分类为情景/语义/程序/情感/反思记忆
- 🔒 **AES-256 加密** - 您的原始记忆内容使用 AES-256-GCM 加密存储
- 🌐 **开源可审计** - 100% 开源代码，接受社区审计，确保没有后门
- 🛡️ **隐私安全** - 机器隔离，验证码认领机制

## 完整示例

```python
from memoryx import connect_memory

# 连接记忆系统
memory = connect_memory()

# 存储不同类型的记忆
memory.add(
    content="用户是Python开发者",
    category="semantic"  # 语义记忆
)

memory.add(
    content="用户昨天去了北京",
    category="episodic"  # 情景记忆
)

# 列出所有记忆
memories = memory.list(limit=10)

# 搜索相关记忆
results = memory.search("用户职业")
for item in results["data"]["data"]:
    print(f"- {item['content']}")

# 删除记忆
memory.delete("memory_id_here")

# 获取认领验证码
code = memory.get_claim_code()
print(f"认领验证码: {code}")
```

## 安装

```bash
pip install memoryx
```

## 认领机器

Agent 注册后，访问 [t0ken.ai/agent-register](https://t0ken.ai/agent-register) 输入验证码认领这台机器。

## 安全与开源

**🔒 AES-256 加密存储**
- 您的原始记忆内容使用 AES-256-GCM 加密存储
- 每个用户拥有独立的加密密钥
- 服务端永不触碰明文，保障数据安全

**🌐 100% 开源可审计**
- 完整的开源代码：[github.com/t0ken-ai/MemoryX](https://github.com/t0ken-ai/MemoryX)
- 接受社区审计，确保没有后门
- 您可以查看、验证甚至改进我们的加密实现
- 许可证：MIT

## 文档

详细文档请访问: https://docs.t0ken.ai
# Test sync - Sun Feb 15 00:06:27 CST 2026
