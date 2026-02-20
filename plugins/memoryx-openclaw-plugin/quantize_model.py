"""
potion-multilingual-128M ONNX模型量化工具

功能：将FP32的potion-multilingual-128M ONNX模型转换为INT8量化版本
用途：减少模型体积(489MB -> 123MB)，便于在浏览器/边缘设备部署

使用方法：
    python quantize_model.py

输入模型：potion-multilingual-128M/model.onnx (FP32, 489MB)
输出模型：potion-multilingual-128M/model_int8.onnx (INT8, 123MB)

量化效果：
    - 原始模型: 489MB
    - 量化后: 123MB  
    - 压缩比: 4x
    - 精度损失: <1% (相似度0.7077 vs 原始0.7083)
    - 推理速度: ~9ms (比API快很多)

测试方法：
    python test_quantized.py

注意事项：
    - 量化后精度略有下降，但相似度误差<1%
    - 推理速度：INT8比FP32慢约20倍(8.89ms vs 0.44ms)
    - 但仍比调用远程API快很多(通常100-500ms)
"""

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper, helper, AttributeProto
import os

INPUT_PATH = os.path.join(os.path.dirname(__file__), "models/potion-multilingual-128M/model.onnx")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "models/potion-multilingual-128M/model_int8.onnx")


def quantize_weight(weight_fp32):
    """
    将FP32权重量化为UINT8
    
    量化公式: quantized = round(fp32 / scale + zero_point)
    反量化公式: fp32 = (quantized - zero_point) * scale
    """
    min_val = weight_fp32.min()
    max_val = weight_fp32.max()
    
    scale = (max_val - min_val) / 255.0
    zero_point = int(round(-min_val / scale))
    zero_point = max(0, min(255, zero_point))
    
    weight_scaled = weight_fp32 / scale + zero_point
    weight_uint8 = np.clip(np.round(weight_scaled), 0, 255).astype(np.uint8)
    
    return weight_uint8, np.float32(scale), np.uint8(zero_point)


def update_node_inputs(node, old_name, new_name):
    """更新节点及其子图中的输入引用"""
    for i in range(len(node.input)):
        if node.input[i] == old_name:
            node.input[i] = new_name
    
    for attr in node.attribute:
        if attr.type == AttributeProto.GRAPH:
            update_graph_inputs(attr.g, old_name, new_name)


def update_graph_inputs(graph, old_name, new_name):
    """递归更新图中所有节点的输入引用"""
    for node in graph.node:
        update_node_inputs(node, old_name, new_name)


def quantize_model():
    """主量化流程"""
    print(f"读取模型: {INPUT_PATH}")
    print(f"原始大小: {os.path.getsize(INPUT_PATH) / 1024 / 1024:.1f} MB")
    
    model = onnx.load(INPUT_PATH)
    graph = model.graph
    
    target_weight_name = "embedding_bag.weight"
    quantized_info = None
    
    for init in graph.initializer:
        if init.name == target_weight_name:
            weight_fp32 = numpy_helper.to_array(init)
            print(f"原始权重: shape={weight_fp32.shape}, dtype={weight_fp32.dtype}")
            
            weight_uint8, scale, zero_point = quantize_weight(weight_fp32)
            print(f"量化权重: shape={weight_uint8.shape}, dtype={weight_uint8.dtype}")
            
            quantized_info = {
                'uint8': weight_uint8,
                'scale': scale,
                'zero_point': zero_point,
                'shape': list(weight_fp32.shape),
                'original_name': target_weight_name
            }
            break
    
    if not quantized_info:
        print("错误: 未找到 embedding_bag.weight")
        return
    
    print("\n修改模型...")
    
    new_initializers = []
    for init in graph.initializer:
        if init.name != target_weight_name:
            new_initializers.append(init)
    
    quantized_weight_init = helper.make_tensor(
        name="embedding_bag.weight.quantized",
        data_type=TensorProto.UINT8,
        dims=quantized_info['shape'],
        vals=quantized_info['uint8'].tobytes(),
        raw=True
    )
    new_initializers.append(quantized_weight_init)
    
    scale_init = helper.make_tensor(
        name="embedding_bag.weight.scale",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[quantized_info['scale']],
        raw=False
    )
    new_initializers.append(scale_init)
    
    zp_init = helper.make_tensor(
        name="embedding_bag.weight.zero_point",
        data_type=TensorProto.UINT8,
        dims=[1],
        vals=[int(quantized_info['zero_point'])],
        raw=False
    )
    new_initializers.append(zp_init)
    
    del graph.initializer[:]
    graph.initializer.extend(new_initializers)
    
    dequant_node = helper.make_node(
        "DequantizeLinear",
        inputs=["embedding_bag.weight.quantized", "embedding_bag.weight.scale", "embedding_bag.weight.zero_point"],
        outputs=["embedding_bag.weight.dequant"],
        name="DequantizeLinear_embedding_bag"
    )
    
    new_nodes = [dequant_node]
    for node in graph.node:
        update_node_inputs(node, target_weight_name, "embedding_bag.weight.dequant")
        new_nodes.append(node)
    
    del graph.node[:]
    graph.node.extend(new_nodes)
    
    print("验证模型...")
    try:
        onnx.checker.check_model(model)
        print("✓ 模型验证通过")
    except Exception as e:
        print(f"模型验证警告: {str(e)[:100]}...")
    
    print(f"\n保存量化模型: {OUTPUT_PATH}")
    onnx.save(model, OUTPUT_PATH)
    print(f"量化后大小: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f} MB")
    print(f"压缩比: {os.path.getsize(INPUT_PATH) / os.path.getsize(OUTPUT_PATH):.1f}x")


if __name__ == "__main__":
    quantize_model()
