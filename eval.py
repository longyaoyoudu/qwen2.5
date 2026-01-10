from evalscope import TaskConfig, run_task
import logging

# 启用详细日志
logging.basicConfig(level=logging.INFO)

task_cfg = TaskConfig(
    model='./output_zero3',
    api_url='http://127.0.0.1:8000/v1/chat/completions',
    eval_type='openai_api',
    datasets=[
        'data_collection',
    ],
    dataset_args={
        'data_collection': {
            'split': 'test',  # 使用测试集
            'limit': 10,      # 限制样本数量（先测试少量数据）
            'dataset_id': '/data/AIfusion/Tony2016Edu/Ruilong_Jin/Program/qwen_program/data/qwen3_test.jsonl',
            'filters': {'remove_until':'</think>'}  #过滤掉思考内容
        }
    },
    eval_batch_size=128,
    generation_config={
        'max_tokens': 10000, #最大生成token数，建议设置较大值避免输出截断
        'temperature': 0.6, #采样温度（qwen报告推荐值）
        'top_p': 0.95, #top-p采样（qwen报告推荐值）
        'top_k': 20, #top-k采样（qwen报告推荐值）
        'n': 1 #每个请求产生的回复数量
    },
    timeout=60000, #超时时间
    stream=True, #是否使用流式输出
    limit=100, #设置为2000条数据进行测试

)

run_task(task_cfg=task_cfg)

# try:
#     result = run_task(task_cfg=task_cfg)
#     print("评估成功!")
#     print(f"结果: {result}")
# except Exception as e:
#     print(f"评估失败: {e}")
#     import traceback
#     traceback.print_exc()