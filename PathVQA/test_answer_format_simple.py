import sys
import os
sys.path.append('/nas/leiwenhui/tys/PathVQA/MUMC/')

# 模拟原始dataset的答案格式
def test_answer_format():
    # 模拟原始dataset返回的答案格式
    sample_answer = ["in the canals of hering[SEP]"]  # 这是一个包含单个答案的列表
    
    print("=== 测试答案格式处理 ===")
    print(f"原始答案格式: {sample_answer}")
    print(f"类型: {type(sample_answer)}")
    print(f"长度: {len(sample_answer)}")
    
    # 模拟特征提取时的处理
    if isinstance(sample_answer, list) and len(sample_answer) > 0:
        true_answer = sample_answer[0]
        print(f"提取的真实答案: {true_answer}")
        print(f"真实答案类型: {type(true_answer)}")
    else:
        print("答案格式错误")

if __name__ == "__main__":
    test_answer_format() 