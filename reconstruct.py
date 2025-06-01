import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct data")
    parser.add_argument(
        "--raw_file",
        type=str,
        default="processed_data/homographic.json",
        help="Path to the input file",
    )
    parser.add_argument(
        "--generate_file",
        type=str,
        default="qwen2.5-3b-homographic-cn.json",
        help="Path to the DPO file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="processed_data/dpo_homographic.json",
        help="Path to the output file",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    with open(args.raw_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    dpo_data = []
    with open(args.generate_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            for item in raw:
                if item['prompt'] in entry['prompt']:
                    if "\n\n" in entry["predict"]:
                        # 以“\n\n”分割
                        parts = entry["predict"].split("\n\n")
                        # 获取开始和结束为“”或“”的那一个
                        for part in parts:
                            if part.startswith("“") or part.endswith("”"):
                                # 获取“”内的内容
                                entry["predict"] = part[1:-1]
                    
                    dpo_data.append({
                        "conversations": [
                            {
                                "from": "human",
                                "value": item["prompt"]
                            }
                        ],
                        "chosen": {
                            "from": "human",
                            "value": item["chosen"]
                        },
                        "generate": {
                            "from": "gpt",
                            "value": entry["predict"]
                        },
                        "key_words": item["key_words"]
                    })
                    break


    # 将数据写入文件
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)