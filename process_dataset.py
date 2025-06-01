import json
import os
import argparse

def process_dataset(input_path, output_dir, repeat_times):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    categories = {
        'homographic': {'data': [], 'keywords': []},
        'homophonic': {'data': [], 'keywords': []}
    }

    # 处理homographic类
    if 'homographic' in dataset:
        for item in dataset['homographic']:
            keywords = item['key_words']['pun_words']
            categories['homographic']['keywords'].append(keywords)
            prompt = f"根据上述词义，请为我编写⼀个幽默的句⼦，并使⽤“{keywords[0]}”这个词在⼀个句⼦中⼀次或多次。这个句⼦应该能够引发笑声或娱乐，并展示“{keywords[0]}”这个词语的不同含义在幽默语境下的创意⽤法。你可以尝试在句⼦中加⼊意想不到的情节或转折，以增加幽默效果。请确保句⼦仍然通顺、⾃然，并具有良好的语法和逻辑结构。"
            categories['homographic']['data'].append({
                "prompt": prompt,
                "chosen": item['content'],
                "key_words": item['key_words']  # 添加key_words字段
            })

    # 处理homophonic类
    if 'homophonic' in dataset:
        for item in dataset['homophonic']:
            pun = item['key_words'].get('pun_words', [])
            alternative = item['key_words'].get('alternative_words', [])
            
            if len(pun) >= 1 and len(alternative) >= 1:
                keywords = [pun[0], alternative[0]]
                categories['homophonic']['keywords'].append(keywords)
                
                prompt = f"请为我编写⼀个幽默的句⼦，其中包含词语“{keywords[0]}”和“{keywords[1]}”。这个句⼦应该能够引发笑声或娱乐，并展示“{keywords[0]}”和“{keywords[1]}”这两个同音词语在幽默语境下的创意⽤法。你可以尝试在句⼦中加⼊意想不到的情节或转折，以增加幽默效果。请确保句⼦仍然通顺、⾃然，并具有良好的语法和逻辑结构。"
                categories['homophonic']['data'].append({
                    "prompt": prompt,
                    "chosen": item['content'],
                    "key_words": item['key_words']  # 添加key_words字段
                })

    for category, data in categories.items():
        json_path = os.path.join(output_dir, f'{category}_cn.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            datas = data['data'] * repeat_times
            json.dump(datas, f, ensure_ascii=False, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Process ChinesePun dataset.")
    parser.add_argument('--input_path', type=str, default='./ChinesePun.json', help='Path to the input dataset file.')
    parser.add_argument('--output_dir', type=str, default='./processed_data', help='Directory to save the processed data.')
    parser.add_argument('--repeat_times', type=int, default=1, help='Number of times to repeat each entry in the dataset.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    process_dataset(args.input_path, args.output_dir, args.repeat_times)