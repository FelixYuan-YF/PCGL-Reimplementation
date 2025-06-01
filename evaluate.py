import json
from collections import Counter
import jieba  # 中文分词
import argparse

def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def calculate_avg_len(file):
    lens = []

    for entry in file:
        sentence = entry.get("generate", {}).get("value", "")

        lens.append(len(sentence))

    avg_len = sum(lens) / len(lens) if lens else 0
    
    return avg_len

def calculate_dist(file):
    corpus_unigrams = []
    corpus_bigrams = []
    sentence_unigrams = []
    sentence_bigrams = []

    for entry in file:
        sentence = entry.get("generate", {}).get("value", "")

        # 中文分词
        words = list(jieba.cut(sentence))

        corpus_unigrams.extend(words)
        corpus_bigrams.extend([tuple(words[i:i+2]) for i in range(len(words)-1)])

        sentence_unigrams.append(len(Counter(words))/len(words) if words else 0)
        sentence_bigrams.append(len(Counter([tuple(words[i:i+2]) for i in range(len(words)-1)])) / len(words) if words else 0)
    
    # 计算Dist-1和Dist-2
    corpus_dist1 = len(Counter(corpus_unigrams)) / len(corpus_unigrams) if corpus_unigrams else 0
    corpus_dist2 = len(Counter(corpus_bigrams)) / len(corpus_bigrams) if corpus_bigrams else 0
    sentence_dist1 = sum(sentence_unigrams) / len(sentence_unigrams) if sentence_unigrams else 0
    sentence_dist2 = sum(sentence_bigrams) / len(sentence_bigrams) if sentence_bigrams else 0
    
    return corpus_dist1, corpus_dist2, sentence_dist1, sentence_dist2

def calculate_structure_success(file):
    rates = []
    for entry in file:
        key_words = entry.get("key_words", {})
        pun_words = key_words.get("pun_words", [])
        alt_words = key_words.get("alternative_words", [])
        pun_label = " / ".join(pun_words) if pun_words else "N/A"
        sentence = entry.get("generate", {}).get("value", "")
        if alt_words:
            success = any(alt_word in sentence for alt_word in alt_words) and any(pun_word in sentence for pun_word in pun_words)
        else:
            success = any(pun_word in sentence for pun_word in pun_words)

        rates.append({
            "pun_words": pun_label,
            "success": int(success),
        })
    return rates

def parse_args():
    parser = argparse.ArgumentParser(description="the script for compute success rate")
    parser.add_argument('input_file', type=str, default='processed_data/dpo_homographic.json', help='input file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    data_entries = load_json(args.input_file)

    avg_length = calculate_avg_len(data_entries)

    corpus_dist1, corpus_dist2, sentence_dist1, sentence_dist2 = calculate_dist(data_entries)

    success = calculate_structure_success(data_entries)

    total = len(success)
    success_count = sum(r['success'] for r in success)
    success_rate = success_count / total if total else 0

    print(f"Total: {total}")
    print(f"Average Length: {avg_length:.4f}")
    print("Corpus Diversity:")
    print(f"Dist-1: {corpus_dist1:.4f}, Dist-2: {corpus_dist2:.4f}")
    print("Sentence Diversity:")
    print(f"Dist-1: {sentence_dist1:.4f}, Dist-2: {sentence_dist2:.4f}")
    print(f"Structure Success Count: {success_count}")
    print(f"Structure Success Rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()