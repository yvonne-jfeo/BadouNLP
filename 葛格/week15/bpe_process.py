import os
import re
import json
from collections import defaultdict
from pathlib import Path

class BPEProcessor:
    def __init__(self, vocab_size=30000, save_dir="bpe_model", unk_token="[UNK]"):
        self.vocab_size = vocab_size
        self.save_dir = save_dir
        self.unk_token = unk_token
        self.vocab = set()
        self.merges = {}  
        self.id_to_token = {}
        self.token_to_id = {}
        
        os.makedirs(save_dir, exist_ok=True)
    
    def _get_pairs(self, word):
        pairs = set()
        if len(word) < 2:
            return pairs
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _count_pair_frequencies(self, corpus_subwords, corpus_counts):
        """统计所有子词对的频率（修复：接收单独的词频字典）"""
        pair_freq = defaultdict(int)
        for word, subwords in corpus_subwords.items():
            count = corpus_counts[word]  
            if len(subwords) < 2:
                continue
            pairs = self._get_pairs(subwords)
            for pair in pairs:
                pair_freq[pair] += count 
        return pair_freq
    
    def _preprocess_text(self, text):
        """预处理文本：优化中文支持"""
        text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z0-9,.!?])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z0-9,.!?])([\u4e00-\u9fa5])', r'\1 \2', text)
        
        # 英文小写化（不影响中文）
        def lowercase_english(match):
            return match.group(1).lower() if match.group(1) else ""
        text = re.sub(r'([a-zA-Z]+)', lowercase_english, text)
        
        # 标点符号前后加空格
        text = re.sub(r'([,.!?;:"()])', r' \1 ', text)
        
        # 去除多余空格
        return re.sub(r'\s+', ' ', text).strip()
    
    def load_corpus_from_folder(self, folder_path):
        """从文件夹加载所有TXT文件作为语料（分离词与词频）"""
        corpus_counts = defaultdict(int)  # 词: 出现次数（整数）
        corpus_subwords = {}  # 词: 子词列表（字符级）
        
        for file_path in Path(folder_path).glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                processed_text = self._preprocess_text(text)
                
                # 分词：中文按字符拆分，英文按空格拆分
                words = []
                for token in processed_text.split():
                    if re.search(r'[\u4e00-\u9fa5]', token):
                        words.extend(list(token))  # 中文拆分为单个字符
                    else:
                        words.append(token)  # 英文/标点保留整体
                
                # 统计词频并初始化子词列表
                for word in words:
                    word_with_end = word + '</w>'  # 添加结束标记
                    corpus_counts[word_with_end] += 1
                    # 初始化子词列表（字符级）
                    if word_with_end not in corpus_subwords:
                        corpus_subwords[word_with_end] = [c for c in word_with_end]
                
                print(f"已处理文件: {file_path}, 词数量: {len(words)}")
            
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
        
        print(f"总语料词类型数: {len(corpus_counts)}")
        return corpus_counts, corpus_subwords
    
    def train(self, folder_path):
        """训练BPE模型（使用分离的词频和子词列表）"""
        # 加载语料：分离词频和子词列表
        corpus_counts, corpus_subwords = self.load_corpus_from_folder(folder_path)
        
        # 初始化词汇表（包含未登录词标记）
        self.vocab.add(self.unk_token)
        for word in corpus_subwords:
            for char in word:
                self.vocab.add(char)
        
        current_vocab_size = len(self.vocab)
        print(f"初始词汇表大小: {current_vocab_size} (含未登录词标记)")
        
        iteration = 0
        while current_vocab_size < self.vocab_size:
            iteration += 1
            
            # 传入两个字典：子词列表和词频（修复核心问题）
            pair_freq = self._count_pair_frequencies(corpus_subwords, corpus_counts)
            if not pair_freq:
                break  # 没有可合并的子词对
            
            best_pair = max(pair_freq, key=pair_freq.get)
            merged = ''.join(best_pair)
            self.merges[best_pair] = merged
            self.vocab.add(merged)
            
            # 更新语料中的子词
            for word in corpus_subwords:
                subwords = corpus_subwords[word]
                i = 0
                while i < len(subwords) - 1:
                    if subwords[i] == best_pair[0] and subwords[i+1] == best_pair[1]:
                        subwords = subwords[:i] + [merged] + subwords[i+2:]
                    else:
                        i += 1
                corpus_subwords[word] = subwords
            
            current_vocab_size = len(self.vocab)
            
            if iteration % 100 == 0:
                print(f"迭代 {iteration}, 合并: {best_pair} -> {merged}, 词汇表大小: {current_vocab_size}")
        
        # 创建id映射
        sorted_vocab = sorted(self.vocab)
        self.id_to_token = {i: token for i, token in enumerate(sorted_vocab)}
        self.token_to_id = {token: i for i, token in self.id_to_token.items()}
        
        print(f"训练完成! 最终词汇表大小: {len(self.vocab)}, 合并次数: {len(self.merges)}")
        self.save_model()
        
        return self
    
    def encode(self, text):
        if not self.token_to_id:
            raise ValueError("模型尚未训练或加载，请先训练模型")
        
        processed_text = self._preprocess_text(text)
        words = []
        for token in processed_text.split():
            if re.search(r'[\u4e00-\u9fa5]', token):
                words.extend(list(token))
            else:
                words.append(token)
        
        encoded = []
        for word in words:
            token = list(word + '</w>')
            
            # 应用合并规则
            for (a, b), merged in reversed(self.merges.items()):
                i = 0
                while i < len(token) - 1:
                    if token[i] == a and token[i+1] == b:
                        token = token[:i] + [merged] + token[i+2:]
                    else:
                        i += 1
            
            # 处理未登录词
            token_ids = []
            for t in token:
                token_ids.append(self.token_to_id.get(t, self.token_to_id[self.unk_token]))
            
            encoded.extend(token_ids)
        
        return encoded
    
    def decode(self, token_ids):
        if not self.id_to_token:
            raise ValueError("模型尚未训练或加载，请先训练模型")
        
        tokens = []
        for id in token_ids:
            tokens.append(self.id_to_token.get(id, self.unk_token))
        
        text = ''.join(tokens).replace('</w>', ' ')
        return re.sub(r'\s+', ' ', text).strip()
    
    def save_model(self):
        """保存合并规则和词汇表"""
        merges_list = [[k[0], k[1], v] for k, v in self.merges.items()]
        with open(os.path.join(self.save_dir, 'merges.json'), 'w', encoding='utf-8') as f:
            json.dump(merges_list, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(self.save_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存至 {self.save_dir} 目录")
    
    def load_model(self):
        """加载合并规则和词汇表"""
        try:
            with open(os.path.join(self.save_dir, 'merges.json'), 'r', encoding='utf-8') as f:
                merges_list = json.load(f)
                self.merges = {(item[0], item[1]): item[2] for item in merges_list}
            
            with open(os.path.join(self.save_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
                self.token_to_id = json.load(f)
                self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
                self.vocab = set(self.token_to_id.keys())
            
            print(f"模型已从 {self.save_dir} 目录加载，词汇表大小: {len(self.vocab)}, 合并规则数量: {len(self.merges)}")
            return True
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            return False


if __name__ == "__main__":
    CORPUS_FOLDER = "/home/usr3/pycharm_projects/self_project/bpe/Heroes" 
    VOCAB_SIZE = 10000
    MODEL_DIR = "bpe_rag_model"
    
    bpe_processor = BPEProcessor(vocab_size=VOCAB_SIZE, save_dir=MODEL_DIR)
    
    # 训练新模型
    print("开始训练BPE模型...")
    bpe_processor.train(CORPUS_FOLDER)
    
    # 测试编码解码
    test_text = "这是一个测试文本，用于演示BPE的编码和解码功能。"
    print("\n测试文本:", test_text)
    
    encoded = bpe_processor.encode(test_text)
    print(f"编码结果 (前20个id): {encoded[:20]}...")
    print(f"编码长度: {len(encoded)}")
    
    decoded = bpe_processor.decode(encoded)
    print("解码结果:", decoded)
