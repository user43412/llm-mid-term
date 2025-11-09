# dataloader.py

import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import re


class IWSLT2017Dataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', source_lang: str = 'en', target_lang: str = 'de',
                 max_length: int = 100, min_freq: int = 2,
                 source_vocab: Optional[Dict[str, int]] = None,
                 target_vocab: Optional[Dict[str, int]] = None):
        """
        IWSLT2017 数据集加载器

        Args:
            data_dir: 数据目录路径
            split: 数据集分割 ('train', 'valid', 'test')
            source_lang: 源语言
            target_lang: 目标语言
            max_length: 最大序列长度
            min_freq: 词频阈值
            source_vocab: 预训练的源语言词汇表（用于验证/测试集）
            target_vocab: 预训练的目标语言词汇表（用于验证/测试集）
        """
        self.data_dir = data_dir
        self.split = split
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length
        self.min_freq = min_freq

        # 初始化词汇表
        if source_vocab is not None and target_vocab is not None:
            # 使用预训练的词汇表（验证/测试集）
            self.source_vocab = source_vocab
            self.target_vocab = target_vocab
        else:
            # 初始化空词汇表（训练集）
            self.source_vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
            self.target_vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}

        self.source_index2word = {v: k for k, v in self.source_vocab.items()}
        self.target_index2word = {v: k for k, v in self.target_vocab.items()}

        # 加载数据
        self.data = self._load_data()

        # 只有训练集才构建词汇表
        if split == 'train':
            self._build_vocab()
            # 更新index2word映射
            self.source_index2word = {v: k for k, v in self.source_vocab.items()}
            self.target_index2word = {v: k for k, v in self.target_vocab.items()}

        print(f"Loaded {len(self.data)} {split} samples")
        print(f"Source vocab size: {len(self.source_vocab)}")
        print(f"Target vocab size: {len(self.target_vocab)}")

    def _load_data(self) -> List[Tuple[str, str]]:
        """加载数据文件"""
        data = []

        if self.split == 'train':
            # 加载训练数据
            source_file = os.path.join(self.data_dir,
                                       f'train.tags.{self.source_lang}-{self.target_lang}.{self.source_lang}')
            target_file = os.path.join(self.data_dir,
                                       f'train.tags.{self.source_lang}-{self.target_lang}.{self.target_lang}')

            if os.path.exists(source_file) and os.path.exists(target_file):
                data = self._load_plain_text(source_file, target_file)
            else:
                raise FileNotFoundError(f"Training files not found in {self.data_dir}")

        elif self.split == 'valid':
            # 加载验证数据 - 只使用dev2010
            source_file = os.path.join(self.data_dir,
                                       f'IWSLT17.TED.dev2010.{self.source_lang}-{self.target_lang}.{self.source_lang}.xml')
            target_file = os.path.join(self.data_dir,
                                       f'IWSLT17.TED.dev2010.{self.source_lang}-{self.target_lang}.{self.target_lang}.xml')

            if os.path.exists(source_file) and os.path.exists(target_file):
                data = self._load_xml_data(source_file, target_file)
            else:
                raise FileNotFoundError(f"Validation files not found in {self.data_dir}")

        else:  # test
            # 测试集应该严格隔离，通常只使用特定的测试年份
            # 这里我们只使用tst2014作为示例，实际使用时应该根据评估需求选择
            test_year = 2014  # 可以选择特定的测试年份
            source_file = os.path.join(self.data_dir,
                                       f'IWSLT17.TED.tst{test_year}.{self.source_lang}-{self.target_lang}.{self.source_lang}.xml')
            target_file = os.path.join(self.data_dir,
                                       f'IWSLT17.TED.tst{test_year}.{self.source_lang}-{self.target_lang}.{self.target_lang}.xml')

            if os.path.exists(source_file) and os.path.exists(target_file):
                data = self._load_xml_data(source_file, target_file)
            else:
                # 如果指定的测试文件不存在，尝试其他年份
                test_years = [2010, 2011, 2012, 2013, 2014, 2015]
                for year in test_years:
                    source_file = os.path.join(self.data_dir,
                                               f'IWSLT17.TED.tst{year}.{self.source_lang}-{self.target_lang}.{self.source_lang}.xml')
                    target_file = os.path.join(self.data_dir,
                                               f'IWSLT17.TED.tst{year}.{self.source_lang}-{self.target_lang}.{self.target_lang}.xml')

                    if os.path.exists(source_file) and os.path.exists(target_file):
                        data = self._load_xml_data(source_file, target_file)
                        print(f"Using test year: {year}")
                        break
                else:
                    raise FileNotFoundError(f"No test files found in {self.data_dir}")

        return data

    def _load_plain_text(self, source_file: str, target_file: str) -> List[Tuple[str, str]]:
        """加载纯文本格式的训练数据"""
        data = []

        with open(source_file, 'r', encoding='utf-8') as f_src, \
                open(target_file, 'r', encoding='utf-8') as f_tgt:

            src_lines = f_src.readlines()
            tgt_lines = f_tgt.readlines()

            for src_line, tgt_line in zip(src_lines, tgt_lines):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()

                # 跳过元数据行（以<开头的行）
                if src_line.startswith('<') or tgt_line.startswith('<'):
                    continue

                # 跳过空行
                if not src_line or not tgt_line:
                    continue

                data.append((src_line, tgt_line))

        return data

    def _load_xml_data(self, source_file: str, target_file: str) -> List[Tuple[str, str]]:
        """加载XML格式的验证/测试数据"""
        data = []

        try:
            # 解析源语言XML文件
            src_tree = ET.parse(source_file)
            src_root = src_tree.getroot()

            # 解析目标语言XML文件
            tgt_tree = ET.parse(target_file)
            tgt_root = tgt_tree.getroot()

            # 提取seg标签内容
            src_segments = []
            tgt_segments = []

            # 从源文件提取segments
            for doc in src_root.findall('.//doc'):
                for seg in doc.findall('.//seg'):
                    src_segments.append(seg.text.strip() if seg.text else "")

            # 从目标文件提取segments
            for doc in tgt_root.findall('.//doc'):
                for seg in doc.findall('.//seg'):
                    tgt_segments.append(seg.text.strip() if seg.text else "")

            # 对齐segments
            for src_seg, tgt_seg in zip(src_segments, tgt_segments):
                if src_seg and tgt_seg:  # 跳过空文本
                    data.append((src_seg, tgt_seg))

        except Exception as e:
            print(f"Error loading XML files: {e}")

        return data

    def _build_vocab(self):
        """构建词汇表（仅在训练集上）"""
        from collections import Counter

        # 统计词频
        source_word_freq = Counter()
        target_word_freq = Counter()

        for src_text, tgt_text in self.data:
            src_words = self._tokenize(src_text)
            tgt_words = self._tokenize(tgt_text)

            source_word_freq.update(src_words)
            target_word_freq.update(tgt_words)

        # 构建源语言词汇表
        vocab_index = len(self.source_vocab)
        for word, freq in source_word_freq.items():
            if freq >= self.min_freq and word not in self.source_vocab:
                self.source_vocab[word] = vocab_index
                vocab_index += 1

        # 构建目标语言词汇表
        vocab_index = len(self.target_vocab)
        for word, freq in target_word_freq.items():
            if freq >= self.min_freq and word not in self.target_vocab:
                self.target_vocab[word] = vocab_index
                vocab_index += 1

    def _tokenize(self, text: str) -> List[str]:
        """简单的文本分词"""
        # 转换为小写，分割单词
        text = text.lower()
        words = re.findall(r'\w+|[^\w\s]', text)
        return words

    def _text_to_indices(self, text: str, vocab: Dict[str, int]) -> List[int]:
        """将文本转换为索引序列"""
        words = self._tokenize(text)
        indices = [vocab.get(word, vocab['<unk>']) for word in words]

        # 添加开始和结束标记
        indices = [vocab['<sos>']] + indices + [vocab['<eos>']]

        # 截断或填充到最大长度
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
            indices[-1] = vocab['<eos>']  # 确保最后一个token是<eos>
        else:
            indices.extend([vocab['<pad>']] * (self.max_length - len(indices)))

        return indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        # 转换为索引序列
        src_indices = self._text_to_indices(src_text, self.source_vocab)
        tgt_indices = self._text_to_indices(tgt_text, self.target_vocab)

        # 创建注意力掩码
        src_mask = [1 if token != self.source_vocab['<pad>'] else 0 for token in src_indices]
        tgt_mask = [1 if token != self.target_vocab['<pad>'] else 0 for token in tgt_indices]

        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
            'src_ids': torch.tensor(src_indices, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_indices, dtype=torch.long),
            'src_mask': torch.tensor(src_mask, dtype=torch.long),
            'tgt_mask': torch.tensor(tgt_mask, dtype=torch.long)
        }

    def get_vocab_sizes(self):
        """获取词汇表大小"""
        return len(self.source_vocab), len(self.target_vocab)

    def get_vocabs(self):
        """获取词汇表"""
        return self.source_vocab, self.target_vocab


def create_data_loader(data_dir: str, split: str, batch_size: int = 32, shuffle: bool = True,
                       source_lang: str = 'en', target_lang: str = 'de',
                       source_vocab: Optional[Dict[str, int]] = None,
                       target_vocab: Optional[Dict[str, int]] = None,
                       **kwargs):
    """创建数据加载器"""
    dataset = IWSLT2017Dataset(
        data_dir=data_dir,
        split=split,
        source_lang=source_lang,
        target_lang=target_lang,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        **kwargs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and split == 'train',  # 只有训练集shuffle
        num_workers=0  # 为了简化，设为0
    )

    return dataloader, dataset


def create_data_loaders(data_dir: str, batch_size: int = 32,
                        source_lang: str = 'en', target_lang: str = 'de',
                        max_length: int = 100, min_freq: int = 2):
    """
    创建完整的数据加载器（训练、验证、测试）
    确保正确的数据流，避免数据泄露
    """
    # 1. 首先创建训练集
    train_loader, train_dataset = create_data_loader(
        data_dir=data_dir,
        split='train',
        batch_size=batch_size,
        shuffle=True,
        source_lang=source_lang,
        target_lang=target_lang,
        max_length=max_length,
        min_freq=min_freq
    )

    # 获取训练集的词汇表
    source_vocab, target_vocab = train_dataset.get_vocabs()

    # 2. 创建验证集，使用训练集的词汇表
    valid_loader, valid_dataset = create_data_loader(
        data_dir=data_dir,
        split='valid',
        batch_size=batch_size,
        shuffle=False,
        source_lang=source_lang,
        target_lang=target_lang,
        max_length=max_length,
        source_vocab=source_vocab,
        target_vocab=target_vocab
    )

    # 3. 创建测试集，使用训练集的词汇表
    test_loader, test_dataset = create_data_loader(
        data_dir=data_dir,
        split='test',
        batch_size=batch_size,
        shuffle=False,
        source_lang=source_lang,
        target_lang=target_lang,
        max_length=max_length,
        source_vocab=source_vocab,
        target_vocab=target_vocab
    )

    return (train_loader, valid_loader, test_loader,
            train_dataset, valid_dataset, test_dataset)


# 使用示例
if __name__ == "__main__":
    # 测试数据加载器
    data_dir = "data/en-de"  # 根据你的实际路径调整

    # 使用新的安全数据加载方式
    train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset = create_data_loaders(
        data_dir=data_dir,
        batch_size=16,
        max_length=50,
        min_freq=2
    )

    # 获取一个训练batch进行测试
    batch = next(iter(train_loader))
    print("Batch keys:", batch.keys())
    print("Source IDs shape:", batch['src_ids'].shape)
    print("Target IDs shape:", batch['tgt_ids'].shape)
    print("Source mask shape:", batch['src_mask'].shape)

    # 打印一个样本
    print("\nSample source text:", batch['src_text'][0])
    print("Sample target text:", batch['tgt_text'][0])
    print("Source indices:", batch['src_ids'][0])
    print("Target indices:", batch['tgt_ids'][0])

    # 验证词汇表一致性
    print(f"\n词汇表大小检查:")
    print(f"训练集源语言词汇表大小: {len(train_dataset.source_vocab)}")
    print(f"验证集源语言词汇表大小: {len(valid_dataset.source_vocab)}")
    print(f"测试集源语言词汇表大小: {len(test_dataset.source_vocab)}")
    print("所有词汇表大小应该相同，避免数据泄露")