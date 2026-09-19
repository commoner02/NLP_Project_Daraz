import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Any, Union
from collections import Counter
import numpy as np

class BanglaVocab:
    def __init__(self, max_size: int = 15000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        
        self.vocab_size = 2
        
    def fit(self, texts: List[str]):
        """Build vocabulary from a list of strings."""
        word_freqs = Counter()
        for text in texts:
            words = str(text).split()
            word_freqs.update(words)
            
        # Sort by frequency and filter by min_freq
        valid_words = [word for word, count in word_freqs.most_common() if count >= self.min_freq]
        
        # Truncate to max_size
        valid_words = valid_words[:self.max_size - 2]
        
        for word in valid_words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                
    def transform(self, text: str, max_length: int) -> List[int]:
        """Convert string to list of indices with padding/truncation."""
        words = str(text).split()
        indices = [self.word2idx.get(w, self.word2idx[self.unk_token]) for w in words]
        
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices += [self.word2idx[self.pad_token]] * (max_length - len(indices))
            
        return indices
        
    def transform_batch(self, texts: List[str], max_length: int) -> torch.Tensor:
        """Convert a list of strings to a padded tensor."""
        batch = [self.transform(t, max_length) for t in texts]
        return torch.tensor(batch, dtype=torch.long)

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab: BanglaVocab, max_length: int):
        self.texts = texts if isinstance(texts, list) else list(texts)
        self.labels = labels if isinstance(labels, list) else list(labels)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_indices = self.vocab.transform(self.texts[idx], self.max_length)
        # Note: we might want labels as long for CrossEntropy or float for BCE
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(self.labels[idx])

def create_dataloader(texts, labels, vocab: BanglaVocab, max_length: int, batch_size: int, shuffle: bool = True) -> DataLoader:
    """Helper to create a DataLoader from raw texts and labels."""
    dataset = TextDataset(texts, labels, vocab, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

