import json

from pywin.framework.interact import valueFormatOutputError
from transformers import BertTokenizer, BertTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, processors
from pathlib import Path
from typing import List

custom_special_tokens = [
    "[TISSUE]",  # Tissue type
    "[SPATIAL]",  # spatial coords
    "[CANCER]",  # cancer score
    "[IS_BORDER]", "[NOT_BORDER]",  # whether is border or not
    "[GENE]",  # prefixes expressed genes
    "[REACT]",  # Cell reactivity score
    "[NEIGHBOR]",  # Neighbourhood data
    "[MITO_HIGH]", "[MITO_MED]", "[MITO_LOW]"  # binned mitochondrial activity levels
]


class TissueTokenizer():
    def __init__(self, base_model_name: str = 'bert-base-uncased'):
        print('\n\n------------ Starting Tokenization --\n')
        self.base_tokenizer = BertTokenizer.from_pretrained(base_model_name)
        self.special_tokens = custom_special_tokens

        # get the original vocab
        self.vocab = self.base_tokenizer.get_vocab()
        self.original_vocab_size = len(self.vocab)
        self.gene_tokens = []

    def add_gene_tokens(self, gene_names: List[str]):
        """Add all genes from the dataset as vocab for the model"""
        n_genes = len(gene_names)

        # Calculate vocab stats
        stats = {
            "base_vocab_size": self.original_vocab_size,
            "n_special_tokens": len(self.special_tokens),
            'n_genes': n_genes,
            "total_vocab_size": self.original_vocab_size + len(self.special_tokens) + n_genes
        }

        # Create gene tokens with prefix
        self.gene_tokens = [f'gene_{gene}' for gene in gene_names]

        # Add to vocabulary
        for token in self.gene_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.tokenizer = self._create_tokenizer()

        # Add additional stats
        stats['final_vocab_size'] = len(self.vocab)
        stats['vocab_utilization'] = len(self.vocab) / 30000 * 100
        print("\nVocabulary Statistics:")
        print(f"Base vocabulary size: {stats['base_vocab_size']}")
        print(f"Number of special tokens: {stats['n_special_tokens']}")
        print(f"Number of genes added: {stats['n_genes']}")
        print(f"Final vocabulary size: {stats['final_vocab_size']}")
        print(f"Vocabulary utilization: {stats['vocab_utilization']:.1f}%")

    def _create_tokenizer(self, tokenizer_path: str = 'tokenizer.json'):
        print('--> Creating tokenizer')
        # Add special tokens
        for token in self.special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Create wordpiece tokenizer
        custom_tokenizer = Tokenizer(models.WordPiece(
            vocab=self.vocab,
            unk_token="[UNK]"
        ))

        # Configure normalization to preserve case and numbers
        custom_tokenizer.normalizer = normalizers.Sequence([
            normalizers.Replace(pattern=r"(-?\d+\.\d+)", content=" \\1 "),  # Preserve decimal numbers
            normalizers.Replace(pattern=r"(-?\d+)", content=" \\1 "),  # Preserve integers
            normalizers.Replace(pattern=r"[_]", content=" _ "),  # Split on underscores
            normalizers.NFD(),
            normalizers.StripAccents()
        ])

        # Configure pre-tokenization to preserve sequences
        custom_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Split(
                pattern=r"([^\w\s])",  # Split on non-word, non-space characters
                behavior="isolated",  # Keep the splits
                invert=False)
        ])

        # Configure decoder
        custom_decoder = decoders.WordPiece(prefix="##")

        # Configure post-processing
        custom_tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", self.vocab["[CLS]"]),
                ("[SEP]", self.vocab["[SEP]"])
            ]
        )

        # Create fast tokenizer wrapper
        fast_tokenizer = BertTokenizerFast(
            tokenizer_object=custom_tokenizer,
            vocab=self.vocab,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]"
        )

        # Add all special tokens
        all_special_tokens = self.special_tokens + self.gene_tokens
        special_tokens = {"additional_special_tokens": all_special_tokens}
        fast_tokenizer.add_special_tokens(special_tokens)
        return fast_tokenizer

    @classmethod
    def load_tokenizer(cls, tokenizer_dir: str = 'tokenizer/_internal'):
        """Load tokenizer from disk"""
        instance = cls(base_model_name='bert-base-uncased')
        instance.tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)

        # Load vocab info
        with open(Path(tokenizer_dir) / 'vocab_info.json', 'r') as f:
            vocab_info = json.load(f)
            instance.gene_tokens = vocab_info['gene_tokens']

        return instance

    def save_tokenizer(self, output_dir: str = ''):
        """Save the tokenizer files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)

        # Save current vocab info
        vocab_info = {
            "original_vocab_size": self.original_vocab_size,
            "final_vocab_size": len(self.vocab),
            "special_tokens": self.special_tokens,
            "gene_tokens_len": len(self.gene_tokens),
            "gene_tokens": self.gene_tokens
        }

        with open(Path(output_dir) / "vocab_info.json", 'w') as f:
            json.dump(vocab_info, f, indent=2)

    def get_gene_token(self, gene_name: str):
        """Get the gene token for a given gene name"""
        token = f'gene_{gene_name}'
        if token not in self.vocab:
            raise ValueError(F'Gene {gene_name} not found in vocabulary.')
