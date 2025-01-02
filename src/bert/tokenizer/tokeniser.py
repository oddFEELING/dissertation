import json

from pywin.framework.interact import valueFormatOutputError
from transformers import BertTokenizer, BertTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, processors
from pathlib import Path
from typing import List, Optional

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
        # Add special tokens to vocab first
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
            # Preserve gene tokens (do this first to prevent number matching)
            normalizers.Replace(
                pattern=r"(gene_[A-Z0-9]+)",  # Matches gene_SYMBOL format
                content=" \\1 "
            ),
            # Preserve tissue types and other underscore tokens
            normalizers.Replace(
                pattern=r"([a-zA-Z]+_[a-zA-Z]+)",  # Matches word_word format
                content=" \\1 "
            ),
            # Handle spatial coordinates and other numbers
            normalizers.Replace(pattern=r"(-?\d+\.\d+)", content=" \\1 "),  # Decimal numbers
            normalizers.Replace(pattern=r"(-?\d+)", content=" \\1 "),  # Integers
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
        custom_tokenizer.decoder = custom_decoder

        # Add special tokens to the tokenizer's special tokens map
        special_tokens_map = {
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]"
        }

        # Configure post-processing
        custom_tokenizer.post_processor = processors.TemplateProcessing(
            single="$A",
            pair="$A $B",
            special_tokens=[
                ("[CLS]", self.vocab["[CLS]"]),
                ("[SEP]", self.vocab["[SEP]"])
            ]
        )

        # Create fast tokenizer wrapper
        fast_tokenizer = BertTokenizerFast(
            tokenizer_object=custom_tokenizer,
            vocab=self.vocab,
            **special_tokens_map
        )

        # Add all special tokens ensuring they're treated as special
        all_special_tokens = self.special_tokens + self.gene_tokens
        special_tokens = {
            "additional_special_tokens": all_special_tokens,
            **special_tokens_map
        }
        fast_tokenizer.add_special_tokens(special_tokens)
        
        # Ensure special tokens are never split
        fast_tokenizer.add_tokens(all_special_tokens, special_tokens=True)
        
        return fast_tokenizer

    def _validate_spatial_coordinate(self, value_str: str) -> Optional[str]:
        """Validate spatial coordinate is within -100 to 100 range"""
        try:
            value = float(value_str)
            if -100 <= value <= 100:
                return value_str
            return "[UNK]"
        except ValueError:
            return "[UNK]"

    def validate_token_sequence(self, text: str) -> bool:
        """Validate a token sequence for proper formatting and value ranges"""
        # Split text into tokens
        tokens = text.split()
        
        # Track if we're expecting spatial coordinates
        expecting_spatial = False
        
        for i, token in enumerate(tokens):
            if token == "[SPATIAL]":
                expecting_spatial = True
                continue
                
            # Validate the next two tokens after [SPATIAL] as coordinates
            if expecting_spatial and i < len(tokens) - 1:
                coord1 = self._validate_spatial_coordinate(token)
                coord2 = self._validate_spatial_coordinate(tokens[i + 1])
                if coord1 == "[UNK]" or coord2 == "[UNK]":
                    print(f"Invalid spatial coordinates: {token}, {tokens[i + 1]}")
                    return False
                expecting_spatial = False
                
            # Validate gene tokens
            if token.startswith("gene_") and token not in self.vocab:
                print(f"Invalid gene token: {token}")
                return False

        # Verify tokenization preserves the sequence
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        decoded = self.tokenizer.decode(encoded, skip_special_tokens=False)

        print(f"Sequence: {text} \nEncoded: {encoded}\nDecoded: {decoded}")
        
        if text.strip() != decoded.strip():
            print(f"Tokenization changed the sequence:\nOriginal: {text}\nDecoded:  {decoded}")
            return False
            
        return True

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
        return token

    def verify_special_tokens(self):
        """Verify all special tokens and sequence components"""
        print("\nVerifying token handling...")
        
        # Test full sequence
        test_sequence = "[CLS] [TISSUE] brain_cancer [SPATIAL] -16.49 82.62 [SEP]"
        print("\nTesting full sequence handling:")
        encoded = self.tokenizer.encode(test_sequence, add_special_tokens=False)
        decoded = self.tokenizer.decode(encoded, skip_special_tokens=False)
        print(f"Sequence:\nOriginal: {test_sequence}\nDecoded:  {decoded}")
        assert test_sequence.strip() == decoded.strip(), "Sequence was not preserved!"
        
        # Test coordinate handling
        test_coords = ["-99.5", "0.0", "99.99", "-100", "100"]
        print("\nTesting coordinate handling:")
        for coord in test_coords:
            encoded = self.tokenizer.encode(coord, add_special_tokens=False)
            decoded = self.tokenizer.decode(encoded, skip_special_tokens=False)
            print(f"Coordinate {coord} -> {encoded} -> {decoded}")
            assert decoded.strip() == coord, f"Coordinate {coord} was not preserved!"

        # Test underscore token handling
        test_underscore_tokens = ["brain_cancer", "liver_cancer", "lung_tissue"]
        print("\nTesting underscore token handling:")
        for token in test_underscore_tokens:
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            decoded = self.tokenizer.decode(encoded, skip_special_tokens=False)
            print(f"Underscore token {token} -> {encoded} -> {decoded}")
            assert len(encoded) == 1, f"Underscore token {token} was split!"

        # Test gene token handling
        if self.gene_tokens:
            test_genes = self.gene_tokens[:5]  # Test first 5 genes
            print("\nTesting gene token handling:")
            for gene in test_genes:
                encoded = self.tokenizer.encode(gene, add_special_tokens=False)
                decoded = self.tokenizer.decode(encoded, skip_special_tokens=False)
                print(f"Gene {gene} -> {encoded} -> {decoded}")
                assert len(encoded) == 1, f"Gene token {gene} was split!"
        
        # Test special tokens
        print("\nTesting special tokens:")
        for token in self.special_tokens:
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            decoded = self.tokenizer.decode(encoded, skip_special_tokens=False)
            print(f"{token} -> {encoded} -> {decoded}")
            assert len(encoded) == 1, f"Token {token} was split!"

        print("\nAll verification tests passed successfully!")
