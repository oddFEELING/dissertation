from transformers import BertTokenizer, BertTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, processors
from typing import List, Dict, Optional
import json
import numpy as np
from pathlib import Path
import anndata as ad
from pydantic import BaseModel


class NeighborInteractivity(BaseModel):
    interactivity_score: float
    location: List[float]
    distance: float
    angle: float


class VariableGene(BaseModel):
    gene_name: str
    expression_level: float
    dispersion_level: float


class ModelFeatures(BaseModel):
    tissue_type: str
    spatial_coords: List[float]
    cancer_score: float
    top_var_genes: List[VariableGene]
    cell_reactivity: float
    neighbor_interactivities: Optional[List[NeighborInteractivity]]
    mito_activity: float
    is_border: bool


class SpatialBertTokenizer:
    def __init__(self, base_model_name: str = 'bert-base-uncased'):
        # Initialize base tokenizer
        self.base_tokenizer = BertTokenizer.from_pretrained(base_model_name)

        # Define special tokens
        self.special_tokens = [
            "[TISSUE]",  # Tissue type
            "[SPATIAL]",  # spatial coords
            "[CANCER]",  # cancer score
            "[IS_BORDER]", "[NOT_BORDER]",  # whether is border or not
            "[GENE]",  # prefixes expressed genes
            "[REACT]",  # Cell reactivity score
            "[NEIGHBORS]",  # Neighbourhood data
            "[MITO_HIGH]", "[MITO_MED]", "[MITO_LOW]"  # binned mitochondrial activity levels
        ]

        # Get the original vocabulary
        self.vocab = self.base_tokenizer.get_vocab()
        self.original_vocab_size = len(self.vocab)
        self.gene_tokens = []

    def add_gene_tokens(self, adata: ad.AnnData) -> Dict:
        """Add all genes from AnnData as special tokens with gene_ prefix"""
        # Extract gene names from AnnData
        gene_names = list(adata.var_names)
        n_genes = len(gene_names)

        # Calculate vocabulary statistics
        stats = {
            'base_vocab_size': self.original_vocab_size,
            'n_special_tokens': len(self.special_tokens),
            'n_genes': n_genes,
            'total_vocab_size': self.original_vocab_size + len(self.special_tokens) + n_genes
        }

        # Create gene tokens with prefix
        self.gene_tokens = [f"gene_{gene}" for gene in gene_names]

        # Add to vocabulary
        for token in self.gene_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Update tokenizer with new vocabulary
        self.tokenizer = self._create_custom_tokenizer()

        # Add additional statistics
        stats['final_vocab_size'] = len(self.vocab)
        stats['vocab_utilization'] = len(self.vocab) / 30000 * 100  # Standard BERT vocab size is 30k

        print("\nVocabulary Statistics:")
        print(f"Base vocabulary size: {stats['base_vocab_size']}")
        print(f"Number of special tokens: {stats['n_special_tokens']}")
        print(f"Number of genes added: {stats['n_genes']}")
        print(f"Final vocabulary size: {stats['final_vocab_size']}")
        print(f"Vocabulary utilization: {stats['vocab_utilization']:.1f}%")

        return stats

    def _create_custom_tokenizer(self) -> BertTokenizerFast:
        """Create a custom tokenizer based on BERT with our special tokens and gene tokens"""
        # Add standard special tokens first
        for token in self.special_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Create WordPiece tokenizer
        custom_tokenizer = Tokenizer(models.WordPiece(
            vocab=self.vocab,
            unk_token="[UNK]"
        ))

        # Configure normalization to preserve case and numbers
        custom_tokenizer.normalizer = normalizers.Sequence([
            normalizers.Replace(pattern=r"(-?\d+\.\d+)", replacement=" \\1 "),  # Preserve decimal numbers
            normalizers.Replace(pattern=r"(-?\d+)", replacement=" \\1 "),  # Preserve integers
            normalizers.Replace(pattern=r"[_]", replacement=" _ "),  # Split on underscores
            normalizers.NFD(),
            normalizers.StripAccents()
        ])

        # Configure pre-tokenization to preserve special sequences
        custom_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Split(
                pattern=r"([^\w\s])",  # Split on non-word, non-space characters
                behavior="isolated",  # Keep the splits
                invert=False
            )
        ])

        # Configure decoder
        custom_tokenizer.decoder = decoders.WordPiece(prefix="##")

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
        fast_tokenizer.add_special_tokens({'additional_special_tokens': all_special_tokens})

        return fast_tokenizer

    def save_tokenizer(self, output_dir: str) -> None:
        """Save the tokenizer files"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_dir)

        # Save vocab info
        vocab_info = {
            'original_vocab_size': self.original_vocab_size,
            'final_vocab_size': len(self.vocab),
            'special_tokens': self.special_tokens,
            'gene_tokens': self.gene_tokens
        }
        with open(Path(output_dir) / 'vocab_info.json', 'w') as f:
            json.dump(vocab_info, f, indent=2)

    @classmethod
    def load_tokenizer(cls, tokenizer_dir: str) -> 'SpatialBertTokenizer':
        """Load a saved tokenizer"""
        instance = cls(base_model_name='bert-base-uncased')
        instance.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)

        # Load vocab info
        with open(Path(tokenizer_dir) / 'vocab_info.json', 'r') as f:
            vocab_info = json.load(f)
            instance.gene_tokens = vocab_info['gene_tokens']

        return instance

    def get_gene_token(self, gene_name: str) -> str:
        """Get the gene token for a given gene name"""
        token = f"gene_{gene_name}"
        if token not in self.vocab:
            raise ValueError(f"Gene {gene_name} not found in vocabulary")
        return token


class SpatialCorpusGenerator:
    def __init__(self, tokenizer: SpatialBertTokenizer):
        self.tokenizer = tokenizer
        self.mito_thresholds = {
            'high': 3.0,
            'med': 1.5
        }

    def get_mito_token(self, mito_activity: float) -> str:
        """Determine appropriate mitochondrial activity token"""
        if mito_activity >= self.mito_thresholds['high']:
            return "[MITO_HIGH]"
        elif mito_activity >= self.mito_thresholds['med']:
            return "[MITO_MED]"
        else:
            return "[MITO_LOW]"

    def format_spatial_coords(self, coords: List[float]) -> str:
        """Format spatial coordinates"""
        return f"{coords[0]:.3f}_{coords[1]:.3f}"

    def generate_sequence(self, features: ModelFeatures) -> str:
        """Generate a structured sequence from model features"""
        # Start sequence
        sequence_parts = ["[CLS]"]

        # Add tissue type
        sequence_parts.append(f"[TISSUE] {features.tissue_type}")

        # Add spatial coordinates (separated x y)
        sequence_parts.append(f"[SPATIAL] {features.spatial_coords[0]:.3f} {features.spatial_coords[1]:.3f}")

        # Add cancer score
        sequence_parts.append(f"[CANCER] {features.cancer_score:.3f}")

        # Add gene information
        for gene in features.top_var_genes[:5]:
            sequence_parts.append(
                f"[GENE] {gene.gene_name}_{gene.expression_level:.3f}_{gene.dispersion_level:.3f}"
            )

        # Add border status
        sequence_parts.append("[IS_BORDER]" if features.is_border else "[NOT_BORDER]")

        # Add neighbor information
        sorted_neighbors = sorted(
            features.neighbor_interactivities,
            key=lambda x: x.interactivity_score,
            reverse=True
        )
        for neighbor in sorted_neighbors[:5]:
            sequence_parts.append(
                f"[NEIGHBORS] {neighbor.location[0]:.3f} {neighbor.location[1]:.3f} "
                f"{neighbor.distance:.3f} {neighbor.angle:.3f} {neighbor.interactivity_score:.3f}"
            )

        # Add mitochondrial activity bin
        sequence_parts.append(self.get_mito_token(features.mito_activity))

        # End sequence
        sequence_parts.append("[SEP]")

        return " ".join(sequence_parts)

    def generate_corpus(self, features_list: List[ModelFeatures],
                        output_dir: str,
                        include_metadata: bool = True) -> None:
        """Generate and tokenize corpus from features list"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sequences = []
        tokenized_data = []
        metadata = {
            'total_sequences': len(features_list),
            'tissue_types': {},
            'mito_distribution': {'high': 0, 'med': 0, 'low': 0},
            'border_distribution': {'border': 0, 'non_border': 0},
            'sequence_length_stats': {
                'min': float('inf'),
                'max': 0,
                'avg': 0
            }
        }

        for features in features_list:
            # Generate sequence
            sequence = self.generate_sequence(features)
            sequences.append(sequence)

            # Tokenize sequence
            tokenized = self.tokenizer.tokenizer(
                sequence,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            tokenized_data.append({
                'input_ids': tokenized['input_ids'].tolist(),
                'attention_mask': tokenized['attention_mask'].tolist()
            })

            # Update metadata
            if include_metadata:
                seq_length = len(tokenized['input_ids'][0])
                metadata['sequence_length_stats']['min'] = min(
                    metadata['sequence_length_stats']['min'],
                    seq_length
                )
                metadata['sequence_length_stats']['max'] = max(
                    metadata['sequence_length_stats']['max'],
                    seq_length
                )
                metadata['sequence_length_stats']['avg'] += seq_length

                metadata['tissue_types'][features.tissue_type] = \
                    metadata['tissue_types'].get(features.tissue_type, 0) + 1

                if features.mito_activity >= self.mito_thresholds['high']:
                    metadata['mito_distribution']['high'] += 1
                elif features.mito_activity >= self.mito_thresholds['med']:
                    metadata['mito_distribution']['med'] += 1
                else:
                    metadata['mito_distribution']['low'] += 1

                if features.is_border:
                    metadata['border_distribution']['border'] += 1
                else:
                    metadata['border_distribution']['non_border'] += 1

        # Finalize metadata
        if include_metadata and features_list:
            metadata['sequence_length_stats']['avg'] /= len(features_list)

        # Save all files
        with open(Path(output_dir) / 'sequences.txt', 'w') as f:
            f.write('\n'.join(sequences))

        with open(Path(output_dir) / 'tokenized_sequences.json', 'w') as f:
            json.dump(tokenized_data, f)

        if include_metadata:
            with open(Path(output_dir) / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Create example features
    example_features = ModelFeatures(
        tissue_type='lung_cancer',
        spatial_coords=[-89.922, 37.680],
        cancer_score=0.640,
        top_var_genes=[
            VariableGene(gene_name='TMSB4X', expression_level=0.86, dispersion_level=1.0),
            VariableGene(gene_name='MT-CO3', expression_level=0.74, dispersion_level=0.71)
        ],
        cell_reactivity=33.319,
        neighbor_interactivities=[
            NeighborInteractivity(
                interactivity_score=95.097,
                location=[-88.241, 35.077],
                distance=3.099,
                angle=-0.997
            ),
            NeighborInteractivity(
                interactivity_score=95.097,
                location=[-88.241, 35.077],
                distance=3.099,
                angle=-0.997
            )
        ],
        mito_activity=2.690,
        is_border=False
    )

    # Initialize tokenizer and corpus generator
    tokenizer = SpatialBertTokenizer()
    corpus_gen = SpatialCorpusGenerator(tokenizer)

    # Generate corpus
    output_dir = "spatial_bert_data"
    corpus_gen.generate_corpus([example_features], output_dir)

    # Save tokenizer
    tokenizer.save_tokenizer(output_dir)

    # Print sample sequence and its tokenization
    sequence = corpus_gen.generate_sequence(example_features)
    print("\nSample sequence:")
    print(sequence)

    corpus_gen.generate_corpus([example_features], output_dir='tokens', include_metadata=True)
    tokens = tokenizer.tokenizer.encode(sequence)
    decoded = tokenizer.tokenizer.decode(tokens)
    print("\nTokenized and decoded:")
    print(decoded)
