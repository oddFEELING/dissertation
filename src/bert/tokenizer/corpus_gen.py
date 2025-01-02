import json
from xml.etree.ElementInclude import include

import numpy as np
from typing import List
from src.bert.tokenizer.tokeniser import custom_special_tokens
from src.bert.data_prep import NeighborInteractivity, ModelFeatures, VariableGene


class CorpusGenerator:
    def __init__(self):
        print('\n\n------------ Starting Corpus Genertator  --\n')
        self.special_tokens = custom_special_tokens

        self.mito_thresholds = {
            "high": 3.0,
            "med": 1.5,
        }

    def get_mito_token(self, mito_activity: float) -> str:
        """Determine appropriate mitochondrial activity token"""
        if mito_activity >= self.mito_thresholds['high']:
            return "[MITO_HIGH]"
        elif mito_activity >= self.mito_thresholds['med']:
            return "[MITO_MED]"
        else:
            return "[MITO_LOW]"

    def format_spatial_coords(self, coords: List[float]):
        """Format spatial coordinates"""
        return f"{coords[0]:.2f} {coords[1]:.2f}"

    def format_gene_sequence(self, gene: VariableGene) -> str:
        """Format a single gene's info"""
        return f'[GENE] gene_{gene.gene_name} {gene.expression_level:.2f} {gene.dispersion_level:.2f}'

    def format_neighbors_sequence(self, neighbors: List[NeighborInteractivity]) -> str:
        """Format neighbourhood information"""
        neighbor_strs = []
        for n in neighbors:
            neighbor_str = f'[NEIGHBOR] {n.location[0]:.2f} {n.location[1]:.2f} {n.distance} {n.angle} {n.interactivity_score}'
            neighbor_strs.append(neighbor_str)

        return " ".join(neighbor_strs)

    def generate_sequence(self, features: ModelFeatures) -> str:
        """Generate a structured sequence from the model features"""

        sequence_parts = [
            "[CLS]",

            # Tissue information
            f'[TISSUE] {features.tissue_type}',

            # Spatial coordinates
            f'[SPATIAL] {self.format_spatial_coords(features.spatial_coords)}',

            # Cancer score
            f"[CANCER] {features.cancer_score}",

            # Reactivity score
            f"[REACT] {features.cell_reactivity:.2f}",

            # Gene information
            " ".join([
                self.format_gene_sequence(gene)
                for gene in features.top_var_genes
            ]),

            # Check if cell is at the border
            "[IS_BORDER]" if features.is_border else "[NOT_BORDER]",

            # Neighbourhood cells
            self.format_neighbors_sequence(features.neighbor_interactivities),

            # Mito activity score
            self.get_mito_token(features.mito_activity),

            # End of sequence
            "[SEP]"
        ]

        return " ".join(sequence_parts)

    def generate_corpus(self, features_list: List[ModelFeatures],
                        output_file: str = 'tokenizer/corpus.txt', include_meta: bool = True):
        """Generate corpus of sequences from list of model features"""
        print(f'--> Generating corpus - allow metadata ({"true" if include_meta else "false"})')
        sequences = []
        metadata = {
            "total_sequences": len(features_list),
            "tissue_types": {},
            "avg_cancer_score": 0.0,
            "mito_distribution": {'high': 0, "med": 0, "low": 0},
            "border_distribution": {"border": 0, "non_border": 0},
            "token_statistics": {token: 0 for token in self.special_tokens}
        }

        for features in features_list:
            sequence = self.generate_sequence(features)
            sequences.append(sequence)

            if include_meta:
                # update metadata
                metadata['tissue_types'][features.tissue_type] = \
                    metadata['tissue_types'].get(features.tissue_type, 0) + 1
                metadata['avg_cancer_score'] += features.cancer_score

                # Update mito distribution
                if features.mito_activity > self.mito_thresholds['high']:
                    metadata['mito_distribution']['high'] += 1
                elif features.mito_activity >= self.mito_thresholds['med']:
                    metadata['mito_distribution']['med'] += 1
                else:
                    metadata['mito_distribution']['low'] += 1

                # update border distribution
                if features.is_border:
                    metadata['border_distribution']['border'] += 1
                else:
                    metadata['border_distribution']['non_border'] += 1

                # Count special tokens
                for token in self.special_tokens:
                    metadata['token_statistics'][token] += sequence.count(token)

        # Finalize metadata
        if include_meta and features_list:
            metadata['avg_cancer_score'] /= len(features_list)

        print('--> Saving corpus to disk')
        # Save sequences
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sequences))

        # Write metadata
        if include_meta:
            metadata_file = output_file.replace('.txt', '.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
