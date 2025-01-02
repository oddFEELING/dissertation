import torch 

class MaskingStrategy:
    def __init__(self, tokenizer, mask_token_id):
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        
        # Define token groups and their masking probabilities
        self.special_tokens = {
            '[TISSUE]': 0.05,
            '[SPATIAL]': 0.05,
            '[CANCER]': 0.05,
            '[REACT]': 0.05,
            '[GENE]': 0.05,
            '[NEIGHBOR]': 0.05,
            '[MITO_HIGH]': 0.05,
            '[MITO_MED]': 0.05,
            '[MITO_LOW]': 0.05,
            '[IS_BORDER]': 0.05,
            '[NOT_BORDER]': 0.05
        }
        
        # Higher probability for tissue types to learn better
        self.tissue_prob = 0.2
        
        # Standard probability for values
        self.value_prob = 0.15
        
        # Lower probability for gene names as they're important identifiers
        self.gene_prob = 0.1

    def should_mask(self, token: str, prev_token: str = None) -> float:
        """Determine if a token should be masked based on its type and context"""
        # Special tokens have their own probabilities
        if token in self.special_tokens:
            return self.special_tokens[token]
            
        # After [TISSUE], it's a tissue type
        if prev_token == '[TISSUE]':
            return self.tissue_prob
            
        # After [GENE], it's a gene name followed by values
        if prev_token == '[GENE]':
            return self.gene_prob
            
        # After [SPATIAL], [CANCER], [REACT], or in [NEIGHBOR] sequence, these are numerical values
        if prev_token in ['[SPATIAL]', '[CANCER]', '[REACT]', '[NEIGHBOR]']:
            return self.value_prob
            
        # Default to standard MLM probability
        return 0.15

    def apply_masking(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply masking to the input sequence based on token types"""
        device = input_ids.device  # Get the device of input tensor
        labels = input_ids.clone()
        probability_matrix = torch.zeros_like(input_ids, dtype=torch.float, device=device)
        
        # Convert input_ids to tokens for context-aware masking
        tokens = [self.tokenizer.decode([token_id.item()]) for token_id in input_ids]
        
        for i in range(len(tokens)):
            prev_token = tokens[i-1] if i > 0 else None
            prob = self.should_mask(tokens[i], prev_token)
            probability_matrix[i] = prob
            
        # Create masking matrix
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Don't mask [CLS], [SEP], or [PAD] tokens
        special_tokens_mask = torch.tensor(
            [self.tokenizer.decode([token_id.item()]) in ['[CLS]', '[SEP]', '[PAD]'] for token_id in input_ids],
            dtype=torch.bool,
            device=device  # Create tensor on the same device
        )
        masked_indices.masked_fill_(special_tokens_mask, False)
        
        # Mask tokens
        input_ids[masked_indices] = self.mask_token_id
        
        # Create labels (-100 for non-masked tokens)
        labels[~masked_indices] = -100
        
        return input_ids, labels
