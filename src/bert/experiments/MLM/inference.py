import torch
from src.bert.tokenizer.tokeniser import TissueTokenizer
from transformers import BertForMaskedLM
from rich.pretty import pprint


class MLMInference:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load tokeniser and model
        self.tokenizer = TissueTokenizer.load_tokenizer(tokenizer_path)
        self.model = BertForMaskedLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict(self, sequence: str, k: int = 5):
        """Predict masked tokens in the sequence
        :param sequence: to predict masked tokens in
        :param k: Number of predictions to return
        """
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer.tokenizer(
                sequence,
                return_tensors='pt',
                padding=True
            ).to(self.device)

            # Get predictions
            outputs = self.model(**inputs)
            predictions = outputs.logits

            # Find positions of [MASK} tokens
            mask_token_id = self.tokenizer.tokenizer.mask_token_id
            mask_positions = (inputs.input_ids == mask_token_id).nonzero()

            results = []
            for pos in mask_positions:
                batch_idx, token_idx = pos

                # Get top k predictions for this position
                probs = torch.nn.functional.softmax(predictions[batch_idx, token_idx], dim=-1)
                top_k = torch.topk(probs, k)

                # Convert to tokens
                predicted_tokens = [
                    (self.tokenizer.tokenizer.decode([token_id.item()]), prob.item())
                    for token_id, prob in zip(top_k.indices, top_k.values)
                ]

                results.append((token_idx.item(), predicted_tokens))

            return results


if __name__ == '__main__':
    inference = MLMInference(
        model_path='../../outputs/BEST_MODEL/',
        tokenizer_path='../../tokenizer/_internal'
    )

    sequence = "[CLS] [TISSUE] lung_cancer [SPATIAL] [mask] 63.602 [CANCER] 0.68 [REACT] 59.02 [GENE] gene_MT-CO3 0.800 0.710 [GENE] gene_TMSB4X 0.750 1.000 [GENE] gene_TFRC 0.630 1.250 [GENE] gene_SFTPA1 0.620 2.630 [GENE] gene_KRT15 0.620 1.400 [NOT_BORDER] [NEIGHBOR] 39.49749666988194 61.00685378590077 3.09 -1.00 70.80521953105927 [NEIGHBOR] 39.49749666988194 66.19614882506528 3.09 1.00 70.0 [NEIGHBOR] 36.13522575903724 61.00685378590077 3.09 -2.15 70.0 [NEIGHBOR] 36.14441229158055 66.20430809399477 3.09 2.14 69.95754756501067 [NEIGHBOR] 34.45409030361489 63.601501305483026 3.36 3.14 64.36625088705267 [MITO_MED] [SEP]"
    predictions = inference.predict(sequence, 5)

    pprint(predictions)
