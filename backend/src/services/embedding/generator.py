"""Embedding Generator using PubMedBERT"""

import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from src.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate embeddings using PubMedBERT model.

    Optimized for biomedical text understanding.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model name (default: PubMedBERT)
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.embedding_dim = settings.EMBEDDING_DIMENSION

        # Load tokenizer and model
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)

        logger.info(f"Model loaded on {self.device}")

    def encode(
        self,
        text: str,
        max_length: int = 512,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            max_length: Maximum token length
            normalize: Whether to L2 normalize the embedding

        Returns:
            Embedding vector (768,)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        ).to(self.device)

        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        embedding = embedding.squeeze()

        if normalize:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def batch_encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            max_length: Maximum token length
            normalize: Whether to L2 normalize embeddings
            show_progress: Show progress bar

        Returns:
            Embedding matrix (n_texts, 768)
        """
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Generating embeddings")
            except ImportError:
                pass

        for i in iterator:
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms

        return embeddings

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))


# Singleton instance for reuse
_generator_instance: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create singleton embedding generator."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = EmbeddingGenerator()
    return _generator_instance
