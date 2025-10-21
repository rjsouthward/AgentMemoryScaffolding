"""Text embedding encoder using LiteLLM.

This module provides the Encoder class for converting text into vector embeddings
using various language models through the LiteLLM interface. It includes features
for text truncation, batching, and concurrent processing to optimize embedding
generation for large text collections.

The encoder is specifically optimized for the text-embedding-3-small model from OpenAI
with hardcoded best practice constants for batch sizes and token limits.
"""

import litellm
import asyncio
import tiktoken


class Encoder:
    def __init__(self, model_name: str, **kwargs):
        """Initialize the text encoder.

        Args:
            model_name: Name of the embedding model to use (e.g., 'text-embedding-3-small')
            **kwargs: Additional arguments to pass to the embedding model
        """
        self.model_name = model_name
        self.kwargs = kwargs

        # Setup tokenizer for text truncation
        # Note: Uses text-embedding-3-small tokenizer regardless of model for consistency
        self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

        # Hardcoded best practice constants optimized for text-embedding-3-small
        self.max_batch_size = 128  # Maximum texts per API call
        self.max_tokens_per_input = 8192  # Maximum tokens per individual text
        self.max_concurrent = 4  # Maximum concurrent API calls

        # Semaphore to limit concurrent requests and prevent rate limiting
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

    def _truncate_text(self, text: str) -> str:
        """Truncate text if it exceeds the model's token limit.

        Args:
            text: Input text to potentially truncate

        Returns:
            Truncated text that fits within the model's token limit
        """
        tokens = self.tokenizer.encode(text)

        # Only truncate if text exceeds the maximum token limit
        if len(tokens) > self.max_tokens_per_input:
            tokens = tokens[:self.max_tokens_per_input]  # Keep first N tokens
            text = self.tokenizer.decode(tokens)  # Convert back to text

        return text

    def _batch_texts(self, texts: list[str]) -> list[list[str]]:
        """Split texts into batches that respect the maximum batch size.

        Args:
            texts: List of input texts to batch

        Returns:
            List of text batches, each containing at most max_batch_size texts
        """
        # First truncate all texts to ensure they fit within token limits
        truncated = [self._truncate_text(t) for t in texts]

        # Split into batches of maximum allowed size
        return [
            truncated[i:i + self.max_batch_size]
            for i in range(0, len(truncated), self.max_batch_size)
        ]

    async def aencode(self, input: list[str]) -> list[list[float]]:
        """Asynchronously encode a list of texts into vector embeddings.

        This method efficiently processes large lists of texts by:
        1. Truncating texts that exceed token limits
        2. Batching inputs to respect API limits (<=128 per request)
        3. Running batches concurrently with rate limiting

        Args:
            input: List of text strings to encode

        Returns:
            List of embedding vectors (one per input text)

        Raises:
            Exception: If any batch encoding fails
        """
        # Split texts into appropriately sized batches
        batches = self._batch_texts(input)

        async def encode_batch(batch: list[str]) -> list[list[float]]:
            """Encode a single batch of texts with rate limiting."""
            async with self.semaphore:  # Limit concurrent requests
                try:
                    # Call the embedding API with the batch
                    output = await litellm.aembedding(
                        model=self.model_name,
                        input=batch,
                        **self.kwargs
                    )
                    # Extract embeddings from the response
                    embeddings = [d["embedding"] for d in output.data]
                    return embeddings
                except Exception as e:
                    print(f"Error encoding batch: {e}")
                    raise

        # Process all batches concurrently and wait for completion
        results = await asyncio.gather(*(encode_batch(batch) for batch in batches))

        # Flatten the list of batch results into a single list of embeddings
        all_embeddings = [emb for batch_embs in results for emb in batch_embs]
        return all_embeddings
