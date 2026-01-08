"""Utility for transcribing batches of in-memory audio arrays using Whisper."""

from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperTranscriber:
    """Wrapper around HuggingFace Whisper pipeline to transcribe audio batches.

    The class initializes the model & processor once so multiple calls to
    `transcribe` are cheap.
    """

    def __init__(
        self,
        model_dir: Path,
        device: Optional[str] = None,
        use_flash_attention_2: bool = True,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_dir = model_dir
        self.use_flash_attention_2 = use_flash_attention_2
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def load_model(self):
        # Load model & processor directly on GPU
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_dir,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=(
                "flash_attention_2" if self.use_flash_attention_2 else None
            ),
            device_map=self.device,
        )
        
        # Clear forced_decoder_ids to avoid conflict with language parameter
        if hasattr(self.model, 'generation_config') and self.model.generation_config:
            self.model.generation_config.forced_decoder_ids = None
        
        self.processor = AutoProcessor.from_pretrained(self.model_dir)

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            chunk_length_s=30.0,
        )

    def transcribe(
        self,
        audio_arrays: List[np.ndarray],
        sample_rate: int,
        *,
        batch_size: int = 4,
        max_new_tokens: Optional[int] = None,
        language: Optional[str] = None,
        task: Optional[str] = None,  # e.g. "translate" or "transcribe"
        return_timestamps: bool = False,
    ) -> List[str]:
        """Transcribe a single batch of audio arrays.

        Parameters
        ----------
        audio_arrays: List[np.ndarray]
            Each element is a 1D float waveform (any dtype convertible to float32)
        sample_rate: int
            The sampling rate corresponding to every provided array.
        batch_size: int
            Internal batching for the pipeline (tune for memory/perf).
        max_new_tokens: Optional[int]
            Limit generation length; if None default model settings apply.
        language: Optional[str]
            Force language (e.g., "en"). If None, model will attempt detection.
        task: Optional[str]
            Whisper task override ("translate" or "transcribe").
        return_timestamps: bool
            If True, request timestamp information (if supported by pipeline version).

        Returns
        -------
        List[str]
            Transcribed text for each input audio array in order.
        """
        if not audio_arrays:
            return []

        # Normalize arrays & construct pipeline inputs, tracking indices
        inputs = []
        valid_indices = []

        for i, arr in enumerate(audio_arrays):
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr)
            # Ensure mono float32
            if arr.ndim > 1:
                # simple average to mono
                arr = arr.mean(axis=-1)
            arr = arr.astype(np.float32, copy=False)

            # Skip empty arrays
            if arr.size == 0:
                print(f"Warning: Skipping empty audio array at index {i}")
                continue

            inputs.append({"array": arr, "sampling_rate": sample_rate})
            valid_indices.append(i)

        # If all arrays were empty, return empty strings for each
        if not inputs:
            print("Warning: All audio arrays were empty")
            return [""] * len(audio_arrays)

        generate_kwargs = {}
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_new_tokens
        if language is not None:
            generate_kwargs["language"] = language
            # Clear forced_decoder_ids to avoid conflict with language parameter
            self.model.generation_config.forced_decoder_ids = None
        if task is not None:
            generate_kwargs["task"] = task
        if return_timestamps:
            # Newer transformers use return_timestamps; fallback safe if unused.
            generate_kwargs["return_timestamps"] = True

        outputs = self._pipe(
            inputs,
            batch_size=min(batch_size, len(inputs)),
            generate_kwargs=generate_kwargs or None,
        )

        # Normalize single vs multi return structure
        if isinstance(outputs, dict):  # single item path
            outputs = [outputs]

        # Extract texts from valid outputs
        valid_texts: List[str] = []
        for out in outputs:
            if return_timestamps and isinstance(out.get("chunks"), list):
                # join chunk texts if timestamp mode used
                chunk_text = " ".join(c.get("text", "").strip() for c in out["chunks"])  # type: ignore[index]
                valid_texts.append(chunk_text.strip())
            else:
                valid_texts.append(out.get("text", "").strip())

        # Reconstruct full results list, inserting empty strings for skipped arrays
        texts = [""] * len(audio_arrays)
        for i, valid_idx in enumerate(valid_indices):
            texts[valid_idx] = valid_texts[i]

        return texts


if __name__ == "__main__":  # simple manual smoke test
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("audio_file", type=Path, help="Path to audio file")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./data/models/whisper-large-v3"),
        help="Path to Whisper model directory",
    )
    args = parser.parse_args()

    transcriber = WhisperTranscriber(model_dir=args.model_dir)
    transcriber.load_model()
    arr, sr = librosa.load(args.audio_file)
    result = transcriber.transcribe([arr], sr, language="en", task="transcribe")
    print(f"Transcription: {result[0]}")
