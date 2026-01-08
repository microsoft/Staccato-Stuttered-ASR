from pathlib import Path
from typing import Any, List, Optional

import dspy
import librosa
import numpy as np
from dotenv import load_dotenv

from approaches.vanilla_whisper.whisper import WhisperTranscriber
from evaluation.inference_models.asr_model_base import ASRModelBase

# Default model directory (relative to project root)
DEFAULT_WHISPER_MODEL_DIR = Path("./data/models/whisper-large-v3")


class TranscribeStutteredSpeechSignature(dspy.Signature):
    """You are an expert speech therapist with 20 years of experience helping people who stutter. You are tasked with transcribing a recording of a person who stutters speaking English. People who stutter speak with involuntary sound repetitions, word repetitions, prolongations, and blocks. You must transcribe the words that the speaker INTENDED to say, excluding involuntary disfluencies. You are also given an initial transcription of the recording that was produced by Whisper, an automatic speech recognition model. There may be errors in this transcription because Whisper was not trained on speech data of people who stutter, and is known to have poor accuracy on stuttered speech."""

    stuttered_speech_audio: dspy.Audio = dspy.InputField()
    initial_transcription: str = dspy.InputField()

    revised_transcription: str = dspy.OutputField()


class TranscribeStutteredSpeechModule(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(TranscribeStutteredSpeechSignature)

    def forward(
        self, audio_array: np.ndarray, sample_rate: int, initial_transcription: str
    ) -> str:
        dspy_audio = dspy.Audio.from_array(audio_array, sample_rate)
        output = self.cot(
            stuttered_speech_audio=dspy_audio,
            initial_transcription=initial_transcription,
        )
        return output.revised_transcription


class DspyPipeline(ASRModelBase):
    model: Any
    processor: Any

    def __init__(
        self,
        model_name: str = "dspy_pipe",
        model_dir: Optional[Path] = None,
        lm: str = "gpt-4o-mini-audio-preview-2024-12-17",
        use_flash_attention_2: bool = True,
    ):
        if model_dir is None:
            model_dir = DEFAULT_WHISPER_MODEL_DIR
        super().__init__(model_name, model_dir)
        self.lm = lm
        self.use_flash_attention_2 = use_flash_attention_2
        load_dotenv()

    def load_model(self):
        self.whisper_model = WhisperTranscriber(
            model_dir=self.model_dir,
            use_flash_attention_2=self.use_flash_attention_2,
        )
        self.whisper_model.load_model()
        dspy.configure(lm=dspy.LM(model=self.lm))
        self.transcribe_stuttered_speech_module = TranscribeStutteredSpeechModule()

    def transcribe(
        self,
        audio_arrays: List[np.ndarray],
        sample_rate: int,
    ) -> List[str]:
        whisper_transcriptions = self.whisper_model.transcribe(
            audio_arrays, sample_rate, language="en"
        )
        examples = [
            dspy.Example(
                audio_array=audio_array,
                sample_rate=sample_rate,
                initial_transcription=whisper_transcription,
            ).with_inputs("audio_array", "sample_rate", "initial_transcription")
            for whisper_transcription, audio_array in zip(
                whisper_transcriptions, audio_arrays
            )
        ]
        results = self.transcribe_stuttered_speech_module.batch(examples=examples)
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe stuttered speech")
    parser.add_argument(
        "--audio-files",
        nargs="+",
        type=Path,
        help="Paths to audio files to transcribe",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_WHISPER_MODEL_DIR,
        help="Path to Whisper model directory",
    )
    parser.add_argument(
        "--lm",
        type=str,
        default="gpt-4o-mini-audio-preview-2024-12-17",
        help="Language model to use for refinement",
    )
    args = parser.parse_args()

    pipe = DspyPipeline(model_dir=args.model_dir, lm=args.lm)
    pipe.load_model()

    if args.audio_files:
        audio_arrays = [librosa.load(wav_path)[0] for wav_path in args.audio_files]
        sr = librosa.load(args.audio_files[0])[1]
        result = pipe.transcribe(audio_arrays, sr)
        for path, text in zip(args.audio_files, result):
            print(f"{path}: {text}")
