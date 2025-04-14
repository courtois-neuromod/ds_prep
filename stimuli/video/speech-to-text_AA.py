import argparse
import json
from argparse import Namespace
from pathlib import Path

import assemblyai as aai
from moviepy.editor import VideoFileClip


def extract_wav(segment_file: Path, audio_dir: Path) -> Path:
    """Converts mkv files to wav files.

    Args:
        segment_file: path to the mkv file
        audio_dir: path to store wav file

    Return:
        wav_path: path to the wav file
    """
    segment_name = segment_file.stem
    wav_path = Path(audio_dir) / f"{segment_name}.wav"
    with VideoFileClip(str(segment_file)) as clip:
        clip.audio.write_audiofile(wav_path)
    return wav_path


def audio_to_transcrip(
    json_dir: Path,
    wav_path: Path,
    api_token: str,
) -> Path:
    """Extracts word timestamps from the wav files.

    Args:
        json_dir: path to the folder to store formatted,
        AssemblyAI output.
        wav_path: path to the audio file
        api_token: token to the AssemblyAI api

    Return:
        json_path: path to the json file which contains,
        words and timestamps
    """
    segment_name = wav_path.stem
    wav_path = str(wav_path)
    json_path = json_dir / f"{segment_name}.json"
    print("Connecting to the Assembly AI...")
    aai.settings.api_key = f"{api_token}"
    config = aai.TranscriptionConfig(language_code="fr")  # change this to en_us or fr
    transcriber = aai.Transcriber(config=config)
    print("Transcribing your audio file...")
    transcript = transcriber.transcribe(wav_path)
    json_results = {
        "results": {
            "channels": [
                {
                    "alternatives": [
                        {
                            "transcript": transcript.text,
                            "words": [
                                {
                                    "word": x.text,
                                    "onset": float(x.start) / 1000,
                                    "offset": float(x.end) / 1000,
                                    "confidence": x.confidence,
                                }
                                for x in transcript.words
                            ],
                        },
                    ],
                },
            ],
        },
    }

    with open(json_path, "w") as outfile:
        json.dump(json_results, outfile)
        print("Saved your transcribed json file...")
    return json_path


def parse_arguments() -> Namespace:
    """Function to parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process video files, extract wav files, and word onsets.",
    )
    parser.add_argument("--data_dir", help="where the mkv/wav files are")
    parser.add_argument("--output_dir", help="where to store the outputs")
    parser.add_argument("--file_type", help="the type of the audio file (mkv or wav)")
    parser.add_argument("--stimuli_name", help="name of the stimuli folder")
    parser.add_argument("--api_key", help="AssemblyAI API key.")
    return parser.parse_args()


def main() -> None:
    """Main function of the script.

    Args:
        --data_dir: path to the mkv files
        --output_dir: path to the store outputs
        --file_type : type of the audio/movie file
        --stimuli_name: name of the stimuli folder
        --api_key: AssemblyAI API key.
    """
    args = parse_arguments()
    movie_dir = Path(args.data_dir)
    json_dir = (
        Path(args.output_dir) / "transcript" / "word_timestamps" / args.stimuli_name
    )
    json_dir.mkdir(parents=True, exist_ok=True)

    if args.file_type == "mkv":
        # If the mkv files exist, iterate over the mkv files for conversion
        # to wav files.
        wav_dir = Path(args.output_dir) / "audio" / args.stimuli_name
        wav_dir.mkdir(parents=True, exist_ok=True)
        mkv_files = sorted(movie_dir.glob("*.mkv"))

        for segment_file in mkv_files:
            # extract and save the corresponding wav file
            wav_path = extract_wav(
                segment_file,
                wav_dir,
            )
            # extract and save the transcript file
            audio_to_transcrip(
                json_dir,
                wav_path,
                args.api_key,
            )
    elif args.file_type == "wav":
        wav_files = sorted(movie_dir.glob("*.wav"))
        for segment_file in wav_files:
            audio_to_transcrip(
                json_dir,
                segment_file,
                args.api_key,
            )


if __name__ == "__main__":
    main()
