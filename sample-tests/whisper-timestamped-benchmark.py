import argparse
import datetime
from enum import Enum
import json
import math
import mimetypes
import os
import re
import subprocess
import time
import traceback

class MatchMode(Enum):
    segment = 'segment' # Get the timestamp range of the full segment
    word = 'word' # Get the timestamps of only the matched pattern or word
    def __str__(self):
        return self.value

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            prog='Transcriber',
            description='Transcribes stuff.')
        parser.add_argument('--benchmark', action='store_false', help='Do benchmarks')
        parser.add_argument('--condition_on_previous', action='store_true', help='Whether to provide the previous output as a prompt for the next window.')
        parser.add_argument('--detect_disfluencies', action='store_true', help='Enables disfluency detection.')
        parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='Cuda device to use.')
        parser.add_argument('--language', type=str, default='en', choices=['auto', 'en', 'ja', 'fr'], help='Transcription language. Usually faster if you specify. Capability depends on the model.')
        parser.add_argument('--match_mode', type=MatchMode, default='word', choices=list(MatchMode), help='Whether to match timestamps for the whole segment or just the word.')
        parser.add_argument('--model', type=str, default='large-v3-turbo', help='Whisper model to use.')
        parser.add_argument('--no_vad', action='store_false', help='Do not use Voice Activity Detection')
        parser.add_argument('--naive', action='store_true', help='Use naive approach for word-level timestamps (takes twice as long)')
        parser.add_argument('--prompt', type=str, help='Initial prompt to use for transcription.')

        args, unknown_args = parser.parse_known_args()
        if help in args:
            parser.print_help()

        # Flip some negative flags for clarity
        vad = not args.no_vad

        # Traverse all arguments for files and directories
        input_files = []
        for arg in unknown_args:
            if os.path.isfile(arg):
                input_files.append(arg)
            elif os.path.isdir(arg):
                input_files.extend([os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(arg) for f in filenames])

        # Take only videos from all listed files and sort
        try:
            import natsort
            input_videos = natsort.natsorted([file for file in input_files if mimetypes.guess_type(file)[0].split('/')[0] == 'video'])
        except ImportError: # Fallback to standard library sort
            print('natsort not found. Falling back to standard sort. Try "pip install natsort" if you want files to be naturally sorted.')
            input_videos = sorted([file for file in input_files if mimetypes.guess_type(file)[0].split('/')[0] == 'video'])
        
        #print('Loading whisper...')
        import whisper_timestamped as whisper
        whisper_model = whisper.load_model(args.model, device=args.device, download_root="./models")
        for video in input_files:
            print(f'Transcribing file "{video}"')
            audio = whisper.load_audio(video)
            benchmarks = []
            benchmark_loops = 5
            num_loops = benchmark_loops if args.benchmark else 1
            for i in range(0, num_loops):
                start_timestamp = time.perf_counter()
                result = whisper.transcribe_timestamped(whisper_model,
                    audio,
                    naive_approach=args.naive,
                    initial_prompt=args.prompt,
                    vad='auditok' if vad else False,
                    beam_size=5,
                    best_of=5,
                    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                    language=None if args.language == 'auto' else args.language,
                    condition_on_previous_text=args.condition_on_previous,
                    detect_disfluencies=args.detect_disfluencies)
                end_timestamp = time.perf_counter()
                elapsed = end_timestamp - start_timestamp  # seconds (float)
                benchmarks.append(elapsed)
                if args.benchmark:
                    print(f'Benchmark {i+1} of {benchmark_loops} elapsed time: {elapsed} seconds.')
            if args.benchmark:
                print('Elapsed times:')
                print(benchmarks)
            
            #transcript_filename = os.path.splitext(video)[0] + '.json'
            #print(f'Saving transcript to file {transcript_filename}')
            #with open(transcript_filename, 'w', encoding='utf-8') as fout:
            #    json.dump(result, fout, ensure_ascii=False, indent=2)
            #output = os.path.splitext(video)[0]
            #transcript_filename = ''
            #filename_count = 0
            #while True:
            #    # Name the output after the input and appended with the language info. Add numbers if there is a conflict.
            #    base_path = os.path.dirname(output) + (os.path.sep if os.path.dirname(output) != "" else "") + os.path.basename(output)
            #    transcript_filename = '{}.{}'.format(base_path, result['language'])
            #    if filename_count > 0:
            #        transcript_filename += '-{}'.format(filename_count)
            #    transcript_filename += '.vtt'
            #    if os.path.isfile(transcript_filename):
            #        filename_count += 1 # Try to deconflict the file name by finding a different file name
            #    else:
            #        break
            #with open(transcript_filename, 'w', encoding='utf-8') as fout:
                #from whisper_timestamped.make_subtitles import write_vtt
                #print(f'Saving transcript to file {transcript_filename}')
                #write_vtt(result['segments'], fout)
    except argparse.ArgumentError as e:
        print(e)
    except Exception:
        print(traceback.format_exc())
