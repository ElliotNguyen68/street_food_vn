from typing import Union, List, Dict
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from tqdm import tqdm
from pathlib import Path
import os

def can_translate_to(available_transcripts, languages: str):
    translated_languages = [x["language_code"] for x in available_transcripts._translation_languages]
    for language in languages:
        if language in translated_languages:
            return True
    return False

def has_transcript(video_ids, languages: Union[str, List[str]]="en"):
    if not isinstance(languages, list):
        languages = [languages]

    available = []
    for video_id in tqdm(video_ids):
        try: 
            available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            if can_translate_to(available_transcripts, languages):
                available.append(video_id)
        except:
            # print("NOT FOUND transcript", video_id)
            pass

    return available

def get_transcripts(video_ids, language: str = "en") -> Dict[str, Union[str, Dict[str, str]]]:
    available_videos = has_transcript(video_ids, language)
    # transcripts = []
    for video_id in tqdm(available_videos):
        transcript = YouTubeTranscriptApi.list_transcripts(video_id)
        if language in list(transcript._manually_created_transcripts.keys()) or language in list(transcript._generated_transcripts.keys()):
            # if not generator:
            #     transcripts.append({
            #         "video_id": video_id,
            #         "transcript": YouTubeTranscriptApi.get_transcript(video_id, languages=(language,))
            #     })
            # else:
            yield {
                    "video_id": video_id,
                    "transcript": YouTubeTranscriptApi.get_transcript(video_id, languages=(language,))
                }
        else:
            manually, generated = list(transcript._manually_created_transcripts.keys()), list(transcript._generated_transcripts.keys())
            available =  manually[0] if len(manually) > 0 else generated[0]
            available = transcript.find_transcript([available]).translate(language)
            # if not generator:
            #     transcripts.append({
            #         "video_id": video_id,
            #         "transcript": available.fetch()
            #     })
            # else:
            yield {
                    "video_id": video_id,
                    "transcript": available.fetch()
                }
    # if not generator:
    #     return transcripts

def to_doc(transcripts: Dict[str, str], save_dir: str):
    if not os.path.exists(save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=False)
        
    for video_trans in transcripts:
        video_id, transcript = video_trans["video_id"], video_trans["transcript"]
        doc = " ".join(text["text"] for text in transcript)
        with open(os.path.join(save_dir, f"trans_{video_id}.txt"), "w", encoding="utf-8") as fr:
            fr.write(doc)  

if __name__ == '__main__':
    video_ids = [
        "sxSokGqIOms", 
        "4MEng1Jjcyw",
        "Yy4JhoTfq38"
    ]
    print(to_doc(get_transcripts(video_ids), "./datasets/transcripts"))
