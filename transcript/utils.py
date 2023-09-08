from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from typing import Union
from tqdm import tqdm

def can_translate_to(available_transcripts, languages: str):
    translated_languages = [x["language_code"] for x in available_transcripts._translation_languages]
    for language in languages:
        if language in translated_languages:
            return True
    return False

def has_transcript(video_ids, languages: Union[str, list[str]]="en"):
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

def get_transcripts(video_ids, language: str = "en"):
    available_videos = has_transcript(video_ids, language)
    transcripts = []
    for video_id in tqdm(available_videos):
        transcript = YouTubeTranscriptApi.list_transcripts(video_id)
        if language in list(transcript._manually_created_transcripts.keys()) or language in list(transcript._generated_transcripts.keys()):
            transcripts.append({
                "video_id": video_id,
                "transcript": YouTubeTranscriptApi.get_transcript(video_id, languages=(language,))
            })
        else:
            manually, generated = list(transcript._manually_created_transcripts.keys()), list(transcript._generated_transcripts.keys())
            available =  manually[0] if len(manually) > 0 else generated[0]
            available = transcript.find_transcript([available]).translate(language)
            transcripts.append({
                "video_id": video_id,
                "transcipt": available.fetch()
            })
    return transcripts

if __name__ == '__main__':
    video_ids = [
        # "sxSokGqIOms", 
        # "4MEng1Jjcyw",
        "Yy4JhoTfq38"
    ]
    print(get_transcripts(video_ids))