from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from typing import Union
from tqdm import tqdm
def has_transcript(video_ids, languages: Union[str, list[str]]="en"):
    if not isinstance(languages, list):
        languages = [languages]

    available = []
    for video_id in tqdm(video_ids):
        try: 
            available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            if available_transcripts.find_transcript(languages):
                available.append(video_id)
        except:
            pass

    return available

    # print(transcript_results)


if __name__ == '__main__':
    available = has_transcript(
        [
            "sxSokGqIOms", 
            "Yy4JhoTfq38"
        ]
    )
    print(available)


    
