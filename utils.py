import pandas as pd
import isodate
from googleapiclient.discovery import build

def get_channel_id_from_url(url, youtube):
    """
    Extracts the channel ID from a YouTube channel URL.
    Supports: /channel/, /@username/, /c/ and /user/ types.
    """
    if "channel/" in url:
        return url.split("channel/")[-1].split("/")[0]
    elif "@" in url:
        username = url.split("@")[-1].split("/")[0]
    else:
        username = url.strip("/").split("/")[-1]

    # Search channel by username
    response = youtube.search().list(
        part='snippet',
        q=username,
        type='channel',
        maxResults=1
    ).execute()

    if response.get('items'):
        return response['items'][0]['snippet']['channelId']
    return None

# def fetch_channel_data(channel_id, api_key, max_videos=30):
#     """
#     Fetches up to `max_videos` uploaded video data from a channel.
#     Returns a DataFrame with title, stats, duration, publish time, etc.
#     """
#     youtube = build('youtube', 'v3', developerKey=api_key)
#     try:
#         channel_res = youtube.channels().list(
#             part='contentDetails,snippet',
#             id=channel_id
#         ).execute()

#         uploads_id = channel_res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
#         channel_title = channel_res['items'][0]['snippet']['title']

#     except Exception as e:
#         print(f"Error fetching channel info: {e}")
#         return pd.DataFrame()

#     videos = []
#     video_count = 0
#     next_page_token = None

#     while video_count < max_videos:
#         try:
#             pl_res = youtube.playlistItems().list(
#                 part='snippet',
#                 playlistId=uploads_id,
#                 maxResults=50,
#                 pageToken=next_page_token
#             ).execute()
#         except Exception as e:
#             print(f"Error fetching playlist: {e}")
#             break

#         for item in pl_res.get('items', []):
#             if video_count >= max_videos:
#                 break

#             video_id = item['snippet']['resourceId']['videoId']
#             title = item['snippet']['title']
#             publish_time = item['snippet']['publishedAt']

#             vid_res = youtube.videos().list(
#                 part='statistics,contentDetails',
#                 id=video_id
#             ).execute()

#             if not vid_res.get('items'):
#                 continue

#             stats = vid_res['items'][0].get('statistics', {})
#             content = vid_res['items'][0].get('contentDetails', {})

#             duration_iso = content.get('duration', 'PT0S')
#             duration = isodate.parse_duration(duration_iso).total_seconds()

#             videos.append({
#                 'Channel': channel_title,
#                 'Title': title,
#                 'Views': int(stats.get('viewCount', 0)),
#                 'Likes': int(stats.get('likeCount', 0)),
#                 'Comments': int(stats.get('commentCount', 0)),
#                 'Duration': duration_iso,
#                 'Duration_sec': duration,
#                 'Publish Time': publish_time
#             })

#             video_count += 1

#         next_page_token = pl_res.get('nextPageToken')
#         if not next_page_token:
#             break

#     return pd.DataFrame(videos)
def fetch_channel_data(channel_id, api_key, max_videos=30):
    """
    Fetches up to `max_videos` uploaded video data from a channel.
    Returns a DataFrame with title, stats, duration, publish time, etc.
    """
    youtube = build('youtube', 'v3', developerKey=api_key)
    try:
        channel_res = youtube.channels().list(
            part='contentDetails,snippet',
            id=channel_id
        ).execute()

        uploads_id = channel_res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        channel_title = channel_res['items'][0]['snippet']['title']

    except Exception as e:
        print(f"Error fetching channel info: {e}")
        return pd.DataFrame()

    videos = []
    video_count = 0
    next_page_token = None

    while video_count < max_videos:
        try:
            pl_res = youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_id,
                maxResults=50,
                pageToken=next_page_token
            ).execute()
        except Exception as e:
            print(f"Error fetching playlist: {e}")
            break

        for item in pl_res.get('items', []):
            if video_count >= max_videos:
                break

            video_id = item['snippet']['resourceId']['videoId']
            title = item['snippet']['title']
            publish_time = item['snippet']['publishedAt']

            vid_res = youtube.videos().list(
                part='statistics,contentDetails',
                id=video_id
            ).execute()

            if not vid_res.get('items'):
                continue

            stats = vid_res['items'][0].get('statistics', {})
            content = vid_res['items'][0].get('contentDetails', {})

            duration_iso = content.get('duration', 'PT0S')
            duration = isodate.parse_duration(duration_iso).total_seconds()

            videos.append({
                'Video ID': video_id,  # ✅ This line added
                'Channel': channel_title,
                'Title': title,
                'Views': int(stats.get('viewCount', 0)),
                'Likes': int(stats.get('likeCount', 0)),
                'Comments': int(stats.get('commentCount', 0)),
                'Duration': duration_iso,
                'Duration_sec': duration,
                'Publish Time': publish_time
            })

            video_count += 1

        next_page_token = pl_res.get('nextPageToken')
        if not next_page_token:
            break

    return pd.DataFrame(videos)

def fetch_multi_year_video_data(channel_ids, api_key, start_year=2023, end_year=2025, max_videos=1000):
    """
    Fetches video data from multiple channels across multiple years.
    Filters based on year range and tags Shorts.
    """
    from googleapiclient.discovery import build
    import isodate
    import pandas as pd
    from datetime import datetime

    youtube = build('youtube', 'v3', developerKey=api_key)
    all_data = []

    for channel_id in channel_ids:
        try:
            # Get upload playlist ID and channel name
            ch_res = youtube.channels().list(
                part='contentDetails,snippet',
                id=channel_id
            ).execute()

            uploads_id = ch_res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            channel_name = ch_res['items'][0]['snippet']['title']
        except Exception as e:
            print(f"Failed to fetch channel {channel_id}: {e}")
            continue

        video_count = 0
        next_page_token = None

        while video_count < max_videos:
            try:
                pl_res = youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_id,
                    maxResults=50,
                    pageToken=next_page_token
                ).execute()
            except:
                break

            for item in pl_res.get("items", []):
                if video_count >= max_videos:
                    break

                snippet = item["snippet"]
                vid_id = snippet["resourceId"]["videoId"]
                title = snippet["title"]
                publish_time = snippet["publishedAt"]
                publish_date = pd.to_datetime(publish_time)
                year = publish_date.year

                if year < start_year or year > end_year:
                    continue

                vid_res = youtube.videos().list(
                    part="statistics,contentDetails,snippet",
                    id=vid_id
                ).execute()

                if not vid_res.get("items"):
                    continue

                video = vid_res["items"][0]
                stats = video.get("statistics", {})
                content = video.get("contentDetails", {})
                duration = isodate.parse_duration(content.get("duration", "PT0S")).total_seconds()

                all_data.append({
                    "Channel": channel_name,
                    "Title": title,
                    "Video ID": vid_id,
                    "Views": int(stats.get("viewCount", 0)),
                    "Likes": int(stats.get("likeCount", 0)),
                    "Comments": int(stats.get("commentCount", 0)),
                    "Duration (sec)": duration,
                    "Publish Time": publish_date,
                    "isShort": duration < 60
                })
                video_count += 1

            next_page_token = pl_res.get("nextPageToken")
            if not next_page_token:
                break

    return pd.DataFrame(all_data)
