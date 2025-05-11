from dotenv import load_dotenv
import os
import requests
import argparse
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import time
import random
from functools import wraps
from datetime import datetime
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.http import NBAStatsHTTP
from requests.exceptions import Timeout, ConnectionError, RequestException

NBAStatsHTTP.nba_response.timeout = 60
MAX_RETRIES = 3
base_dir = './weather'

load_dotenv() 

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

API_KEY = os.getenv("API_KEY")
search_url = "http://api.weatherapi.com/v1/search.json"

def retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except (Timeout, ConnectionError, RequestException) as e:
                if attempt == MAX_RETRIES:
                    print(f"Max retries reached. Last error: {e}")
                    raise
                sleep_time = 60
                print(f"Attempt {attempt} failed: {e}. Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            except Exception as e:
                print(f"Error: {e}")
                raise
    return wrapper

def get_game_ids(season, league_id='00', season_type='Regular Season'):
    games = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable=league_id,
        season_type_nullable=season_type
    )
    game_ids = games.get_data_frames()[0]['GAME_ID'].dropna().unique().tolist()
    return game_ids

def get_location_coords(arena_name):
    params = {"key": API_KEY, "q": arena_name}
    res = requests.get(search_url, params=params)
    data = res.json()
    if data:
        location = data[0]  
        return location['lat'], location['lon']
    return None, None

@retry
def get_weather(latitude, longitude, date):

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "daily": ["temperature_2m_max", "rain_sum", "showers_sum", "snowfall_sum", "precipitation_sum", "precipitation_hours", "precipitation_probability_max", "wind_speed_10m_max", "temperature_2m_min", "apparent_temperature_mean"],
        "timezone": "Europe/London",
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    return daily

def create_weather_df(team_dir, schedule_df):
    weather_data = []
    for _, row in schedule_df.iterrows():
        location = row['arenaName'] + ', ' + row['arenaCity']
        lat, lon = get_location_coords(location)
        t = datetime.strptime(row['gameDate'].strip(), "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%d")
        print(location, t)
        if lat and lon:
            daily = get_weather(lat, lon, t)
            if daily: 
                weather_data.append({
                    "gameId": row['gameId'],
                    "arenaName": row['arenaName'],
                    "date": row['gameDate'],
                    "temperature_2m_max": daily.Variables(0).ValuesAsNumpy()[0],
                    "rain_sum": daily.Variables(1).ValuesAsNumpy()[0],
                    "showers_sum": daily.Variables(2).ValuesAsNumpy()[0],
                    "snowfall_sum": daily.Variables(3).ValuesAsNumpy()[0],
                    "precipitation_sum": daily.Variables(4).ValuesAsNumpy()[0],
                    "precipitation_hours": daily.Variables(5).ValuesAsNumpy()[0],
                    "precipitation_probability_max": daily.Variables(6).ValuesAsNumpy()[0],
                    "wind_speed_10m_max": daily.Variables(7).ValuesAsNumpy()[0],
                    "temperature_2m_min": daily.Variables(8).ValuesAsNumpy()[0],
                    "apparent_temperature_mean": daily.Variables(9).ValuesAsNumpy()[0]
                })
            else:
                print(f"Warning: Forecast data missing for {row['gameId']} on {row['gameDate']}")
                weather_data.append({
                    "gameId": row['gameId'],
                    "arenaName": row['arenaName'],
                    "date": row['gameDate'],
                    "temperature_2m_max": None,
                    "rain_sum": None,
                    "showers_sum": None,
                    "snowfall_sum": None,
                    "precipitation_sum": None,
                    "precipitation_hours": None,
                    "precipitation_probability_max": None,
                    "wind_speed_10m_max": None,
                    "temperature_2m_min": None,
                    "apparent_temperature_mean": None
                })
        time.sleep(1)


    weather_df = pd.DataFrame(weather_data)
    weather_file = os.path.join(base_dir, team_dir.strip("/"), f'weather.csv')
    weather_df.to_csv(weather_file, index=False)

def main():
    parser = argparse.ArgumentParser(description="Get weather data")
    parser.add_argument('--season', type=str, required=True)
    parser.add_argument('--team-dir', type=str, default=None)
    args = parser.parse_args()
    
    season = args.season
    
    team_dir = args.team_dir

    game_ids = get_game_ids(season)  
    schedule_file = os.path.join('./schedules', team_dir.strip("/"), f'schedule.csv')  
    schedule_df = pd.read_csv(schedule_file)
    schedule_df['gameId'] = schedule_df['gameId'].astype(str).str.zfill(10)
    schedule_df = schedule_df[schedule_df['gameId'].isin(game_ids)]

    create_weather_df(team_dir, schedule_df)

if __name__ == "__main__":

    main()