import argparse
import pandas as pd
import time
import random
from functools import wraps
import os
from nba_api.stats.endpoints import scheduleleaguev2, leaguegamefinder
from nba_api.stats.library.http import NBAStatsHTTP
from requests.exceptions import Timeout, ConnectionError, RequestException

NBAStatsHTTP.nba_response.timeout = 60
MAX_RETRIES = 3
base_dir = './schedules'

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

@retry
def get_schedules(season: str, league_id='00', season_dir=None):
    print(f"Get game IDs for season: {season}")
    schedules = scheduleleaguev2.ScheduleLeagueV2(
        season=season,
        league_id=league_id,
    )
    game_schedules = schedules.get_data_frames()[0]
    schedule_files = os.path.join(base_dir, season_dir.strip("/"), 'schedule.csv')
                
    game_schedules.to_csv(schedule_files, index=False)

def main():
    parser = argparse.ArgumentParser(description="Get game schedules")
    parser.add_argument('--season', type=str, required=True)
    parser.add_argument('--season-dir', type=str, default=None)
    args = parser.parse_args()
    
    season = args.season
    
    season_dir = args.season_dir 
    
    get_schedules(season=season, season_dir=season_dir)
    
if __name__ == "__main__":
    main()