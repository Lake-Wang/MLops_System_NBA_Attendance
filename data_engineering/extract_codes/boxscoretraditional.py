import argparse
import pandas as pd
import time
import random
from functools import wraps
import os
from nba_api.stats.endpoints import boxscoretraditionalv3, leaguegamefinder
from nba_api.stats.library.http import NBAStatsHTTP
from requests.exceptions import Timeout, ConnectionError, RequestException

NBAStatsHTTP.nba_response.timeout = 60
MAX_RETRIES = 3
base_dir = './boxscoretraditional'

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
def get_boxscore_data(game_id):
    boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    return boxscore.get_data_frames()

def process_boxscores(game_ids, base_dir, player_dir, team_dir):
    if isinstance(game_ids, str):
        game_ids = [game_ids]
    
    successful_games = 0
    
    for ix, game_id in enumerate(game_ids):
        try:
            data_frames = get_boxscore_data(game_id)
            if len(data_frames) >= 3:
                player_df = data_frames[1]
                team_df = data_frames[2]
            
                player_file = os.path.join(base_dir, player_dir.strip("/"), f'player_stats_{game_id}.csv')
                team_file = os.path.join(base_dir, team_dir.strip("/"), f'team_stats_{game_id}.csv')
                
                player_df.to_csv(player_file, index=False)
                team_df.to_csv(team_file, index=False)
                
                successful_games += 1

                print(f"Saved stats for game {game_id}")
            else:
                print(f"Skipping game {game_id}")
        except Exception as e:
            print(f"Failed to save game {game_id}: {e}")
    return successful_games

def main():
    parser = argparse.ArgumentParser(description="Get box scores data")
    parser.add_argument('--season', type=str, required=True)
    parser.add_argument('--player-dir', type=str, default=None)
    parser.add_argument('--team-dir', type=str, default=None)
    args = parser.parse_args()
    
    season = args.season
    
    player_dir = args.player_dir 
    team_dir = args.team_dir

    game_ids = get_game_ids(season)    
    successful_games = process_boxscores(game_ids, base_dir, player_dir, team_dir)
    
    print(f"Saved {successful_games} out of {len(game_ids)} games")

if __name__ == "__main__":
    main()