import os
import numpy as np
import pandas as pd
import glob

def get_dfs(seasons_dir, stat_dirs, schedule_dir, weather_dir, y_dir):
    combined_stats = {stat_dir: [] for stat_dir in stat_dirs}
    attendance_all = []
    schedule_all = []
    weather_all = []

    for season in seasons_dir:
        for stat_dir in stat_dirs:
            stats_files = glob.glob(os.path.join(stat_dir, season, "team/team_stats_*.csv"))
            stats_dfs = []
            for file in stats_files:
                df = pd.read_csv(file)
                if df.isnull().values.any():
                    print(f"Stats file with NaNs: {file}")
                stats_dfs.append(df)
            combined = pd.concat(stats_dfs, ignore_index=True)
            combined_stats[stat_dir].append(combined)

        attendance_files = glob.glob(os.path.join(y_dir, season, "team_stats_*.csv"))
        attendance_dfs = []
        for file in attendance_files:
            df = pd.read_csv(file)
            attendance_dfs.append(df)
        attendance_all.append(pd.concat(attendance_dfs, ignore_index=True))

        schedule_file = os.path.join(schedule_dir, season, "schedule.csv")
        if os.path.exists(schedule_file):
            df = pd.read_csv(schedule_file)
        schedule_all.append(df)

        weather_file = os.path.join(weather_dir, season, "weather.csv")
        if os.path.exists(weather_file):
            df = pd.read_csv(weather_file)
        weather_all.append(df)

    final_stats = {os.path.basename(stat_dir.rstrip('/\\')): pd.concat(dfs, ignore_index=True) for stat_dir, dfs in combined_stats.items()}

    final_attendance = pd.concat(attendance_all, ignore_index=True)
    final_schedule = pd.concat(schedule_all, ignore_index=True)
    final_weather = pd.concat(weather_all, ignore_index=True)

    return final_stats, final_attendance, final_schedule, final_weather

def feature_engineer_data(full_stats_df):
    df_with_games = full_stats_df.copy()
    df_with_games['game_number'] = df_with_games.groupby(['teamId', 'seasonYear']).cumcount() + 1

    game_teams = {}
    for game_id in df_with_games["gameId"].unique():
        home_team = df_with_games[(df_with_games["gameId"] == game_id) & (df_with_games["is_home"] == True)]["teamId"].values
        away_team = df_with_games[(df_with_games["gameId"] == game_id) & (df_with_games["is_home"] == False)]["teamId"].values
        
        if len(home_team) > 0 and len(away_team) > 0:
            game_teams[game_id] = {"home": home_team[0], "away": away_team[0]}

    df_with_opponent = df_with_games.copy()

    opponent_map = {}
    for game_id in df_with_opponent["gameId"].unique():
        game_df = df_with_opponent[df_with_opponent["gameId"] == game_id]
        if len(game_df) == 2:
            teams = game_df["teamId"].tolist()
            opponent_map[(game_id, teams[0])] = teams[1]
            opponent_map[(game_id, teams[1])] = teams[0]

    df_with_opponent["opp_teamId"] = df_with_opponent.apply(lambda row: opponent_map.get((row["gameId"], row["teamId"]), np.nan), axis=1)

    df_sorted = df_with_opponent.sort_values(["teamId", "game_number"])
    numerical_cols = [
        'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage', 'threePointersMade', 'threePointersAttempted',
        'threePointersPercentage', 'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage', 'reboundsOffensive', 
        'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal', 'points', 
        'plusMinusPoints', 'estimatedOffensiveRating', 'offensiveRating', 'estimatedDefensiveRating', 'defensiveRating',
        'estimatedNetRating', 'netRating', 'assistPercentage', 'assistToTurnover', 'assistRatio', 'offensiveReboundPercentage', 
        'defensiveReboundPercentage', 'reboundPercentage', 'estimatedTeamTurnoverPercentage', 'turnoverRatio',
        'effectiveFieldGoalPercentage', 'trueShootingPercentage', 'usagePercentage', 'estimatedUsagePercentage', 'estimatedPace', 'pace',
        'pacePer40', 'possessions', 'PIE', 'freeThrowAttemptRate', 'teamTurnoverPercentage', 'oppEffectiveFieldGoalPercentage',
        'oppFreeThrowAttemptRate', 'oppTeamTurnoverPercentage', 'oppOffensiveReboundPercentage'
    ]

    df_sorted['games_played'] = df_sorted.groupby(['teamId', 'seasonYear']).cumcount()
    for col in numerical_cols:
        n_samples = df_sorted[col].shape[0]
        df_sorted[col] = df_sorted[col] + np.random.normal(loc=0, scale=1, size=n_samples)
        df_sorted[f"{col}_season_avg"] = df_sorted.groupby(['teamId', 'seasonYear'])[col].transform(lambda x: x.expanding().mean().shift(1))
        df_sorted[f"{col}_past_5_avg"] = df_sorted.groupby(['teamId', 'seasonYear'])[col].transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))

    opponent_stats = df_sorted.copy()
    opponent_stats = opponent_stats.drop(columns='opp_teamId')
    opponent_stats = opponent_stats.rename(columns={"teamId": "opp_teamId"})
    opp_cols = {col: f"opp_{col}" for col in opponent_stats.columns if col not in ["gameId", "opp_teamId"]}
    opponent_stats = opponent_stats.rename(columns=opp_cols)

    final_df = df_sorted.merge(opponent_stats, left_on=["gameId", "opp_teamId"], right_on=["gameId", "opp_teamId"], how="left")
    final_df = final_df[final_df['is_home']==True]
    return final_df

def get_final_dfs(seasons_dir, stat_dirs, schedule_dir, weather_dir, y_dir):
    
    final_stats, attendance_df, schedule_df, weather_df = get_dfs(seasons_dir, stat_dirs, schedule_dir, weather_dir, y_dir)
    
    traditional_df = final_stats['boxscoretraditional']
    advanced_df = final_stats['boxscoreadvanced']
    fourfactor_df = final_stats['boxscorefourfactor']

    full_stats_df = traditional_df.merge(advanced_df, on=['gameId', 'teamId', 'teamCity', 'teamName', 'teamTricode', 'teamSlug','minutes'], how='left', suffixes=('', '_dup'))
    full_stats_df = full_stats_df.merge(fourfactor_df, on=['gameId', 'teamId', 'teamCity', 'teamName', 'teamTricode', 'teamSlug','minutes'], how='left', suffixes=('', '_dup'))
    for col in full_stats_df.columns:
        dup_col = f"{col}_dup"
        if dup_col in full_stats_df.columns:
            if full_stats_df[col].equals(full_stats_df[dup_col]):
                full_stats_df.drop(columns=[dup_col], inplace=True)
                
    full_stats_df = full_stats_df.sort_values(by='gameId')
    
    full_stats_df = full_stats_df.merge(schedule_df[['gameId', 'seasonYear', 'homeTeam_teamTricode']], on='gameId', how='left')
    full_stats_df['is_home'] = full_stats_df['teamTricode'] == full_stats_df['homeTeam_teamTricode'] 
    full_stats_df = full_stats_df.drop(columns=['teamSlug', 'minutes', 'homeTeam_teamTricode'])

    final_df = feature_engineer_data(full_stats_df)

    model_1_df = final_df.copy()

    attendance_df = final_df.merge(attendance_df, on = 'gameId', how = 'left', suffixes=('', '_dup'))[['gameId', 'ATTENDANCE']]

    schedule_df['gameId'] = schedule_df['gameId'].astype(str).str.zfill(10).astype(int)
    schedule_df = final_df.merge(schedule_df, on = 'gameId', how = 'left', suffixes=('', '_dup'))
    for col in schedule_df.columns:
        dup_col = f"{col}_dup"
        if dup_col in schedule_df.columns:
            if schedule_df[col].equals(schedule_df[dup_col]):
                schedule_df.drop(columns=[dup_col], inplace=True)
    schedule_df = schedule_df[['gameId', 'seasonYear', 'gameDate', 'teamTricode', 'opp_teamTricode']]

    model_2_df = schedule_df.merge(attendance_df,  on = 'gameId', how = 'left')
    model_2_df = model_2_df.merge(weather_df, left_on= ['gameId', 'gameDate'], right_on= ['gameId', 'date'], how='left')
    model_2_df = model_2_df.drop(columns='date')
    model_2_df['precipitation_probability_max'] = model_2_df['precipitation_probability_max'].fillna(0)
    return model_1_df, model_2_df

def main(base_dir, subdirs, seasons_dir, stat_dirs, schedule_dir, weather_dir, y_dir):
    model_1_df, model_2_df = get_final_dfs(seasons_dir, stat_dirs, schedule_dir, weather_dir, y_dir)

    full_df = model_1_df.merge(model_2_df, on=['gameId', 'seasonYear', 'teamTricode', 'opp_teamTricode'], how='inner')
    full_df = full_df.dropna()

    model1_data = full_df[model_1_df.columns]
    model2_data = full_df[model_2_df.columns]

    Y_model1 = model1_data[['plusMinusPoints', 'seasonYear']]
    Y_train_model1 = Y_model1.loc[(Y_model1['seasonYear'] == '2022-23') | (Y_model1['seasonYear'] == '2023-24')]
    Y_train_model1 = Y_train_model1.drop('seasonYear', axis=1)
    Y_test_model1 = Y_model1.loc[Y_model1['seasonYear'] == '2024-25']
    Y_test_model1 = Y_test_model1.drop('seasonYear', axis=1)

    X_model1 = model1_data[[col for col in model_1_df.columns if 'avg' in col]]
    X_model1 = pd.concat([model1_data['gameId'], model1_data['seasonYear'], X_model1], axis=1)

    X_train_model1 = X_model1.loc[(X_model1['seasonYear'] == '2022-23') | (X_model1['seasonYear'] == '2023-24')]
    X_train_model1 = X_train_model1.drop('seasonYear', axis=1)
    X_test_model1 = X_model1.loc[X_model1['seasonYear'] == '2024-25']
    X_test_model1 = X_test_model1.drop('seasonYear', axis=1)

    Y_model2 = model2_data[['ATTENDANCE', 'seasonYear']]
    Y_train_model2 = Y_model2.loc[(Y_model2['seasonYear'] == '2022-23') | (Y_model2['seasonYear'] == '2023-24')]
    Y_train_model2 = Y_train_model2.drop('seasonYear', axis=1)
    Y_test_model2 = Y_model2.loc[Y_model2['seasonYear'] == '2024-25']
    Y_test_model2 = Y_test_model2.drop('seasonYear', axis=1)

    home_dummies = pd.get_dummies(model2_data['teamTricode'], prefix='team', drop_first=True).astype(float)
    X_model2 = pd.concat([model2_data, home_dummies], axis=1)
    X_model2 = X_model2.drop('ATTENDANCE', axis=1)

    X_train_model2 = X_model2.loc[(X_model2['seasonYear'] == '2022-23') | (X_model2['seasonYear'] == '2023-24')]
    X_train_model2 = X_train_model2.select_dtypes(include=['float', 'int'])
    X_test_model2 = X_model2.loc[X_model2['seasonYear'] == '2024-25']
    X_test_model2 = X_test_model2.select_dtypes(include=['float', 'int'])

    train_dir = os.path.join(base_dir, subdirs[0])
    test_dir = os.path.join(base_dir, subdirs[1])

    model1_data.to_csv(os.path.join(train_dir, 'full_stats.csv'), index=False)
    Y_train_model1.to_csv(os.path.join(train_dir, 'Y_train_model1.csv'), index=False)
    Y_test_model1.to_csv(os.path.join(test_dir, 'Y_test_model1.csv'), index=False)
    
    X_train_model1.to_csv(os.path.join(train_dir, 'X_train_model1.csv'), index=False)
    X_test_model1.to_csv(os.path.join(test_dir, 'X_test_model1.csv'), index=False)

    Y_train_model2.to_csv(os.path.join(train_dir, 'Y_train_model2.csv'), index=False)
    Y_test_model2.to_csv(os.path.join(test_dir, 'Y_test_model2.csv'), index=False)
    
    X_model2.to_csv(os.path.join(train_dir, 'full_attendance.csv'), index=False)
    X_train_model2.to_csv(os.path.join(train_dir, 'X_train_model2.csv'), index=False)
    X_test_model2.to_csv(os.path.join(test_dir, 'X_test_model2.csv'), index=False)

if __name__ == "__main__":
    base_dir = "/data/nba_data/online"

    subdirs = ["train", "test"]

    season_dirs = ['season2223', 'season2324', 'season2425']
    stat_dirs = ['./boxscoretraditional', './boxscoreadvanced', './boxscorefourfactor']
    y_dir = './boxscoresummary'
    schedule_dir = './schedules'
    weather_dir = './weather'

    main(base_dir, subdirs, season_dirs, stat_dirs, schedule_dir, weather_dir, y_dir)
    
