import pandas as pd
import json
import os
import py7zr


def get_names(event):
    player_names = {-1: "ball"}
    player_positions = {-1: "no_position"}
    team_names = {
        event["home"]["teamid"]: event["home"]["name"],
        event["visitor"]["teamid"]: event["visitor"]["name"],
    }

    visitor_abbreviation = event["visitor"]["abbreviation"]
    home_abbreviation = event["home"]["abbreviation"]

    for player in event["home"]["players"]:
        player_names[player["playerid"]] = (
            player["firstname"] + " " + player["lastname"]
        )
        player_positions[player["playerid"]] = player["position"]

    for player in event["visitor"]["players"]:
        player_names[player["playerid"]] = (
            player["firstname"] + " " + player["lastname"]
        )
        player_positions[player["playerid"]] = player["position"]

    return (
        visitor_abbreviation,
        home_abbreviation,
        team_names,
        player_names,
        player_positions,
    )


def process_game_file(
    game_file,
    data_folder=os.path.join("data", "zipped"),
    output_folder=os.path.join("data", "parquet"),
    columns=None,
):
    if columns is None:
        columns = [
            "game_id",
            "team_id",
            "position",
            "quarter",
            "event_id",
            "moment_id",
            "shot_clock",
            "ball",
        ]

    with py7zr.SevenZipFile(os.path.join(data_folder, game_file), mode="r") as z:
        zip_data = z.read()
        names = z.getnames()
        if len(names) == 0:
            return
        name = names[0]
        if zip_data and name.endswith(".json"):
            json_file_name = name
            data_str = zip_data[json_file_name].read().decode("utf-8")
            game_data = json.loads(data_str)

            data = []
            (
                visitor_abbreviation,
                home_abbreviation,
                team_names,
                player_names,
                player_positions,
            ) = get_names(game_data["events"][0])

            removed_elements = 0
            counted_elements = 0

            for event in game_data["events"]:
                for moment in event["moments"]:
                    if len(moment[5]) != 11:
                        removed_elements += 1
                        continue

                    row_elements = {}
                    player_pos = False
                    if "game_id" in columns:
                        row_elements["game_name"] = (
                            visitor_abbreviation + "@" + home_abbreviation
                        )
                    if "team_id" in columns:
                        row_elements["home_team"] = team_names[event["home"]["teamid"]]
                        row_elements["visitor_team"] = team_names[
                            event["visitor"]["teamid"]
                        ]
                    if "position" in columns:
                        player_pos = True
                    if "quarter" in columns:
                        row_elements["quarter"] = moment[0]
                    if "event_id" in columns:
                        row_elements["event_id"] = event["eventId"]
                    if "moment_id" in columns:
                        row_elements["moment_id"] = moment[1]
                    if "shot_clock" in columns:
                        row_elements["shot_clock"] = moment[3]

                    row_elements["game_clock"] = moment[2]

                    if "ball" in columns:
                        row_elements["ball_x"] = moment[5][0][2]
                        row_elements["ball_y"] = moment[5][0][3]
                        row_elements["ball_radius"] = moment[5][0][4]

                    counter = 0
                    for player in moment[5][1:]:
                        row_elements[f"player_{counter}_name"] = player_names[player[1]]
                        row_elements[f"player_{counter}_id"] = player[1]
                        row_elements[f"x_{counter}"] = player[2]
                        row_elements[f"y_{counter}"] = player[3]
                        if player_pos:
                            row_elements[f"role_{counter}"] = player_positions[
                                player[1]
                            ]
                        counter += 1

                    data.append(row_elements)
                    counted_elements += 1

            df = pd.DataFrame(data)
            if len(data) == 0:
                raise Exception(f"No moments in {game_file}")
            print(f"Removed {removed_elements} elements from {game_file}")
            print(f"Counted {counted_elements} elements from {game_file}")
            json_file_name = f"{visitor_abbreviation}@{home_abbreviation}_{game_data['gamedate']}.json"
            filename = f"{visitor_abbreviation}@{home_abbreviation}_{game_data['gamedate']}.parquet"
            df.to_parquet(os.path.join(output_folder, filename))


def unzip_all(data_folder, output_folder):
    data_folder = data_folder
    os.makedirs(output_folder, exist_ok=True)
    game_files = [f.name for f in os.scandir(data_folder)]

    for idx, game_file in enumerate(game_files):
        try:
            process_game_file(
                game_file, data_folder=data_folder, output_folder=output_folder
            )
        except Exception as e:
            print(f"Error in file {game_file} : {e}")
        print(f"percentage completed: {(idx + 1) / len(game_files) * 100:.2f}%")
