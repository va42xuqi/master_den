import glob
import pandas as pd
import os

STEPS_IN = -1
STEPS_OUT = 1
MIN_SEQUENCE_LENGTH = 125
MAX_ALLOWED_SEQ_LEN = 3000
MAX_VELOCITY = 11.11  # 36.6 feet per second
SHUFFLE = True
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
BATCH_SIZE = 64
NUM_WORKERS = 0
TARGET_DATA = "LAL"
OBJECT_AMOUNT = 10  # 2 Teams * 5 Player
# USE_ALL = True  # True = PLAYER_DIRECTORY, False = PLAYER

# ANIMATION
FPS = 25
FRAMES = 500
FRAME_OFFSET = 0
HEIGHT: float = 15.24  # 50 feet
WIDTH: float = 28.65  # 94 feet
frame_size = [HEIGHT, WIDTH]

# categorial sizes
HEIGHT_CATEGORIES = int(HEIGHT * 100)  # Height categories (0 to 1524)
WIDTH_CATEGORIES = int(WIDTH * 100)  # Width categories (0 to 2865)
VELOCITY_CATEGORIES = int(MAX_VELOCITY * 100)  # Velocity categories (0 to 1111)

### LSTM ###
LSTM_LAYERS = 2
LSTM_HIDDEN_SIZE = 64
DROPOUT = 0.2
LEARNING_RATE = 0.001
### LMU ###
MEMORY_SIZE = 16
LMU_HIDDEN_SIZE = 16
THETA = 125
LEARN_A = False
LEARN_B = False
### CNN ###
CNN_HIDDEN_SIZE = 64
CNN_DROPOUT = 0.2
### LINEAR ###
LINEAR_HIDDEN_SIZE = 64
LINEAR_DROPOUT = 0.2
### RET_NET ###
RET_NET_HIDDEN_SIZE = 64
RET_NET_DROPOUT = 0.2
### BITNET ###
BITNET_HIDDEN_SIZE = 64
BITNET_DROPOUT = 0.2
### LUKAS_CNN ###
LUKAS_CNN_HIDDEN_SIZE = 64
LUKAS_CNN_DROPOUT = 0.2
### TRANSFORMER ###
EMBEDDING_SIZE = 256
TRAFO_HIDDEN_SIZE = EMBEDDING_SIZE * 4
NHEAD = 8
NBLOCK = 6
NUM_ENCODER_LAYERS = 6
DIM_FEEDFORWARD = 256
DIM_GEN = 64
####

GAMES = [
    "DAL@LAL_2015-11-01",
    "DEN@LAL_2015-11-03",
    "DET@LAL_2015-11-15",
    "GSW@LAL_2016-01-05",
    "IND@LAL_2015-11-29",
    "LAC@LAL_2015-12-25",
    "LAL@ATL_2015-12-04",
    "LAL@BKN_2015-11-06",
    "LAL@BOS_2015-12-30",
    "LAL@CHA_2015-12-28",
    "LAL@DAL_2015-11-13",
    "LAL@DEN_2015-12-22",
    "LAL@DET_2015-12-06",
    "LAL@GSW_2015-11-24",
    "LAL@MEM_2015-12-27",
    "LAL@MIA_2015-11-10",
    "LAL@OKC_2015-12-19",
    "LAL@ORL_2015-11-11",
    "LAL@PHI_2015-12-01",
    "LAL@PHX_2015-11-16",
    "LAL@POR_2015-11-28",
    "LAL@POR_2016-01-23",
    "LAL@SAC_2015-10-30",
    "LAL@SAC_2016-01-07",
    "LAL@SAS_2015-12-11",
    "LAL@WAS_2015-12-02",
    "MIL@LAL_2015-12-15",
    "MIN@LAL_2015-10-28",
    "NOP@LAL_2016-01-12",
    "OKC@LAL_2015-12-23",
    "OKC@LAL_2016-01-08",
    "PHI@LAL_2016-01-01",
    "PHX@LAL_2016-01-03",
    "POR@LAL_2015-11-22",
    "SAC@LAL_2016-01-20",
    "SAS@LAL_2016-01-22",
    "TOR@LAL_2015-11-20",
]


def get_path():
    # get path from this file
    p = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(p, "..", "..", "..")
    p = os.path.join(p, "data", "nba")
    return p


def get_load_path():
    p = get_path()
    p = os.path.join(p, "data_raw")
    if "beegfs" in p:
        p = os.path.realpath(p)
    p = os.path.join(p, "parquet")
    return p


def get_raw_path():
    p = get_path()
    p = os.path.join(p, "data_raw")
    if "beegfs" in p:
        p = os.path.realpath(p)
    p = os.path.join(p, "zipped")
    return p


def get_store_path():
    p = get_path()
    return p


path = get_path()

GAME = GAMES[0]
PLAYER = [os.path.join(path, GAME + ".parquet")]
PLAYER_DIRECTORY = [
    os.path.join(path, file) for file in os.listdir(path) if "LAL" in file
]


def get_name():
    name = f"{TARGET_DATA}"
    return name


def search_player(name, path):
    data_list = glob.glob(path + "/*")
    result = []
    for file in data_list:
        data = pd.read_parquet(file)
        for i in range(10):
            if data[f"player_{i}"].isin([name]).any():
                print(f"{name} is in {file}")
                result.append(file)
                break
    return result


def get_dataloader(mode="train", arch="lstm", min_sequence_length=MIN_SEQUENCE_LENGTH):
    from project.scenes.nba.dataset import CustomNBADataloader

    return CustomNBADataloader(
        split=[0.95, 1, 0] if mode == "train" else [0, 0, 1],
        name=(get_name() + "_train" if mode == "train" else get_name() + "_test"),
        path=get_store_path(),
        batch_size=(
            16
            if arch in ["transformer", "tft", "ostf", "os_bitnet", "uni_bitnet", "gpt"]
            else BATCH_SIZE
        ),
        min_sequence_length=min_sequence_length,
    )


def func_color(axs, i, input_plot, target_plot, rest):
    if i == 0:
        axs.plot(
            target_plot[0, 0, :],
            target_plot[0, 1, :],
            label="Ground Truth",
            color="green",
        )
        axs.plot(
            input_plot[0, 0, :],
            input_plot[0, 1, :],
            label="Target Input",
            color="darkgreen",
        )
    elif i == 1:
        axs.plot(
            input_plot[i, 0, :],
            input_plot[i, 1, :],
            label="Team",
            color="royalblue",
            alpha=0.5,
        )
    elif i == 5:
        axs.plot(
            input_plot[i, 0, :],
            input_plot[i, 1, :],
            label="Enemy",
            color="firebrick",
            alpha=0.5,
        )
    elif i < 5:
        axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], color="royalblue", alpha=0.5)
    elif i < 10:
        axs.plot(input_plot[i, 0, :], input_plot[i, 1, :], color="firebrick", alpha=0.5)
    else:
        axs.plot(
            input_plot[i, 0, :],
            input_plot[i, 1, :],
            label="Ball",
            color="black",
            alpha=0.5,
        )


def convert_to_centered(data):
    data = data.clone()
    data[..., 0] = data[..., 0] - WIDTH / 2
    data[..., 1] = data[..., 1] - HEIGHT / 2
    return data


def get_goal_position():
    basket_offset = 0.1016  # 4 inches
    # relative to middle point:
    # left basket: y = 0, x = - WIDTH / 2 + basket_offset
    # right basket: y = 0, x = WIDTH / 2 - basket_offset
    return [[-WIDTH / 2 + basket_offset, 0], [WIDTH / 2 - basket_offset, 0]]
