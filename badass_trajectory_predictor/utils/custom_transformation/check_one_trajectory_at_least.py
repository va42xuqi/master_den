from ..dataset_base import BaseTransformation


class CheckTrajectory(BaseTransformation):

    def __init__(self, x_and_y=True, search_index=0):
        super().__init__()
        self.x_and_y = x_and_y
        self.search_index = search_index

    def forward(self, x, y, start_pos):
        if x is None:
            return None, None, None

        for i in range(x.shape[0]):
            trajectory_found = True
            for time in range(x.shape[2]):
                if x[i, self.search_index, time] == 0:
                    trajectory_found = False
                    break

            if trajectory_found and self.x_and_y:
                for time in range(y.shape[2]):
                    if y[i, self.search_index, time] == 0:
                        trajectory_found = False
                        break

            if trajectory_found:
                return x, y, start_pos

        return None, None, None
