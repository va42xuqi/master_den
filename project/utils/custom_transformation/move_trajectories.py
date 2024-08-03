from ..dataset_base import BaseTransformation


class MoveTrajectories(BaseTransformation):
    def __init__(self, x_and_y=True, search_index=0, move_to=0):
        super().__init__()
        self.x_and_y = x_and_y
        self.search_index = search_index
        self.move_to = move_to

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
                # Swap the trajectory to the beginning
                x[[self.move_to, i]] = x[[i, self.move_to]]
                y[[self.move_to, i]] = y[[i, self.move_to]]
                start_pos[[self.move_to, i]] = start_pos[[i, self.move_to]]
                return x, y, start_pos

        return None, None, None
