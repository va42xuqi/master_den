import numpy as np
import torch
from ..dataset_base import BaseTransformation


# v = s / t
def velocity_to_position(vx, vy, t, x0, y0):
    return vx * t + x0, vy * t + y0  # v = s / t


def velocity_vector_to_position_vector(v, t, x0, y0):
    if len(v.shape) == 4:
        res = torch.zeros((v.shape[0], v.shape[1], 2, v.shape[3]), dtype=v.dtype)
        for i in range(v.shape[3]):
            position = velocity_to_position(v[:, :, 0, i], v[:, :, 1, i], t[i], x0, y0)
            res[:, :, 0, i], res[:, :, 1, i] = position
            x0, y0 = position

        return res
    else:
        res = torch.zeros((v.shape[0], 2, v.shape[2]), dtype=v.dtype)
        for i in range(v.shape[2]):
            position = velocity_to_position(v[:, 0, i], v[:, 1, i], t[i], x0, y0)
            res[:, 0, i], res[:, 1, i] = position
            x0, y0 = position

        return res


def position_to_distance(x, y, axis=0) -> np.ndarray:
    dx = x - y
    dist = np.sqrt(np.sum(dx**2, axis=axis))
    return dist


class VelocityTransformation(BaseTransformation):

    def __init__(self, max_velocity=25, unit="m/s", has_ball=True):
        super().__init__()
        self.has_ball = has_ball
        if unit == "m/s":
            self.max_velocity = max_velocity
        elif unit == "km/h":
            self.max_velocity = max_velocity / 3.6

    def forward(self, x, y, startpos=None):

        if x is None or y is None:
            return None, None, None

        for outer in range(x.shape[0]):
            time = abs(x[outer][2][1:] - x[outer][2][:-1])
            x[outer][:2, :-1] = (x[outer][:2, 1:] - x[outer][:2, :-1]) / time

            time = abs(y[outer][2][1:] - y[outer][2][:-1])
            y[outer][:2, 1:] = (y[outer][:2, 1:] - y[outer][:2, :-1]) / time

            time = abs(x[outer][2][-1] - y[outer][2][0])
            y[outer][:2, 0] = (y[outer][:2, 0] - x[outer][:2, -1]) / time

        x = x[:, :, : x.shape[2] - 1]
        if self.max_velocity > 0:
            if not self.check_velocity(x, self.has_ball) or not self.check_velocity(
                y, self.has_ball
            ):
                return None, None, None

        return x, y, startpos

    def check_velocity(self, x, has_ball=False):
        if not has_ball:
            velocity = torch.sqrt(x[:, 0, :] ** 2 + x[:, 1, :] ** 2)
        else:
            velocity = torch.sqrt(x[:-1, 0, :] ** 2 + x[:-1, 1, :] ** 2)
        return (velocity <= self.max_velocity).all()


if __name__ == "__main__":
    vec = VelocityTransformation(100000)
    t1 = torch.zeros((2, 3, 10))
    t2 = torch.zeros((2, 3, 4))

    for k in range(2):
        for i in range(10):
            t1[k][0][i] = i * 2
            t1[k][1][i] = 0
            t1[k][2][i] = i

    t1[0][0][1] = 10

    for k in range(2):
        for i in range(10, 14):
            t2[k][0][i - 10] = i + 9
            t2[k][1][i - 10] = 0
            t2[k][2][i - 10] = i

    print(t1)
    print(t2)
    res1, res2 = vec.forward(t1, t2)
    print(res1)
    print(res2)

    x1 = np.array(([2, 2, 3], [2, 4, 3]))
    y1 = np.array(([4, 2, 4], [4, 4, 4]))

    # print(res1)

    # test velocity_to_position
    v = torch.zeros((2, 1, 2, 10))  # 2 batch, 1 person, 2 velocity, 10 time steps
    v[0, 0, 0, 0] = 1
    v[0, 0, 1, 5] = 1
    v[1, 0, 0, 2] = 4
    t = [2 for i in range(10)]
    x0 = torch.Tensor(([0], [0]))
    y0 = torch.Tensor(([0], [0]))
    res = velocity_vector_to_position_vector(v, t, x0, y0)
    # print(res)
