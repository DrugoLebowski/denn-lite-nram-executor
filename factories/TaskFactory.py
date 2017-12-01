# Project
from Singleton import Singleton
from tasks.Task import Task
from tasks.TaskCopy import TaskCopy
from tasks.TaskAccess import TaskAccess
from tasks.TaskIncrement import TaskIncrement

class TaskFactory(metaclass=Singleton):

    @staticmethod
    def create(name: str, batch_size: int, max_int: int, num_regs: int, timestep: int) -> Task:
        return {
            "task_copy": TaskCopy(batch_size, max_int, num_regs, timestep),
            "task_access": TaskAccess(batch_size, max_int, num_regs, timestep),
            "task_increment": TaskIncrement(batch_size, max_int, num_regs, timestep),
        }[name]
