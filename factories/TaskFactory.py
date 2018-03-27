# Project
from Singleton import Singleton
from tasks.Task import Task
from tasks.TaskCopy import TaskCopy
from tasks.TaskAccess import TaskAccess
from tasks.TaskIncrement import TaskIncrement
from tasks.TaskSwap import TaskSwap
from tasks.TaskReverse import TaskReverse
from tasks.TaskPermutation import TaskPermutation
from tasks.TaskListK import TaskListK
from tasks.TaskListSearch import TaskListSearch
from tasks.TaskMerge import TaskMerge
from tasks.TaskWalkBST import TaskWalkBST
from tasks.TaskSum import TaskSum
from tasks.TaskProduct import TaskProduct

class TaskFactory(metaclass=Singleton):

    @staticmethod
    def create(name: str, batch_size: int, max_int: int, num_regs: int, timestep: int, sequence_size: int) -> Task:
        return {
            "task_copy": TaskCopy(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_access": TaskAccess(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_increment": TaskIncrement(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_swap": TaskSwap(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_reverse": TaskReverse(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_permutation": TaskPermutation(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_list_k": TaskListK(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_list_search": TaskListSearch(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_merge": TaskMerge(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_walkbst": TaskWalkBST(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_sum": TaskSum(batch_size, max_int, num_regs, timestep, sequence_size),
            "task_product": TaskProduct(batch_size, max_int, num_regs, timestep, sequence_size),
        }[name]
