import threading

class SequenceGenerator:
    _instance = None
    _current_seq = 0
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SequenceGenerator, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # 注意：由于单例模式，__init__ 方法可能会被多次调用，
        # 但我们只需要初始化一次，所以这里不需要设置初始序列号。
        pass

    def next_seq(self):
        with self._lock:
            self._current_seq += 1
            return str(self._current_seq)
#
# # 使用示例
# seq_gen = SequenceGenerator()
# print(seq_gen.next_seq())  # 输出: 1
# print(seq_gen.next_seq())  # 输出: 2
#
# # 即使尝试创建新的实例，也会返回相同的实例
# another_seq_gen = SequenceGenerator()
# print(another_seq_gen is seq_gen)  # 输出: True
# print(another_seq_gen.next_seq())  # 输出: 3