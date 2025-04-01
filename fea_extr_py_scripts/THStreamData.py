from collections import deque

class THStreamDataPayload:
    def __init__(self, rgb_data, point_data, face_data, limb_data, ext_data, ext_desc):
        self.rgb_data = rgb_data
        self.point_data = point_data
        self.face_data = face_data
        self.limb_data = limb_data
        self.ext_data = ext_data
        self.ext_desc = ext_desc

    def __str__(self):
        # 使用 f-string 格式化字符串，使其更易于阅读
        return (f"THStreamDataPayload(rgb_data={self.rgb_data}, "
                f"point_data={self.point_data}, "
                f"face_data={self.face_data}, "
                f"limb_data={self.limb_data}, "
                f"ext_data={self.ext_data}, "
                f"ext_desc='{self.ext_desc}')")

class THDataWarehouse:
    def __init__(self, capacity):
        self.capacity = capacity
        self.warehouse = deque(maxlen=capacity)

    def add_item(self, payload: THStreamDataPayload):
        self.warehouse.append(payload)

    def get_items(self) -> THStreamDataPayload:
        try:
            return self.warehouse.popleft()
        except IndexError:
            return THStreamDataPayload()

    def get_size(self) -> int:
        return len(self.warehouse)

    def __str__(self):
        """提供仓库的字符串表示，便于打印和调试"""
        return f"DataWarehouse(capacity={self.capacity}, queue=[{', '.join(payload.description() for payload in self.warehouse)}])"