import math
import heapq

class AStarPlanner:
    def __init__(self, width=800, height=600, resolution=10, robot_radius=15, margin=30):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.margin = margin
        self.cols = int(width / resolution)
        self.rows = int(height / resolution)
        self.grid_blocked = None  # 缓存静态障碍物网格

    def _rect_distance(self, cx, cy, rect):
        rx, ry, rw, rh = rect
        closest_x = max(rx, min(cx, rx + rw))
        closest_y = max(ry, min(cy, ry + rh))
        return math.hypot(cx - closest_x, cy - closest_y)

    def plan(self, start_pos, end_pos, obstacles):
        start_c = int(min(max(start_pos[0] / self.resolution, 0), self.cols - 1))
        start_r = int(min(max(start_pos[1] / self.resolution, 0), self.rows - 1))
        end_c = int(min(max(end_pos[0] / self.resolution, 0), self.cols - 1))
        end_r = int(min(max(end_pos[1] / self.resolution, 0), self.rows - 1))

        if (start_c, start_r) == (end_c, end_r):
            return [end_pos]

        # 障碍物网格预计算 (添加缓存机制，仅计算一次)
        if self.grid_blocked is None:
            self.grid_blocked = [[False for _ in range(self.rows)] for _ in range(self.cols)]
            safe_dist = self.robot_radius + self.margin
            for c in range(self.cols):
                cx = c * self.resolution + self.resolution / 2.0
                for r in range(self.rows):
                    cy = r * self.resolution + self.resolution / 2.0
                    for obs in obstacles:
                        if self._rect_distance(cx, cy, obs) < safe_dist:
                            self.grid_blocked[c][r] = True
                            break

        # A* 搜索
        open_set = []
        heapq.heappush(open_set, (0, (start_c, start_r)))
        came_from = {}
        g_score = {(start_c, start_r): 0}
        
        # 为了应对目标被框在障碍物内的情况，记录一个离目标最近的点
        best_node = (start_c, start_r)
        best_h = math.hypot(start_c - end_c, start_r - end_r)
        
        motions = [(-1, 0, 10), (1, 0, 10), (0, -1, 10), (0, 1, 10),
                   (-1, -1, 14), (-1, 1, 14), (1, -1, 14), (1, 1, 14)]

        path_found = False

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == (end_c, end_r):
                best_node = current
                path_found = True
                break

            for dc, dr, cost in motions:
                nc, nr = current[0] + dc, current[1] + dr

                if 0 <= nc < self.cols and 0 <= nr < self.rows and not self.grid_blocked[nc][nr]:
                    tentative_g_score = g_score[current] + cost
                    
                    if tentative_g_score < g_score.get((nc, nr), float('inf')):
                        came_from[(nc, nr)] = current
                        g_score[(nc, nr)] = tentative_g_score
                        h_score = math.hypot(nc - end_c, nr - end_r) * 10
                        
                        if h_score / 10 < best_h:
                            best_h = h_score / 10
                            best_node = (nc, nr)
                            
                        heapq.heappush(open_set, (tentative_g_score + h_score, (nc, nr)))

        # 回溯路径
        path_grid = []
        current = best_node
        while current in came_from:
            path_grid.append(current)
            current = came_from[current]
        path_grid.append((start_c, start_r))
        path_grid.reverse()

        # 转换为连续坐标，使用格点中心
        path_continuous = []
        for c, r in path_grid:
            px = c * self.resolution + self.resolution / 2.0
            py = r * self.resolution + self.resolution / 2.0
            path_continuous.append((px, py))
            
        # 优化：强制将最终点替换为精准的目标坐标（如果可以找到端点）
        if path_found and path_continuous:
             path_continuous[-1] = end_pos

        if not path_continuous:
             return [start_pos]
             
        return path_continuous

def get_lookahead_waypoint(robot_pos, path, lookahead_dist=40.0):
    """
    沿着路径往后寻找第一个距离当前位置大于 lookahead_dist 的点
    """
    if not path:
        return robot_pos
        
    rx, ry = robot_pos
    
    for pt in path:
        dist = math.hypot(pt[0] - rx, pt[1] - ry)
        if dist >= lookahead_dist:
            return pt
            
    # 如果路径上所有点都很近，直接返回最后一个点
    return path[-1]
