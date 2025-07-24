import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import networkx as nx

# 定义复数算子 (120°相位旋转)
alpha = np.exp(-1j * 2 * np.pi / 3)  # e^{-j2π/3}
gamma = np.array([
    [1, alpha ** 2, alpha],
    [alpha, 1, alpha ** 2],
    [alpha ** 2, alpha, 1]
])


# IEEE 13节点系统参数
class IEEE13System:
    def __init__(self):
        self.lines = []  # 线路数据
        self.loads = []   # 负荷数据
        self.transformers = []  # 变压器数据
        self.buses = set()  # 所有节点集合
        self._build_data()
        self._build_topology()
        self.voltage_levels = {
            "SourceBus": 115.0,
            "650": 115.0,
            "632": 4.16,
            "633": 4.16,
            "634": 0.48,
            "645": 4.16,
            "646": 4.16,
            "670": 4.16,
            "671": 4.16,
            "675": 4.16,
            "680": 4.16,
            "684": 4.16,
            "692": 4.16,
            "611": 0.48,
            "652": 0.48
        }
        # 设置电压等级基准值
        self.voltage_bases = [115.0, 4.16, 0.48]  # kV
        self.base_power = 1000.0  # kVA (基准功率)



    def _build_data(self):
        self.lines = [
            # (名称, 相数, 起始节点, 起始相别, 终止节点, 终止相别, 长度, 长度单位,  R矩阵, X矩阵, C矩阵, 阻抗单位, 开关标识)

            # 三相线路
            ('650632', 3, '650', ['1', '2', '3'], '632', ['1', '2', '3'], 2000, 'ft',
             [0.3465, 0.1560, 0.3375, 0.1580, 0.1535, 0.3414],
             [1.0179, 0.5017, 1.0478, 0.4236, 0.3849, 1.0348],
             None, 'mi',False),

            ('632670', 3, '632', ['1', '2', '3'], '670', ['1', '2', '3'], 667, 'ft',
             [0.3465, 0.1560, 0.3375, 0.1580, 0.1535, 0.3414],
             [1.0179, 0.5017, 1.0478, 0.4236, 0.3849, 1.0348],
             None, 'mi',False),

            ('670671', 3, '670', ['1', '2', '3'], '671', ['1', '2', '3'], 1333, 'ft',
             [0.3465, 0.1560, 0.3375, 0.1580, 0.1535, 0.3414],
             [1.0179, 0.5017, 1.0478, 0.4236, 0.3849, 1.0348],
             None, 'mi',False),

            ('671680', 3, '671', ['1', '2', '3'], '680', ['1', '2', '3'], 1000, 'ft',
             [0.3465, 0.1560, 0.3375, 0.1580, 0.1535, 0.3414],
             [1.0179, 0.5017, 1.0478, 0.4236, 0.3849, 1.0348],
             None, 'mi',False),

            ('632633', 3, '632', ['1', '2', '3'], '633', ['1', '2', '3'], 500, 'ft',
             [0.7526, 0.1580, 0.7475, 0.1560, 0.1535, 0.7436],
             [1.1814, 0.4236, 1.1983, 0.5017, 0.3849, 1.2112],
             None, 'mi',False),

            ('692675', 3, '692', ['1', '2', '3'], '675', ['1', '2', '3'], 500, 'ft',
             [0.791721, 0.318476, 0.781649, 0.28345, 0.318476, 0.791721],
             [0.438352, 0.0276838, 0.396697, -0.0184204, 0.0276838, 0.438352],
             [383.948, 0, 383.948, 0, 0, 383.948], 'mi',False),

            # # 两相线路
            # ('632645', 2, '632', ['2', '3'], '645', ['2', '3'], 500, 'ft',
            #  [1.3238, 0.2066, 1.3294],
            #  [1.3569, 0.4591, 1.3471],
            #  None, 'mi',False),
            # ('645646', 2, '645', ['2', '3'], '646', ['2', '3'], 300, 'ft',
            #  [1.3238, 0.2066, 1.3294],
            #  [1.3569, 0.4591, 1.3471],
            #  None, 'mi', False),
            # 两相线路改单相
            ('632645', 1, '632', ['2'], '645', ['2'], 500, 'ft',
             [1.3294],
             [1.3471],
             None, 'mi', False),
            ('645646', 1, '645', ['2'], '646', ['2'], 300, 'ft',
             [1.3294],
             [1.3471],
             None, 'mi', False),

            ('671684', 2, '671', ['1', '3'], '684', ['1', '3'], 300, 'ft',
             [1.3238, 0.2066, 1.3294],
             [1.3569, 0.4591, 1.3471],
             None, 'mi',False),

            # 单相线路
            ('684611', 1, '684', ['3'], '611', ['3'], 300, 'ft',
             [1.3292],
             [1.3475],
             None, 'mi',False),

            ('684652', 1, '684', ['1'], '652', ['1'], 800, 'ft',
             [1.3425],
             [0.5124],
             [236], 'mi',False),

            #开关线路
            ('671692', 3, '671', ['1', '2', '3'], '692', ['1', '2', '3'], 0, 'ft',
            [0.0001,0,0.0001,0,0,0.0001],
            [0,0,0,0,0,0],
            None,
             'mi',True) , # 新增开关标识

            ('633634', 3, '633', ['1', '2', '3'], '634', ['1', '2', '3'], 0, 'ft',
             [0.0000, 0, 0.0000, 0, 0, 0.0000],
             [0, 0, 0, 0, 0, 0],
             None,
             'mi', False),

            ('SourceBus650', 3, 'SourceBus', ['1', '2', '3'], '650', ['1', '2', '3'], 0, 'ft',
             [0.0000, 0, 0.0000, 0, 0, 0.0000],
             [0, 0, 0, 0, 0, 0],
             None,
             'mi', False),


        ]

        self.loads = [
        # (bus名称, 相别, Phases, Conn, Model, kV, kW, kvar)
        # 三相平衡负荷
        ('671', 'abc', 3, 'Delta', 1, 4.16, 1155, 660),
        # 单相负荷（634节点）
        ('634', 'a', 1, 'Wye', 1, 0.277, 160, 110),
        ('634', 'b', 1, 'Wye', 1, 0.277, 120, 90),
        ('634', 'c', 1, 'Wye', 1, 0.277, 120, 90),
        # 单相负荷（其他节点）
        ('645', 'b', 1, 'Wye', 1, 2.4, 170, 125),
        ('646', 'b', 1, 'Delta', 2, 4.16, 230, 132),
        ('692', 'c', 1, 'Delta', 5, 4.16, 170, 151),
        # 三相不平衡负荷（675节点）
        ('675', 'a', 1, 'Wye', 1, 2.4, 485, 190),
        ('675', 'b', 1, 'Wye', 1, 2.4, 68, 60),
        ('675', 'c', 1, 'Wye', 1, 2.4, 290, 212),
        # 其他单相负荷
        ('611', 'c', 1, 'Wye', 5, 2.4, 170, 80),
        ('652', 'a', 1, 'Wye', 2, 2.4, 128, 86),
        ('670', 'a', 1, 'Wye', 1, 2.4, 17, 10),
        ('670', 'b', 1, 'Wye', 1, 2.4, 66, 38),
        ('670', 'c', 1, 'Wye', 1, 2.4, 117, 68)
    ]

        self.transformers = [
            # (名称, 高压侧节点, 低压侧节点, 高压侧连接方式, 低压侧连接方式,
            #  高压侧电压, 低压侧电压, 容量, 高压侧电阻%, 高压侧电抗%, 低压侧电阻%, 低压侧电抗%)
            ('Sub', 'SourceBus', '650', 'delta', 'wye', 115.0, 4.16, 5000, 0.5, 4, 0.5, 4),
            ('XFM1', '633', '634', 'wye', 'wye', 4.16, 0.480, 500, 0.55, 1, 0.55, 1)
        ]
        # 收集所有节点
        self.buses = set()
        for line in self.lines:
            self.buses.add(line[2])  # 起始节点
            self.buses.add(line[4])  # 终止节点
        for load in self.loads:
            self.buses.add(load[0])   # 负荷节点
        for xfm in self.transformers:
            self.buses.add(xfm[1])    # 高压侧节点
            self.buses.add(xfm[2])    # 低压侧节点

    def  _build_topology(self):
        # 初始化数据结构
        self.children = defaultdict(list)  # 每个节点的子节点
        self.parent = {}  # 每个节点的父节点
        self.path_to_root = {}  # 每个节点到根节点的路径
        self.node_phases = defaultdict(set)  # 每个节点的相位集合
        self.node_voltages = {}  # 每个节点的额定电压（线电压，kV）
        self.downstream_buses = defaultdict(set)  # 每个节点的下游节点
        self.line_impedances = {}  # 线路阻抗矩阵缓存
        self.transformer_info = {}  # 变压器信息缓存，键为(高压节点,低压节点)

        # 处理线路
        for line in self.lines:
            (name, nphases, start_bus, start_phases, end_bus, end_phases,
             length, length_unit, r_matrix, x_matrix, c_matrix, imp_unit, is_switch) = line

            # 数字相位到字母的映射
            phase_mapping = {'1': 'a', '2': 'b', '3': 'c'}
            start_phase_letters = sorted([phase_mapping[p] for p in start_phases])
            end_phase_letters = sorted([phase_mapping[p] for p in end_phases])

            # 更新节点相位
            self.node_phases[start_bus] |= set(start_phase_letters)
            self.node_phases[end_bus] |= set(end_phase_letters)

            # 构建连接关系（线路）
            self.parent[end_bus] = start_bus
            self.children[start_bus].append(end_bus)

            # 计算并缓存线路阻抗和相位信息
            Z_matrix, length_mi, _ = self.create_line_impedance_matrix(line)
            self.line_impedances[(start_bus, end_bus)] = (Z_matrix, length_mi, start_phase_letters)

            # 确保节点已添加到下游字典
            if start_bus not in self.downstream_buses:
                self.downstream_buses[start_bus] = set()
            if end_bus not in self.downstream_buses:
                self.downstream_buses[end_bus] = set()

        # 处理变压器
        for xfm in self.transformers:
            name, hv_bus, lv_bus, hv_conn, lv_conn, hv_kv, lv_kv, kva, hv_r, hv_x, lv_r, lv_x = xfm
            # 变压器高压侧和低压侧都是三相
            self.node_phases[hv_bus] |= {'a', 'b', 'c'}
            self.node_phases[lv_bus] |= {'a', 'b', 'c'}
            self.node_voltages[hv_bus] = hv_kv
            self.node_voltages[lv_bus] = lv_kv

            # 变压器连接关系（高压侧到低压侧）
            self.parent[lv_bus] = hv_bus
            self.children[hv_bus].append(lv_bus)

            # 存储变压器信息
            self.transformer_info[(hv_bus, lv_bus)] = {
                'name': name,
                'hv_conn': hv_conn,
                'lv_conn': lv_conn,
                'hv_kv': hv_kv,
                'lv_kv': lv_kv,
                'kva': kva,
                'hv_r_pct': hv_r,
                'hv_x_pct': hv_x,
                'lv_r_pct': lv_r,
                'lv_x_pct': lv_x
            }

            # 确保节点已添加到下游字典
            if hv_bus not in self.downstream_buses:
                self.downstream_buses[hv_bus] = set()
            if lv_bus not in self.downstream_buses:
                self.downstream_buses[lv_bus] = set()

        # 确定根节点（假设SourceBus是根）
        self.root_bus = 'SourceBus'
        self.node_voltages[self.root_bus] = 115.0  # 根节点电压

        # 确保根节点在下游字典中
        if self.root_bus not in self.downstream_buses:
            self.downstream_buses[self.root_bus] = set()

        # 重置下游节点集合（重要！）
        self.downstream_buses = defaultdict(set)

        # 计算下游节点集合（使用修复的递归方法）
        self._compute_downstream_buses(self.root_bus)

        # 计算到根节点的路径
        for bus in self.buses:
            path = []
            current = bus
            while current != self.root_bus and current in self.parent:
                parent = self.parent[current]
                # 判断连接是线路还是变压器
                if (parent, current) in self.line_impedances:
                    conn_type = 'line'
                elif (parent, current) in self.transformer_info:
                    conn_type = 'transformer'
                else:
                    conn_type = 'unknown'
                path.append((parent, current, conn_type))
                current = parent
            self.path_to_root[bus] = list(reversed(path))  # 从根到当前节点

        # 计算下游节点集合（递归）
        self._compute_downstream_buses(self.root_bus)

    def get_node_voltage(self, bus):
        """获取节点的基准线电压 (kV)"""
        return self.voltage_levels.get(bus, 4.16)  # 默认4.16kV

    def _compute_downstream_buses(self, bus):
        """递归计算下游节点集合"""
        # 确保当前节点包含在自身下游
        self.downstream_buses[bus].add(bus)

        # 递归处理所有子节点
        for child in self.children.get(bus, []):
            # 确保子节点已初始化
            if child not in self.downstream_buses:
                self.downstream_buses[child] = set()

            # 递归计算子节点的下游
            self._compute_downstream_buses(child)

            # 将子节点的下游添加到当前节点的下游
            self.downstream_buses[bus] |= self.downstream_buses[child]

    def create_line_impedance_matrix(self, line_data):
        # 解包线路数据
        (name, nphases, start_bus, start_phases, end_bus, end_phases,
         length, length_unit, r_matrix, x_matrix, c_matrix, imp_unit, is_switch) = line_data

        # 数字相位到字母的映射
        phase_mapping = {'1': 'a', '2': 'b', '3': 'c'}
        phase_letters = [phase_mapping[p] for p in start_phases]

        # 处理开关线路 - 理想短路线
        if is_switch:
            # 使用极小的阻抗值 (0.0001 + j0)
            z = 0.0001
            if nphases == 1:
                return np.array([[complex(z, 0)]]), 0, phase_letters
            elif nphases == 2:
                return np.array([
                    [complex(z, 0), complex(0, 0)],
                    [complex(0, 0), complex(z, 0)]
                ]), 0, phase_letters
            elif nphases == 3:
                return np.array([
                    [complex(z, 0), complex(0, 0), complex(0, 0)],
                    [complex(0, 0), complex(z, 0), complex(0, 0)],
                    [complex(0, 0), complex(0, 0), complex(z, 0)]
                ]), 0, phase_letters

        # 处理普通线路
        # 单位转换 (英尺转英里)
        if length_unit == 'ft' and imp_unit == 'mi':
            length_mi = length / 5280  # 1英里=5280英尺
        else:
            length_mi = length  # 单位相同或未知时保持原值

        # 根据相数构建阻抗矩阵
        if nphases == 1:
            Z_matrix = np.array([
                [complex(r_matrix[0], x_matrix[0])]
            ])
        elif nphases == 2:
            Z_matrix = np.array([
                [complex(r_matrix[0], x_matrix[0]), complex(r_matrix[1], x_matrix[1])],
                [complex(r_matrix[1], x_matrix[1]), complex(r_matrix[2], x_matrix[2])]
            ])
        elif nphases == 3:
            # 注意：元素顺序为 [Raa, Rab, Rbb, Rac, Rbc, Rcc]
            Z_matrix = np.array([
                [complex(r_matrix[0], x_matrix[0]), complex(r_matrix[1], x_matrix[1]),
                 complex(r_matrix[3], x_matrix[3])],
                [complex(r_matrix[1], x_matrix[1]), complex(r_matrix[2], x_matrix[2]),
                 complex(r_matrix[4], x_matrix[4])],
                [complex(r_matrix[3], x_matrix[3]), complex(r_matrix[4], x_matrix[4]),
                 complex(r_matrix[5], x_matrix[5])]
            ])
        else:
            raise ValueError(f"不支持的相数: {nphases}")

        # 考虑实际长度
        Z_matrix *= length_mi

        return Z_matrix, length_mi, phase_letters

    def get_phase_index(self, phase_code):
        """
        根据相位代码获取相位索引 - 针对单相和三相优化
        """
        # 标准化相位代码（小写、去除空格）
        phase_code = str(phase_code).lower().strip()

        # 单相映射
        if phase_code in ['a', '1']:
            return [0]  # A相
        elif phase_code in ['b', '2']:
            return [1]  # B相
        elif phase_code in ['c', '3']:
            return [2]  # C相

        # 三相负荷
        if phase_code in ['abc', '3ph', 'three']:
            return [0, 1, 2]  # 所有三相

        # 默认返回空列表
        print(f"警告: 无法识别的相位代码: '{phase_code}'")
        return []

    def get_all_impedances(self):
        """
        获取系统中所有线路和变压器的阻抗信息
        保持原有数据结构不变

        返回:
            dict: 包含两个键的字典:
                - 'lines': 列表，每个元素是原线路元组
                - 'transformers': 列表，每个元素是原变压器元组
        """
        # 确保数据已构建
        if not hasattr(self, 'lines'):
            self._build_data()

        # 提取阻抗相关信息
        line_impedances = []
        for line in self.lines:
            # 提取线路名称、R矩阵和X矩阵
            name = line[0]
            r_matrix = line[8]
            x_matrix = line[9]
            line_impedances.append((name, r_matrix, x_matrix))

        transformer_impedances = []
        for tf in self.transformers:
            # 提取变压器名称和阻抗值
            name = tf[0]
            impedance = tf[5]
            transformer_impedances.append((name, impedance))

        return {
            "lines": line_impedances,
            "transformers": transformer_impedances
        }

    def to_per_unit(self, value, v_base, is_impedance=False):
        """转换为标幺值"""
        S_base = 100  # kVA系统基准

        if is_impedance:
            # 阻抗标幺值转换: Z_pu = Z_actual * S_base / V_base^2
            if np.isscalar(value):
                return value * S_base / (v_base ** 2)
            else:
                return value * S_base / (v_base ** 2)
        else:
            # 功率标幺值转换: S_pu = S_actual / S_base
            return value / S_base



    def get_downstream_buses(self, bus):
        """获取节点的下游节点集合"""
        return self.downstream_buses.get(bus, set())


    def get_line_edges(self):
        """获取所有线路连接边"""
        return [(line[2], line[4]) for line in self.lines]  # (start_bus, end_bus)


    def test_downstream_nodes(self):
        """
        测试并输出所有节点的下游节点
        """
        # 确保拓扑已经构建
        if not hasattr(self, 'downstream_buses'):
            self._build_topology()

        print("\n" + "=" * 50)
        print("节点下游关系测试")
        print("=" * 50)

        # 按节点名称排序
        all_buses = sorted(self.buses, key=lambda x: (isinstance(x, int), x))

        for bus in all_buses:
            downstream = self.downstream_buses.get(bus, set())

            # 格式化输出
            print(f"节点 {bus} 的下游节点: {', '.join(str(node) for node in sorted(downstream))}")

            # 检查是否有子节点但下游节点为空（应包含自身）
            if bus in self.children and not downstream:
                print(f"警告: 节点 {bus} 有子节点但下游节点为空!")

            # 检查下游节点是否包含自身
            if bus not in downstream:
                print(f"警告: 节点 {bus} 的下游节点不包含自身!")

        print("\n关键节点检查:")
        # 检查根节点
        root_downstream = self.downstream_buses.get(self.root_bus, set())
        print(f"根节点 {self.root_bus} 的下游节点数: {len(root_downstream)} (应包含所有节点)")

        # 检查叶节点（没有子节点的节点）
        leaf_nodes = [bus for bus in self.buses if bus not in self.children or not self.children[bus]]
        for leaf in leaf_nodes:
            leaf_downstream = self.downstream_buses.get(leaf, set())
            if leaf_downstream != {leaf}:
                print(f"警告: 叶节点 {leaf} 的下游节点应为自身, 实际为: {leaf_downstream}")
            else:
                print(f"叶节点 {leaf} 的下游节点正确: {leaf_downstream}")

        print("\n" + "=" * 50)
        print("拓扑测试完成")
        print("=" * 50)


    def get_path_to_root(self, bus):
        """获取某个节点到根节点的路径"""
        return self.path_to_root.get(bus, [])
    def get_all_paths_to_root(self):
        """获取所有节点到根节点的路径"""
        paths = {}
        for bus in self.buses:
            paths[bus] = self.get_path_to_root(bus)
        return paths

    def test_get_all_impedances(self):
        # 获取所有阻抗
        all_impedances = self.get_all_impedances()

        # 输出线路阻抗
        print("\n=== 线路阻抗 ===")
        for name, r_matrix, x_matrix in all_impedances["lines"]:
            print(f"{name}:")
            print(f"  R矩阵: {r_matrix}")
            print(f"  X矩阵: {x_matrix}")

        # 输出变压器阻抗
        print("\n=== 变压器阻抗 ===")
        for name, impedance in all_impedances["transformers"]:
            print(f"{name}: {impedance:.4f} Ω")

        return all_impedances


if __name__ == "__main__":
    TestSystem = IEEE13System()
    TestSystem.test_downstream_nodes()

    # 获取所有节点的路径
    all_paths = TestSystem.get_all_paths_to_root()
    for bus, path in all_paths.items():
        print(f"\n节点 {bus} 的路径:")
        for segment in path:
            print(f"  {segment[0]} → {segment[1]} ({segment[2]})")


    TestSystem.test_downstream_nodes()

    impedances = TestSystem.test_get_all_impedances()

    print(impedances)
