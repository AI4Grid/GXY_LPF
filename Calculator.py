import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from System_IEEE13bus import IEEE13System
from collections import deque
import pprint
class LPFCalculator:
    def __init__(self, system):
        """
        初始化线性化潮流计算器

        参数:
        system: IEEE13System实例
        """
        self.system = system
        self.alpha = np.exp(-1j * 2 * np.pi / 3)  # α = e^{-j2π/3}
        self.beta = np.diag([self.alpha, 1, self.alpha ** 2])  # β = diag(α, 1, α²)
        self.gamma = np.array([
            [1, self.alpha ** 2, self.alpha],
            [self.alpha, 1, self.alpha ** 2],
            [self.alpha ** 2, self.alpha, 1]
        ])

        # 初始化节点相位配置
        self.bus_phases = {}      # 节点相位数
        self.bus_phase_letters = {}  # 节点相位字母标识
        self.voltage_bases = {}   # 节点电压等级

        # 存储计算结果
        self.lambda_ij = {}  # 线路功率流Λ_ij
        self.S_ij = {}  # 复功率流矩阵S_ij
        self.v_j = {}  # 节点电压v_j
        self.transformer_info = {}  # 确保这里初始化了transformer_info

        # 设置电压等级基准值
        # self.voltage_bases = [115.0, 4.16, 0.48]  # kV
        self.base_power = 1000.0  # kVA (基准功率)
        # 电压等级基准值 (kV)
        self.voltage_bases = {
            # 源系统 (115kV)
            'SourceBus': 115.0,

            # 4.16kV系统 (Sub变压器次级侧)
            '650': 4.16,  # Sub变压器次级侧
            '632': 4.16,  # 主母线
            '645': 4.16,  # 645节点
            '646': 4.16,  # 646节点
            '670': 4.16,  # 670节点
            '671': 4.16,  # 671节点
            '680': 4.16,  # 680节点
            '684': 4.16,  # 684节点
            '611': 4.16,  # 单相负载节点 (连接到684的C相)
            '692': 4.16,  # 692节点
            '652': 4.16,  # 652节点
            '675': 4.16,  # 675节点
            'RG60': 4.16,  # 特殊设备节点

            # 0.48kV系统 (XFM1变压器次级侧)
            '633': 4.16,  # XFM1变压器高压侧 (4.16kV)
            '634': 0.48,  # XFM1变压器低压侧 (0.48kV)

        }
        # 添加源节点电压属性
        self.source_voltage_vector = None
        self.reference_voltage = None
        self.source_angle = 0  # 根据配置设置30度相移
        self.line_info = {}
        self.parent_map = {}
        self.v_matrices = {}
        self.Wij_dict = {}
        self.children_map = {}
        # 电容补偿数据
        self.capacitor_data = {
            '675': {'a': 200, 'b': 200, 'c': 200},  # kVAR
            '611': {'c': 100}  # kVAR
        }

        # 计算基准阻抗
        self.base_impedances = {
            voltage: (voltage ** 2) * 1000 / self.base_power  # (kV^2 * 1000)/kVA = Ω
            for voltage in set(self.voltage_bases.values())
        }#{0.48: 0.2304, 4.16: 17.305600000000002, 115.0: 13225.0}

        # 打印基准值
        print(f"\n基准设置:")
        print(f"基准功率: {self.base_power} kVA")
        for level, impedance in self.base_impedances.items():
            print(f"电压等级 {level}kV: 基准阻抗 = {impedance:.4f} Ω")

    def to_per_unit(self, value, bus=None, voltage_level=None):
        """
        将实际值转换为标幺值

        参数:
        value: 实际值 (阻抗为Ω, 功率为kW, 电压为kV)
        bus: 节点名称 (用于确定电压等级)
        voltage_level: 直接指定电压等级 (kV)

        返回:
            标幺值
        """
        # 确定电压等级
        voltage_level = self.voltage_bases.get(bus, None)

        # 对于功率: S_pu = S_actual / S_base
        if isinstance(value, (int, float, complex)):
            # 功率的标幺值转换只除以基准功率
            return value / self.base_power

        # 对于阻抗: Z_pu = Z_actual / Z_base
        # 计算基准阻抗
        Z_base = ((voltage_level * 1000) ** 2) / (self.base_power * 1000)

        if isinstance(value, complex):
            # 复数阻抗
            return complex(
                value.real / Z_base,
                value.imag / Z_base
            )
        else:
            # 其他类型直接返回
            return value / Z_base

    def convert_impedance_to_ohms(self, r_list, x_list, length, unit):
        """
        将每英里阻抗转换为实际阻抗值(Ω)
        :param r_list: 电阻列表 (ohms/mile)
        :param x_list: 电抗列表 (ohms/mile)
        :param length: 线路长度
        :param unit: 长度单位 ('ft' 或 'mi')
        :return: 实际阻抗值 (Ω)
        """
        if unit == 'ft':
            # 英尺转换为英里: 1英里 = 5280英尺
            length_miles = length / 5280.0
        elif unit == 'mi':
            length_miles = length
        else:
            print(f"警告: 未知单位 '{unit}'，假设为英里")
            length_miles = length

        # 计算实际阻抗值
        r_actual = [r * length_miles for r in r_list]
        x_actual = [x * length_miles for x in x_list]

        # 创建复数阻抗列表
        impedance_actual = [complex(r, x) for r, x in zip(r_actual, x_actual)]

        return impedance_actual

    def _get_phase_matrix(self, phases):
        """
        获取相位选择矩阵 (Φ)

        参数:
        phases: 相位列表，如['a','b','c']或['a','c']

        返回:
        Φ: 相位选择矩阵 (3xN, N为相位数)
        """
        phase_mapping = {'a': 0, 'b': 1, 'c': 2}
        indices = [phase_mapping[p] for p in sorted(phases)]
        N = len(indices)
        phi = np.zeros((3, N))
        for i, idx in enumerate(indices):
            phi[idx, i] = 1
        return phi

    def calculate_lambda(self):
        """
        计算线路功率流Λ_ij (公式12a)
        """
        # 相位映射字典
        phase_mapping = {'1': 'a', '2': 'b', '3': 'c'}

        # 初始化Λ_ij字典
        self.lambda_ij = {}

        # 添加调试输出
        print("\n" + "=" * 50)
        print("开始计算Λ_ij")
        print("=" * 50)

        # 遍历所有线路
        for (start, end) in self.system.get_line_edges():
            # 查找对应的线路对象
            line = next((l for l in self.system.lines if l[2] == start and l[4] == end), None)
            if not line:
                continue

            # 获取线路相位配置
            phases = line[3]  # start_phases 列表
            phase_letters = sorted([phase_mapping[p] for p in phases])
            n_phases = len(phases)

            # 创建相位选择矩阵 (n_system_phases × n_line_phases)
            phi_ij = self._get_phase_matrix(phase_letters)

            # 获取下游节点集合
            downstream = self.system.get_downstream_buses(end)

            # 初始化Λ_ij为0 (与线路相位数相同)
            lambda_ij = np.zeros(n_phases, dtype=complex)

            # # 添加线路调试输出
            print(f"\n线路 {start}→{end} ({n_phases}相): {phase_letters}")
            print(f"下游节点: {sorted(downstream)}")
            line_total_load = 0 + 0j  # 跟踪线路总负荷

            # 累加下游所有负荷
            for bus in downstream:
                # 获取节点所有负荷
                bus_loads = [load for load in self.system.loads if load[0] == bus]

                # 获取该节点的电容补偿数据（如果有）
                cap_data = self.capacitor_data.get(bus, {})
                print(f"处理节点 {bus} 的负荷和电容补偿: {cap_data}")

                for load_idx, load in enumerate(bus_loads):
                    _, phase_code, _, _, _, _, P, Q = load

                    # 检查是否有该相位的电容补偿
                    cap_q = cap_data.get(phase_mapping.get(phase_code, ''), 0)
                    if cap_q:
                        print(f"  应用电容补偿: {phase_code}相补偿 {cap_q} kVAR")
                        Q -= cap_q  # 直接减去电容补偿的无功

                    # 计算复功率 (kW + jkvar)
                    S = complex(P, Q)# 转换为W
                    line_total_load += S

                    # 获取负荷相位索引
                    load_phase_indices = self.system.get_phase_index(phase_code)

                    # 添加负荷调试输出
                    print(
                        f"  - 负荷 {load_idx + 1}: {P}kW + j{Q}kvar (相位 '{phase_code}') -> S = {S.real:.1f}kW + j{S.imag:.1f}kvar")

                    # 如果没有相位索引，跳过该负荷
                    if not load_phase_indices:
                        print(f"    警告: 无法识别的相位代码 '{phase_code}'，跳过")
                        continue

                    # 创建负荷相位向量 (3×1)
                    S_vec = np.zeros(3, dtype=complex)
                    for idx in load_phase_indices:
                        S_vec[idx] = S / len(load_phase_indices)  # 将负荷平均分配到各相

                    # 投影到线路相位空间
                    projected_S = phi_ij.T @ S_vec  # (n_line_phases × 1)
                    lambda_ij += projected_S

                    # 添加详细输出
                    print(
                        f"    投影后: {[x for x in projected_S.real]}kW + j{[x for x in projected_S.imag]}kvar")

            # 线路循环结束输出
            print(
                f"线路 {start}→{end} 下游总负荷: {line_total_load.real:.1f}kW + j{line_total_load.imag:.1f}kvar")
            print(
                f"线路 {start}→{end} 的Λ_ij = {[x for x in lambda_ij.real]}kW + j{[x for x in lambda_ij.imag]}kvar")
            print("-" * 50)
            self.lambda_ij[(start, end)] = lambda_ij

        # 计算系统总负荷（直接求和所有负荷）
        total_system_load = 0 + 0j
        for load in self.system.loads:
            _, _, _, _, _, _, P, Q = load
            total_system_load += complex(P, Q)
        print(f"系统总负荷: {total_system_load.real}kW + j{total_system_load.imag}kvar")
        print("-" * 50)
        return self.lambda_ij

    def calculate_complex_power_flow(self):
        """
        计算复功率流矩阵S_ij (公式12b)
        S_ij = γ^{Φ_ij} diag(Λ_ij)
        输出单位：kW + j kvar
        """
        # 确保γ矩阵已定义
        if not hasattr(self, 'gamma'):
            alpha = np.exp(-1j * 2 * np.pi / 3)
            self.gamma = np.array([
                [1, alpha ** 2, alpha],
                [alpha, 1, alpha ** 2],
                [alpha ** 2, alpha, 1]
            ], dtype=complex)

        self.S_ij = {}

        # 遍历所有线路的Λ_ij
        for line_key, lambda_ij in self.lambda_ij.items():
            # 查找对应的线路数据
            line_data = None
            for line in self.system.lines:
                # 检查两种可能的键格式：线路名称或(起始节点, 终止节点)
                if line[0] == line_key or (line[2], line[4]) == line_key:
                    line_data = line
                    break

            if not line_data:
                print(f"警告: 未找到线路 {line_key} 的数据，跳过")
                continue

            # 提取线路信息
            line_name = line_data[0]
            phases = line_data[1]  # 相数
            phase_list = line_data[3]  # 起始相别

            # 数字到字母的映射
            num_to_letter = {'1': 'a', '2': 'b', '3': 'c'}
            phase_letters = [num_to_letter[p] for p in phase_list]

            # 获取起始节点和终止节点
            from_bus = line_data[2]
            to_bus = line_data[4]

            print(f"\n线路 {line_name} ({from_bus}→{to_bus}) {phases}相 相位配置: {phase_letters}")
            print(f"  Λ_ij (kW + j kvar): {[f'{x.real:.2f}{x.imag:+.2f}j' for x in lambda_ij]}")

            # 根据相数创建γ^{Φ_ij}矩阵
            if phases == 1:
                # 单相线路
                gamma_phi = np.array([[1.0 + 0j]])
                print("  使用单位矩阵 γ^{Φ_ij}: [[1.0]]")
            elif phases == 2:
                # 两相线路 - 根据实际相位选择子矩阵
                phase_indices = []
                for phase in phase_letters:
                    if phase == 'a':
                        phase_indices.append(0)
                    elif phase == 'b':
                        phase_indices.append(1)
                    elif phase == 'c':
                        phase_indices.append(2)

                # 选择对应的2x2子矩阵
                gamma_phi = self.gamma[np.ix_(phase_indices, phase_indices)]
                print("  使用γ子矩阵 γ^{Φ_ij}:")
                self._print_matrix(gamma_phi)
            else:
                # 三相线路 - 使用完整γ矩阵
                gamma_phi = self.gamma
                print("  使用完整γ矩阵 γ^{Φ_ij}:")
                self._print_matrix(gamma_phi)


            # 转换为标幺值
            lambda_ij_pu = [self.to_per_unit(val, from_bus) for val in lambda_ij]

            # 创建对角矩阵 diag(Λ_ij)
            diag_lambda = np.zeros((phases, phases), dtype=complex)
            for i in range(phases):
                diag_lambda[i, i] = lambda_ij_pu[i]
            print("  diag(Λ_ij):")
            self._print_matrix(diag_lambda)

            # 计算S_ij = γ^{Φ_ij} diag(Λ_ij)
            S_ij = gamma_phi @ diag_lambda

            # 存储结果（标幺值）
            self.S_ij[line_name] = S_ij

            # 格式化输出S_ij矩阵
            print(f"  S_ij (标幺值):")
            for i in range(S_ij.shape[0]):
                row = []
                for j in range(S_ij.shape[1]):
                    real = S_ij[i, j].real
                    imag = S_ij[i, j].imag
                    # 处理正负号显示
                    if imag >= 0:
                        row.append(f"{real:.2f} + j{imag:.2f}")
                    else:
                        row.append(f"{real:.2f} - j{-imag:.2f}")
                print("    " + "  ".join(row))

        return self.S_ij

    def calculate_voltages(self):
        """
        计算所有节点的电压矩阵 (标幺值系统)
        """
        # 1. 设置源节点电压参数
        base_kv = 115.0  # 基准线电压 (kV)
        pu_value = 1.0001  # 标幺值
        source_angle = 0  # 度

        # 2. 计算基准值 (实际值)
        V_base_line = base_kv * 1000  # V (线电压)
        V_base_phase = V_base_line / np.sqrt(3)  # V (相电压)

        # 3. 创建实际电压向量
        angle_rad = np.radians(source_angle)
        V_a = V_base_phase * pu_value * np.exp(1j * angle_rad)
        V_b = V_base_phase * pu_value * np.exp(1j * (angle_rad - 2 * np.pi / 3))
        V_c = V_base_phase * pu_value * np.exp(1j * (angle_rad - 4 * np.pi / 3))

        # 确保是二维数组
        V0_actual = np.array([[V_a], [V_b], [V_c]])  # 形状为 (3,1)

        # 4. 计算标幺值电压向量
        V0_pu = V0_actual / V_base_phase  # 现在形状是 (3,1)

        # 5. 计算标幺值电压矩阵 - 正确维度
        self.v_j['SourceBus'] = V0_pu @ V0_pu.conj().T

        print("\n源节点电压矩阵 (标幺值):")
        self._print_matrix(self.v_j['SourceBus'])

        # print("\n=== 处理变压器 ===")
        # self._process_transformers()
        # print(f"变压器处理完成。变压器数量: {len(self.transformer_info)}")

        # 3. 构建网络拓扑
        print("\n=== 构建网络拓扑 ===")
        # 创建变压器映射表
        transformer_map = {}

        print("\n=== 处理线路 ===")
        print(f"系统中有 {len(self.system.lines)} 条线路")
        # 在电压计算循环之前
        for idx, line in enumerate(self.system.lines):
            line_name = line[0]
            phases = line[1]
            from_bus = line[2]
            to_bus = line[4]
            r_list = line[8]  # 电阻列表 (ohms/mile)
            x_list = line[9]  # 电抗列表 (ohms/mile)
            phase_list = line[3]
            length = line[6]  # 线路长度
            unit = line[7]  # 长度单位

            # 打印线路基本信息
            print(f"\n处理线路 #{idx + 1}: {line_name}")
            print(f"  起始总线: {from_bus}, 终止总线: {to_bus}")
            print(f"  相位数: {phases}")
            print(f"  相位列表: {phase_list}")
            print(f"  长度: {length}{unit}")
            print(f"  原始电阻列表: {r_list}")
            print(f"  原始电抗列表: {x_list}")

            # # # 确定电压等级
            voltage_level = self.voltage_bases[to_bus]
            # 转换阻抗为实际值(Ω)
            impedance_actual = self.convert_impedance_to_ohms(r_list, x_list, length, unit)

            # 打印转换后的实际阻抗
            print(f"  实际阻抗 (Ω): {[f'{z.real:.6f}{z.imag:+.6f}j' for z in impedance_actual]}")

            # 构建阻抗矩阵（实际值）
            Z_matrix_actual = np.zeros((phases, phases), dtype=complex)

            # 填充阻抗矩阵
            if phases == 3:
                # 三相线路阻抗矩阵
                Z_matrix_actual[0, 0] = impedance_actual[0]
                Z_matrix_actual[0, 1] = impedance_actual[1]
                Z_matrix_actual[0, 2] = impedance_actual[2]
                Z_matrix_actual[1, 0] = impedance_actual[1]
                Z_matrix_actual[1, 1] = impedance_actual[3]
                Z_matrix_actual[1, 2] = impedance_actual[4]
                Z_matrix_actual[2, 0] = impedance_actual[2]
                Z_matrix_actual[2, 1] = impedance_actual[4]
                Z_matrix_actual[2, 2] = impedance_actual[5]
            elif phases == 2:
                # 两相线路阻抗矩阵
                Z_matrix_actual[0, 0] = impedance_actual[0]
                Z_matrix_actual[0, 1] = impedance_actual[1]
                Z_matrix_actual[1, 0] = impedance_actual[1]
                Z_matrix_actual[1, 1] = impedance_actual[2]
            else:  # 单相
                Z_matrix_actual[0, 0] = impedance_actual[0]

            # 转换为标幺值
            Z_matrix_pu = self.to_per_unit(Z_matrix_actual, bus=from_bus, voltage_level=voltage_level)

            # 数字到字母的映射
            num_to_letter = {'1': 'a', '2': 'b', '3': 'c'}
            phase_letters = [num_to_letter[p] for p in phase_list]

            # 记录线路信息
            line_key = (from_bus, to_bus)
            # 检查是否为变压器线路
            is_transformer = (from_bus, to_bus) in transformer_map

            # 存储线路信息，添加变压器标志
            self.line_info[line_key] = (line_name, Z_matrix_pu, phases, phase_letters, voltage_level, is_transformer)

            self.parent_map[to_bus] = from_bus

            # 打印标幺值阻抗矩阵
            print(f"  标幺值阻抗矩阵:")
            self._print_matrix(Z_matrix_pu)

        # 4. 为每个节点计算路径
        print("\n=== 计算节点路径 ===")
        node_paths = {}
        print(f"系统中有 {len(self.system.buses)} 个总线节点")

        for bus in self.system.buses:
            print(f"\n计算节点 '{bus}' 的路径:")

            if bus == 'SourceBus':
                node_paths[bus] = []
                print("  源节点，路径为空列表")
                continue

            path = []
            current = bus
            print(f"  当前节点: {current}")

            while current != 'SourceBus':
                print(f"    处理 {current} 的父节点...")
                parent = self.parent_map.get(current)

                if parent is None:
                    print(f"    警告: 节点 {current} 没有找到父节点!")
                    break

                print(f"    父节点: {parent}")
                path.append((parent, current))
                print(f"    添加路径段: {parent} → {current}")
                current = parent
                print(f"    移动到父节点: {current}")

            # 反转路径列表使其从源节点开始
            reversed_path = list(reversed(path))
            node_paths[bus] = reversed_path
            print(f"  最终路径 (源节点开始): {reversed_path}")

        # 打印所有节点路径
        print("\n=== 所有节点路径汇总 ===")
        for bus, path in node_paths.items():
            if path:
                path_str = " → ".join([f"{p[0]}-{p[1]}" for p in path])
                print(f"{bus}: {path_str}")
            else:
                print(f"{bus}: 源节点 (无路径)")
##########################################################################################################
        print("\n=== 相位推断开始 ===")
        inferred_phases = {}
        inferred_phase_letters = {}

        # 步骤2: 从线路信息推断
        print("\n步骤2: 从线路信息推断")
        print("遍历所有线路信息...")
        for line_key in self.line_info:
            # line_name, z_matrix, phases, phase_letters, voltage_level = self.line_info[line_key]

            line_name, z_matrix, phases, phase_letters, voltage_level, is_transformer = self.line_info[line_key]

            from_bus, to_bus = line_key
            print(f"  线路: {line_name} ({from_bus}->{to_bus}) - {phases}相 ({', '.join(phase_letters)})")

            # 上游节点
            if from_bus not in inferred_phases:
                inferred_phases[from_bus] = phases
                inferred_phase_letters[from_bus] = phase_letters
                print(f"    → 设置上游节点 {from_bus}: {phases}相 ({', '.join(phase_letters)})")
            else:
                print(f"    → 上游节点 {from_bus} 已设置为 {inferred_phases[from_bus]}相")

            # 下游节点
            if to_bus not in inferred_phases:
                inferred_phases[to_bus] = phases
                inferred_phase_letters[to_bus] = phase_letters
                print(f"    → 设置下游节点 {to_bus}: {phases}相 ({', '.join(phase_letters)})")
            else:
                print(f"    → 下游节点 {to_bus} 已设置为 {inferred_phases[to_bus]}相")

        # 步骤3: 精简相位推断 - 按节点分组处理
        print("\n步骤3: 从负荷信息推断")
        print("遍历所有负荷信息...")
        # 创建一个字典来收集每个节点的相位信息
        node_phases = {}

        for load in self.system.loads:
            bus = load[0]  # 负荷所在节点
            phase_code = str(load[1])  # 相位代码
            phases = load[2]  # 相位数
            print(f"  负荷在节点 {bus}: 相位代码='{phase_code}', 相位数={phases}")

            # 初始化节点的相位集合
            if bus not in node_phases:
                node_phases[bus] = set()
                print(f"    → 初始化节点 {bus} 的相位集合")

            # 解析相位代码，添加到相位集合
            if phase_code in ['abc', '3']:
                node_phases[bus].update(['a', 'b', 'c'])
                print(f"    → 添加三相(a,b,c)到节点 {bus}")
            elif phase_code in ['a', 'b', 'c']:
                node_phases[bus].add(phase_code)
                print(f"    → 添加单相({phase_code})到节点 {bus}")
            elif phase_code in ['ab', 'bc', 'ac']:
                node_phases[bus].update(list(phase_code))
                print(f"    → 添加两相({phase_code})到节点 {bus}")
            else:
                # 无法解析，使用相位数
                if phases == 1:
                    node_phases[bus].add('a')
                    print(f"    → 无法解析代码，添加单相(a)到节点 {bus}")
                elif phases == 2:
                    node_phases[bus].update(['b', 'c'])
                    print(f"    → 无法解析代码，添加两相(b,c)到节点 {bus}")
                else:
                    node_phases[bus].update(['a', 'b', 'c'])
                    print(f"    → 无法解析代码，添加三相(a,b,c)到节点 {bus}")

        # 根据收集到的相位信息确定节点配置
        print("\n合并负荷相位信息...")
        for bus, phase_set in node_phases.items():
            phases = len(phase_set)
            phase_letters = sorted(phase_set) if phase_set else ['a']
            print(f"  节点 {bus}: 收集的相位 = {phase_set} => {phases}相 ({', '.join(phase_letters)})")

            # 如果节点有多个单相负荷，合并为多相配置
            if phases == 0:
                phases = 1
                phase_letters = ['a']
                print(f"    → 无相位信息，使用默认单相(a)")

            # 存储推断结果
            inferred_phases[bus] = phases
            inferred_phase_letters[bus] = phase_letters
            print(f"    → 存储推断: {phases}相 ({', '.join(phase_letters)})")

        # 步骤4: 特殊节点处理
        print("\n步骤4: 特殊节点处理")
        inferred_phases['SourceBus'] = 3
        inferred_phase_letters['SourceBus'] = ['a', 'b', 'c']
        print("  源节点 SourceBus 设置为3相 (a,b,c)")

        # 步骤5: 确定每个节点的相位
        print("\n步骤5: 确定最终节点相位")
        print("遍历所有节点...")
        for bus in self.system.buses:
            if bus == 'SourceBus':
                self.bus_phases[bus] = 3
                self.bus_phase_letters[bus] = ['a', 'b', 'c']
                print(f"  {bus}: 源节点 -> 固定为3相 (a,b,c)")
                continue

            # 检查节点是否有路径定义
            if bus in node_paths and node_paths[bus]:
                last_segment = node_paths[bus][-1]
                print(f"  {bus}: 路径存在，最后一段: {last_segment}")

                # 检查线路信息是否存在
                if last_segment in self.line_info:
                    _, _, phases, phase_letters, _ ,_= self.line_info[last_segment]
                    self.bus_phases[bus] = phases
                    self.bus_phase_letters[bus] = phase_letters
                    print(f"    → 使用线路配置: {phases}相 ({', '.join(phase_letters)})")
                else:
                    # 如果线路信息缺失，使用推断配置
                    if bus in inferred_phases:
                        self.bus_phases[bus] = inferred_phases[bus]
                        self.bus_phase_letters[bus] = inferred_phase_letters[bus]
                        print(
                            f"    → 警告: 线路信息缺失，使用推断配置: {inferred_phases[bus]}相 ({', '.join(inferred_phase_letters[bus])})")
                    else:
                        # 没有推断信息，使用默认值
                        self.bus_phases[bus] = 1
                        self.bus_phase_letters[bus] = ['a']
                        print(f"    → 警告: 线路信息缺失且无推断配置，使用默认单相(a)")
            else:
                # 如果没有路径信息，使用推断配置
                print(f"  {bus}: 无路径信息")
                if bus in inferred_phases:
                    self.bus_phases[bus] = inferred_phases[bus]
                    self.bus_phase_letters[bus] = inferred_phase_letters[bus]
                    print(f"    → 使用推断配置: {inferred_phases[bus]}相 ({', '.join(inferred_phase_letters[bus])})")
                else:
                    # 没有推断信息，使用默认值
                    self.bus_phases[bus] = 1
                    self.bus_phase_letters[bus] = ['a']
                    print(f"    → 使用默认单相(a)")

        # 打印相位配置信息
        print("\n最终节点相位配置:")
        for bus, phases in self.bus_phases.items():
            letters = self.bus_phase_letters.get(bus, ['?'])
            print(f"  {bus}: {phases}相 ({', '.join(letters)})")
######################################################################################################################
        source_angle_shift = 0  #角度偏移
        source_pu = 1.0001  # 从配置文件读取的标幺值基准
        angle_shift_rad = np.radians(source_angle_shift)

        # 计算三相电压向量 (考虑角度偏移)
        v_a = source_pu * np.exp(1j * angle_shift_rad)
        v_b = source_pu * np.exp(1j * (angle_shift_rad - np.radians(120)))
        v_c = source_pu * np.exp(1j * (angle_shift_rad + np.radians(120)))

        # 形成源节点电压矩阵 v_j = V * V^H
        V_source = np.array([v_a, v_b, v_c])
        self.v_j['SourceBus'] = np.outer(V_source, V_source.conj())

        print(f"源节点电压矩阵 (考虑角度偏移):")
        self._print_matrix(self.v_j['SourceBus'], precision=6)
        print(f"  a相: {np.abs(v_a):.6f} ∠ {np.degrees(np.angle(v_a)):.2f}°")
        print(f"  b相: {np.abs(v_b):.6f} ∠ {np.degrees(np.angle(v_b)):.2f}°")
        print(f"  c相: {np.abs(v_c):.6f} ∠ {np.degrees(np.angle(v_c)):.2f}°")

        for bus in self.system.buses:
            if bus == 'SourceBus':
                continue

            # 获取节点信息
            phases = self.bus_phases[bus]
            phase_letters = self.bus_phase_letters[bus]
            path = node_paths[bus]
            voltage_level = self.voltage_bases[bus]

            # # 打印节点信息标题
            # print(f"节点 {bus} ,{phases}相 ({', '.join(phase_letters)})")
            # print(f"\n{'=' * 80}")
            # print(f"计算节点 {bus} 电压")
            # print(f"{'-' * 80}")
            # print(f"• 电压等级: {voltage_level}kV")
            # print(f"• 相位配置: {phases}相 ({', '.join(phase_letters)})")
            # print(f"• 路径: {' → '.join([f'{from_bus}-{to_bus}' for from_bus, to_bus in path])}")
            # print(f"• 路径长度: {len(path)}个线段")
            # print(f"{'-' * 80}")

            # 初始化压降矩阵 (全零)
            v_drop = np.zeros((phases, phases), dtype=complex)
            # print(f"初始压降矩阵 (全零):")
            # self._print_matrix(v_drop, precision=6)

            # 沿路径累加电压压降
            # print(f"\n开始沿路径计算压降...")
            for seg_idx, (from_bus, to_bus) in enumerate(path):
                line_key = (from_bus, to_bus)
                if line_key in self.line_info:
                    line_name, z_matrix_pu, line_phases, line_phase_letters, line_voltage_level, is_transformer = self.line_info[
                        line_key]
                    if is_transformer:
                        # print(f"\n线段 {seg_idx + 1}/{len(path)}: {line_name} ({from_bus}→{to_bus})")
                        # print(f"  └── 变压器线路，压降设置为零")
                        continue

                    # 获取该线路的S_ij
                    if line_name in self.S_ij:
                        S_ij = self.S_ij[line_name]
                        z_H = z_matrix_pu.conj().T

                        # 计算压降项: S_ij z_ij^H + z_ij S_ij^H
                        drop_matrix = S_ij @ z_H + z_matrix_pu @ S_ij.conj().T

                        # # 打印线段信息
                        # print(f"\n线段 {seg_idx + 1}/{len(path)}: {line_name} ({from_bus}→{to_bus})")
                        # print(f"  └── 电压等级: {line_voltage_level}kV")
                        # print(f"  └── 线路阻抗矩阵 (标幺值):")
                        # self._print_matrix(z_matrix_pu, precision=6)
                        # print(f"  └── S_ij 矩阵 (标幺值):")
                        # self._print_matrix(S_ij, precision=6)
                        # print(f"  └── 压降矩阵 (标幺值):")
                        # self._print_matrix(drop_matrix, precision=6)

                        # 尝试投影
                        try:
                            # print(f"  └── 投影到目标相位: {', '.join(phase_letters)}")
                            projected_drop = self.project_matrix(
                                drop_matrix,
                                line_phase_letters,
                                phase_letters
                            )

                            # print(f"  └── 投影后压降矩阵 (标幺值):")
                            # self._print_matrix(projected_drop, precision=6)

                            # 累加投影后的压降矩阵
                            v_drop += projected_drop

                            # # 打印累加结果
                            # print(f"  └── 当前累加压降矩阵 (标幺值):")
                            # self._print_matrix(v_drop, precision=6)
                            # print(f"  └── 最大压降值: {np.max(np.abs(v_drop)):.6f} 标幺值")

                        except Exception as e:
                            print(f"  └── 投影错误: {e}")
                            print(f"        └── 源相位: {line_phase_letters}")
                            print(f"        └── 目标相位: {phase_letters}")
                    else:
                        print(f"  └── 警告: 线路 {line_name} 的功率矩阵 S_ij 缺失!")
                else:
                    print(f"  └── 警告: 线路 {from_bus}→{to_bus} 的信息缺失!")
                    pass

            # # 源节点电压投影
            # print(f"\n准备投影源节点电压到目标相位...")
            try:
                # print(f"源节点电压矩阵 (标幺值):")
                # self._print_matrix(self.v_j['SourceBus'], precision=6)

                v0_projected = self.project_matrix(
                    self.v_j['SourceBus'],
                    ['a', 'b', 'c'],  # 源节点总是三相
                    phase_letters
                )

                # print(f"投影到目标相位后的源节点电压矩阵 (标幺值):")
                # self._print_matrix(v0_projected, precision=6)

                # 计算最终电压: v_j = v0_projected - v_drop
                v_j = v0_projected - v_drop
                #
                # print(f"节点 {bus} 最终电压矩阵v_j (标幺值):")
                # self._print_matrix(v_j, precision=6)

                # 存储结果
                self.v_j[bus] = v_j

                # 直接从对角线获取电压幅值（标幺值）
                diag_values = np.diag(v_j)

                # 确保对角线元素为实数（理论上应该是实数）
                diag_real = np.real(diag_values)

                # 计算电压幅值（标幺值）
                voltage_magnitudes_pu = np.sqrt(np.abs(diag_real))

                # 打印电压标幺值（kV）
                # print("\n电压标幺值 :")
                # for i, phase in enumerate(phase_letters):
                #     mag = voltage_magnitudes_pu[i]
                #     print(f"  {phase}相: {mag:.6f} ")
                # 创建相位值列表
                pu_values = []
                for i, phase in enumerate(phase_letters):
                    mag = voltage_magnitudes_pu[i]
                    pu_values.append(f"{mag:.6f}")

                print(f"节点 {bus} ,{phases}相 ({', '.join(phase_letters)}),电压标幺值: ({', '.join(pu_values)})")

                # 计算实际电压值（kV）
                actual_voltages = voltage_magnitudes_pu * voltage_level

                # 打印实际电压值（kV）
                # print("\n实际电压幅值 (kV):")
                # for i, phase in enumerate(phase_letters):
                #     mag = actual_voltages[i]
                #     print(f"  {phase}相: {mag:.6f} kV")

                # 创建相位值列表
                phase_values = []
                for i, phase in enumerate(phase_letters):
                    mag = actual_voltages[i]
                    phase_values.append(f"{mag:.6f}")

                # # 输出为(,,)格式
                # print(f"实际电压幅值 (kV): ({', '.join(phase_values)})")


            except Exception as e:
                print(f"投影源节点电压错误: {e}")

        print(f"{'=' * 80}")

        return self.v_j
    def _print_matrix(self, matrix, precision=4, is_complex=False):
        """
        打印矩阵，支持复数和实数矩阵
        """
        for i in range(matrix.shape[0]):
            row_str = ""
            for j in range(matrix.shape[1]):
                if is_complex:
                    # 处理复数
                    real = matrix[i, j].real
                    imag = matrix[i, j].imag
                    if imag >= 0:
                        row_str += f"{real:.{precision}f}+{imag:.{precision}f}j  "
                    else:
                        row_str += f"{real:.{precision}f}{imag:.{precision}f}j  "
                else:
                    # 处理实数
                    row_str += f"{matrix[i, j]:.{precision}f}  "
            print(row_str)

    def _process_transformers(self):
        """
        处理变压器，包括阻抗转换和连接方式
        """
        print("\n处理变压器...")
        if not hasattr(self, 'transformer_info'):
            self.transformer_info = {}

        for trans in self.system.transformers:
            # 确保变压器元组有足够的元素
            if len(trans) < 12:
                print(f"警告: 变压器数据不完整，跳过: {trans}")
                continue

            # 解析变压器参数
            name = trans[0]
            hv_bus = trans[1]
            lv_bus = trans[2]
            hv_conn = trans[3]
            lv_conn = trans[4]
            hv_kV = trans[5]  # 高压侧额定电压 (kV)
            lv_kV = trans[6]  # 低压侧额定电压 (kV)
            kVA = trans[7]  # 额定容量 (kVA)
            R1 = trans[8]  # 高压侧电阻百分比
            X1 = trans[9]  # 高压侧电抗百分比
            R2 = trans[10]  # 低压侧电阻百分比
            X2 = trans[11]  # 低压侧电抗百分比

            print(f"  变压器 {name}: {hv_kV}kV → {lv_kV}kV ({kVA}kVA)")

            # 计算变比
            turn_ratio = hv_kV / lv_kV

            # 计算变压器阻抗 (标幺值)
            Z_base_hv = (hv_kV ** 2) * 1000 / kVA  # 高压侧基准阻抗
            Z_pu = complex(R1, X1) / 100  # 百分比转换为标幺值

            print(f"    高压侧基准阻抗: {Z_base_hv:.4f} Ω")
            print(f"    阻抗标幺值: {Z_pu.real:.4f} + j{Z_pu.imag:.4f}")

            # 根据连接方式创建转换矩阵
            if hv_conn == 'delta' and lv_conn == 'wye':
                # Δ-Y变压器
                conversion_matrix = np.array([
                    [1, -1, 0],
                    [0, 1, -1],
                    [-1, 0, 1]
                ]) / np.sqrt(3)
            elif hv_conn == 'wye' and lv_conn == 'wye':
                # Y-Y变压器
                conversion_matrix = np.eye(3)
            else:
                # 其他连接方式暂不支持
                conversion_matrix = np.eye(3)
                print(f"    警告: 不支持的连接方式 {hv_conn}-{lv_conn}，使用单位矩阵")

            # 记录变压器信息
            self.transformer_info[name] = {
                'hv_bus': hv_bus,
                'lv_bus': lv_bus,
                'hv_kV': hv_kV,  # 确保这里有正确的键名
                'lv_kV': lv_kV,  # 确保这里有正确的键名
                'turn_ratio': turn_ratio,
                'Z_pu': Z_pu,
                'conversion_matrix': conversion_matrix
            }

            print(f"  已存储变压器信息")

    def project_matrix(self, source_matrix, source_phases, target_phases):
        """

        将矩阵从源相位空间投影到目标相位空间

        参数:
        matrix: 源矩阵 (维度: len(src_phases) x len(src_phases))
        src_phases: 源相位列表 (如 ['a', 'b', 'c'])
        dst_phases: 目标相位列表 (如 ['a', 'c'])

        返回:
            投影后的矩阵 (维度: len(dst_phases) x len(dst_phases))
        """

        # 系统标准相位顺序
        system_phases = ['a', 'b', 'c']

        # 创建3x3的源矩阵（缺失相位设为零）
        full_source = np.zeros((3, 3), dtype=complex)
        source_indices = [system_phases.index(p) for p in source_phases]
        for i, si in enumerate(source_indices):
            for j, sj in enumerate(source_indices):
                full_source[si, sj] = source_matrix[i, j]

        # 提取目标相位子集
        target_indices = [system_phases.index(p) for p in target_phases]
        projected = full_source[np.ix_(target_indices, target_indices)]

        return projected

    def run(self):
        """执行LPF计算流程"""
        print("开始线性化潮流计算...")
        self.calculate_lambda()
        print("步骤1: 线路功率流Λ_ij计算完成")
        self.calculate_complex_power_flow()
        print("步骤2: 复功率流矩阵S_ij计算完成")
        self.calculate_voltages()
        print("步骤4: 节点电压计算完成")

        print("LPF计算完成！")
        # self._print_matrix(self.v_j)
        # pprint.pprint(self.v_j,
        #               indent=2,
        #               width=100,
        #               depth=3,
        #               compact=False)

if __name__ == "__main__":
    # 创建系统实例（需要根据您的实际实现）
    TestSystem = IEEE13System()  # 假设您有这个类

    # 创建LPFCalculator实例
    Test_Calculator = LPFCalculator(TestSystem)

    # 运行计算
    Test_Calculator.run()



