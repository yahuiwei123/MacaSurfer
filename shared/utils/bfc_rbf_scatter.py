import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import gaussian_filter, label
import argparse

# Load MRI Image
def load_mri_image(file_path):
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)
    return image, image_array

def image_preprocess(image, label):
    # mask image through label
    image = np.where(label > 0, image, 0)

    # 检查标签区域是否有有效体素
    label_mask = label > 0
    if np.sum(label_mask) == 0:
        raise ValueError("Label mask is empty, no valid voxels found.")

    # normalize image intensities
    min_scale = np.min(image[label_mask])
    max_scale = np.max(image[label_mask])
    intensity_range = max_scale - min_scale

    # 避免除零错误：如果强度范围太小，直接返回原图像归一化到[1e-3, 1]
    if intensity_range < 1e-6:
        print(f"Warning: Image intensity range in label area is too small ({intensity_range:.6f}), skipping normalization.")
        norm = np.clip(image, 1e-3, 1)
        return norm, 0.0, 1.0

    norm = (image - min_scale) / intensity_range

    # obtain mean value of csf
    norm = np.clip(norm, 1e-3, 1)

    return norm, min_scale, max_scale

class ScatterBiasFieldCorrectorFast:
    def __init__(self, image, labels, mask=None,
                 n_points=1024,  # 总控制点数量
                 max_iter=30, tol=1e-4,
                 use_soft_labels=True, soft_sigma=0.5,
                 rbf_sigma=0.15,  # RBF核宽度，调大增加平滑性
                 k_neighbors=8,  # 每个体素只考虑最近的K个点，速度和网格版一致
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        加速版散点偏置场矫正，计算量和网格版同量级
        利用RBF局部性，每个体素只计算最近K个点，速度提升64倍以上
        """
        self.device = device
        print(f"Using device: {self.device}")
        print(f"Fast mode enabled: each voxel uses top {k_neighbors} nearest control points")

        self.image_np = image
        self.labels_np = labels
        self.mask_np = mask if mask is not None else (labels > 0).astype(np.uint8)
        self.n_points = n_points
        self.max_iter = max_iter
        self.tol = tol
        self.use_soft_labels = use_soft_labels
        self.soft_sigma = soft_sigma
        self.rbf_sigma = rbf_sigma
        self.k_neighbors = min(k_neighbors, n_points)

        # 图像尺寸
        self.H, self.W, self.D = image.shape

        # 预处理标签
        self.classes = np.unique(labels[labels > 0])
        self.num_classes = len(self.classes)

        # 生成软标签
        if self.use_soft_labels:
            self.soft_labels_np = self._generate_soft_labels()
        else:
            self.soft_labels_np = np.zeros((self.num_classes, self.H, self.W, self.D))
            for i, c in enumerate(self.classes):
                self.soft_labels_np[i] = (labels == c).astype(float)

        # 初始化随机分散控制点
        self._init_adaptive_scatter_points()

        # 提取掩码区域的坐标
        mask_indices = np.where(self.mask_np > 0)
        self.image = torch.tensor(self.image_np[mask_indices], dtype=torch.float32, device=self.device)
        self.soft_labels = torch.tensor(self.soft_labels_np[:, mask_indices[0], mask_indices[1], mask_indices[2]],
                                       dtype=torch.float32, device=self.device)
        self.coords = torch.tensor(
            np.stack([
                mask_indices[0] / (self.H - 1),
                mask_indices[1] / (self.W - 1),
                mask_indices[2] / (self.D - 1)
            ], axis=1), dtype=torch.float32, device=self.device
        )

        # 批处理大小
        self.batch_size = 20000

    def _generate_soft_labels(self):
        """生成平滑的软标签"""
        soft_labels = np.zeros((self.num_classes, self.H, self.W, self.D))
        for i, c in enumerate(self.classes):
            class_mask = (self.labels_np == c).astype(float)
            soft_labels[i] = gaussian_filter(class_mask, sigma=self.soft_sigma)
        # 归一化
        total = np.sum(soft_labels, axis=0, keepdims=True) + 1e-10
        soft_labels /= total
        return soft_labels

    def _compute_variance_map(self):
        """计算初始类内方差图，用于自适应点采样"""
        class_means = []
        for i in range(self.num_classes):
            weights = self.soft_labels_np[i]
            mean = np.sum(self.image_np * weights) / (np.sum(weights) + 1e-10)
            class_means.append(mean)

        residual_map = np.zeros_like(self.image_np)
        for i in range(self.num_classes):
            mean = class_means[i]
            residual = (self.image_np - mean) ** 2 * self.soft_labels_np[i]
            residual_map += residual

        residual_map = gaussian_filter(residual_map, sigma=0.3)
        return residual_map

    def _init_adaptive_scatter_points(self):
        """自适应采样控制点：20%均匀随机撒点，80%放在方差最大的分散区域"""
        mask = self.mask_np > 0
        coords_flat = np.argwhere(mask)  # (N, 3) -> [z, y, x]
        n_total = self.n_points
        n_uniform = int(n_total * 0.2)  # 60%均匀点
        n_high_var = n_total - n_uniform  # 40%高方差区域点

        # 1. 采样80%均匀分布的点
        uniform_idx = np.random.choice(len(coords_flat), size=n_uniform, replace=False)
        uniform_coords = coords_flat[uniform_idx]

        # 2. 采样20%高方差区域的点，分散在不同区域
        high_var_coords = []
        if n_high_var > 0:
            # 计算类内方差图
            variance_map = self._compute_variance_map()

            # 非极大值抑制，找到多个不重叠的高方差区域中心
            from scipy.ndimage import maximum_filter, label

            # 3x3x3最大值滤波，找到局部极大值点
            max_filtered = maximum_filter(variance_map, size=5)
            local_maxima = (variance_map == max_filtered) & mask
            labeled_maxima, num_maxima = label(local_maxima)

            # 只保留前20个最大的局部极大值点所在区域
            max_values = []
            for region_id in range(1, num_maxima + 1):
                region_mask = labeled_maxima == region_id
                max_val = np.max(variance_map[region_mask])
                max_values.append((max_val, region_id))

            # 按方差从大到小排序，取前n_high_var*2个区域，保证分散
            max_values.sort(reverse=True, key=lambda x: x[0])
            top_regions = [rid for (val, rid) in max_values[:min(len(max_values), n_high_var*2)]]

            # 从每个高方差区域采样点
            points_per_region = max(1, n_high_var // len(top_regions))
            remaining_points = n_high_var

            for region_id in top_regions:
                if remaining_points <= 0:
                    break
                region_mask = labeled_maxima == region_id
                # 膨胀区域，获得更大的采样范围
                from scipy.ndimage import binary_dilation
                region_mask = binary_dilation(region_mask, iterations=2) & mask
                region_coords = np.argwhere(region_mask)

                if len(region_coords) > 0:
                    # 每个区域采样不超过points_per_region个点
                    sample_num = min(points_per_region, remaining_points, len(region_coords))
                    sample_idx = np.random.choice(len(region_coords), size=sample_num, replace=False)
                    high_var_coords.extend(region_coords[sample_idx])
                    remaining_points -= sample_num

            # 如果还有剩余点数，从方差最高的体素中采样
            if remaining_points > 0:
                # 获取所有脑内体素的方差值
                masked_variance = variance_map[mask]
                # 取前5%高方差的体素
                threshold = np.percentile(masked_variance, 95)
                high_var_mask = (variance_map > threshold) & mask
                high_var_all_coords = np.argwhere(high_var_mask)

                if len(high_var_all_coords) > 0:
                    sample_idx = np.random.choice(len(high_var_all_coords), size=remaining_points, replace=False)
                    high_var_coords.extend(high_var_all_coords[sample_idx])

        # 合并两部分点
        all_coords = np.concatenate([uniform_coords, np.array(high_var_coords)], axis=0)

        # 计算brainmask的边界范围，控制点不能超出这个范围
        mask_coords = np.argwhere(self.mask_np > 0)
        z_min, z_max = mask_coords[:,0].min(), mask_coords[:,0].max()
        y_min, y_max = mask_coords[:,1].min(), mask_coords[:,1].max()
        x_min, x_max = mask_coords[:,2].min(), mask_coords[:,2].max()

        # 保存归一化后的边界
        self.mask_bounds = torch.tensor([
            [z_min/(self.H-1), z_max/(self.H-1)],
            [y_min/(self.W-1), y_max/(self.W-1)],
            [x_min/(self.D-1), x_max/(self.D-1)]
        ], dtype=torch.float32, device=self.device)

        # 全局打乱
        np.random.shuffle(all_coords)

        # 转换到[0,1]归一化坐标
        self.init_points = np.stack([
            all_coords[:, 0] / (self.H - 1),
            all_coords[:, 1] / (self.W - 1),
            all_coords[:, 2] / (self.D - 1)
        ], axis=1)

        print(f"Adaptive scatter points initialized: {self.n_points} points")
        print(f"  Uniform points: {n_uniform}, High variance points: {len(high_var_coords)}")
        print(f"  Brain mask bounds (z,y,x):")
        print(f"    Z: [{self.mask_bounds[0,0]:.3f}, {self.mask_bounds[0,1]:.3f}]")
        print(f"    Y: [{self.mask_bounds[1,0]:.3f}, {self.mask_bounds[1,1]:.3f}]")
        print(f"    X: [{self.mask_bounds[2,0]:.3f}, {self.mask_bounds[2,1]:.3f}]")
        print(f"  Points range:")
        print(f"    Z: [{self.init_points[:,0].min():.3f}, {self.init_points[:,0].max():.3f}]")
        print(f"    Y: [{self.init_points[:,1].min():.3f}, {self.init_points[:,1].max():.3f}]")
        print(f"    X: [{self.init_points[:,2].min():.3f}, {self.init_points[:,2].max():.3f}]")

        # 转换为可优化参数
        self.points = torch.nn.Parameter(
            torch.tensor(self.init_points, dtype=torch.float32, device=self.device)
        )
        # 控制点偏置值初始化为1
        self.point_values = torch.nn.Parameter(
            torch.ones(self.n_points, dtype=torch.float32, device=self.device)
        )

    def _eval_bias_field_batch(self, coords_batch):
        """加速版RBF插值：每个体素只计算最近K个控制点，速度和网格版一致"""
        # 计算每个体素到所有控制点的距离 (B, P)
        dist = torch.sum((coords_batch.unsqueeze(1) - self.points.unsqueeze(0)) ** 2, dim=2)

        # 只保留最近的K个点，其他点距离太远权重可以忽略
        topk_dist, topk_idx = torch.topk(dist, k=self.k_neighbors, largest=False, dim=1)
        topk_values = self.point_values[topk_idx]  # (B, K)

        # 高斯RBF核，只计算最近K个点
        rbf_weights = torch.exp(-topk_dist / (2 * self.rbf_sigma ** 2))
        # 插值得到偏置值
        bias = torch.sum(rbf_weights * topk_values, dim=1) / (torch.sum(rbf_weights, dim=1) + 1e-10)
        return bias

    def _eval_bias_field(self):
        """批处理计算全图偏置场"""
        num_voxels = self.coords.shape[0]
        bias = torch.zeros(num_voxels, device=self.device)

        for i in range(0, num_voxels, self.batch_size):
            end_idx = min(i + self.batch_size, num_voxels)
            coords_batch = self.coords[i:end_idx]
            bias[i:end_idx] = self._eval_bias_field_batch(coords_batch)

        return bias

    def _compute_objective(self):
        """计算目标函数：类内方差 + 正则项"""
        bias = self._eval_bias_field()
        corrected = self.image / torch.clamp(bias, 1e-10, 1e10)

        total_variance = 0.0
        # 先计算所有类的体素级残差，用于生成自适应权重
        all_residuals = torch.zeros_like(self.image)
        class_weights = []

        # 第一步：计算初始残差图
        for i in range(self.num_classes):
            weights = self.soft_labels[i]
            weight_sum = torch.sum(weights)
            class_weights.append(weights)
            if weight_sum < 1e-3:
                continue
            mean = torch.sum(corrected * weights) / weight_sum
            residual = weights * (corrected - mean) ** 2
            all_residuals += residual

        # 生成自适应权重：残差越大，权重越高
        # 对残差开方避免权重差异过大导致优化不稳定
        adaptive_weight = torch.pow(all_residuals + 1e-10, 0.5)
        # 归一化权重，保证损失量级和原来一致
        adaptive_weight = adaptive_weight / torch.mean(adaptive_weight)
        # 限制最大权重为20倍，避免个别异常点主导损失
        adaptive_weight = torch.clamp(adaptive_weight, min=0.2, max=20.0)

        # 第二步：自适应加权计算总方差
        for i in range(self.num_classes):
            weights = class_weights[i]
            weight_sum = torch.sum(weights)
            if weight_sum < 1e-3:
                continue
            # 使用自适应加权计算均值和方差
            weighted_sum = torch.sum(corrected * weights * adaptive_weight)
            weighted_total = torch.sum(weights * adaptive_weight)
            if weighted_total < 1e-3:
                mean = torch.sum(corrected * weights) / weight_sum
            else:
                mean = weighted_sum / weighted_total
            variance = torch.sum(weights * adaptive_weight * (corrected - mean) ** 2) / weight_sum
            total_variance += variance * weight_sum

        # 正则1：偏置场能量正则，约束接近1
        reg_bias = 0.1 * torch.sum((bias - 1.0) ** 2)

        # 正则2：控制点间距正则，避免点过于集中
        # 计算点之间的最小距离
        pairwise_dist = torch.cdist(self.points, self.points) + torch.eye(self.n_points, device=self.device) * 1e3
        min_dist, _ = torch.min(pairwise_dist, dim=1)
        reg_spacing = 0.01 * torch.sum(1.0 / (min_dist ** 2 + 1e-3))

        # # 正则3：控制点值空间平滑正则（基于空间K近邻）
        # # 找到每个点的空间最近K个邻居，约束相邻点值接近
        # k_smooth = self.k_neighbors
        # pairwise_dist = torch.cdist(self.points, self.points) + torch.eye(self.n_points, device=self.device) * 1e3
        # _, neighbor_idx = torch.topk(pairwise_dist, k=k_smooth, largest=False, dim=1)
        # neighbor_values = self.point_values[neighbor_idx]  # (P, K)
        # reg_value_smooth = 0.1 * torch.mean((self.point_values.unsqueeze(1) - neighbor_values) ** 2)

        # 正则5：偏置场梯度平滑正则（新增）
        # 直接约束偏置场的梯度，保证整体平滑
        bias_grad = torch.abs(torch.gradient(bias.reshape(-1), spacing=1.0/self.H)[0])
        reg_bias_grad = 5e1 * torch.mean(bias_grad)

        # 正则4：边界惩罚，防止点跑出mask范围，比单纯clamp更有效
        # 计算每个点到边界的距离，越靠近边界惩罚越大
        z_low_penalty = torch.sum(torch.relu(self.mask_bounds[0, 0] - self.points[:, 0]) ** 2)
        z_high_penalty = torch.sum(torch.relu(self.points[:, 0] - self.mask_bounds[0, 1]) ** 2)
        y_low_penalty = torch.sum(torch.relu(self.mask_bounds[1, 0] - self.points[:, 1]) ** 2)
        y_high_penalty = torch.sum(torch.relu(self.points[:, 1] - self.mask_bounds[1, 1]) ** 2)
        x_low_penalty = torch.sum(torch.relu(self.mask_bounds[2, 0] - self.points[:, 2]) ** 2)
        x_high_penalty = torch.sum(torch.relu(self.points[:, 2] - self.mask_bounds[2, 1]) ** 2)
        reg_boundary = 10.0 * (z_low_penalty + z_high_penalty + y_low_penalty + y_high_penalty + x_low_penalty + x_high_penalty)

        return total_variance + reg_bias + reg_spacing + reg_boundary + reg_bias_grad

    def fit(self):
        """联合优化控制点位置和数值"""
        prev_obj = float('inf')

        # 动态加点相关初始化
        self.prev_residual_map = None
        self.initial_n_points = self.n_points
        self.max_add_points = int(self.initial_n_points * 0.5)  # 最多加20%的点
        self.added_points_count = 0

        # 同时优化所有参数
        self.points.requires_grad_(True)
        self.point_values.requires_grad_(True)

        optimizer = torch.optim.Adam([
            {'params': [self.point_values], 'lr': 0.01},  # 降低学习率，减少值的振荡
            {'params': [self.points], 'lr': 0.001}
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        for iter in range(self.max_iter):
            print(f"Iteration {iter+1}/{self.max_iter}")
            epoch_loss = 0.0

            for step in range(40):
                optimizer.zero_grad()
                loss = self._compute_objective()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.point_values, max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.points, max_norm=0.03)

                optimizer.step()

                # 约束点在brainmask范围内，不会跑到脑外
                with torch.no_grad():
                    # 分别对每个维度clamp到mask的边界范围
                    self.points.data[:, 0] = torch.clamp(self.points.data[:, 0],
                                                       self.mask_bounds[0, 0], self.mask_bounds[0, 1])
                    self.points.data[:, 1] = torch.clamp(self.points.data[:, 1],
                                                       self.mask_bounds[1, 0], self.mask_bounds[1, 1])
                    self.points.data[:, 2] = torch.clamp(self.points.data[:, 2],
                                                       self.mask_bounds[2, 0], self.mask_bounds[2, 1])
                    # 轻量级间距约束，不再做O(P²)检查，靠正则项保证
                    pass

                current_loss = loss.item()
                epoch_loss = current_loss
                if step % 10 == 0:
                    print(f"    Step {step}/40, loss: {current_loss:.6f}")

            scheduler.step()

            # 打印当前控制点范围，检查是否超出边界
            with torch.no_grad():
                z_min, z_max = self.points[:, 0].min().item(), self.points[:, 0].max().item()
                y_min, y_max = self.points[:, 1].min().item(), self.points[:, 1].max().item()
                x_min, x_max = self.points[:, 2].min().item(), self.points[:, 2].max().item()
            print(f"  Points range after epoch {iter+1}:")
            print(f"    Z: [{z_min:.3f}, {z_max:.3f}] (mask bounds: [{self.mask_bounds[0,0]:.3f}, {self.mask_bounds[0,1]:.3f}])")
            print(f"    Y: [{y_min:.3f}, {y_max:.3f}] (mask bounds: [{self.mask_bounds[1,0]:.3f}, {self.mask_bounds[1,1]:.3f}])")
            print(f"    X: [{x_min:.3f}, {x_max:.3f}] (mask bounds: [{self.mask_bounds[2,0]:.3f}, {self.mask_bounds[2,1]:.3f}])")

            obj_change = abs(prev_obj - epoch_loss) / (abs(prev_obj) + 1e-10)
            print(f"  Epoch objective: {epoch_loss:.6f}, change: {obj_change:.6f}\n")

            if obj_change < self.tol and iter > 5:
                print(f"Converged after {iter+1} iterations!")
                break

            prev_obj = epoch_loss

    def save_control_points(self, points, output_path, reference_image):
        """保存散点控制点为3D NIfTI图像"""
        point_img = np.zeros((self.H, self.W, self.D), dtype=np.float32)
        points_np = points.copy()

        # 转换到原图像坐标
        coords = (points_np * np.array([self.H-1, self.W-1, self.D-1])).astype(int)

        for (z, y, x) in coords:
            z_clamp = max(0, min(self.H-1, z))
            y_clamp = max(0, min(self.W-1, y))
            x_clamp = max(0, min(self.D-1, x))
            # 标记3x3x3的点
            for dz in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if 0 <= z_clamp + dz < self.H and 0 <= y_clamp + dy < self.W and 0 <= x_clamp + dx < self.D:
                            point_img[z_clamp + dz, y_clamp + dy, x_clamp + dx] = 1.0

        img = sitk.GetImageFromArray(point_img)
        img.CopyInformation(reference_image)
        sitk.WriteImage(img, output_path)
        print(f"Scatter control points saved to {output_path}")

    def get_bias_field(self):
        """获取完整全图3D偏置场，包含脑内和脑外所有位置（优化过程仍仅使用脑内体素）"""
        # 生成全图所有坐标（包含脑外区域）
        z_grid, y_grid, x_grid = np.meshgrid(
            np.arange(self.H),
            np.arange(self.W),
            np.arange(self.D),
            indexing='ij'  # 保持z,y,x顺序与原图像一致
        )

        # 转换为[0,1]范围的归一化坐标
        coords = torch.tensor(
            np.stack([
                z_grid.flatten()/(self.H-1),
                y_grid.flatten()/(self.W-1),
                x_grid.flatten()/(self.D-1)
            ], axis=1),
            dtype=torch.float32, device=self.device
        )

        # 批处理计算全图所有位置的偏置值
        bias = torch.zeros(coords.shape[0], device=self.device)
        infer_batch_size = self.batch_size // 4  # 调整批大小避免显存不足
        for i in range(0, coords.shape[0], infer_batch_size):
            end_idx = min(i + infer_batch_size, coords.shape[0])
            bias[i:end_idx] = self._eval_bias_field_batch(coords[i:end_idx])

        # 转换回3D图像格式
        bias_np = bias.detach().cpu().numpy()
        bias_field = bias_np.reshape(self.H, self.W, self.D)

        # 可选：轻微高斯平滑保证脑内外偏置场平滑过渡
        bias_field = gaussian_filter(bias_field, sigma=0.2)

        return bias_field

    def get_corrected_image(self):
        """获取矫正后图像"""
        bias = self.get_bias_field()
        corrected = self.image_np / np.clip(bias, 1e-10, 1e10)
        corrected[self.mask_np == 0] = self.image_np[self.mask_np == 0]
        return corrected

def main(args):
    # path to input image file
    input_path = args.input_img
    label_path = args.input_lab
    region_path = args.input_msk

    # path to output image file
    corrected_output_path = args.output_img
    bias_output_path = args.output_bias

    # load MRI image
    reference_image, image = load_mri_image(input_path)
    _, label = load_mri_image(label_path)
    _, region = load_mri_image(region_path)

    # ==================== B-spline单标签矫正分支 ====================
    if args.use_label_value is not None:
        print(f"Using B-spline bias field estimation with only label value: {args.use_label_value}")

        # 转换为SimpleITK图像
        itk_image = reference_image
        itk_mask = sitk.GetImageFromArray((label == args.use_label_value).astype(np.uint8))
        itk_mask.CopyInformation(itk_image)

        # 稍微平滑mask，避免边缘跳变
        itk_mask = sitk.DiscreteGaussian(itk_mask, 0.5)
        itk_mask = sitk.RescaleIntensity(itk_mask, 0, 1)

        # N4BiasFieldCorrection参数设置（基于B样条）
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaskLabel(1)
        corrector.SetNumberOfHistogramBins(200)

        # 设置B样条网格大小
        grid_size = [args.bspline_grid_size] * 3
        corrector.SetSplineOrder(3)
        corrector.SetNumberOfControlPoints(grid_size)

        # 多分辨率设置
        corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
        corrector.SetConvergenceThreshold(1e-6)

        # 执行矫正（只在指定标签区域优化）
        print("Running B-spline bias field correction...")
        corrected_image = corrector.Execute(itk_image, itk_mask)

        # 获取偏置场
        bias_field = corrector.GetLogBiasFieldAsImage(itk_image)
        bias_field = sitk.Exp(bias_field)  # 转换为真实偏置场（不是log域）

        # 保存结果
        sitk.WriteImage(corrected_image, corrected_output_path)
        sitk.WriteImage(bias_field, bias_output_path)

        print(f"Bias field saved to {bias_output_path}")
        print(f"Corrected image saved to {corrected_output_path}")
        return

    # ==================== 原有RBF矫正分支 ====================
    # 预处理
    region = np.where(region > 0, 1, 0)
    image_norm, min_scale, max_scale = image_preprocess(image, label)

    # 初始化矫正器
    corrector = ScatterBiasFieldCorrectorFast(
        image_norm, label, mask=region,
        n_points=args.n_points,
        max_iter=args.max_iter,
        tol=args.tol,
        use_soft_labels=not args.no_soft_labels,
        soft_sigma=args.soft_sigma,
        rbf_sigma=args.rbf_sigma,
        k_neighbors=args.k_neighbors,
        device=args.device
    )

    # Debug模式保存初始点
    if args.debug:
        init_path = "init_scatter_points.nii.gz" if args.output_points is None else args.output_points + "_init.nii.gz"
        corrector.save_control_points(corrector.init_points, init_path, reference_image)

    # 执行矫正
    corrector.fit()

    # Debug模式保存最终点
    if args.debug:
        final_path = "final_scatter_points.nii.gz" if args.output_points is None else args.output_points + "_final.nii.gz"
        corrector.save_control_points(corrector.points.detach().cpu().numpy(), final_path, reference_image)

    # 获取结果
    bias_field_norm = corrector.get_bias_field()
    corrected_norm = corrector.get_corrected_image()

    # 恢复到原始强度范围
    corrected_array = corrected_norm * (max_scale - min_scale) + min_scale
    corrected_array[label == 0] = image[label == 0]

    # 保存偏置场
    bias_image = sitk.GetImageFromArray(bias_field_norm.astype(np.float32))
    bias_image.CopyInformation(reference_image)
    sitk.WriteImage(bias_image, bias_output_path)
    print(f"Bias field saved to {bias_output_path}")

    # 保存矫正后图像
    corrected_image = sitk.GetImageFromArray(corrected_array.astype(np.float32))
    corrected_image.CopyInformation(reference_image)
    sitk.WriteImage(corrected_image, corrected_output_path)
    print(f"Corrected image saved to {corrected_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast GPU-accelerated bias field correction with scatter adaptive control points")
    parser.add_argument("--input_img", type=str, required=True, help="input image path")
    parser.add_argument("--input_lab", type=str, required=True, help="input label image path")
    parser.add_argument("--input_msk", type=str, required=True, help="input region mask path")

    parser.add_argument("--n_points", type=int, default=48, help="total number of scatter control points")
    parser.add_argument("--rbf_sigma", type=float, default=0.05, help="RBF kernel sigma (0.05-0.2)")
    parser.add_argument("--k_neighbors", type=int, default=48, help="number of nearest neighbors per voxel (K=48 gives same speed as grid version)")
    parser.add_argument("--batch_size", type=int, default=5000, help="batch size for voxel processing")
    parser.add_argument("--max_iter", type=int, default=25, help="maximum iterations")
    parser.add_argument("--tol", type=float, default=1e-6, help="convergence tolerance")
    parser.add_argument("--no_soft_labels", action='store_true', help="use hard labels instead of soft labels")
    parser.add_argument("--soft_sigma", type=float, default=0.5, help="sigma for soft label smoothing")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="computation device: 'cuda' or 'cpu'")
    parser.add_argument("--debug", action='store_true', help="enable debug mode, save control points")
    parser.add_argument("--output_points", type=str, default=None,
                       help="output prefix for control point images")
    parser.add_argument("--use_label_value", type=int, default=None,
                       help="Use only specified label value for B-spline bias field estimation (skip RBF method, e.g. 3 for gray matter)")
    parser.add_argument("--bspline_grid_size", type=int, default=5,
                       help="B-spline control point grid size per dimension, larger = more flexible bias field")

    parser.add_argument("--output_img", type=str, required=True, help="corrected image output path")
    parser.add_argument("--output_bias", type=str, required=True, help="bias field output path")

    args = parser.parse_args()

    main(args=args)
