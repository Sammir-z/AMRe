"""
特征可视化脚本
使用t-SNE绘制模态特征演变图，按类别标签着色显示聚类效果
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle
import os
import argparse
from typing import Dict, List
import matplotlib.cm as cm


def load_features(feature_dir: str, epoch: int) -> Dict:
    """加载指定epoch的特征"""
    feature_path = os.path.join(feature_dir, f'features_epoch_{epoch}.pkl')
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Features file not found: {feature_path}")
    
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def reduce_dimensions(features: np.ndarray, method: str = 'tsne', 
                     n_components: int = 2, perplexity: int = 30,
                     random_state: int = 42) -> np.ndarray:
    """
    使用t-SNE或PCA进行降维
    
    Args:
        features: 高维特征 (N, D)
        method: 'tsne' 或 'pca'
        n_components: 降维后的维度
        perplexity: t-SNE的perplexity参数
        random_state: 随机种子
        
    Returns:
        降维后的特征 (N, n_components)
    """
    if method == 'tsne':
        # 如果特征维度太高，先用PCA降到50维
        if features.shape[1] > 50:
            print(f"    Feature dim {features.shape[1]} > 50, applying PCA first...")
            pca = PCA(n_components=50, random_state=random_state)
            features = pca.fit_transform(features)
        
        print(f"    Applying t-SNE (perplexity={perplexity})...")
        reducer = TSNE(n_components=n_components, perplexity=perplexity, 
                      random_state=random_state, n_iter=1000)
    else:
        print(f"    Applying PCA...")
        reducer = PCA(n_components=n_components, random_state=random_state)
    
    reduced_features = reducer.fit_transform(features)
    return reduced_features


def plot_single_epoch(ax, features_dict: Dict[str, np.ndarray], 
                     labels: np.ndarray,
                     modality_names: List[str], 
                     epoch: int,
                     marker_size: int = 8,
                     alpha: float = 0.7):
    """
    绘制单个epoch的特征分布（按模态着色）
    
    Args:
        ax: matplotlib轴对象
        features_dict: 包含各模态2D特征的字典
        labels: 类别标签（未使用）
        modality_names: 模态名称列表
        epoch: epoch编号
        marker_size: 点的大小
        alpha: 透明度
    """
    # 设置背景为白色
    ax.set_facecolor('white')
    
    # 定义模态颜色
    modality_colors = {
        'Fusion': '#FF8C00',   # 深橙色
        'Audio': '#1E90FF',    # 道奇蓝
        'Visual': '#32CD32',   # 酸橙绿
        'Image': '#32CD32',    # 酸橙绿（与Visual相同）
        'Text': '#DC143C',     # 深红色
    }
    
    # 绘制各模态特征（每个模态使用固定颜色）
    for name in modality_names:
        ax.scatter(features_dict[name][:, 0], 
                  features_dict[name][:, 1],
                  c=modality_colors.get(name, '#808080'),  # 默认灰色
                  s=marker_size, 
                  alpha=alpha * 0.5,
                  edgecolors='none',
                  label=name)
    
    # 绘制fusion特征
    ax.scatter(features_dict['fusion'][:, 0], 
              features_dict['fusion'][:, 1],
              c=modality_colors['Fusion'],
              s=marker_size * 1.5,
              alpha=alpha,
              edgecolors='none',
              label='Fusion',
              zorder=3)
    
    # 设置标题和样式
    ax.set_title(f'Epoch={epoch}', fontsize=14, fontweight='bold', 
                pad=10, color='black')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 隐藏边框
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_feature_evolution(feature_dir: str, 
                          epochs: List[int],
                          modality_names: List[str],
                          save_path: str = None,
                          method: str = 'tsne',
                          perplexity: int = 30,
                          figsize: tuple = (18, 10),
                          max_samples: int = 1000):
    """
    绘制特征演变图（类似论文Figure 4）
    
    Args:
        feature_dir: 特征保存目录
        epochs: 要绘制的epoch列表，如 [0, 10, 20, 40, 70, 90]
        modality_names: 模态名称列表
        save_path: 保存路径
        method: 降维方法 ('tsne' 或 'pca')
        perplexity: t-SNE的perplexity参数
        figsize: 图像大小
        max_samples: 每个epoch最多使用的样本数
    """
    # 创建子图
    n_epochs = len(epochs)
    n_cols = 3
    n_rows = (n_epochs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                            facecolor='white', dpi=100)
    axes = axes.flatten() if n_epochs > 1 else [axes]
    
    # 为每个epoch绘制特征分布
    for idx, epoch in enumerate(epochs):
        print(f"\nProcessing Epoch {epoch}...")
        
        # 加载特征
        try:
            features_data = load_features(feature_dir, epoch)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            axes[idx].text(0.5, 0.5, f'Epoch={epoch}\nData not found', 
                         ha='center', va='center', fontsize=12)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue
        
        # 随机采样（如果样本太多）
        n_samples = features_data['fusion'].shape[0]
        if n_samples > max_samples:
            print(f"  Sampling {max_samples} from {n_samples} samples...")
            indices = np.random.choice(n_samples, max_samples, replace=False)
            features_data = {k: v[indices] for k, v in features_data.items()}
        
        # 分别对fusion特征和各模态特征进行降维（维度可能不同）
        reduced_dict = {}
        
        # 降维fusion特征
        print(f"  Reducing fusion features (dim={features_data['fusion'].shape[1]})...")
        reduced_dict['fusion'] = reduce_dimensions(features_data['fusion'], 
                                                   method=method, 
                                                   perplexity=perplexity)
        
        # 检查所有模态特征是否维度相同
        modality_dims = [features_data[name].shape[1] for name in modality_names]
        if len(set(modality_dims)) == 1:
            # 所有模态维度相同，可以合并降维
            print(f"  Reducing modality features together (dim={modality_dims[0]})...")
            modality_features = np.concatenate([features_data[name] for name in modality_names], axis=0)
            reduced_modalities = reduce_dimensions(modality_features, method=method, perplexity=perplexity)
            
            # 分离各模态
            n_per_modality = features_data[modality_names[0]].shape[0]
            for i, name in enumerate(modality_names):
                start_idx = i * n_per_modality
                reduced_dict[name] = reduced_modalities[start_idx:start_idx + n_per_modality]
        else:
            # 维度不同，分别降维
            print(f"  Modality dimensions differ: {dict(zip(modality_names, modality_dims))}")
            for name in modality_names:
                print(f"  Reducing {name} features (dim={features_data[name].shape[1]})...")
                reduced_dict[name] = reduce_dimensions(features_data[name], 
                                                      method=method, 
                                                      perplexity=perplexity)
        
        # 绘制（使用小点和高透明度）
        plot_single_epoch(axes[idx], reduced_dict, features_data['labels'],
                         modality_names, epoch, 
                         marker_size=8, alpha=0.7)
    
    # 隐藏多余的子图
    for idx in range(n_epochs, len(axes)):
        axes[idx].axis('off')
    
    # 添加图例（显示各模态颜色）
    import matplotlib.patches as mpatches
    modality_colors = {
        'Fusion': '#FF8C00',
        'Audio': '#1E90FF',
        'Visual': '#32CD32',
        'Image': '#32CD32',
        'Text': '#DC143C',
    }
    
    legend_elements = [mpatches.Patch(facecolor=modality_colors['Fusion'], 
                                     label='Fusion', edgecolor='none')]
    for name in modality_names:
        legend_elements.append(
            mpatches.Patch(facecolor=modality_colors.get(name, '#808080'), 
                          label=name, edgecolor='none')
        )
    
    fig.legend(handles=legend_elements, loc='upper right', 
              fontsize=12, frameon=True, ncol=len(modality_names)+1,
              bbox_to_anchor=(0.98, 0.98), framealpha=0.9)
    
    # 添加简洁的总标题
    plt.suptitle('Evolution of Modality-Specific Feature Distributions',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=1.5, h_pad=2.5, w_pad=2.0)
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize feature evolution over training epochs')
    parser.add_argument('--feature_dir', type=str, required=True,
                       help='Directory containing saved features')
    parser.add_argument('--epochs', type=int, nargs='+', default=[1, 10, 20, 30, 40, 50, 60],
                       help='Epochs to visualize')
    parser.add_argument('--modalities', type=str, nargs='+', 
                       default=['Visual', 'Audio'],
                       help='Modality names, e.g., Visual Audio or Image Text')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the figure')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'],
                       help='Dimensionality reduction method')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--max_samples', type=int, default=2000,
                       help='Maximum samples per epoch')
    
    args = parser.parse_args()
    
    # 如果没有指定保存路径，自动生成
    if args.save_path is None:
        dataset_name = os.path.basename(args.feature_dir).split('_')[0]
        args.save_path = os.path.join(args.feature_dir, 
                                     f'feature_evolution_{dataset_name}.png')
    
    print("="*80)
    print("Feature Evolution Visualization")
    print("="*80)
    print(f"Feature directory: {args.feature_dir}")
    print(f"Epochs to plot: {args.epochs}")
    print(f"Modalities: {args.modalities}")
    print(f"Method: {args.method}")
    print(f"Save path: {args.save_path}")
    print("="*80)
    
    plot_feature_evolution(
        feature_dir=args.feature_dir,
        epochs=args.epochs,
        modality_names=args.modalities,
        save_path=args.save_path,
        method=args.method,
        perplexity=args.perplexity,
        max_samples=args.max_samples
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
