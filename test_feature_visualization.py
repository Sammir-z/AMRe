"""
特征可视化快速测试脚本
用于验证特征收集和可视化功能是否正常工作
"""

import torch
import numpy as np
import os
import sys
import pickle

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.feature_extractor import FeatureCollector


def test_feature_collector():
    """测试特征收集器"""
    print("="*60)
    print("测试特征收集器")
    print("="*60)
    
    # 创建临时目录
    save_dir = "test_features"
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化收集器
    modality_names = ["Visual", "Audio"]
    collector = FeatureCollector(save_dir, modality_names)
    
    # 模拟几个batch的特征
    batch_size = 32
    feature_dim = 128
    n_batches = 5
    
    print(f"\n模拟收集 {n_batches} 个batch的特征...")
    for epoch in [0, 10, 20]:
        print(f"\nEpoch {epoch}:")
        for batch in range(n_batches):
            # 模拟特征
            fusion_feat = torch.randn(batch_size, feature_dim)
            visual_feat = torch.randn(batch_size, feature_dim)
            audio_feat = torch.randn(batch_size, feature_dim)
            labels = torch.randint(0, 6, (batch_size,))
            
            # 收集特征
            collector.collect_features(
                epoch=epoch,
                fusion_features=fusion_feat,
                modality_features=[visual_feat, audio_feat],
                labels=labels
            )
        
        # 保存epoch特征
        collector.save_epoch_features(epoch)
        
        # 验证文件是否存在
        saved_file = os.path.join(save_dir, f"features_epoch_{epoch}.pkl")
        assert os.path.exists(saved_file), f"文件未创建: {saved_file}"
        
        # 加载并验证
        loaded_data = collector.load_epoch_features(epoch)
        expected_samples = batch_size * n_batches
        
        assert loaded_data['fusion'].shape[0] == expected_samples, \
            f"样本数不匹配: {loaded_data['fusion'].shape[0]} vs {expected_samples}"
        assert loaded_data['Visual'].shape[0] == expected_samples
        assert loaded_data['Audio'].shape[0] == expected_samples
        
        print(f"  ✓ Epoch {epoch} 特征保存和加载成功")
        print(f"    总样本数: {expected_samples}")
        print(f"    特征维度: {feature_dim}")
    
    print("\n✓ 特征收集器测试通过!")
    
    # 清理
    import shutil
    shutil.rmtree(save_dir)
    print(f"✓ 清理临时目录: {save_dir}")


def test_visualization_pipeline():
    """测试可视化流程"""
    print("\n" + "="*60)
    print("测试可视化流程")
    print("="*60)
    
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        print("✓ 所需库已安装")
    except ImportError as e:
        print(f"✗ 缺少依赖库: {e}")
        print("请安装: pip install scikit-learn matplotlib")
        return
    
    # 创建模拟数据
    print("\n创建模拟特征数据...")
    n_samples = 200
    n_features = 64
    
    # 模拟3个模态的特征（有一定聚类结构）
    np.random.seed(42)
    
    # 创建3个聚类中心
    centers = [
        np.array([0, 0]),
        np.array([5, 5]),
        np.array([10, 0])
    ]
    
    features_2d = []
    labels = []
    for i, center in enumerate(centers):
        cluster_points = np.random.randn(n_samples // 3, 2) + center
        features_2d.append(cluster_points)
        labels.extend([i] * (n_samples // 3))
    
    features_2d = np.vstack(features_2d)
    labels = np.array(labels)
    
    # 升维到高维空间（模拟真实特征）
    random_matrix = np.random.randn(2, n_features)
    fusion_features = features_2d @ random_matrix + np.random.randn(n_samples, n_features) * 0.1
    
    # 为不同模态添加一些噪声
    visual_features = fusion_features + np.random.randn(n_samples, n_features) * 0.5
    audio_features = fusion_features + np.random.randn(n_samples, n_features) * 0.5
    
    print(f"  Fusion特征: {fusion_features.shape}")
    print(f"  Visual特征: {visual_features.shape}")
    print(f"  Audio特征: {audio_features.shape}")
    
    # 测试PCA降维
    print("\n测试PCA降维...")
    pca = PCA(n_components=2, random_state=42)
    fusion_2d_pca = pca.fit_transform(fusion_features)
    print(f"  ✓ PCA降维成功: {fusion_2d_pca.shape}")
    
    # 测试t-SNE降维
    print("\n测试t-SNE降维（可能需要几秒）...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=300)
    fusion_2d_tsne = tsne.fit_transform(fusion_features[:100])  # 只用部分数据测试
    print(f"  ✓ t-SNE降维成功: {fusion_2d_tsne.shape}")
    
    # 测试绘图
    print("\n测试绘图功能...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PCA结果
    axes[0].scatter(fusion_2d_pca[:, 0], fusion_2d_pca[:, 1], 
                   c=labels, cmap='viridis', alpha=0.6)
    axes[0].set_title('PCA')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # t-SNE结果
    axes[1].scatter(fusion_2d_tsne[:, 0], fusion_2d_tsne[:, 1], 
                   c=labels[:100], cmap='viridis', alpha=0.6)
    axes[1].set_title('t-SNE')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.tight_layout()
    
    # 保存测试图像
    test_save_path = "test_visualization.png"
    plt.savefig(test_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 测试图像已保存: {test_save_path}")
    print(f"  提示: 可以打开 {test_save_path} 查看效果")
    
    print("\n✓ 可视化流程测试通过!")
    
    # 清理
    if os.path.exists(test_save_path):
        print(f"✓ 可以删除测试图像: {test_save_path}")


def main():
    print("\n" + "="*60)
    print("特征可视化功能测试")
    print("="*60)
    
    # 测试1: 特征收集器
    try:
        test_feature_collector()
    except Exception as e:
        print(f"\n✗ 特征收集器测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 可视化流程
    try:
        test_visualization_pipeline()
    except Exception as e:
        print(f"\n✗ 可视化流程测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    print("\n如果所有测试通过，说明特征可视化功能可以正常使用。")
    print("现在可以在训练脚本中启用 --save_features True 来保存特征。")
    print("="*60)


if __name__ == '__main__':
    main()
