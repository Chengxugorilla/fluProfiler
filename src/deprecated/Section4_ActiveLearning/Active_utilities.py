import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def cluster_virus(df, vector, drop_duplicate=False):

    if drop_duplicate:
        df_arr = pd.DataFrame(vector, columns=[f'col_{i}' for i in range(vector.shape[1])])
        df_merged = pd.concat([df.reset_index(drop=True), df_arr], axis=1)
        df_merged["virusDate"] = pd.to_datetime(df_merged["virusDate"], errors="coerce")
        
        aggregate_dict = {col: 'first' for col in df_merged.columns if col != 'seq_c'}
        aggregate_dict['virusDate'] = 'min'
        df_agg = df_merged.groupby('seq_c').agg(aggregate_dict).reset_index()
        df = df_agg.iloc[:, :-2560]
        vector = np.array(df_agg.iloc[:, -2560:])

    # elbow_method(vector)
    
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(vector)

    k = 2
    labels = KMeans(n_clusters=k, random_state=441).fit_predict(X_2d)
    
    df_plot = pd.DataFrame(X_2d, columns=['x', 'y'])
    df_plot['cluster'] = labels.astype(str)
    df_plot['type'] = df['Type'].tolist()
    df_plot['virusDate'] = df['virusDate']
    df_plot['virusIslID'] = df['virusIslID'].tolist()

    return df_plot

def plot_virus_scatter(df_plot, color_map=None, save_path=None):
    plt.figure(figsize=(5, 3.5))

    order = ['Crick', 'CDC', 'CNIC']  # 你想要的顺序

    for t in order:
        subdf = df_plot[df_plot['type'] == t]  # 筛选对应类型的数据
        if len(subdf) == 0:
            continue  # 防止某类不存在时报错

        plt.scatter(subdf["x"], subdf["y"],
                    label=t,
                    c=color_map.get(t, "gray"),
                    s=8, alpha=0.45, lw=0)

    # ===== 样式 =====
    plt.xlabel("component-1", fontsize=14)
    plt.ylabel("component-2", fontsize=14)
    
    plt.legend(frameon=False, fontsize=10, loc='upper left', bbox_to_anchor=(0.98, 1), 
    markerscale=2, alignment='left')

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    if save_path is not None:
        ext = os.path.splitext(save_path)[1].lower()

        if ext == '.png':
            plt.savefig(save_path, dpi=600, transparent=True,
                        pil_kwargs={"compress_level": 0}, bbox_inches="tight")
        elif ext == '.svg':
            plt.savefig(save_path, format='svg', bbox_inches="tight")
    else:
        plt.show()

def plot_virus_scatter_time(df_plot, save_path=None):
    """
    根据virusDate由浅到深绘制散点（自然色渐变：蓝→绿→黄）。
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # 确保时间格式并去掉NaT
    df_plot = df_plot.copy()
    df_plot["virusDate"] = pd.to_datetime(df_plot["virusDate"], errors="coerce")
    df_plot = df_plot.dropna(subset=["virusDate"])
    if df_plot.empty:
        raise ValueError("virusDate 列全部为空或无法解析为日期")

    # 按时间排序（保证后期点覆盖早期点）
    df_plot = df_plot.sort_values("virusDate")

    # 时间归一化（排除极端值，避免色带压缩）
    vmin = df_plot["virusDate"].quantile(0.05).timestamp()
    vmax = df_plot["virusDate"].quantile(0.95).timestamp()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # 自然色 colormap
    cmap = cm.get_cmap("cividis")

    # 绘制散点
    sc = ax.scatter(df_plot["x"], df_plot["y"],
                    c=df_plot["virusDate"].apply(lambda x: x.timestamp()),
                    cmap=cmap, norm=norm,
                    s=12, alpha=0.6, lw=0)

    # 颜色条
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(pd.to_datetime(ticks, unit="s").strftime("%Y"))  # 只显示年份
    cbar.set_label("virusDate", fontsize=10)

    # 样式
    ax.set_xlabel("component-1", fontsize=14)
    ax.set_ylabel("component-2", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, transparent=True,
                    bbox_inches="tight", pil_kwargs={"compress_level": 0})
    else:
        plt.show()

def save_max_per_file(emb_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for key, emb in emb_dict.items():
        # emb: 形状 (N, D) 的 tensor
        # torch.max(dim=0) 返回 (values, indices)，我们只要 values
        max_vec = emb.max(dim=0).values          # 形状 (D,)

        out_path = os.path.join(out_dir, f"{key}_max.pt")
        torch.save(max_vec.cpu(), out_path)      # 建议先 .cpu() 再保存
