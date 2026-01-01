import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import datetime
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# ---------------------------------------------------------
# 0. ページ設定
# ---------------------------------------------------------
st.set_page_config(page_title="Ultimate Sci-Stat & Graph Engine", layout="wide")

# タイトルと説明
st.title("🔬 Ultimate Sci-Stat & Graph Engine")
st.markdown("""
**統計解析から論文グレードのグラフ作成までを自動化する統合ツール**
1. データを入力 → 2. 自動診断と統計解析 → 3. 有意差バー付きグラフの自動生成
""")

# ---------------------------------------------------------
# 1. サイドバー設定 (グラフとデザイン)
# ---------------------------------------------------------
with st.sidebar:
    st.header("🛠️ グラフ設定")
    
    with st.expander("📈 グラフの種類", expanded=True):
        graph_type = st.selectbox("形式", ["棒グラフ (Bar)", "箱ひげ図 (Box)", "バイオリン図 (Violin)"])
        if "棒" in graph_type:
            error_type = st.radio("エラーバー", ["SD (標準偏差)", "SEM (標準誤差)"])
        else:
            error_type = "None"
        
    with st.expander("🎨 デザイン調整", expanded=True):
        fig_title = st.text_input("図のタイトル", value="Experiment Result")
        y_axis_label = st.text_input("Y軸ラベル", value="Relative Value")
        bar_width = st.slider("棒/箱の太さ", 0.1, 1.0, 0.6)
        dot_size = st.slider("ドットサイズ (0で非表示)", 0, 100, 20)
        fig_height = st.slider("画像の高さ", 3.0, 10.0, 5.0)
        
    st.divider()
    st.markdown("### 📢 Notice")
    st.caption("本ツールはベータ版です。論文等に使用する際は開発者までご連絡ください。")

# ---------------------------------------------------------
# 2. データ入力セクション
# ---------------------------------------------------------
if 'g_count' not in st.session_state: st.session_state.g_count = 3

st.subheader("1. データ入力")
col_ctrl, _ = st.columns([1, 5])
with col_ctrl:
    if st.button("＋ 群を追加"): st.session_state.g_count += 1
    if st.session_state.g_count > 2 and st.button("－ 群を削除"): st.session_state.g_count -= 1

# 動的カラム生成
data_dict = {}
cols = st.columns(min(st.session_state.g_count, 4))
for i in range(st.session_state.g_count):
    with cols[i % 4]:
        def_name = f"Group {i+1}"
        name = st.text_input(f"名前 {i+1}", value=def_name, key=f"n{i}")
        raw = st.text_area(f"データ {i+1}", height=120, key=f"d{i}", placeholder="10.5\n12.3\n...")
        vals = [float(x.strip()) for x in raw.replace(',', '\n').split('\n') if x.strip()]
        if len(vals) > 0: data_dict[name] = vals

st.divider()

# ---------------------------------------------------------
# 3. 統計解析エンジン (Logic Core)
# ---------------------------------------------------------
# 有意差ラベル生成関数
def get_sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

sig_pairs = [] # 有意差ペアを保存するリスト [{'g1':Name, 'g2':Name, 'label':'*', 'p':0.03}, ...]

if len(data_dict) >= 2:
    st.header("2. 統計解析レポート")
    
    # データの準備
    group_names = list(data_dict.keys())
    all_values = list(data_dict.values())
    
    # A. 正規性診断 (Shapiro-Wilk)
    all_normal = True
    for v in all_values:
        if len(v) >= 3:
            _, p_s = stats.shapiro(v)
            if p_s <= 0.05: all_normal = False
            
    # B. 等分散性診断 (Levene)
    try:
        _, p_lev = stats.levene(*all_values)
        is_equal_var = (p_lev > 0.05)
    except:
        is_equal_var = True # データ不足時など

    # C. 検定の自動選択と実行
    method_name = ""
    p_global = 1.0
    
    # --- 2群比較 ---
    if len(data_dict) == 2:
        g1, g2 = all_values[0], all_values[1]
        if all_normal:
            method_name = "Student's t-test" if is_equal_var else "Welch's t-test"
            _, p_global = stats.ttest_ind(g1, g2, equal_var=is_equal_var)
        else:
            method_name = "Mann-Whitney U test"
            _, p_global = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            
        if p_global < 0.05:
            sig_pairs.append({'g1': group_names[0], 'g2': group_names[1], 'label': get_sig_label(p_global), 'p': p_global})

    # --- 3群以上比較 ---
    else:
        if all_normal and is_equal_var:
            # Parametric: ANOVA + Tukey
            method_name = "One-way ANOVA + Tukey's HSD"
            _, p_global = stats.f_oneway(*all_values)
            
            if p_global < 0.05:
                flat_data = [v for sub in all_values for v in sub]
                labels = [n for n, sub in data_dict.items() for _ in sub]
                res = pairwise_tukeyhsd(flat_data, labels)
                
                # Tukey結果の抽出
                df_res = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
                for _, row in df_res.iterrows():
                    if row['reject']:
                        sig_pairs.append({'g1': row['group1'], 'g2': row['group2'], 'label': get_sig_label(row['p-adj']), 'p': row['p-adj']})
        else:
            # Non-parametric: Kruskal-Wallis + Dunn
            method_name = "Kruskal-Wallis + Dunn's test"
            _, p_global = stats.kruskal(*all_values)
            
            if p_global < 0.05:
                dunn = sp.posthoc_dunn(all_values, p_adjust='bonferroni')
                dunn.columns = group_names
                dunn.index = group_names
                
                # ペアごとの抽出
                for i in range(len(group_names)):
                    for j in range(i+1, len(group_names)):
                        n1, n2 = group_names[i], group_names[j]
                        p_val = dunn.loc[n1, n2]
                        if p_val < 0.05:
                            sig_pairs.append({'g1': n1, 'g2': n2, 'label': get_sig_label(p_val), 'p': p_val})

    # レポート表示
    st.success(f"**採用された手法: {method_name}**")
    st.write(f"全体P値: {p_global:.4e} ({'有意差あり' if p_global < 0.05 else '有意差なし'})")
    
    with st.expander("詳細な解析レポート (先生への説明用)"):
        report = f"""
        1. データ診断:
           正規性: {'あり (パラメトリック検定推奨)' if all_normal else 'なし (ノンパラメトリック検定推奨)'}
           等分散性: {'あり' if is_equal_var else 'なし'}
        
        2. 選択された検定: {method_name}
           理由: データの分布とバラツキに基づき、最も妥当な手法を自動選択しました。
           
        3. 結果:
           Global P-value: {p_global:.4e}
           有意差のあるペア: {len(sig_pairs)} 組
        """
        st.text_area("レポート", report, height=200)

    st.divider()

# ---------------------------------------------------------
# 4. グラフ描画エンジン (Visualization Core)
# ---------------------------------------------------------
if len(data_dict) >= 1:
    st.header("3. グラフ生成 (Auto-Labeling)")
    
    try:
        # matplotlib設定
        plt.rcParams['font.family'] = 'sans-serif'
        fig, ax = plt.subplots(figsize=(6, fig_height))
        
        # 配色と座標
        x_positions = np.arange(len(group_names))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(group_names)))
        
        # Y軸の最大値を計算 (バーの高さ調整用)
        max_val = -np.inf
        for v in all_values:
            if len(v) > 0: max_val = max(max_val, max(v))
        if max_val == -np.inf: max_val = 1
        
        # --- A. プロット描画 ---
        for i, (name, vals) in enumerate(data_dict.items()):
            if len(vals) == 0: continue
            vals = np.array(vals)
            
            mean_v = np.mean(vals)
            std_v = np.std(vals, ddof=1) if len(vals) > 1 else 0
            sem_v = std_v / np.sqrt(len(vals)) if len(vals) > 0 else 0
            err = sem_v if error_type == "SEM" else std_v
            
            if "棒" in graph_type:
                ax.bar(i, mean_v, width=bar_width, color=colors[i], edgecolor='black', alpha=0.7, zorder=1)
                ax.errorbar(i, mean_v, yerr=err, fmt='none', color='black', capsize=5, zorder=2)
            elif "箱" in graph_type:
                ax.boxplot(vals, positions=[i], widths=bar_width, patch_artist=True,
                           boxprops=dict(facecolor=colors[i]), medianprops=dict(color='black'), showfliers=False)
            elif "バイオリン" in graph_type:
                parts = ax.violinplot(vals, positions=[i], widths=bar_width, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)
            
            # ドットプロット (Jitter)
            if dot_size > 0:
                noise = np.random.normal(0, 0.04, len(vals))
                ax.scatter(x_positions[i] + noise, vals, s=dot_size, color='white', edgecolor='gray', zorder=3, alpha=0.8)

        # --- B. 有意差バーの自動描画 (Auto-Bracket) ---
        # バーの高さを管理
        y_step = max_val * 0.15 # バーごとの高さの積み上げ幅
        current_y = max_val * 1.1 # 最初のバーの高さ
        
        # 有意差ペアをループ
        for pair in sig_pairs:
            try:
                idx1 = group_names.index(pair['g1'])
                idx2 = group_names.index(pair['g2'])
                
                # 描画座標
                x1, x2 = idx1, idx2
                bar_h = current_y
                col_h = max_val * 0.03 # 脚の長さ
                
                # コの字型ライン
                ax.plot([x1, x1, x2, x2], [bar_h-col_h, bar_h, bar_h, bar_h-col_h], lw=1.5, c='black')
                # ラベル (*, **, ***)
                ax.text((x1+x2)/2, bar_h, pair['label'], ha='center', va='bottom', fontsize=14)
                
                # 次のバーのために高さを上げる
                current_y += y_step
            except:
                pass

        # --- C. レイアウト仕上げ ---
        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_names, fontsize=12)
        ax.set_ylabel(y_axis_label, fontsize=12)
        ax.set_title(fig_title, fontsize=14)
        ax.set_ylim(0, current_y * 1.05) # 上限をバーに合わせて調整
        
        # 枠線をシンプルに
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
        
        # ダウンロード
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button("📥 高解像度画像を保存 (PNG)", data=img_buf, file_name=f"result_{now_str}.png", mime="image/png")
        
    except Exception as e:
        st.error(f"描画エラー: {e}")

else:
    st.info("データが入力されていません。")
